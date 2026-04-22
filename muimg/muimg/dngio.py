"""DNG file format support for Allsky.

This module handles DNG file creation and related functionality.
"""
import io
import logging
import imagecodecs
import numpy as np
from datetime import datetime

from enum import Enum, IntEnum
from pathlib import Path
from tifffile import COMPRESSION, TiffFile, TiffPage, TiffWriter
from typing import Optional, Union, List, Dict, Tuple, Any, Type, IO
from dataclasses import dataclass, replace

from . import raw_render
from .raw_render import DemosaicAlgorithm

# Import metadata classes from tiff_metadata module
from .tiff_metadata import (
    MetadataTags,
    TIFF_TAG_TYPE_REGISTRY,
    XmpMetadata,
    convert_tag_value,
    resolve_tag,
    filter_tags_by_ifd_category,
)
# tifffile followups:
# - dng_validate expects SubIFD NextIFD == 0, but tifffile writes NextIFD chaining for SubIFDs and does not expose a supported way to force it to zero.
# - Copying compressed tiled pages is not always possible (e.g. tile size / alignment constraints) and can require a decode + re-encode fallback, which currently emits a warning.

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

class RawStageSelector(str, Enum):
    """Raw processing stage selectors."""
    RAW = "raw"
    LINEARIZED = "linearized"
    LINEARIZED_PLUS_OPS = "linearized_plus_ops"
    
    @classmethod
    def lookup(cls, value: str) -> "RawStageSelector":
        """Look up enum member by string value."""
        from .common import enum_from_string
        return enum_from_string(cls, value)

class SubFileType(IntEnum):
    """NewSubFileType values from DNG spec."""
    MAIN_IMAGE = 0
    PREVIEW_IMAGE = 1
    TRANSPARENCY_MASK = 4
    DEPTH_MAP = 8
    ALT_PREVIEW_IMAGE = 65537
    
    @classmethod
    def lookup(cls, value: int) -> Optional["SubFileType"]:
        """Look up enum member by integer value."""
        from .common import enum_from_value
        return enum_from_value(cls, value)



# =============================================================================
# Core DNG Classes
# =============================================================================

class DngPage(TiffPage):
    """TiffPage subclass with DNG-specific functionality.
    
    Provides convenient access to DNG tags with automatic translation,
    parent IFD tag inheritance, and raw data extraction methods.
    
    Inherits all TiffPage attributes and methods. Created by "upgrading"
    an existing TiffPage instance.
    """
    
    @dataclass(frozen=True)
    class _RawStage:
        data: np.ndarray
        cfa_pattern: Optional[str] = None
    
    def __new__(cls, *args, **kwargs):
        """Create DngPage instance without calling TiffPage.__init__."""
        return object.__new__(cls)
    
    def __init__(self, tiff_page: TiffPage):
        """Initialize DngPage by copying TiffPage state.
        
        Args:
            tiff_page: The TiffPage to upgrade
        """
        self.__dict__.update(tiff_page.__dict__)
        # Cache whether this is IFD0 (check if it's the first page in parent)
        parent_file = self.__dict__.get('parent')
        if parent_file and hasattr(parent_file, 'pages') and parent_file.pages:
            self._is_ifd0 = parent_file.pages[0] is tiff_page
        else:
            self._is_ifd0 = True  # No parent or pages, assume IFD0
    
    @property
    def parent(self) -> "DngFile":
        """Return parent DngFile (overrides TiffPage.parent for type consistency)."""
        # Access the underlying TiffFile parent and ensure it's a DngFile
        tiff_parent = self.__dict__['parent']
        if not isinstance(tiff_parent, DngFile):
            # This shouldn't happen in normal usage, but handle it gracefully
            return tiff_parent
        return tiff_parent
    
    @parent.setter
    def parent(self, value):
        """Allow TiffPage to set parent during initialization."""
        self.__dict__['parent'] = value
    
    @property
    def ifd0(self) -> "DngPage":
        """Return IFD0 (returns self if this page IS IFD0)."""
        # Use DngFile.ifd0 property which properly wraps pages[0]
        return self.parent.ifd0

    @property
    def is_ifd0(self) -> bool:
        """True if this page is IFD0 (top-level IFD, not a SubIFD)."""
        return self._is_ifd0
    
    @property
    def photometric_name(self) -> Optional[str]:
        """Photometric interpretation string (e.g., 'CFA', 'LINEAR_RAW', 'RGB')."""
        if self.photometric is not None:
            return self.photometric.name
        return None
    
    @property
    def is_cfa(self) -> bool:
        """True if this page contains CFA (Bayer) raw data."""
        return self.photometric_name == "CFA"
    
    @property
    def is_linear_raw(self) -> bool:
        """True if this page contains LINEAR_RAW (demosaiced) data."""
        return self.photometric_name == "LINEAR_RAW"
    
    @property
    def is_main_image(self) -> bool:
        """True if this page is the main image (NewSubfileType == 0)."""
        value = self.get_tag("NewSubfileType")
        if value is None:
            return False
        return value == SubFileType.MAIN_IMAGE
    
    @property
    def is_preview(self) -> bool:
        """True if this page is a preview image."""
        value = self.get_tag("NewSubfileType") 
        if value is None:
            return False
        return value in (SubFileType.PREVIEW_IMAGE, SubFileType.ALT_PREVIEW_IMAGE)

    def get_rendered_size(self) -> Tuple[int, int]:
        """Get the final dimensions after DefaultCrop is applied.
        
        Returns:
            Tuple of (width, height) after crop is applied.
            If DefaultCropSize is not present, returns (imagewidth, imagelength).
        """
        crop_size = self.get_tag("DefaultCropSize")
        if crop_size is not None:
            return int(crop_size[0]), int(crop_size[1])
        return self.imagewidth or 0, self.imagelength or 0

    def get_xmp(self) -> Optional[XmpMetadata]:
        """Return XMP metadata as an `XmpMetadata` object."""
        xmp = self.get_tag("XMP")
        return xmp

    def _get_tag_object(self, tag: Union[str, int]) -> Optional[tuple]:
        """Internal helper to get tag object and metadata with normalized values.
        
        All tag values are normalized to system byte order for consistent internal
        representation. This includes:
        - Multi-byte typed arrays (SHORT, LONG, FLOAT, etc.)
        - ProfileGainTableMap/ProfileGainTableMap2 binary blobs
        
        Args:
            tag: Either a numeric tag code (int) or tag name string
        
        Returns:
            Tuple of (tag_id, tag_name, registry_spec, raw_tag, normalized_value) or None if not found.
        """        
        tag_id, tag_name, spec = resolve_tag(tag)
        
        if tag_id is None:
            logger.warning(f"Tag '{tag}' not found in LOCAL_TIFF_TAGS.")
            return None
        
        raw_tag = None
        if tag_id in self.tags:
            raw_tag = self.tags[tag_id]
        
        if raw_tag is None:
            return None
        
        # First handle PGTM special case (needs explicit source/target byte order)
        from .raw_render import transcode_pgtm_if_needed
        val = transcode_pgtm_if_needed(raw_tag.code, raw_tag.value, self.parent.byteorder, '=')

        # Then normalize any multi-byte arrays
        from .tiff_metadata import normalize_array_to_target_byteorder
        normalized_val = normalize_array_to_target_byteorder(val, '=')
        
        return (tag_id, tag_name, spec, raw_tag, normalized_val)

    def get_tag(
        self,
        tag: Union[str, int],
        return_type: Optional[type] = None,
    ) -> Optional[Any]:
        """Get tag value with automatic or specified type conversion.
        
        Args:
            tag: Either a numeric tag code (int) or tag name string
            return_type: Optional type to convert to (int, float, str, tuple, list, etc.)
        
        Returns:
            Tag value (converted based on return_type), or None if not found.
            
        See also:
            get_raw_tag: Returns raw tag value without any conversion.
        """
        tag_info = self._get_tag_object(tag)
        if tag_info is None:
            return None
        
        _, tag_name, spec, raw_tag, normalized_val = tag_info
        return convert_tag_value(
            tag_name, normalized_val, raw_tag.dtype, raw_tag.count, spec, return_type)

    def get_raw_tag(self, tag: Union[str, int]) -> Optional[Any]:
        """Get raw tag value without any type conversion.
        
        Args:
            tag: Either a numeric tag code (int) or tag name string
        
        Returns:
            Raw tag value as stored, or None if not found.
            
        See also:
            get_tag: Returns tag value with automatic or specified type conversion.
        """
        tag_info = self._get_tag_object(tag)
        return tag_info[4] if tag_info else None
    
    def get_time_from_tags(self, time_type: str = "original") -> Optional[datetime]:
        """Extract datetime from EXIF time tags with subseconds and timezone.
        
        This is the inverse of MetadataTags.add_time_tags(). Reads the datetime,
        subsecond, and timezone offset tags and combines them into a single
        datetime object.
        
        Args:
            time_type: Which set of tags to read:
                - "original" (default): DateTimeOriginal, SubsecTimeOriginal, OffsetTimeOriginal
                - "digitized": DateTimeDigitized, SubsecTimeDigitized, OffsetTimeDigitized
                - "modified": DateTime, SubsecTime, OffsetTime
                - "preview": PreviewDateTime (no subsec/offset tags)
                
        Returns:
            datetime object with microseconds and timezone if available, or None if
            the datetime tag is not present or cannot be parsed.
            
        Example:
            capture_time = page.get_time("original")
            if capture_time:
                print(f"Captured at {capture_time}")
        """
        from .tiff_metadata import _get_time_impl
        return _get_time_impl(self, time_type)


    def get_ifd0_tags(self, convert_exif: bool = True) -> MetadataTags:
        """Return a copy of IFD0 tags as a MetadataTags object.
        
        Args:
            convert_exif: If True (default), convert ExifTag dictionary to individual TIFF tags.
        """
        return self.ifd0.get_page_tags(convert_exif=convert_exif)
    
    def get_page_tags(self, convert_exif: bool = True) -> MetadataTags:
        """Return a copy of all page-level tags as a MetadataTags object.
        
        All multi-byte arrays are normalized to system byte order for consistent
        internal representation. This includes:
        - Multi-byte typed arrays (SHORT, LONG, FLOAT, etc.)
        - ProfileGainTableMap/ProfileGainTableMap2 binary blobs
        
        Args:
            convert_exif: If True (default), convert ExifTag dictionary to individual TIFF tags. 
        """
        tags = MetadataTags()
        
        # Use _get_tag_object to get normalized values for each tag
        for tag_code in self.tags.keys():
            tag_info = self._get_tag_object(tag_code)
            if tag_info is not None:
                _, _, _, raw_tag, normalized_value = tag_info
                tags.add_raw_tag(tag_code, raw_tag.dtype, raw_tag.count, normalized_value)
        
        # Apply EXIF conversion if requested
        if convert_exif:
            from .tiff_metadata import _convert_exif_dict_to_tags
            _convert_exif_dict_to_tags(tags)
        
        return tags
    
    def _decode_tiled(self, decode_func) -> np.ndarray:
        """Decode tiled compressed image data with error handling.
        
        Args:
            decode_func: Function to decode tile data (e.g., imagecodecs.jpegxl_decode).
                        Invalid tiles that fail to decode are filled with zeros.
        
        Returns:
            Decoded image array.
        """
        fh = self.parent.filehandle
        
        # Read all segments using tifffile's read_segments API
        segments = list(fh.read_segments(
            self.dataoffsets,
            self.databytecounts,
            sort=True
        ))
        
        if len(segments) == 1:
            # Single tile/strip
            compressed_data, _ = segments[0]
            return decode_func(compressed_data)
        else:
            # Multiple tiles - decode each and assemble
            tile_width = self.tilewidth
            tile_height = self.tilelength
            img_width = self.imagewidth
            img_height = self.imagelength
            samples = self.samplesperpixel or 1
            
            # Determine output dtype from first valid tile
            dtype = None
            for i, (tile_data, _) in enumerate(segments):
                try:
                    first_tile = decode_func(tile_data)
                    dtype = first_tile.dtype
                    break
                except Exception:
                    continue
            
            # If no valid tiles found, cannot decode
            if dtype is None:
                raise ValueError(
                    f"No valid tiles found in image. "
                    f"All {len(segments)} tiles failed decoding."
                )
            
            # Create output array - handle both 2D (single sample) and 3D (multi-sample)
            if samples == 1:
                output = np.zeros((img_height, img_width), dtype=dtype)
            else:
                output = np.zeros((img_height, img_width, samples), dtype=dtype)
            
            # Decode and place each tile
            tiles_x = (img_width + tile_width - 1) // tile_width
            for i, (tile_data, _) in enumerate(segments):
                try:
                    tile = decode_func(tile_data)
                except Exception as e:
                    logger.warning(f"Failed to decode tile {i}: {e}, filling with zeros")
                    # Skip failed tile - output array already zero-filled
                    continue
                
                ty = (i // tiles_x) * tile_height
                tx = (i % tiles_x) * tile_width
                
                # Handle edge tiles that may be smaller
                th = min(tile_height, img_height - ty)
                tw = min(tile_width, img_width - tx)
                
                output[ty:ty+th, tx:tx+tw] = tile[:th, :tw]
            
            return output
    
    def _decode_jpegxl(self) -> np.ndarray:
        """Decode JPEG XL compressed image data, handling tiled images.
        
        Returns:
            Decoded image array.
        """
        return self._decode_tiled(imagecodecs.jpegxl_decode)
    
    def _decode_jpeg(self) -> np.ndarray:
        """Decode JPEG compressed image data, handling tiled images.
        
        Handles malformed DNG files (e.g., Google Pixel) that may have zero-filled
        padding tiles instead of valid JPEG data. Invalid tiles are caught by the
        JPEG decoder and filled with zeros.
        
        Also handles JPEG's unusual output shape for CFA data: (height, width/2, 2)
        is reshaped to (height, width) to match expected tile dimensions.
        
        Returns:
            Decoded image array.
        """
        tile_height = self.tilelength
        tile_width = self.tilewidth
        
        def decode_jpeg_tile(tile_data: bytes) -> np.ndarray:
            """Decode JPEG tile and reshape if needed.
            
            JPEG encodes CFA data as (height, width/2, 2) where pairs of pixels
            are treated as a single pixel with 2 components. Reshape to expected
            (height, width) tile dimensions.
            """
            tile = imagecodecs.jpeg_decode(tile_data, bitspersample=self.bitspersample)
            
            # Handle JPEG CFA format: (256, 128, 2) -> (256, 256)
            if tile.shape != (tile_height, tile_width):
                if tile.ndim == 3 and tile.shape[2] == 2:
                    tile = tile.reshape(tile_height, tile_width)
                else:
                    # Unexpected shape - return zeros to avoid crashes
                    logger.warning(
                        f"JPEG tile has unexpected shape {tile.shape}, "
                        f"expected ({tile_height}, {tile_width}), returning zeros"
                    )
                    return np.zeros((tile_height, tile_width), dtype=tile.dtype)
            
            return tile
        
        return self._decode_tiled(decode_jpeg_tile)

    def _stage1(self) -> Optional["DngPage._RawStage"]:
        if self.is_cfa:
            if self.compression in (COMPRESSION.JPEGXL, COMPRESSION.JPEGXL_DNG):
                raw_cfa = self._decode_jpegxl()
            elif self.compression == COMPRESSION.JPEG:
                raw_cfa = self._decode_jpeg()
            else:
                raw_cfa = self.asarray()

            col_interleave = self.get_tag("ColumnInterleaveFactor")
            row_interleave = self.get_tag("RowInterleaveFactor")
            if col_interleave == 2 and row_interleave == 2:
                raw_cfa = deswizzle_cfa_data(raw_cfa)

            cfa_str = self.get_tag("CFAPattern", str)
            return self._RawStage(data=raw_cfa, cfa_pattern=cfa_str)

        if self.is_linear_raw:
            if self.compression in (COMPRESSION.JPEGXL, COMPRESSION.JPEGXL_DNG):
                raw_linear = self._decode_jpegxl()
            else:
                raw_linear = self.asarray()
            return self._RawStage(data=raw_linear)

        return None

    def _stage2(
        self, stage1: "DngPage._RawStage", apply_ops2: bool = True
    ) -> "DngPage._RawStage":
        photometric = self.photometric_name
        if photometric == "CFA":
            samples_per_pixel = 1
        else:
            samples_per_pixel = int(self.get_tag("SamplesPerPixel") or 1)

        black_repeat_dim = self.get_tag("BlackLevelRepeatDim")
        if black_repeat_dim is None:
            black_repeat_dim = (1, 1)
        black_repeat_rows = int(black_repeat_dim[0]) if hasattr(black_repeat_dim, "__len__") else 1
        black_repeat_cols = (
            int(black_repeat_dim[1])
            if hasattr(black_repeat_dim, "__len__") and len(black_repeat_dim) > 1
            else 1
        )
        expected_black_size = black_repeat_rows * black_repeat_cols * samples_per_pixel

        black_level_raw = self.get_tag("BlackLevel")
        if black_level_raw is None:
            black_level = np.zeros(expected_black_size, dtype=np.float32)
        else:
            black_level = np.atleast_1d(black_level_raw).astype(np.float32).ravel()
            if len(black_level) != expected_black_size:
                black_level = np.zeros(expected_black_size, dtype=np.float32)

        bits_per_sample_raw = self.get_tag("BitsPerSample")
        if bits_per_sample_raw is None:
            bits_per_sample = 16
        elif isinstance(bits_per_sample_raw, np.ndarray):
            bits_per_sample = int(bits_per_sample_raw.flat[0])
        elif isinstance(bits_per_sample_raw, (list, tuple)):
            bits_per_sample = int(bits_per_sample_raw[0])
        else:
            bits_per_sample = int(bits_per_sample_raw)
        
        # Check if this is float data (SampleFormat = 3)
        sample_format = self.get_tag("SampleFormat")
        if sample_format is not None:
            # SampleFormat can be a single value or array
            if isinstance(sample_format, (list, tuple, np.ndarray)):
                sample_format = sample_format[0]
        
        # Match DNG SDK default: float uses 1.0, integer uses (2^bits - 1)
        # DNG SDK reference: dng_ifd.cpp lines 3466-3468
        if sample_format == 3:
            default_white = 1.0
        else:
            # Integer data (SampleFormat=1) or missing SampleFormat tag
            default_white = float((1 << bits_per_sample) - 1)

        white_level_raw = self.get_tag("WhiteLevel")
        if white_level_raw is None:
            white_level = np.full(samples_per_pixel, default_white, dtype=np.float32)
        else:
            white_level = np.atleast_1d(white_level_raw).astype(np.float32).ravel()
            white_level = np.where(white_level < 0, default_white, white_level)
            if len(white_level) < samples_per_pixel:
                white_level = np.concatenate(
                    [
                        white_level,
                        np.full(samples_per_pixel - len(white_level), default_white, dtype=np.float32),
                    ]
                )

        black_delta_h = self.get_tag("BlackLevelDeltaH")
        black_delta_v = self.get_tag("BlackLevelDeltaV")
        if black_delta_h is not None:
            black_delta_h = np.atleast_1d(black_delta_h).astype(np.float32)
        if black_delta_v is not None:
            black_delta_v = np.atleast_1d(black_delta_v).astype(np.float32)

        linearization_table = self.get_tag("LinearizationTable")
        if linearization_table is not None:
            linearization_table = np.asarray(linearization_table, dtype=np.uint16)

        # Apply OpcodeList1 to raw sensor data (before linearization)
        # Note: OpcodeList1 is always applied (not gated by apply_ops2)
        data = stage1.data
        opcode_list1 = self.get_tag("OpcodeList1")
        if opcode_list1 is not None:
            try:
                opcodes = raw_render.parse_opcode_list(bytes(opcode_list1))
            except Exception as e:
                logger.warning(f"Failed to parse OpcodeList1: {e}")
                opcodes = None
            
            if opcodes:
                try:
                    logger.debug(f"OpcodeList1: {len(opcodes)} opcodes")
                    data = raw_render.apply_opcodes_cfa(data, opcodes)
                except Exception as e:
                    logger.warning(f"Failed to apply OpcodeList1: {e}")

        active_area = self.get_tag("ActiveArea")
        if active_area is not None:
            aa_top, aa_left, aa_bottom, aa_right = active_area
            data = data[aa_top:aa_bottom, aa_left:aa_right]

        normalized = raw_render._raw_render.normalize_raw(
            data=data.astype(np.float32),
            black_level=black_level,
            black_repeat_rows=black_repeat_rows,
            black_repeat_cols=black_repeat_cols,
            samples_per_pixel=samples_per_pixel,
            white_level=white_level,
            black_delta_h=black_delta_h,
            black_delta_v=black_delta_v,
            linearization_table=linearization_table,
        )

        stage2 = self._RawStage(data=normalized, cfa_pattern=stage1.cfa_pattern)

        if not apply_ops2:
            return stage2

        opcode_list2 = self.get_tag("OpcodeList2")
        if opcode_list2 is not None:
            try:
                opcodes = raw_render.parse_opcode_list(bytes(opcode_list2))
            except Exception as e:
                logger.warning(f"Failed to parse OpcodeList2: {e}")
                opcodes = None
            
            if opcodes:
                try:
                    logger.debug(f"OpcodeList2: {len(opcodes)} opcodes")
                    if self.photometric_name == "LINEAR_RAW":
                        data_ops = raw_render.apply_opcodes(stage2.data, opcodes, use_bicubic=False)
                        data_ops = np.clip(data_ops, 0.0, 1.0).astype(np.float32, copy=False)
                        return self._RawStage(data=data_ops)
                    else:
                        data_ops = raw_render.apply_opcodes_cfa(stage2.data, opcodes)
                        data_ops = np.clip(data_ops, 0.0, 1.0).astype(np.float32, copy=False)
                        return self._RawStage(data=data_ops, cfa_pattern=stage2.cfa_pattern)
                except Exception as e:
                    logger.warning(f"Failed to apply OpcodeList2: {e}")
        
        return stage2

    def get_cfa(
        self, stage: RawStageSelector = RawStageSelector.RAW
    ) -> Optional[tuple[np.ndarray, str]]:
        """Extract CFA data and pattern from this page.

        Args:
            stage: Which stage of the raw pipeline to return.

                - RawStageSelector.RAW: Stored samples (decoded).
                - RawStageSelector.LINEARIZED: Raw + linearize + range map
                  (ActiveArea-cropped, float32 in [0, 1]).
                - RawStageSelector.LINEARIZED_PLUS_OPS: LINEARIZED + OpcodeList2 (if present)
                  (float32 in [0, 1]).

        Returns:
            Tuple of (cfa_array, cfa_pattern_str) or None if not a CFA page.
            cfa_pattern_str is e.g., 'RGGB', 'BGGR'.
        """
        stage1 = self._stage1()
        if stage1 is None:
            return None

        if stage == RawStageSelector.RAW:
            stage_out = stage1
        elif stage == RawStageSelector.LINEARIZED:
            stage_out = self._stage2(stage1, apply_ops2=False)
        elif stage == RawStageSelector.LINEARIZED_PLUS_OPS:
            stage_out = self._stage2(stage1, apply_ops2=True)
        else:
            raise ValueError(f"Unknown stage selector: {stage}")

        if stage_out.cfa_pattern is None:
            return None

        return stage_out.data, stage_out.cfa_pattern
    
    def get_linear_raw(
        self, stage: RawStageSelector = RawStageSelector.RAW
    ) -> Optional[np.ndarray]:
        """Extract LINEAR_RAW data from this page.

        Args:
            stage: Which stage of the raw pipeline to return.

                - RawStageSelector.RAW: Stored samples (decoded).
                - RawStageSelector.LINEARIZED: Raw + linearize + range map
                  (ActiveArea-cropped, float32 in [0, 1]).
                - RawStageSelector.LINEARIZED_PLUS_OPS: LINEARIZED + OpcodeList2 (if present)
                  (float32 in [0, 1]).
        
        Returns:
            Raw linear data array or None if not a LINEAR_RAW page.
        """
        if not self.is_linear_raw:
            return None

        stage1 = self._stage1()
        if stage1 is None:
            return None

        if stage == RawStageSelector.RAW:
            return stage1.data
        if stage == RawStageSelector.LINEARIZED:
            return self._stage2(stage1, apply_ops2=False).data
        if stage == RawStageSelector.LINEARIZED_PLUS_OPS:
            return self._stage2(stage1, apply_ops2=True).data
        raise ValueError(f"Unknown stage selector: {stage}")

    def get_camera_rgb_raw(
        self, 
        demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.OPENCV_EA
        ) -> Optional[np.ndarray]:
        """Extract the camera-RGB intermediate from a raw page for the color pipeline.

        This corresponds to the `rgb_camera` input passed into
        `raw_render._render_camera_rgb(...)` during `render_raw()`.

        Returns stage3 (normalized + ActiveArea-cropped), applies OpcodeList3 (if
        present), demosaics (if CFA), applies OpcodeList3 (if present), and applies
        DefaultCrop (if present).

        - If photometric == LINEAR_RAW: returns the stage3 linear RGB.
        - Else: demosaics the stage2 CFA to camera RGB, then applies stage3 operations.

        Args:
            demosaic_algorithm: Demosaic algorithm to use when the source is CFA.

        Returns:
            Camera RGB array (H, W, 3) float32 in [0, 1], or None if extraction fails.
        """
        # Validate this is a raw page
        if not (self.is_cfa or self.is_linear_raw):
            raise ValueError(
                f"get_camera_rgb_raw() requires CFA or LINEAR_RAW page, got {self.photometric_name}"
            )
        
        stage1 = self._stage1()
        if stage1 is None:
            return None
        stage2 = self._stage2(stage1)

        photometric = self.photometric_name

        if photometric == "LINEAR_RAW":
            rgb_camera = stage2.data
            rgb_camera = np.clip(rgb_camera, 0.0, 1.0).astype(np.float32, copy=False)
        else:
            cfa_normalized = stage2.data
            cfa_pattern = stage2.cfa_pattern
            if cfa_pattern is None:
                cfa_pattern = "RGGB"

            rgb_camera = raw_render.demosaic(
                cfa_normalized, cfa_pattern, algorithm=demosaic_algorithm
            )
            rgb_camera = np.clip(rgb_camera, 0.0, 1.0).astype(np.float32, copy=False)

        # Apply OpcodeList3 (Stage3 operations)
        opcode_list3 = self.get_tag("OpcodeList3")
        if opcode_list3 is not None:
            try:
                opcodes = raw_render.parse_opcode_list(bytes(opcode_list3))
            except Exception as e:
                logger.warning(f"Failed to parse OpcodeList3: {e}")
                opcodes = None
            
            if opcodes:
                try:
                    logger.debug(f"OpcodeList3: {len(opcodes)} opcodes")
                    rgb_camera = raw_render.apply_opcodes(rgb_camera, opcodes, use_bicubic=False)
                except Exception as e:
                    logger.warning(f"Failed to apply OpcodeList3: {e}")

        # Apply DefaultCrop
        crop_origin = self.get_tag("DefaultCropOrigin")
        crop_size = self.get_tag("DefaultCropSize")

        if crop_origin is not None and crop_size is not None:
            crop_x = int(crop_origin[0])
            crop_y = int(crop_origin[1])
            crop_w = int(crop_size[0])
            crop_h = int(crop_size[1])
            rgb_camera = rgb_camera[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        return rgb_camera
    
    def decode_to_rgb(
        self,
        output_dtype: type = np.uint8
    ) -> Optional[np.ndarray]:
        """Decode any DNG page to RGB array.
        
        For raw pages (CFA, LINEAR_RAW), renders with default parameters.
        For preview pages (RGB, YCBCR), just decompresses the image data.
        
        Args:
            output_dtype: Output data type (np.uint8 or np.uint16)
            
        Returns:
            RGB image array (H, W, 3) or None if decoding fails
        """
        # Raw pages - render with default parameters
        if self.is_cfa or self.is_linear_raw:
            return self.render_raw(output_dtype=output_dtype)
        
        # Non-raw pages - let tifffile decode
        # Note: tifffile automatically converts JPEG-compressed YCBCR to RGB
        try:
            image = self.asarray()
        except Exception as e:
            logger.error(
                f"Failed to decode page (photometric={self.photometric_name}): {e}"
            )
            return None
        
        # Convert dtype if needed
        if image.dtype != output_dtype:
            image = raw_render.convert_dtype(image, output_dtype)
        
        return image
    
    def render_raw(
        self,
        output_dtype: type = np.uint16,
        demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.OPENCV_EA,
        strict: bool = True,
        use_xmp: bool = True,
        rendering_params: dict = None,
    ) -> "np.ndarray | None":
        """Render raw DNG page to RGB image with optional XMP-based adjustments.
        
        Applies full DNG raw processing pipeline: linearization, black/white level,
        white balance, color matrix, demosaicing, and tone curve. Converts to
        output color space. Supports XMP metadata for white balance, exposure, and
        tone curve adjustments.
        
        For RGB/YCBCR preview pages, use decode() instead.
        
        Args:
            output_dtype: Output data type (np.uint8 or np.uint16)
            demosaic_algorithm: Algorithm for CFA demosaicing ("RCD", "VNG", "AHD")
            strict: If True, raise error on unsupported DNG tags. If False, warn and continue.
            use_xmp: If True, extract rendering parameters from XMP metadata (Temperature,
                    Tint, Exposure2012, ToneCurvePV2012). Default True.
            rendering_params: Optional dict to override rendering parameters.
                See decode_dng() for full list of supported keys.
                Values in rendering_params override XMP metadata.
        
        Returns:
            Rendered RGB image as numpy array with shape (H, W, 3) and specified dtype,
            or None if rendering fails.
        
        Raises:
            UnsupportedDNGTagError: If strict=True and unsupported tags are encountered
            ValueError: If rendering_params contains unsupported parameter names or if
                page is not a raw page (CFA or LINEAR_RAW)
        
        Example:
            # Use XMP metadata from DNG file
            rgb = page.render_raw()
            
            # Override white balance
            rgb = page.render_raw(rendering_params={'temperature': 6500, 'tint': 10})
            
            # Disable XMP, use only DNG tags
            rgb = page.render_raw(use_xmp=False)
        """
        # Validate this is a raw page
        if not (self.is_cfa or self.is_linear_raw):
            raise ValueError(
                f"render_raw() requires CFA or LINEAR_RAW page, got {self.photometric_name}"
            )
        
        try:
            unsupported = raw_render.validate_dng_tags(self, strict=strict)
        except raw_render.UnsupportedDNGTagError:
            raise
        
        if unsupported and not strict:
            logger.warning(
                f"DNG contains unsupported tags (processing anyway): {', '.join(unsupported)}"
            )

        rgb_camera = self.get_camera_rgb_raw(demosaic_algorithm=demosaic_algorithm)
        if rgb_camera is None:
            logger.error("Failed to extract camera RGB from DNG")
            return None
        
        # Get IFD0 for rendering (color profile tags are in IFD0)
        result = raw_render._render_camera_rgb(
            ifd0_tags=self.ifd0,
            raw_ifd_tags=self,
            rgb_camera=rgb_camera,
            output_dtype=output_dtype,
            rendering_params=rendering_params,
            use_xmp=use_xmp,
        )
        return result

class DngFile(TiffFile):

    """A TIFF file with DNG-specific extensions and helper methods."""

    def __init__(self, file, *args, **kwargs):
        # Let TiffFile handle all file types - no eager conversion to BytesIO
        super().__init__(file, *args, **kwargs)

    def nbytes(self) -> int:
        """Get size of DNG file in bytes without loading into memory.
        
        For in-memory DNGs, returns the buffer size. For file-backed DNGs,
        returns the file size without reading the entire file.
        
        Returns:
            Size in bytes
        """
        # If already BytesIO, get its size
        if isinstance(self.filehandle._fh, io.BytesIO):
            return len(self.filehandle._fh.getvalue())
        
        # For file-backed DNGs, use tifffile's FileHandle.size property (no read needed)
        return self.filehandle.size
    
    def write_to(self, destination: Union[str, Path, io.IOBase]) -> None:
        """Write the DNG file to a destination.
        
        For file-backed DNGs, copies directly from source file without loading
        into memory. For in-memory DNGs, writes from the BytesIO buffer.
        
        Args:
            destination: File path or file-like object to write to
            
        Example:
            >>> dng = create_dng_from_array(data_spec)
            >>> dng.write_to("output.dng")
            >>> # Or to an open file
            >>> with open("output.dng", "wb") as f:
            ...     dng.write_to(f)
        """
        import shutil
        
        # Case 1: Already BytesIO (in-memory DNG) - use buffer directly
        if isinstance(self.filehandle._fh, io.BytesIO):
            data = self.filehandle._fh.getvalue()
            if isinstance(destination, (str, Path)):
                with open(destination, 'wb') as f:
                    f.write(data)
            else:
                destination.write(data)
            return
        
        # Case 2: File-backed DNG - optimize based on destination type
        if isinstance(destination, (str, Path)):
            # File-to-file: use shutil.copyfile for optimal performance
            source_path = self.filehandle.path
            shutil.copyfile(source_path, destination)
        else:
            # File-to-stream: read and write in chunks to avoid loading all into memory
            self.filehandle._fh.seek(0)
            shutil.copyfileobj(self.filehandle._fh, destination)

    @property
    def ifd0(self) -> Optional[DngPage]:
        """Return IFD0 as a DngPage, or None if no pages exist."""
        return DngPage(self.pages[0]) if self.pages else None

    def get_flattened_pages(self) -> List[DngPage]:
        """Get all pages as DngPage instances.
        
        Returns:
            List of DngPage objects in flattened order. Tag inheritance
            falls back to IFD0 via TiffPage.parent.
        """
        def build_recursive(pages_list: Optional[List]) -> List[DngPage]:
            """Build DngPage instances from TiffPages recursively."""
            result = []
            if pages_list is None:
                return result
            
            for tiff_page in pages_list:
                dng_page = DngPage(tiff_page)
                result.append(dng_page)
                # Recursively process sub-pages from raw TiffPage
                result.extend(build_recursive(tiff_page.pages))
            
            return result
        
        return build_recursive(self.pages)
    
    def get_main_page(self) -> Optional[DngPage]:
        """Get the main image page.
        
        First looks for a page with NewSubFileType == 0 (explicit main image).
        If not found, falls back to the first CFA or LINEAR_RAW page.
        
        Returns:
            The main image DngPage, or None if not found.
        """
        pages = self.get_flattened_pages()
        
        # First try to find explicit main image (NewSubFileType == 0)
        for page in pages:
            if page.is_main_image:
                return page
        
        # Fall back to first CFA or LINEAR_RAW page
        for page in pages:
            if page.is_cfa or page.is_linear_raw:
                return page
        
        return None

    def _find_closest_raw_page_and_final_dim(
        self, scale: float
    ) -> Optional[Tuple[DngPage, Tuple[int, int]]]:
        """Find the optimal raw page and compute final dimensions for scaling.
        
        Searches all flattened pages for CFA or LINEAR_RAW pages and returns
        the page with the smallest max dimension that is still >= target dimension,
        along with the final target dimensions.
        
        Args:
            scale: Scaling factor (e.g., 0.5 for half size)
            
        Returns:
            Tuple of (optimal_page, (target_width, target_height)), or None
            if no raw pages are found.
        """
        # Get main page to calculate target dimension
        main_page = self.get_main_page()
        if main_page is None:
            return None
        
        # Calculate target max dimension using cropped dimensions
        main_w, main_h = main_page.get_rendered_size()
        main_max_dim = max(main_w, main_h)
        target_max_dim = main_max_dim * scale
        
        pages = self.get_flattened_pages()
        
        # Filter to only raw pages (CFA or LINEAR_RAW)
        raw_pages = [p for p in pages if p.is_cfa or p.is_linear_raw]
        
        if not raw_pages:
            return None
        
        # Find pages that meet or exceed the target dimension (using cropped dimensions)
        candidates = []
        for page in raw_pages:
            page_w, page_h = page.get_rendered_size()
            max_dim = max(page_w, page_h)
            if max_dim >= target_max_dim:
                candidates.append((page, max_dim))
        
        if candidates:
            # Use the smallest page that still meets the target
            optimal_page = min(candidates, key=lambda x: x[1])[0]
        else:
            # No page meets target - use main page
            optimal_page = main_page
        
        # Calculate final target dimensions maintaining aspect ratio
        
        if main_w >= main_h:
            target_w = int(target_max_dim)
            target_h = int(main_h * target_max_dim / main_w)
        else:
            target_h = int(target_max_dim)
            target_w = int(main_w * target_max_dim / main_h)
        
        return optimal_page, (target_w, target_h)

    def get_ifd0_tags(self, convert_exif: bool = True) -> MetadataTags:
        """Return a copy of IFD0 tags as a MetadataTags object.
        
        Args:
            convert_exif: If True (default), convert ExifTag dictionary to individual TIFF tags.
        """
        return self.ifd0.get_ifd0_tags(convert_exif=convert_exif) if self.ifd0 else MetadataTags()

    def get_tag(
        self,
        tag: Union[str, int],
        return_type: Optional[type] = None,
    ) -> Optional[Any]:
        """See `DngPage.get_tag`."""
        return self.ifd0.get_tag(tag, return_type=return_type) if self.ifd0 else None

    def get_xmp(self) -> Optional[XmpMetadata]:
        """See `DngPage.get_xmp`."""
        return self.ifd0.get_xmp() if self.ifd0 else None

    def get_time_from_tags(self, time_type: str = "original") -> Optional[datetime]:
        """See `DngPage.get_time_from_tags`."""
        return self.ifd0.get_time_from_tags(time_type=time_type) if self.ifd0 else None
    
    def get_rendered_size(self) -> Optional[Tuple[int, int]]:
        """See `DngPage.get_rendered_size`."""
        main_page = self.get_main_page()
        return main_page.get_rendered_size() if main_page else None

    def _forward_main_page(self, method_name: str, *args, require=None, **kwargs):
        page = self.get_main_page()
        if page is None:
            return None
        if require is not None and not require(page):
            return None
        method = getattr(page, method_name)
        return method(*args, **kwargs)
        
    def get_cfa(
        self, stage: RawStageSelector = RawStageSelector.RAW
    ) -> Optional[tuple[np.ndarray, str]]:
        """See `DngPage.get_cfa`."""
        return self._forward_main_page(
            "get_cfa",
            stage=stage,
            require=lambda p: p.is_cfa,
        )

    def get_linear_raw(
        self, stage: RawStageSelector = RawStageSelector.RAW
    ) -> Optional[np.ndarray]:
        """See `DngPage.get_linear_raw`."""
        return self._forward_main_page(
            "get_linear_raw",
            stage=stage,
            require=lambda p: p.is_linear_raw,
        )

    def get_camera_rgb_raw(
        self,
        demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.OPENCV_EA,
    ) -> Optional[np.ndarray]:
        """Extract camera-RGB from main raw page with optional scaling.
        
        When scale is provided, automatically selects the optimal SubIFD pyramid
        level to minimize processing overhead, then applies final scaling if needed.
        
        See `DngPage.get_camera_rgb_raw` for full documentation.
        
        Args:
            demosaic_algorithm: Algorithm for CFA demosaicing
        
        Returns:
            Camera RGB array (H, W, 3) float32 in [0, 1], or None if extraction fails.
        """
        return self._forward_main_page(
            "get_camera_rgb_raw",
            demosaic_algorithm=demosaic_algorithm,
        )

    def render_raw(
        self,
        output_dtype: type = np.uint16,
        demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.OPENCV_EA,
        strict: bool = True,
        use_xmp: bool = True,
        rendering_params: dict = None,
        scale: Optional[float] = None,
    ) -> "np.ndarray | None":
        """Render main raw DNG page to RGB image with optional scaling.
        
        When scale is provided, automatically selects the optimal SubIFD pyramid
        level to minimize processing overhead, then applies final scaling if needed.
        
        To render a specific raw page (e.g., LINEAR_RAW preview at a different
        resolution), use get_flattened_pages()[ifd].render_raw() instead.
        
        See `DngPage.render_raw` for full documentation.
        
        Args:
            output_dtype: Output data type (np.uint8 or np.uint16)
            demosaic_algorithm: Algorithm for CFA demosaicing
            strict: If True, raise error on unsupported DNG tags
            use_xmp: If True, extract rendering parameters from XMP metadata
            rendering_params: Optional dict to override rendering parameters
            scale: Optional scaling factor (e.g., 0.5 for half size). If provided,
                selects the closest SubIFD pyramid level >= target dimension, then
                applies final resize with INTER_AREA if needed.
        
        Returns:
            Rendered RGB image or None if rendering fails
        """
        # Determine which page to use for rendering
        main_page = self.get_main_page()

        if scale is None:
            # No scaling - use main page
            render_page = main_page
            if render_page is None:
                return None
            target_w = target_h = None  # No resize needed
        else:
            # Find optimal page and compute final dimensions
            result = self._find_closest_raw_page_and_final_dim(scale)
            if result is None:
                return None
            render_page, (target_w, target_h) = result
        
        # Validate DNG tags
        try:
            unsupported = raw_render.validate_dng_tags(render_page, strict=strict)
        except raw_render.UnsupportedDNGTagError:
            raise
        
        if unsupported and not strict:
            logger.warning(
                f"DNG contains unsupported tags (processing anyway): {', '.join(unsupported)}"
            )
        
        # Get camera RGB raw from render page
        rgb_camera = render_page.get_camera_rgb_raw(demosaic_algorithm=demosaic_algorithm)
        if rgb_camera is None:
            return None
        
        # Apply resize if needed (when scaling and dimensions don't match)
        if target_w is not None and target_h is not None:
            render_w, render_h = render_page.get_rendered_size()
            if render_w != target_w or render_h != target_h:
                import cv2
                rgb_camera = cv2.resize(
                    rgb_camera, (target_w, target_h), interpolation=cv2.INTER_AREA
                )
        
        # Render camera RGB to final output
        # use main_page for raw_ifd in case there is a PGTM, nothing else uses raw_ifd tags in RGB render path
        rgb_image = raw_render._render_camera_rgb(
            ifd0_tags=self.ifd0,
            raw_ifd_tags=main_page, 
            rgb_camera=rgb_camera,
            output_dtype=output_dtype,
            rendering_params=rendering_params,
            use_xmp=use_xmp,
        )
        
        return rgb_image

    def get_preview_rgb(
        self,
        output_dtype: type = np.uint8,
    ) -> Optional[np.ndarray]:
        """Decode preview image from IFD0 to RGB array.
        
        In most DNG files, IFD0 contains a preview image. This method verifies
        IFD0 is a preview (NewSubFileType indicates preview) before decoding.
        
        For accessing other IFD indices, use get_flattened_pages()[ifd].decode_to_rgb().
        
        See `DngPage.decode_to_rgb` for full documentation.
        
        Args:
            output_dtype: Output data type (np.uint8 or np.uint16)
            
        Returns:
            RGB image array or None if IFD0 is not a preview or decoding fails
        """
        ifd0 = self.ifd0
        if ifd0 is None:
            return None
        
        # Only decode if IFD0 is actually a preview image
        if not ifd0.is_preview:
            return None
        
        return ifd0.decode_to_rgb(output_dtype=output_dtype)


# =============================================================================
# Helper Functions
# =============================================================================

def swizzle_cfa_data(raw_data: np.ndarray) -> np.ndarray:
    """Swizzle RGGB CFA data into a 2x2 grid of R, G1, G2, B sub-images."""

    # Pre-allocate the swizzled array with same dtype as input
    h, w = raw_data.shape
    swizzled_data = np.empty((h, w), dtype=raw_data.dtype)
    
    # Calculate half dimensions for direct assignment
    h_half, w_half = h // 2, w // 2
    
    # Extract channels and write directly into pre-allocated array
    # R pixels: top-left quadrant
    swizzled_data[0:h_half, 0:w_half] = raw_data[0::2, 0::2]
    # G1 pixels: top-right quadrant  
    swizzled_data[0:h_half, w_half:w] = raw_data[0::2, 1::2]
    # G2 pixels: bottom-left quadrant
    swizzled_data[h_half:h, 0:w_half] = raw_data[1::2, 0::2]
    # B pixels: bottom-right quadrant
    swizzled_data[h_half:h, w_half:w] = raw_data[1::2, 1::2]

    return swizzled_data

def deswizzle_cfa_data(swizzled_data: np.ndarray) -> np.ndarray:
    """Deswizzle CFA data from a 2x2 grid of R, G1, G2, B sub-images back to RGGB."""
    h_swizzled, w_swizzled = swizzled_data.shape
    if h_swizzled % 2 != 0 or w_swizzled % 2 != 0:
        raise ValueError("Swizzled data dimensions must be even.")

    # Calculate half dimensions for quadrant extraction
    h_half, w_half = h_swizzled // 2, w_swizzled // 2

    # Extract the four channels from the swizzled data
    # R is top-left quadrant
    r_channel = swizzled_data[0:h_half, 0:w_half]
    # G1 (first green) is top-right quadrant
    g1_channel = swizzled_data[0:h_half, w_half:w_swizzled]
    # G2 (second green) is bottom-left quadrant
    g2_channel = swizzled_data[h_half:h_swizzled, 0:w_half]
    # B is bottom-right quadrant
    b_channel = swizzled_data[h_half:h_swizzled, w_half:w_swizzled]

    # Create an empty array for the original interleaved data
    # Its dimensions will be the same as the swizzled_data because each sub-image
    # was H/2 x W/2, and they are re-interleaved into a H x W image.
    original_data = np.empty_like(swizzled_data)

    # Place the channels back into the original RGGB pattern
    original_data[0::2, 0::2] = r_channel  # R pixels
    original_data[0::2, 1::2] = g1_channel  # G1 pixels (top-right G)
    original_data[1::2, 0::2] = g2_channel  # G2 pixels (bottom-left G)
    original_data[1::2, 1::2] = b_channel  # B pixels

    return original_data

def _prepare_ifd_args(
    metadata: MetadataTags,
    compression,
    is_ifd0: bool,
    subfiletype: int,
    photometric: str,
    subifds_count: int = 0,
    compressionargs: Optional[dict] = None,
) -> dict:
    """Prepare TiffWriter IFD args, extracting tags that have kwargs equivalents.
    
    Initializes the args structure for tifffile write, this requires moving some tags
    from metadata into args
    
    Args:
        metadata: MetadataTags instance to use as extratags (modified in-place)
        compression: COMPRESSION enum value for this IFD
        is_ifd0: True if writing to IFD0, False if writing to SubIFD
        subfiletype: NewSubFileType value
        photometric: Photometric interpretation string
        subifds_count: Number of SubIFDs (0 if none)
        compressionargs: Optional compression arguments dict
        
    Returns:
        Dict with complete IFD args including extratags and extracted kwargs
    """
    # Initialize IFD args with required parameters
    ifd_args = {
        "compression": compression,
        "subfiletype": int(subfiletype),
        "photometric": photometric,
    }
    
    if subifds_count:
        ifd_args["subifds"] = int(subifds_count)
    
    if compressionargs:
        ifd_args["compressionargs"] = compressionargs
    
    # Handle auto-generated tags (Software, metadata)
    # DNG spec: these tags only appear in IFD0, not in SubIFDs
    if is_ifd0:
        # IFD0: Extract Software from metadata if present
        if 'Software' in metadata:
            ifd_args['software'] = metadata.get_tag('Software')
    else:
        # SubIFDs: Prevent tifffile from adding these tags
        ifd_args['software'] = False  # Prevents tifffile default value 'tifffile.py'
        ifd_args['metadata'] = None   # Prevents auto-generated {"shape": [...]}
    
    # Extract Resolution (need both X and Y)
    # get_tag returns float for RATIONAL tags with count=1
    x_res = metadata.get_tag('XResolution')
    y_res = metadata.get_tag('YResolution')
    
    if x_res is not None and y_res is not None:
        ifd_args['resolution'] = (x_res, y_res)
    
    # Extract ResolutionUnit
    if 'ResolutionUnit' in metadata:
        ifd_args['resolutionunit'] = metadata.get_tag('ResolutionUnit')
    
    # Extract PlanarConfiguration
    planar_config = metadata.get_tag('PlanarConfiguration')
    if planar_config is not None:
        ifd_args['planarconfig'] = planar_config
    
    # Note: extratags is set by caller after this function returns
    return ifd_args

def _add_required_ifd0_tags(tags: MetadataTags, needs_v1_7_1: bool = False) -> None:
    """Add required DNG IFD0 tags to existing tags in-place.
    
    Tag names are from tifffile.py TiffTagRegistry.
    Tag types are from tifffile.py DATA_DTYPES.
    
    Args:
        tags: MetadataTags to modify in-place
        needs_v1_7_1: Whether we are using 1.7.1 features
    """
    
    # Add required tags if not already set
    if "Orientation" not in tags:
        _ORIENTATION_HORIZONTAL = 1
        tags.add_tag("Orientation", _ORIENTATION_HORIZONTAL)
    
    # Default to sRGB color space with proper matrices for passthrough
    if "ColorMatrix1" not in tags and "ForwardMatrix1" not in tags:
        # ColorMatrix1: XYZ D50 → Camera (sRGB)
        d50_to_d65 = raw_render.compute_bradford_adaptation(raw_render.D50_xy, raw_render.D65_xy)
        prophoto_to_srgb = raw_render.XYZ_D65_TO_SRGB @ d50_to_d65 @ raw_render.PROPHOTO_RGB_TO_XYZ_D50
        xyz_d50_to_srgb = prophoto_to_srgb @ raw_render.XYZ_D50_TO_PROPHOTO_RGB
        tags.add_tag("ColorMatrix1", xyz_d50_to_srgb)

        # ForwardMatrix1: Camera (sRGB) → PCS (XYZ D50)
        d65_to_d50 = raw_render.compute_bradford_adaptation(raw_render.D65_xy, raw_render.D50_xy)
        forward_matrix1 = d65_to_d50 @ raw_render.SRGB_TO_XYZ_D65
        tags.add_tag("ForwardMatrix1", forward_matrix1)
    
    if "CalibrationIlluminant1" not in tags:
        tags.add_tag("CalibrationIlluminant1", raw_render.Illuminant.D50)
    
    if "AsShotWhiteXY" not in tags and "AsShotNeutral" not in tags:
        # D50 white point in xy chromaticity coordinates
        tags.add_tag("AsShotWhiteXY", [0.34567, 0.35850])
    
    if "AnalogBalance" not in tags:
        # Neutral analog balance
        tags.add_tag("AnalogBalance", [1.0, 1.0, 1.0])
    
    def _version_bytes(major: int, minor: int, patch: int, build: int) -> bytes:
        return bytes([major, minor, patch, build])

    # Set DNGVersion and DNGBackwardVersion
    v1_7_1 = _version_bytes(1, 7, 1, 0)
    v1_4_0 = _version_bytes(1, 4, 0, 0)
    
    # If we need 1.7.1 features, enforce both versions to be at least 1.7.1
    if needs_v1_7_1:
        # Set or overwrite DNGVersion to 1.7.1
        if "DNGVersion" not in tags or tuple(tags.get_tag("DNGVersion")) < (1, 7, 1, 0):
            tags.add_tag("DNGVersion", v1_7_1)
        # Set or overwrite DNGBackwardVersion to 1.7.1
        if "DNGBackwardVersion" not in tags or tuple(tags.get_tag("DNGBackwardVersion")) < (1, 7, 1, 0):
            tags.add_tag("DNGBackwardVersion", v1_7_1)
    else:
        # No 1.7.1 features needed - use defaults if not present
        if "DNGVersion" not in tags:
            tags.add_tag("DNGVersion", v1_7_1)
        if "DNGBackwardVersion" not in tags:
            tags.add_tag("DNGBackwardVersion", v1_4_0)


# ProfileIFD is currently in _TIFFWRITER_MANAGED_TAGS to strip it during copy operations
# since we don't support reading or writing it yet. It possibly could be in a simlar 
# way that exif and gps are suported.
#
# TiffFile parses ExifTag/GPSTag using:
# 1. TIFF.TAG_READERS registry maps tag codes to reader functions:
#    34665: read_exif_ifd, 34853: read_gps_ifd
#    - Reads IFD header (number of tags)
#    - Loops through tag entries
#    - Follows offset pointers to read tag values
#    - Returns dict of tag name -> value
#
# To add ProfileIFD support in the future:
# 1. Determine ProfileIFD structure/format from DNG spec or sample files
# 2. Write read_profile_ifd() function similar to read_exif_ifd():
#    def read_profile_ifd(fh, byteorder, dtype, count, offsetsize):
#        return read_tags(fh, byteorder, offsetsize, TIFF.PROFILE_TAGS, maxifds=1)[0]
# 3. Add to TIFF.TAG_READERS: 50933: read_profile_ifd
# 4. Define TIFF.PROFILE_TAGS registry with ProfileIFD tag names
#

# Tags TiffWriter manages automatically - never copy as extratags
_TIFFWRITER_MANAGED_TAGS = {
    'NewSubfileType', 'SubfileType', 'ImageWidth', 'ImageLength',
    'BitsPerSample', 'Compression', 'PhotometricInterpretation',
    'ImageDescription', 'StripOffsets', 'SamplesPerPixel', 'SampleFormat',
    'RowsPerStrip', 'StripByteCounts', 'XResolution', 'YResolution',
    'PlanarConfiguration', 'ResolutionUnit', 'Software',
    'TileWidth', 'TileLength', 'TileOffsets', 'TileByteCounts',
    'SubIFDs', 'ExifTag', 'GPSTag', 'InteroperabilityTag',
    'ProfileIFD',
}

# Digest tags only valid if we do a loss-less copy of the main page
_DIGEST_TAGS = {
    'NewRawImageDigest', 'RawImageDigest', 'OriginalRawFileDigest', 'RawDataUniqueID',
}

# Tags invalidated when transcoding
_COMPRESSION_INVALIDATED_TAGS = {
    'YCbCrCoefficients', 'YCbCrSubSampling', 'YCbCrPositioning',
    'ReferenceBlackWhite', 'JPEGInterchangeFormat',
    'JPEGInterchangeFormatLength', 'JPEGTables', 'JPEGRestartInterval',
    'JPEGLosslessPredictors', 'JPEGPointTransforms',
    'JPEGQTables', 'JPEGDCTables', 'JPEGACTables',
    'JXLDistance', 'JXLEffort', 'JXLDecodeSpeed',
}

def _filter_metadata_tags(
    tags: MetadataTags,
    exclude_names: Optional[set[str]] = None,
) -> None:
    """Filter tags in-place by removing tags in the exclude set.
    
    Args:
        tags: MetadataTags to filter (modified in-place)
        exclude_names: Set of tag names to remove (None = no filtering)
    """
    for tag_name in (exclude_names or set()):
        tags.remove_tag(tag_name)


class PageOp(str, Enum):
    """Page operation mode for DNG writing."""
    COPY = "copy"        # Copy page data as-is, preserving source compression
    TRANSCODE = "transcode"  # Decompress and re-compress with specified compression
    
    @classmethod
    def lookup(cls, value: str) -> "PageOp":
        """Look up enum member by string value."""
        from .common import enum_from_string
        return enum_from_string(cls, value)

@dataclass
class IfdPageSpec:
    """Specification for writing a DNG page from a source DNG file.
    
    Args:
        page: Source DNG page to write
        subfiletype: NewSubFileType value (0=main, 1=preview)
        page_operation: Either PageOp.COPY to preserve source compression,
                       or (PageOp.TRANSCODE, COMPRESSION) to decompress and recompress.
                       When TRANSCODE, compression can be COMPRESSION.NONE for uncompressed.
        compression_args: Args for compression (only used with TRANSCODE mode)
        extratags: Additional metadata tags to add
        strip_tags: Tag names to remove from source
        copy_page_tags: Whether to copy tags from source page
    """
    page: DngPage
    subfiletype: int = 0
    page_operation: Union[PageOp, Tuple[PageOp, COMPRESSION]] = PageOp.COPY
    compression_args: Optional[dict] = None
    extratags: Optional[MetadataTags] = None
    strip_tags: Optional[set[str]] = None
    copy_page_tags: bool = True
    
    def requires_transcode(self) -> bool:
        """Check if this spec requires transcoding (decompression + recompression)."""
        if isinstance(self.page_operation, tuple):
            return True  # Explicit TRANSCODE mode
        # Check if tiled page needs transcode due to unsupported tile configuration
        if self.page.is_tiled:
            tile_h, tile_w = self.page.tilelength, self.page.tilewidth
            return not (
                tile_h <= self.page.imagelength
                and tile_w <= self.page.imagewidth
                and tile_h % 16 == 0
                and tile_w % 16 == 0
            )
        return False
    
    def get_target_compression(self) -> COMPRESSION:
        """Get the target compression for this spec."""
        if isinstance(self.page_operation, tuple):
            _, compression = self.page_operation
            return compression
        return self.page.compression  # COPY mode preserves source
    
    def has_jxl_compression(self) -> bool:
        """Check if this spec uses JXL compression."""
        compression = self.get_target_compression()
        return compression in (COMPRESSION.JPEGXL, COMPRESSION.JPEGXL_DNG)


@dataclass
class IfdDataSpec:
    """Specification for writing a DNG IFD from raw array data.
    
    Args:
        data: Raw image data array
        photometric: Photometric interpretation ("CFA", "LINEAR_RAW", "RGB", "YCBCR")
        subfiletype: NewSubFileType value (0=main, 1=preview)
        cfa_pattern: CFA pattern (only used for photometric="CFA")
        compression: Compression to apply. None means COMPRESSION.NONE (uncompressed).
        compression_args: Args for compression (e.g., {'level': 90} for JPEG)
        extratags: Additional metadata tags to add
        bits_per_sample: Bits per sample (e.g., 10, 12, 14 for raw data). None means
            infer from dtype (uint8→8, uint16→16, float16→16, float32→32). Use this to
            specify non-standard bit depths like 10-bit or 12-bit data stored in uint16 arrays.
    """
    data: np.ndarray
    photometric: str
    subfiletype: int = 0
    cfa_pattern: str = "RGGB"
    compression: Optional[COMPRESSION] = None
    compression_args: Optional[dict] = None
    extratags: Optional[MetadataTags] = None
    bits_per_sample: Optional[int] = None
    
    def requires_transcode(self) -> bool:
        """Check if this spec requires transcoding (always False for array data)."""
        return False
    
    def has_jxl_compression(self) -> bool:
        """Check if this spec uses JXL compression."""
        return self.compression in (COMPRESSION.JPEGXL, COMPRESSION.JPEGXL_DNG)


def write_dng(
    destination_file: Union[str, Path, io.BytesIO],
    *,
    ifd0_spec: Union[IfdPageSpec, IfdDataSpec],
    subifds: List[Union[IfdPageSpec, IfdDataSpec]] = None,
) -> None:
    """Write raw data to a DNG file using tifffile.

    Args:
        destination_file: Path or io.BytesIO object where to save the DNG file.
        ifd0_spec: Specification for IFD0. ifd0_spec.extratags contains IFD0-level
            metadata tags. Use subfiletype to control NewSubFileType (0=Main, 1=Preview).
        subifds: Ordered list of images to write as SubIFDs. Can mix IfdPageSpec
            (for copying from source DNGs) and IfdDataSpec (for writing from arrays).
            Each spec's subfiletype controls its NewSubFileType. Defaults to empty list
            if IFD0 contains the main image (no preview).
    """
    def _write_page_ifd(
        writer: TiffWriter,
        page: "DngPage",
        *,
        raw_ifd_args: dict,
    ) -> None:
        # IFD args are already prepared by caller

        # Uncompressed: use numpy array so tifffile handles byte order
        if page.compression == COMPRESSION.NONE:
            raw_data = page.asarray()
            logger.debug(f"Read uncompressed data: {raw_data.shape} {raw_data.dtype}")
            writer.write(
                data=raw_data,
                bitspersample=page.bitspersample,
                **raw_ifd_args,
            )
            logger.debug("Successfully wrote uncompressed raw data")
        else:
            # Compressed: read raw segments
            samples_per_pixel = page.samplesperpixel if hasattr(page, 'samplesperpixel') else 1
            if samples_per_pixel > 1:
                write_shape = (page.imagelength, page.imagewidth, samples_per_pixel)
            else:
                write_shape = (page.imagelength, page.imagewidth)

            fh = page.parent.filehandle
            compressed_segments = list(
                fh.read_segments(page.dataoffsets, page.databytecounts, sort=True)
            )
            logger.debug(f"Read {len(compressed_segments)} compressed segments from page")

            def compressed_data_iterator():
                try:
                    for segment_data, _index in compressed_segments:
                        yield segment_data
                except GeneratorExit:
                    return

            if page.is_tiled:
                tile_shape = (page.tilelength, page.tilewidth)
                writer.write(
                    data=compressed_data_iterator(),
                    shape=write_shape,
                    dtype=page.dtype,
                    bitspersample=page.bitspersample,
                    tile=tile_shape,
                    **raw_ifd_args,
                )
                logger.debug(
                    f"Successfully copied tiled compressed data ({sum(page.databytecounts)} bytes)"
                )
            else:
                raw_datasize = (
                    page.imagelength
                    * page.imagewidth
                    * samples_per_pixel
                    * (page.bitspersample // 8)
                )
                writer.write(
                    data=compressed_data_iterator(),
                    shape=write_shape,
                    dtype=page.dtype,
                    bitspersample=page.bitspersample,
                    rowsperstrip=raw_datasize,
                    **raw_ifd_args,
                )
                logger.debug(
                    f"Successfully copied stripped compressed data ({sum(page.databytecounts)} bytes)"
                )

    def _prepare_tags_for_write(tags: MetadataTags, target_byteorder: str):
        """Prepare MetadataTags for writing by converting arrays to target byte order.
        
        Args:
            tags: MetadataTags object with values in system byte order
            target_byteorder: Target file's byte order ('>' or '<')
            
        Returns:
            List of tuples (code, dtype, count, value, writeonce) ready for tifffile
        """
        import sys
        from .tiff_metadata import normalize_array_to_target_byteorder
        from .raw_render import transcode_pgtm_if_needed
        
        system_byteorder = '<' if sys.byteorder == 'little' else '>'
        
        result = []
        # Use public iterator which yields (code, dtype, count, value, writeonce)
        for code, dtype, count, value, writeonce in tags:
            # Transcode PGTM if needed (no-op for non-PGTM tags)
            value = transcode_pgtm_if_needed(code, value, system_byteorder, target_byteorder)
            
            # Normalize arrays if needed (no-op for non-arrays or PGTM bytes)
            value = normalize_array_to_target_byteorder(value, target_byteorder)
            
            result.append((code, dtype, count, value, writeonce))
        
        return result

    def _write_ifd_from_spec(
        writer: TiffWriter,
        spec: Union[IfdPageSpec, IfdDataSpec],
        *,
        is_ifd0: bool = False,
        needs_v1_7_1: bool,
        main_spec: Optional[Union[IfdPageSpec, IfdDataSpec]] = None,
    ) -> None:
        # Get subfiletype from spec
        subfiletype = spec.subfiletype
        
        # ==== handle tags for this IFD ====
        extratags = MetadataTags()

        if isinstance(spec, IfdPageSpec):
            # Determine photometric and strip_tags
            photometric = spec.page.photometric_name
            strip_tags = spec.strip_tags or set()
            
            if spec.requires_transcode():
                # Add compression-invalidated tags to skip set when transcoding
                strip_tags = strip_tags | _COMPRESSION_INVALIDATED_TAGS
            
            # Get page tags with inheritance logic
            if spec.copy_page_tags:
                extratags = spec.page.get_page_tags()
                if is_ifd0:
                    # For IFD0, merge both page tags and IFD0-specific tags
                    extratags |= spec.page.get_ifd0_tags()
        else:
            # IfdDataSpec
            photometric = spec.photometric
            strip_tags = _COMPRESSION_INVALIDATED_TAGS
        
        # User-supplied extratags override - call this after copy above to ensure
        # spec tags take precedence
        extratags |= spec.extratags
        
        # Apply category filtering to all spec-supplied tags
        # Build filter categories based on IFD type and photometric
        filter_categories = ["any", "dng_ifd0", "ifd0", "exif", "dng_profile"] if is_ifd0 else ["any"]
        
        # Add preview category if this is a preview IFD
        if subfiletype in (SubFileType.PREVIEW_IMAGE, SubFileType.ALT_PREVIEW_IMAGE):
            filter_categories += ["dng_preview"]
        
        # Add photometric-specific categories
        if photometric == "LINEAR_RAW":
            filter_categories += ["dng_raw"]
        elif photometric == "CFA":
            filter_categories += ["dng_raw", "dng_raw:cfa"]
        # Non-raw photometric types (RGB, YCBCR) only get "any" category
        
        ifd_tags = filter_tags_by_ifd_category(extratags, filter_categories)

        # Add required IFD0 tags
        if is_ifd0:
            # Backstop: check if digest is likely invalid due to data transformation
            digest_invalid = False
            if main_spec is not None:
                # Case 1: main image is data AND (ifd0 is a page with copytags OR main is doing compression)
                if isinstance(main_spec, IfdDataSpec):
                    has_compression = main_spec.compression is not None and main_spec.compression != COMPRESSION.NONE
                    if (isinstance(spec, IfdPageSpec) and spec.copy_page_tags) or has_compression:
                        digest_invalid = True
                # Case 2: main image is a page with transcode
                elif isinstance(main_spec, IfdPageSpec):
                    if main_spec.requires_transcode():
                        digest_invalid = True
                    # Case 3: main image is a page that was originally a preview (not main)
                    elif not main_spec.page.is_main_image:
                        digest_invalid = True
            
            # If digest is invalid, strip it from tags
            if digest_invalid:
                strip_tags = (strip_tags or set()) | _DIGEST_TAGS
            
            _add_required_ifd0_tags(ifd_tags, needs_v1_7_1=needs_v1_7_1)

        # Step 1: Honor user's strip_tags first
        _filter_metadata_tags(ifd_tags, exclude_names=strip_tags)
        
        # ==== handle tifffile args for this IFD ====
        # Determine compression and args based on spec type
        if isinstance(spec, IfdPageSpec) and not spec.requires_transcode():
            # Fast path: COPY mode - use page's existing compression, no args
            compression = spec.page.compression
            compression_args = None
        else:
            # All other cases: determine compression, then normalize
            if isinstance(spec, IfdPageSpec):
                # IfdPageSpec with transcode required
                if isinstance(spec.page_operation, tuple):
                    # TRANSCODE mode - use specified compression
                    _, compression = spec.page_operation
                    compression_args = spec.compression_args
                else:
                    # Unsupported tile config - use default COMPRESSION.NONE
                    compression = COMPRESSION.NONE
                    compression_args = None
            else:
                # IfdDataSpec - use spec's compression
                compression = spec.compression
                compression_args = spec.compression_args
            
            # Normalize compression type and args
            if compression is None or compression == COMPRESSION.NONE:
                # Uncompressed data doesn't need compression args
                compression = COMPRESSION.NONE
                compression_args = None
            elif compression in (COMPRESSION.JPEGXL, COMPRESSION.JPEGXL_DNG):
                # Normalize JXL compression variants to JPEGXL_DNG
                compression = COMPRESSION.JPEGXL_DNG
            elif compression == COMPRESSION.JPEG:
                # Default to lossless JPEG for raw data if no args provided
                if compression_args is None and photometric in ("LINEAR_RAW", "CFA"):
                    compression_args = {'lossless': True}
        
        # Step 2: Extract tags for tifffile args (before filtering managed tags)
        # This extracts and removes Software, XResolution, YResolution, ResolutionUnit, PlanarConfiguration
        ifd_args = _prepare_ifd_args(
            ifd_tags,
            compression,
            is_ifd0,
            subfiletype,
            photometric,
            len(subifds) if is_ifd0 else 0,
            compression_args,
        )
        
        # Step 3: Strip remaining _TIFFWRITER_MANAGED_TAGS
        _filter_metadata_tags(ifd_tags, exclude_names=_TIFFWRITER_MANAGED_TAGS)
        
        # Fast path: copy page data as-is (no transcode)
        if isinstance(spec, IfdPageSpec) and not spec.requires_transcode():
            # Prepare tags for write (convert arrays to target byte order)
            ifd_args["extratags"] = _prepare_tags_for_write(ifd_tags, writer.tiff.byteorder)
            _write_page_ifd(writer, spec.page, raw_ifd_args=ifd_args)
            return

        # Extract data from spec
        if isinstance(spec, IfdPageSpec):
            # IfdPageSpec with TRANSCODE: extract data from page
            # (we only reach here if spec.requires_transcode() is True)
            # Warn only if we're falling back due to unsupported tiles (not explicit transcode)
            if not isinstance(spec.page_operation, tuple) and spec.page.is_tiled:
                tile_shape = (spec.page.tilelength, spec.page.tilewidth)
                logger.warning(
                    "Falling back to decoded DNG write path for tiled page with unsupported tile %s "
                    "(must be <= image size and multiple of 16 for tifffile direct copy).",
                    tile_shape,
                )
            bits_per_sample = int(spec.page.bitspersample)

            if photometric == "CFA":
                cfa_result = spec.page.get_cfa()
                if cfa_result is None:
                    raise ValueError("Failed to extract CFA data from page")
                data, _ = cfa_result
                samples_per_pixel = 1
            elif photometric == "LINEAR_RAW":
                data = spec.page.get_linear_raw()
                if data is None:
                    raise ValueError("Failed to extract LINEAR_RAW data from page")
                samples_per_pixel = 3
            else:
                # Non-raw photometric (RGB, YCBCR, etc.) - use asarray like decode() does
                try:
                    data = spec.page.asarray()
                except Exception as e:
                    raise ValueError(f"Failed to extract data from page (photometric={photometric}): {e}")
                samples_per_pixel = 3
        else:
            # IfdDataSpec: data from array
            if spec.data.dtype not in (np.uint8, np.uint16, np.float16, np.float32):
                raise ValueError(
                    f"Unsupported dtype {spec.data.dtype}. Supported: uint8, uint16, float16, float32"
                )
            
            # Use explicit bits_per_sample if provided, otherwise infer from dtype
            if spec.bits_per_sample is not None:
                bits_per_sample = spec.bits_per_sample
                
                # Validate bits_per_sample is compatible with dtype
                dtype_bits = spec.data.dtype.itemsize * 8
                if spec.data.dtype in (np.float16, np.float32):
                    # Float types: bits_per_sample must match dtype exactly
                    if bits_per_sample != dtype_bits:
                        raise ValueError(
                            f"bits_per_sample={bits_per_sample} incompatible with float dtype "
                            f"{spec.data.dtype} (must be {dtype_bits})"
                        )
                else:
                    # Integer types: bits_per_sample must be <= dtype capacity
                    if bits_per_sample > dtype_bits:
                        raise ValueError(
                            f"bits_per_sample={bits_per_sample} exceeds dtype {spec.data.dtype} "
                            f"capacity ({dtype_bits} bits)"
                        )
                    if bits_per_sample < 8:
                        raise ValueError(f"bits_per_sample must be >= 8, got {bits_per_sample}")
            else:
                bits_per_sample = spec.data.dtype.itemsize * 8
            
            data = spec.data
            
            # Add CFA tags for array data with CFA photometric
            if photometric == "CFA":
                ifd_tags.add_tag("CFAPattern", spec.cfa_pattern)
                ifd_tags.add_tag("CFARepeatPatternDim", (2, 2))
                ifd_tags.add_tag("CFAPlaneColor", bytes([0, 1, 2]))
                samples_per_pixel = 1
            else:
                # linear_raw, rgb, ycbcr all have 3 channels
                samples_per_pixel = 3
        
        datasize = int((data.shape[0] * data.shape[1] * samples_per_pixel * bits_per_sample) / 8)

        # Write IFD with compression
        if compression in (COMPRESSION.JPEGXL, COMPRESSION.JPEGXL_DNG):
            # JPEGXL requires manual encoding with imagecodecs
            jxl_distance = compression_args.get('distance', 0.0) if compression_args else 0.0
            jxl_effort = compression_args.get('effort', 5) if compression_args else 5
            
            if not (0.0 <= jxl_distance <= 15.0):
                logger.warning(f"JXL distance {jxl_distance} is outside the typical range [0.0, 15.0].")

            ifd_tags.add_tag("JXLDistance", jxl_distance)
            ifd_tags.add_tag("JXLEffort", jxl_effort)
            if photometric == "CFA":
                data = swizzle_cfa_data(data)
                ifd_tags.add_tag("ColumnInterleaveFactor", 2)
                ifd_tags.add_tag("RowInterleaveFactor", 2)

            # JXL supports storing bitspersample in the bitstream, but dng_validate and Photoshop
            # dont use this the way we'd expect. On decode, instead of using 
            # JXL_BIT_DEPTH_FROM_CODESTREAM, they request that the jxl decoder return values scaled 
            # to the container bitdepth (eg 16bits for 10bit data). This causes a disconnect
            # with the bits_per_sample in the IFD.
            # So we shift 9-15 bit data to 16-bit before encoding to avoid this. 
            jxl_encode_bits = bits_per_sample
            if 9 <= bits_per_sample <= 15:
                data = data << (16 - bits_per_sample)
                bits_per_sample = 16  # Update for TIFF tag and JXL encoder

            encoded_bytes = imagecodecs.jpegxl_encode(
                data, distance=jxl_distance, effort=jxl_effort, bitspersample=bits_per_sample
            )
            def encoded_data_iterator():
                yield encoded_bytes

            # Prepare tags after all additions are complete
            ifd_args["extratags"] = _prepare_tags_for_write(ifd_tags, writer.tiff.byteorder)
            writer.write(
                data=encoded_data_iterator(),
                shape=data.shape,
                dtype=data.dtype,
                bitspersample=bits_per_sample,
                rowsperstrip=datasize,
                **ifd_args,
            )
        else:
            # All other compression types - let tifffile handle encoding
            # Prepare tags after all additions are complete (including CFA tags)
            ifd_args["extratags"] = _prepare_tags_for_write(ifd_tags, writer.tiff.byteorder)
            writer.write(data, bitspersample=bits_per_sample, rowsperstrip=datasize, **ifd_args)

    # Initialize subifds list
    if subifds is None:
        subifds = []
    
    # Scan all IFD specs to find the main image
    all_specs = [ifd0_spec] + subifds
    main_specs = [spec for spec in all_specs if spec.subfiletype == SubFileType.MAIN_IMAGE]
    
    if len(main_specs) > 1:
        raise ValueError(f"Multiple main images found (subfiletype={SubFileType.MAIN_IMAGE}). Only one main image is allowed.")
    
    main_spec = main_specs[0] if main_specs else None
    
    # Check if any IFD uses JXL compression
    needs_v1_7_1 = any(s.has_jxl_compression() for s in all_specs)

    try:
        with TiffWriter(destination_file, bigtiff=False, byteorder='<') as tif:

            # Write IFD0
            _write_ifd_from_spec(
                tif,
                ifd0_spec,
                is_ifd0=True,
                needs_v1_7_1=needs_v1_7_1,
                main_spec=main_spec,
            )

            # Write subifds
            for spec in subifds:
                _write_ifd_from_spec(
                    tif, spec, needs_v1_7_1=needs_v1_7_1, main_spec=main_spec
                )

        if isinstance(destination_file, Path):
            logger.debug(f"Successfully wrote DNG file to {destination_file}")
        else:
            logger.debug("Successfully wrote DNG file to in-memory buffer")

    except Exception as e:
        logger.error(f"Error saving DNG file with TiffWriter: {e}")
        raise


def create_dng(
    *,
    ifd0_spec: Union[IfdPageSpec, IfdDataSpec],
    subifds: List[Union[IfdPageSpec, IfdDataSpec]] = None,
) -> "DngFile":
    """Create a DNG file in memory and return as DngFile object.
    
    This is a convenience wrapper around write_dng that writes to an in-memory
    BytesIO stream and returns the result as a DngFile.
    
    Args:
        ifd0_spec: Specification for IFD0
        subifds: Ordered list of images to write as SubIFDs
        
    Returns:
        DngFile object loaded from the in-memory DNG
    """
    buffer = io.BytesIO()
    write_dng(destination_file=buffer, ifd0_spec=ifd0_spec, subifds=subifds)
    buffer.seek(0)
    return DngFile(buffer)

@dataclass
class PreviewParams:
    """Parameters for preview generation in DNG files.
    
    If this object is provided (not None), a preview will be generated.
    If None, no preview is generated.
    
    Attributes:
        max_dimension: Maximum dimension for preview (default: 1024)
        compression: Compression type for preview (default: JPEG)
        compression_args: Arguments for compression (default: {'level': 90} for JPEG)
        rendering_params: Override rendering params (Temperature, Tint, etc.)
        use_xmp: Use XMP metadata for preview rendering (default: True)
    """
    max_dimension: int = 1024
    compression: Optional[COMPRESSION] = None
    compression_args: Optional[dict] = None
    rendering_params: Optional[dict] = None
    use_xmp: bool = True


@dataclass
class PyramidParams:
    """Parameters for pyramid level generation in DNG files.
    
    Attributes:
        levels: Number of pyramid levels to generate (0=none)
        compression: Compression type for pyramid levels (default: JPEGXL_DNG)
        compression_args: Arguments for compression (default: {'distance': 0.5, 'effort': 5})
        extratags: Additional metadata tags to add to each pyramid level
    """
    levels: int = 0
    compression: Optional[COMPRESSION] = None
    compression_args: Optional[dict] = None
    extratags: Optional["MetadataTags"] = None


def create_dng_from_array(
    data_spec: IfdDataSpec,
    *,
    preview: Optional[PreviewParams] = None,
    pyramid: Optional[PyramidParams] = None,
) -> "DngFile":
    """Create a DNG file from array data in memory and return as DngFile object.
    
    This is a convenience wrapper around write_dng_from_array that writes to an
    in-memory BytesIO stream and returns the result as a DngFile.
    
    Args:
        data_spec: IfdDataSpec containing raw image data and metadata
        preview: Optional preview generation parameters
        pyramid: Optional pyramid level generation parameters
        
    Returns:
        DngFile object loaded from the in-memory DNG
        
    Example:
        >>> data_spec = IfdDataSpec(data=raw_array, photometric="CFA", ...)
        >>> preview = PreviewParams(max_dimension=1024)  # Presence means generate preview
        >>> dng = create_dng_from_array(data_spec, preview=preview)
    """
    buffer = io.BytesIO()
    write_dng_from_array(
        destination_file=buffer,
        data_spec=data_spec,
        preview=preview,
        pyramid=pyramid,
    )
    buffer.seek(0)
    return DngFile(buffer)


def write_dng_from_array(
    destination_file: Union[str, Path, io.BytesIO],
    data_spec: IfdDataSpec,
    *,
    preview: Optional[PreviewParams] = None,
    pyramid: Optional[PyramidParams] = None,
) -> None:
    """Write raw array data to a DNG file with optional preview and pyramid generation.
    
    Args:
        destination_file: Path or io.BytesIO object where to save the DNG file
        data_spec: IfdDataSpec containing raw image data and metadata
        preview: PreviewParams for preview generation (None = no preview)
        pyramid: PyramidParams for pyramid generation (None = no pyramid)
    """
    
    # Create uncompressed temporary page spec (compression removed)
    temp_data_spec = IfdDataSpec(
        data=data_spec.data,
        photometric=data_spec.photometric,
        cfa_pattern=data_spec.cfa_pattern,
        extratags=data_spec.extratags,
        bits_per_sample=data_spec.bits_per_sample,
    )
    
    # Create in-memory uncompressed DngFile from the temp spec
    dng_file = create_dng(ifd0_spec=temp_data_spec)
    
    # Extract the main page
    main_page = dng_file.get_main_page()
    if main_page is None:
        raise RuntimeError("Failed to create DNG from array data")
    
    # Create IfdPageSpec with TRANSCODE mode
    # Data is already uncompressed in temp DNG, ready for (re)compression
    # Fast path in write_dng_from_page handles compression-only changes without demosaicing
    page_spec = IfdPageSpec(
        page=main_page,
        page_operation=(PageOp.TRANSCODE, data_spec.compression),
        compression_args=data_spec.compression_args,
        extratags=None,  # Already in the page
    )  
    # Delegate to write_dng_from_page with extracted values
    write_dng_from_page(
        destination_file=destination_file,
        page=page_spec,
        preview=preview,
        pyramid=pyramid,
    )


def _generate_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Generate image pyramid levels using 8-tap Lanczos downsampling.
    
    Creates a pyramid where each level is exactly 1/2 x 1/2 of the previous level,
    using an 8-tap Lanczos windowed sinc filter for high-quality downsampling.
    
    Args:
        image: Input image (any dtype, 2D or 3D)
        num_levels: Maximum number of pyramid levels to generate (including level 0)
    
    Returns:
        List of pyramid levels where:
        - levels[0] is the original input image
        - levels[1] is 1/2 x 1/2 of levels[0]
        - levels[2] is 1/2 x 1/2 of levels[1], etc.
        
        Generation stops when reaching num_levels or when min dimension <= 16.
    
    Example:
        >>> img = np.zeros((1000, 800, 3), dtype=np.uint8)
        >>> pyramid = generate_pyramid(img, num_levels=4)
        >>> [p.shape for p in pyramid]
        [(1000, 800, 3), (500, 400, 3), (250, 200, 3), (125, 100, 3)]
    """
    import cv2
    
    def make_lanczos_kernel(a: int = 4) -> np.ndarray:
        """Generate an 8-tap Lanczos kernel for 2:1 downsampling."""
        # Kernel positions for 8-tap, centered between pixels at half-pixel offsets
        positions = np.arange(-a + 0.5, a, 1.0)  # [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        
        # Lanczos function: L(x) = sinc(x) * sinc(x/a)
        # numpy.sinc(x) computes sin(πx)/(πx)
        kernel = np.sinc(positions) * np.sinc(positions / a)
        
        # Normalize to sum to 1.0
        kernel = kernel / kernel.sum()
        
        return kernel.astype(np.float32)
    
    levels = [image]
    
    # Generate 8-tap Lanczos-4 kernel once (reuse for all levels)
    lanczos_kernel = make_lanczos_kernel(a=4)

    current = image
    while len(levels) < num_levels:
        h, w = current.shape[:2]
        
        # Check stopping condition: min dimension <= 16
        if min(h, w) <= 16:
            break
        
        # Calculate next level dimensions (round up on division)
        next_h = (h + 1) // 2
        next_w = (w + 1) // 2
        
        # Check if next level would be too small
        if min(next_h, next_w) <= 16:
            break
        
        # Downsample using 8-tap Lanczos filter: apply separable filter then subsample
        # anchor=(3, 3) aligns the kernel center (between indices 3 and 4) with output pixels
        filtered = cv2.sepFilter2D(current, -1, lanczos_kernel, lanczos_kernel, 
                                   anchor=(3, 3), borderType=cv2.BORDER_REFLECT_101)
        downsampled = filtered[::2, ::2]
        
        levels.append(downsampled)
        current = downsampled
    
    return levels


# Tags applied during stage1 and stage2 processing (rendering operations)
# that must be stripped when extracting processed data via get_camera_rgb()
STAGE1_STAGE2_TAGS = {
    # Stage1 tags (CFA-specific tags from registry with dng_ifd="dng_raw:cfa")
    "OpcodeList1",
    *(tag_name for tag_name, tag_spec in TIFF_TAG_TYPE_REGISTRY.items() if tag_spec.dng_ifd == "dng_raw:cfa"),
    # Interleave factors (applied during CFA JXL encoding)
    "ColumnInterleaveFactor",
    "RowInterleaveFactor",
    # Stage2 tags (rendering operations)
    "BlackLevel",
    "BlackLevelRepeatDim",
    "BlackLevelDeltaH",
    "BlackLevelDeltaV",
    "WhiteLevel",
    "LinearizationTable",
    "ActiveArea",
    "OpcodeList2",
}

# Tags applied during stage3 processing (get_camera_rgb_raw)
# that must be stripped when extracting camera RGB data
STAGE3_TAGS = {
    "OpcodeList3",
    "DefaultCropOrigin",
    "DefaultCropSize",
}


def write_dng_from_page(
    destination_file: Union[str, Path, io.BytesIO],
    page: Union[IfdPageSpec, DngPage],
    *,
    scale: float = 1.0,
    demosaic: bool = False,
    demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.OPENCV_EA,
    preview: Optional[PreviewParams] = None,
    pyramid: Optional[PyramidParams] = None,
    copy_ifd0_tags: bool = True,
    ifd0_extratags: Optional["MetadataTags"] = None,
    ifd0_strip_tags: Optional[set[str]] = None,
) -> None:
    """Write a DNG file from a page with optional transformations and pyramid generation.
    
    Args:
        destination_file: Destination path or io.BytesIO for in-memory output
        page: Source page (IfdPageSpec or DngPage)
        scale: Scale factor for image (default: 1.0)
        demosaic: If True, convert CFA to LINEAR_RAW
        demosaic_algorithm: Demosaic algorithm to use (default: OPENCV_EA)
        preview: PreviewParams for preview generation (None = no preview)
        pyramid: PyramidParams for pyramid generation (None = no pyramid)
        copy_ifd0_tags: Copy IFD0 tags from source (default: True)
        ifd0_extratags: Additional metadata tags to add to IFD0
        ifd0_strip_tags: Tag names to strip from IFD0
    
    Raises:
        ValueError: If input is invalid
        RuntimeError: If DNG processing fails
    """
    
    # Ensure we have an IfdPageSpec
    source_page_spec = page if isinstance(page, IfdPageSpec) else IfdPageSpec(page=page)
    
    if not (source_page_spec.page.is_cfa or source_page_spec.page.is_linear_raw):
        raise ValueError(
            f"Page must be CFA or LINEAR_RAW, got photometric={source_page_spec.page.photometric_name}"
        )

    # do we change the main page pixels?
    main_needs_transform = ((demosaic and source_page_spec.page.is_cfa) or scale != 1.0)

    # generate custom preview tags
    preview_tags = MetadataTags()
    preview_tags.add_tag("PreviewApplicationName", "muimg")
    preview_tags.add_tag("PreviewApplicationVersion", "1.0.0")

    # handle ifd0 tags
    ifd0_tags = MetadataTags()
    if copy_ifd0_tags:
        ifd0_tags = source_page_spec.page.get_ifd0_tags()
        filter_tags_by_ifd_category(ifd0_tags, ["any", "dng_ifd0", "ifd0", "exif", "dng_profile"])
    if preview:
        ifd0_tags |= preview_tags
    _filter_metadata_tags(ifd0_tags, exclude_names=ifd0_strip_tags)
    ifd0_tags |= ifd0_extratags

    # handle main page tags
    main_page_tags = MetadataTags()
    if source_page_spec.copy_page_tags:
        main_page_tags |= source_page_spec.page.get_page_tags()

        main_page_categories = ["any"]
        if source_page_spec.page.is_linear_raw or main_needs_transform:
            main_page_categories += ["dng_raw"]
        elif source_page_spec.page.is_cfa:
            main_page_categories += ["dng_raw", "dng_raw:cfa"]
        filter_tags_by_ifd_category(main_page_tags, main_page_categories)
        _filter_metadata_tags(main_page_tags, exclude_names=source_page_spec.strip_tags)
    if not preview:
        main_page_tags |= ifd0_tags
    main_page_tags |= source_page_spec.extratags    

    # handle pyramid tags
    pyramid_tags = preview_tags | (pyramid.extratags if pyramid else None)

    # If no demosaic/scale needed then the main_spec is the incoming page spec
    if not main_needs_transform:
        logger.info("Using fast path for main spec (no demosaic/scale)")
        main_spec = replace(
            source_page_spec, 
            subfiletype=SubFileType.MAIN_IMAGE, 
            extratags=main_page_tags,
            copy_page_tags=False)
        
        # Fast path: no preview/pyramid - write and return immediately
        if preview is None and not (pyramid and pyramid.levels > 0):
            write_dng(destination_file=destination_file, ifd0_spec=main_spec)
            if isinstance(destination_file, io.BytesIO):
                logger.info("Successfully wrote DNG to stream")
            else:
                logger.info(f"Successfully wrote DNG to {destination_file}")
            return
    else:
        # Only strip stage and digest tags if we transcoded (transformed the data)
        tags_to_strip = STAGE1_STAGE2_TAGS | STAGE3_TAGS | _DIGEST_TAGS
    
        # Filter tags
        _filter_metadata_tags(ifd0_tags, exclude_names=tags_to_strip)
        _filter_metadata_tags(main_page_tags, exclude_names=tags_to_strip)

    # Extract camera RGB (always needed if we didn't take fast return path)
    logger.info(f"Extracting camera RGB (demosaic={demosaic}, scale={scale})")
    camera_rgb = source_page_spec.page.get_camera_rgb_raw(demosaic_algorithm)
    if camera_rgb is None:
        raise RuntimeError("Failed to extract camera RGB")
    
    # Apply scaling if needed
    if scale != 1.0:
        logger.info(f"Applying scale: {scale}")
        h, w = camera_rgb.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        import cv2
        camera_rgb = cv2.resize(camera_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Compute pyramid levels needed and find best level for preview
    num_pyramid_levels = 1  # Level 0 is always the original
    preview_level_idx = 0  # Default to level 0 if no preview
    
    # Calculate levels needed for preview
    if preview:
        h, w = camera_rgb.shape[:2]
        max_dim = max(h, w)
        levels_for_preview = 1
        while max_dim / (2 ** levels_for_preview) > preview.max_dimension:
            levels_for_preview += 1
        # Best preview level is the one just larger than preview.max_dimension
        preview_level_idx = max(0, levels_for_preview - 1)
        num_pyramid_levels = max(num_pyramid_levels, levels_for_preview + 1)
    
    # Take max with requested pyramid levels
    if pyramid:
        num_pyramid_levels = max(num_pyramid_levels, pyramid.levels + 1)
    
    # Generate pyramid
    pyramid_images = _generate_pyramid(camera_rgb, num_pyramid_levels)
    logger.info(f"Generated {len(pyramid_images)} pyramid levels")
    
    # Build main raw data spec from transcoded data if we don't already have a page spec
    if main_needs_transform:
        # Transformed data - extract compression from source
        raw_uint16 = raw_render.convert_dtype(pyramid_images[0], np.uint16)
        if isinstance(source_page_spec.page_operation, tuple):
            # Explicit TRANSCODE mode - use specified compression
            _, main_compression = source_page_spec.page_operation
            main_compression_args = source_page_spec.compression_args
        else:
            # COPY mode - preserve source compression unknown args so set to None
            main_compression = source_page_spec.page.compression
            main_compression_args = None

        main_spec = IfdDataSpec(
            data=raw_uint16,
            photometric="LINEAR_RAW",
            subfiletype=SubFileType.MAIN_IMAGE,
            compression=main_compression,
            compression_args=main_compression_args,
            extratags = main_page_tags
        )
    
    # Build pyramid level specs (levels 1+)
    pyramid_specs = []
    if pyramid and pyramid.levels > 0:
        for level_idx in range(1, len(pyramid_images)):
            pyramid_uint16 = raw_render.convert_dtype(pyramid_images[level_idx], np.uint16)
            pyramid_spec = IfdDataSpec(
                data=pyramid_uint16,
                photometric="LINEAR_RAW",
                subfiletype=SubFileType.PREVIEW_IMAGE,
                compression=pyramid.compression,
                compression_args=pyramid.compression_args,
                extratags=pyramid_tags,
            )
            pyramid_specs.append(pyramid_spec)
    
    # Generate rendered preview if requested
    if not preview:
        # No preview: IFD0 = main, SubIFD0+ = pyramid
        write_dng(
            destination_file=destination_file,
            ifd0_spec=main_spec,
            subifds=pyramid_specs,
        )
    else:
        # Use pre-calculated best pyramid level
        preview_rgb = pyramid_images[preview_level_idx]
        
        logger.info(f"Rendering preview from pyramid level {preview_level_idx} ({preview_rgb.shape[:2]})")
        
        # Render with color transforms
        rendered_preview = raw_render._render_camera_rgb(
            ifd0_tags=ifd0_tags,
            raw_ifd_tags=source_page_spec.page.get_page_tags(),
            rgb_camera=preview_rgb,
            output_dtype=np.uint8,
            rendering_params=preview.rendering_params,
            use_xmp=preview.use_xmp,
        )
        
        # Resize rendered preview if needed
        h, w = rendered_preview.shape[:2]
        max_dim = max(h, w)
        if max_dim > preview.max_dimension:
            scale_factor = preview.max_dimension / max_dim
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            
            import cv2
            rendered_preview = cv2.resize(rendered_preview, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized rendered preview to {rendered_preview.shape[:2]}")
        
        # Create preview spec for IFD0
        preview_spec = IfdDataSpec(
            data=rendered_preview,
            photometric="RGB",
            subfiletype=SubFileType.PREVIEW_IMAGE,
            compression=preview.compression,
            compression_args=preview.compression_args,
            extratags=ifd0_tags,
        )
        
        # Write: IFD0 = preview, SubIFD0 = main, SubIFD1+ = pyramid
        write_dng(
            destination_file=destination_file,
            ifd0_spec=preview_spec,
            subifds=[main_spec] + pyramid_specs,
        )
    
    if isinstance(destination_file, io.BytesIO):
        logger.info("Successfully wrote DNG to stream")
    else:
        logger.info(f"Successfully wrote DNG to {destination_file}")


def create_dng_from_page(
    page: Union[IfdPageSpec, DngPage],
    *,
    scale: float = 1.0,
    demosaic: bool = False,
    demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.OPENCV_EA,
    preview: Optional[PreviewParams] = None,
    pyramid: Optional[PyramidParams] = None,
    copy_ifd0_tags: bool = True,
    ifd0_extratags: Optional["MetadataTags"] = None,
    ifd0_strip_tags: Optional[set[str]] = None,
) -> "DngFile":
    """Create a DNG file from a page in memory and return as DngFile object.
    
    This is a convenience wrapper around write_dng_from_page that writes to an
    in-memory BytesIO stream and returns the result as a DngFile.
    
    Args:
        page: Source page (IfdPageSpec or DngPage)
        scale: Scale factor for image (default: 1.0)
        demosaic: If True, convert CFA to LINEAR_RAW
        demosaic_algorithm: Demosaic algorithm to use (default: OPENCV_EA)
        preview: Optional preview generation parameters
        pyramid: Optional pyramid level generation parameters
        copy_ifd0_tags: Copy IFD0 tags from source (default: True)
        ifd0_extratags: Additional metadata tags to add to IFD0
        ifd0_strip_tags: Tag names to strip from IFD0
        
    Returns:
        DngFile object loaded from the in-memory DNG
        
    Example:
        >>> page = dng_file.get_main_page()
        >>> preview = PreviewParams(compression=COMPRESSION.JPEG)  # Presence means generate
        >>> pyramid = PyramidParams(levels=2, compression=COMPRESSION.JPEGXL_DNG)
        >>> new_dng = create_dng_from_page(page, scale=0.5, preview=preview, pyramid=pyramid)
    """
    buffer = io.BytesIO()
    write_dng_from_page(
        destination_file=buffer,
        page=page,
        scale=scale,
        demosaic=demosaic,
        demosaic_algorithm=demosaic_algorithm,
        preview=preview,
        pyramid=pyramid,
        copy_ifd0_tags=copy_ifd0_tags,
        ifd0_extratags=ifd0_extratags,
        ifd0_strip_tags=ifd0_strip_tags,
    )
    buffer.seek(0)
    return DngFile(buffer)


def decode_dng(
    file: Union[str, Path, IO[bytes], DngFile, DngPage],
    output_dtype: type = np.uint16,
    demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.OPENCV_EA,
    use_coreimage_if_available: bool = False,
    use_xmp: bool = True,
    rendering_params: dict = None,
    strict: bool = True,
) -> tuple[np.ndarray, "MetadataTags"]:
    """
    Decode a DNG file or page to a numpy array with metadata.
    
    Renders raw pages (CFA or LINEAR_RAW) or decodes preview pages (RGB/YCBCR).
    When passed a file path/DngFile, renders the main raw page.
    When passed a DngPage, renders that specific page.
    
    By default uses the Python SDK pipeline. When use_coreimage_if_available=True,
    uses macOS Core Image pipeline if available (supports XMP and rendering params).
    
    Args:
        file: Path to DNG file, file-like object, DngFile instance, or DngPage instance
        output_dtype: Output numpy data type (np.uint8, np.uint16, np.float16, np.float32)
        demosaic_algorithm: Demosaic algorithm for Python pipeline - "RCD" (default), "VNG", etc.
        use_coreimage_if_available: If True, use (MacOS) Core Image pipeline when available
        use_xmp: Whether to read XMP metadata for processing defaults (both pipelines)
        rendering_params: Optional dict to override rendering parameters. Supported keys:
            - 'Temperature': White balance temperature in Kelvin (float)
            - 'Tint': White balance tint adjustment (float)
            - 'Exposure2012': Exposure compensation in stops (float)
            - 'ToneCurvePV2012': Main tone curve as SplineCurve or list of (x,y) points
            - 'ToneCurvePV2012Red': Red channel tone curve
            - 'ToneCurvePV2012Green': Green channel tone curve
            - 'ToneCurvePV2012Blue': Blue channel tone curve
            - 'crlcp:PerspectiveModel': Lens correction profile
            - 'highlight_preserving_exposure': Use highlight preservation (Python pipeline only)
            - 'orientation': EXIF orientation code (Core Image only)
    
    Returns:
        Tuple of (image, metadata):
            - image: RGB image array with shape (height, width, 3) and specified dtype
            - metadata: MetadataTags containing IFD0 tags
    """
    # Normalize DngPage to DngFile for consistent handling
    if isinstance(file, DngPage):
        file = create_dng_from_page(file)
    
    # Create or use DngFile
    dng_file = file if isinstance(file, DngFile) else DngFile(file)
    
    # Extract metadata
    metadata = dng_file.get_ifd0_tags()
    
    # Try Core Image path if requested
    if use_coreimage_if_available:
        try:
            from ._dngio_coreimage import core_image_available, decode_dng_coreimage

            if core_image_available:
                image = decode_dng_coreimage(
                    file=dng_file,
                    use_xmp=use_xmp,
                    output_dtype=output_dtype,
                    rendering_params=rendering_params,
                )
                return image, metadata

            logger.warning(
                "Core Image requested but not available; falling back to Python pipeline."
            )
        except ImportError:
            logger.warning(
                "Core Image requested but PyObjC/Core Image dependencies are not installed; "
                "falling back to Python pipeline."
            )
    
    # Python SDK pipeline
    result = dng_file.render_raw(
        output_dtype=output_dtype,
        demosaic_algorithm=demosaic_algorithm,
        use_xmp=use_xmp,
        rendering_params=rendering_params,
    )
    if result is None:
        raise RuntimeError(f"No main image page found in DNG file: {file}")
    
    return result, metadata
