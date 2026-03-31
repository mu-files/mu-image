"""DNG file format support for Allsky.

This module handles DNG file creation and related functionality.
"""
import io
import logging
import imagecodecs
import numpy as np
from datetime import datetime

from enum import Enum
import os
import subprocess
from pathlib import Path
from tifffile import COMPRESSION, PHOTOMETRIC, TiffFile, TiffPage, TiffWriter, TIFF
from typing import Optional, Union, List, Dict, Tuple, Any, Type, IO
from dataclasses import dataclass

from . import raw_render

logger = logging.getLogger(__name__)

# tifffile followups:
# - dng_validate expects SubIFD NextIFD == 0, but tifffile writes NextIFD chaining for SubIFDs and does not expose a supported way to force it to zero.
# - Copying compressed tiled pages is not always possible (e.g. tile size / alignment constraints) and can require a decode + re-encode fallback, which currently emits a warning.
# - TiffWriter writes float tag array bytes unchanged; when writing BE files we must ensure float arrays are big-endian (see _ensure_float_tags_be).

class RawStageSelector(str, Enum):
    RAW = "raw"
    LINEARIZED = "linearized"
    LINEARIZED_PLUS_OPS = "linearized_plus_ops"

# Import metadata classes from tiff_metadata module
from .tiff_metadata import (
    MetadataTags,
    TagSpec,
    TIFF_DTYPE_TO_STR,
    special_tag_format,
    TIFF_TAG_TYPE_REGISTRY,
    XmpMetadata,
    get_native_type,
    decode_tag_value,
    resolve_tag
)

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
    
    # NewSubFileType values from DNG spec
    SF_MAIN_IMAGE = 0
    SF_PREVIEW_IMAGE = 1
    SF_TRANSPARENCY_MASK = 2
    SF_PREVIEW_MASK = 8
    SF_ALT_PREVIEW_IMAGE = 65537

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
    
    @property
    def ifd0(self) -> Optional[TiffPage]:
        """Return IFD0, or None if this page IS IFD0."""
        page0 = self.parent.pages[0]
        return None if page0 is self else page0

    def get_ifd0_tags(self) -> MetadataTags:
        """Return a copy of IFD0 tags as a MetadataTags object."""
        src = self.ifd0 or self
        tags = MetadataTags()
        for tag in src.tags.values():
            tags.add_raw_tag(tag.code, tag.dtype, tag.count, tag.value)
        return tags
    
    def get_page_tags(self) -> MetadataTags:
        """Return a copy of all page-level tags as a MetadataTags object."""
        tags = MetadataTags()
        for tag in self.tags.values():
            tags.add_raw_tag(tag.code, tag.dtype, tag.count, tag.value)
        return tags
    
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
        return value == self.SF_MAIN_IMAGE
    
    @property
    def is_preview(self) -> bool:
        """True if this page is a preview image."""
        value = self.get_tag("NewSubfileType") 
        if value is None:
            return False
        return value in (self.SF_PREVIEW_IMAGE, self.SF_ALT_PREVIEW_IMAGE)

    def get_xmp(self) -> Optional[XmpMetadata]:
        """Return XMP metadata as an `XmpMetadata` object."""
        xmp = self.get_tag("XMP")
        return xmp
    
    def get_tag(
        self, 
        tag: Union[str, int], 
        return_type: Optional[type] = None
    ) -> Optional[Any]:
        """Get a tag value with type conversion.
        
        Args:
            tag: Either a numeric tag code (int) or tag name string 
                 (e.g., 'ColorMatrix1', 'CFAPattern')
            return_type: Controls type conversion:
                - None (default): auto-convert to native type from TagSpec
                - specific type (str, float, list, etc.): convert to that type
        
        Returns:
            Tag value (converted based on return_type), or None if not found.
            
        See also:
            get_raw_tag: Returns raw tag value without any conversion.
        """
        # Resolve tag to id/name and get registry spec (for shape info)
        if isinstance(tag, int):
            tag_id = tag
            _, tag_name, registry_spec = resolve_tag(tag)
            if tag_name is None:
                tag_name = str(tag)
        else:
            tag_id, tag_name, registry_spec = resolve_tag(tag)
            if tag_id is None:
                logger.warning(f"Tag '{tag}' not found in LOCAL_TIFF_TAGS.")
                return None
        
        # Get value from page, or fall back to IFD0 for global tags
        raw_tag = None
        if tag_id in self.tags:
            raw_tag = self.tags[tag_id]
        elif self.ifd0 is not None and tag_id in self.ifd0.tags:
            raw_tag = self.ifd0.tags[tag_id]
        
        if raw_tag is None:
            return None

        # Check for special formatting (XMP, DNGVersion, etc.)
        special_value = special_tag_format(tag_name, raw_tag.value, raw_tag.dtype, return_type)
        if special_value is not None:
            return special_value

        shape_spec = None
        if registry_spec and registry_spec.shape and registry_spec.count == raw_tag.count:
            shape_spec = TagSpec(TIFF_DTYPE_TO_STR.get(raw_tag.dtype, 'B'), raw_tag.count, registry_spec.shape)

        effective_type = return_type or get_native_type(raw_tag.dtype, raw_tag.count)
        return decode_tag_value(tag_name, raw_tag.value, raw_tag.dtype, shape_spec, effective_type)

    def get_raw_tag(self, tag: Union[str, int]) -> Optional[Any]:
        """Get raw tag value without any type conversion.
        
        Args:
            tag: Either a numeric tag code (int) or tag name string
        
        Returns:
            Raw tag value as stored, or None if not found.
            
        See also:
            get_tag: Returns tag value with automatic or specified type conversion.
        """
        # For raw access, we can work with tags not in the registry
        if isinstance(tag, int):
            tag_id = tag
        else:
            tag_id, _, _ = resolve_tag(tag)
            if tag_id is None:
                logger.warning(f"Tag '{tag}' not found in LOCAL_TIFF_TAGS.")
                return None
        
        if tag_id in self.tags:
            return self.tags[tag_id].value
        
        # Fall back to IFD0 for global tags
        if self.ifd0 is not None and tag_id in self.ifd0.tags:
            return self.ifd0.tags[tag_id].value
        
        return None
    
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
    
    def _decode_jpegxl(self) -> np.ndarray:
        """Decode JPEG XL compressed image data, handling tiled images.
        
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
            return imagecodecs.jpegxl_decode(compressed_data)
        else:
            # Multiple tiles - decode each and assemble
            tile_width = self.tilewidth
            tile_height = self.tilelength
            img_width = self.imagewidth
            img_height = self.imagelength
            samples = self.samplesperpixel or 1
            
            # Determine output dtype from first tile
            first_tile = imagecodecs.jpegxl_decode(segments[0][0])
            dtype = first_tile.dtype
            
            # Create output array - handle both 2D (single sample) and 3D (multi-sample)
            if samples == 1:
                output = np.zeros((img_height, img_width), dtype=dtype)
            else:
                output = np.zeros((img_height, img_width, samples), dtype=dtype)
            
            # Decode and place each tile
            tiles_x = (img_width + tile_width - 1) // tile_width
            for i, (tile_data, _) in enumerate(segments):
                tile = imagecodecs.jpegxl_decode(tile_data)
                
                ty = (i // tiles_x) * tile_height
                tx = (i % tiles_x) * tile_width
                
                # Handle edge tiles that may be smaller
                th = min(tile_height, img_height - ty)
                tw = min(tile_width, img_width - tx)
                output[ty:ty+th, tx:tx+tw] = tile[:th, :tw]
            
            return output

    def _stage1(self) -> Optional["DngPage._RawStage"]:
        if self.is_cfa:
            if self.compression in (COMPRESSION.JPEGXL, COMPRESSION.JPEGXL_DNG):
                raw_cfa = self._decode_jpegxl()
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
        self, stage1: "DngPage._RawStage", apply_ops: bool = True
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

        active_area = self.get_tag("ActiveArea")
        data = stage1.data
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

        if not apply_ops:
            return stage2

        opcode_list2 = self.get_tag("OpcodeList2")
        if opcode_list2 is None:
            return stage2

        try:
            opcodes = raw_render.parse_opcode_list(bytes(opcode_list2))
        except Exception as e:
            logger.warning(f"Failed to parse OpcodeList2: {e}")
            return stage2

        try:
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
            stage_out = self._stage2(stage1, apply_ops=False)
        elif stage == RawStageSelector.LINEARIZED_PLUS_OPS:
            stage_out = self._stage2(stage1, apply_ops=True)
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
            return self._stage2(stage1, apply_ops=False).data
        if stage == RawStageSelector.LINEARIZED_PLUS_OPS:
            return self._stage2(stage1, apply_ops=True).data
        raise ValueError(f"Unknown stage selector: {stage}")

    def get_camera_rgb_raw(self, demosaic_algorithm: str = "OPENCV_EA") -> Optional[np.ndarray]:
        """Extract the camera-RGB intermediate from a raw page for the color pipeline.

        This corresponds to the `rgb_camera` input passed into
        `raw_render._render_camera_rgb(...)` during `render_raw()`.

        Returns stage2 (normalized + ActiveArea-cropped), applies OpcodeList2 (if
        present), and then:

        - If photometric == LINEAR_RAW: returns the stage2 linear RGB.
        - Else: demosaics the stage2 CFA to camera RGB.

        Args:
            demosaic_algorithm: Demosaic algorithm to use when the source is CFA.

        Returns:
            Camera RGB array (H, W, 3) float32 in [0, 1], or None if extraction fails.
        """
        stage1 = self._stage1()
        if stage1 is None:
            return None
        stage2 = self._stage2(stage1)

        photometric = self.photometric_name

        if photometric == "LINEAR_RAW":
            rgb_camera = stage2.data
            rgb_camera = np.clip(rgb_camera, 0.0, 1.0).astype(np.float32, copy=False)
            return rgb_camera

        cfa_normalized = stage2.data
        cfa_pattern = stage2.cfa_pattern
        if cfa_pattern is None:
            cfa_pattern = "RGGB"

        rgb_camera = raw_render.demosaic(
            cfa_normalized, cfa_pattern, algorithm=demosaic_algorithm
        )
        rgb_camera = np.clip(rgb_camera, 0.0, 1.0).astype(np.float32, copy=False)
        return rgb_camera
    
    def get_preview_rgb(
        self,
        output_dtype: type = np.uint8
    ) -> Optional[np.ndarray]:
        """Decode RGB or YCBCR preview page to RGB array.
        
        For pages with PhotometricInterpretation of RGB or YCBCR.
        Does NOT apply raw rendering pipeline - just decompresses and
        converts color space if needed.
        
        This is for already-rendered preview images embedded in the DNG.
        For raw preview pages (LINEAR_RAW), use render_raw() instead.
        
        Args:
            output_dtype: Output data type (np.uint8 or np.uint16)
            
        Returns:
            RGB image array (H, W, 3) or None if not a preview page
            
        Raises:
            ValueError: If page is not RGB or YCBCR photometric type
        """
        photometric = self.photometric_name
        
        if photometric not in ("RGB", "YCBCR"):
            raise ValueError(
                f"get_preview_rgb() requires RGB or YCBCR page, got {photometric}"
            )
        
        # Decode the image data
        # Note: tifffile automatically converts JPEG-compressed YCBCR to RGB
        try:
            image = self.asarray()
        except Exception as e:
            logger.error(f"Failed to decode preview page: {e}")
            return None
        
        # Convert dtype if needed
        if image.dtype != output_dtype:
            image = raw_render.convert_dtype(image, output_dtype)
        
        return image
    
    def render_raw(
        self,
        output_dtype: type = np.uint16,
        demosaic_algorithm: str = "OPENCV_EA",
        strict: bool = True,
        use_xmp: bool = True,
        rendering_params: dict = None,
    ) -> "np.ndarray | None":
        """Render raw DNG page to RGB image with optional XMP-based adjustments.
        
        This method is specifically for raw pages (CFA or LINEAR_RAW). It demosaics
        CFA data (if needed), applies color matrices, tone curves, and converts to
        output color space. Supports XMP metadata for white balance, exposure, and
        tone curve adjustments.
        
        For RGB/YCBCR preview pages, use get_preview_rgb() instead.
        
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
            if unsupported and not strict:
                logger.warning(
                    f"DNG contains unsupported tags (processing anyway): {', '.join(unsupported)}"
                )

            rgb_camera = self.get_camera_rgb_raw(demosaic_algorithm=demosaic_algorithm)
            if rgb_camera is None:
                logger.error("Failed to extract camera RGB from DNG")
                return None

            # Build rendering parameters dict from XMP and overrides (filters out NOOP values)
            extracted_params = raw_render.supported_xmp_to_dict(self) if use_xmp else {}
            
            # Merge rendering_params overrides (with validation)
            if rendering_params is not None:
                # Render-method specific parameters (not from XMP)
                render_specific_params = {'highlight_preserving_exposure'}
                # Combine XMP params and render-specific params
                supported_params = raw_render.SUPPORTED_XMP_PARAMS | render_specific_params
                
                for key, value in rendering_params.items():
                    if key not in supported_params:
                        raise ValueError(
                            f"Unsupported rendering parameter: {key}. "
                            f"Supported: {supported_params}"
                        )
                    extracted_params[key] = value

            result = raw_render._render_camera_rgb(
                page=self,
                rgb_camera=rgb_camera,
                output_dtype=output_dtype,
                rendering_params=extracted_params if extracted_params else None,
            )
            return result

        except raw_render.UnsupportedDNGTagError:
            raise
        except Exception as e:
            logger.error(f"Error rendering DNG: {e}", exc_info=True)
            return None

class DngFile(TiffFile):

    """A TIFF file with DNG-specific extensions and helper methods."""

    def __init__(self, file, *args, **kwargs):
        # Normalize file input to BytesIO for consistent in-memory operation
        if isinstance(file, (str, Path)):
            # Read file into memory
            with open(file, 'rb') as f:
                file_data = f.read()
            file = io.BytesIO(file_data)
        elif isinstance(file, io.IOBase):
            # Ensure we're at the beginning for consistent behavior
            file.seek(0)
        # For any other type, let TiffFile handle it and potentially fail with a clear error
        
        super().__init__(file, *args, **kwargs)

    def _iter_all_pages_recursive(self, pages_list: Optional[List[TiffPage]]):
        """Recursively iterates through all TIFF pages, including nested ones."""
        if pages_list is None:
            return
        for page in pages_list:
            yield page
            if page.pages:  # Check if there are sub-pages
                yield from self._iter_all_pages_recursive(page.pages)

    def _build_dng_pages_recursive(self, pages_list: Optional[List[TiffPage]]) -> List[DngPage]:
        """Build DngPage instances from TiffPages."""
        result = []
        if pages_list is None:
            return result
        
        for tiff_page in pages_list:
            dng_page = DngPage(tiff_page)
            result.append(dng_page)
            
            # Recursively process sub-pages:
            result.extend(self._build_dng_pages_recursive(tiff_page.pages))
        
        return result

    def get_flattened_pages(self) -> List[DngPage]:
        """Get all pages as DngPage instances.
        
        Returns:
            List of DngPage objects in flattened order. Tag inheritance
            falls back to IFD0 via TiffPage.parent.
        """
        return self._build_dng_pages_recursive(self.pages)
    
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

    def _forward_main_page(self, method_name: str, *args, require=None, **kwargs):
        page = self.get_main_page()
        if page is None:
            return None
        if require is not None and not require(page):
            return None
        method = getattr(page, method_name)
        return method(*args, **kwargs)

    def get_ifd0_tags(self) -> MetadataTags:
        """Return a copy of IFD0 tags as a MetadataTags object."""
        if not self.pages:
            return MetadataTags()
        tags = MetadataTags()
        for tag in self.pages[0].tags.values():
            tags.add_raw_tag(tag.code, tag.dtype, tag.count, tag.value)
        return tags

    def get_tag(
        self,
        tag: Union[str, int],
        return_type: Optional[type] = None,
    ) -> Optional[Any]:
        """See `DngPage.get_tag`."""
        return self._forward_main_page(
            "get_tag",
            tag,
            return_type=return_type,
        )

    def get_xmp(self) -> Optional[XmpMetadata]:
        """See `DngPage.get_xmp`."""
        return self._forward_main_page("get_xmp")

    def get_time_from_tags(self, time_type: str = "original") -> Optional[datetime]:
        """See `DngPage.get_time_from_tags`."""
        return self._forward_main_page(
            "get_time_from_tags",
            time_type=time_type,
        )

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

    def get_camera_rgb_raw(self, demosaic_algorithm: str = "OPENCV_EA") -> Optional[np.ndarray]:
        """See `DngPage.get_camera_rgb_raw`."""
        return self._forward_main_page(
            "get_camera_rgb_raw",
            demosaic_algorithm=demosaic_algorithm,
        )

    def get_preview_rgb(
        self,
        output_dtype: type = np.uint8,
    ) -> Optional[np.ndarray]:
        """Decode RGB or YCBCR preview page from IFD0 to RGB array.
        
        In most DNG files, IFD0 contains a preview image. This method decodes
        that preview. If IFD0 is a raw page instead, raises ValueError.
        
        For accessing preview pages at other IFD indices, use
        get_flattened_pages()[ifd].get_preview_rgb().
        
        See `DngPage.get_preview_rgb` for full documentation.
        
        Args:
            output_dtype: Output data type (np.uint8 or np.uint16)
            
        Returns:
            RGB image array or None if decoding fails
            
        Raises:
            ValueError: If IFD0 is a raw page instead of a preview
        """
        pages = self.get_flattened_pages()
        if not pages:
            raise ValueError("DNG file has no pages")
        
        ifd0 = pages[0]
        if ifd0.is_cfa or ifd0.is_linear_raw:
            raise ValueError(
                f"IFD0 is a raw page ({ifd0.photometric_name}), not a preview. "
                f"Use render_raw() for raw pages or access preview pages via "
                f"get_flattened_pages()[ifd].get_preview_rgb()"
            )
        
        return ifd0.get_preview_rgb(output_dtype=output_dtype)

    def render_raw(
        self,
        output_dtype: type = np.uint16,
        demosaic_algorithm: str = "OPENCV_EA",
        strict: bool = True,
        use_xmp: bool = True,
        rendering_params: dict = None,
    ) -> "np.ndarray | None":
        """Render main raw DNG page to RGB image.
        
        To render a specific raw page (e.g., LINEAR_RAW preview at a different
        resolution), use get_flattened_pages()[ifd].render_raw() instead.
        
        See `DngPage.render_raw` for full documentation.
        
        Args:
            output_dtype: Output data type (np.uint8 or np.uint16)
            demosaic_algorithm: Algorithm for CFA demosaicing
            strict: If True, raise error on unsupported DNG tags
            use_xmp: If True, extract rendering parameters from XMP metadata
            rendering_params: Optional dict to override rendering parameters
        
        Returns:
            Rendered RGB image or None if rendering fails
        """
        return self._forward_main_page(
            "render_raw",
            output_dtype=output_dtype,
            demosaic_algorithm=demosaic_algorithm,
            strict=strict,
            use_xmp=use_xmp,
            rendering_params=rendering_params,
        )


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


def _ensure_float_tags_be(tags: MetadataTags) -> MetadataTags:
    """Convert float arrays to big-endian for TiffWriter.
    
    TiffWriter passes numpy array bytes unchanged, so we must ensure
    float arrays are in big-endian byte order to match our BE file header.
    """
    for code, dtype, count, value, _ in tags:
        if dtype == 11:  # FLOAT
            if isinstance(value, (np.ndarray, np.floating)) and value.dtype.byteorder != '>':
                tags._tags[code] = tags.StoredTag(code, dtype, count, value.astype('>f4'))
    return tags


def _prepare_auto_args(metadata: MetadataTags, is_ifd0: bool) -> dict:
    """Prepare tifffile auto-generated tag parameters.
    
    Controls Software, ImageDescription, and metadata parameters to ensure
    these tags only appear in IFD0, not in SubIFDs (DNG spec requirement).
    
    Args:
        metadata: MetadataTags instance (modified in-place to remove handled tags)
        is_ifd0: True if writing to IFD0, False if writing to SubIFD
        
    Returns:
        Dict with 'software' and 'metadata' kwargs for TiffWriter.write()
    """
    auto_args = {}
    
    if is_ifd0:
        # IFD0: Extract Software from metadata if present
        if 'Software' in metadata:
            auto_args['software'] = metadata.get_tag('Software')
            metadata.remove_tag('Software')
    else:
        # SubIFDs: Prevent tifffile from adding these tags - DNG spec only allows them in ifd0
        auto_args['software'] = False  # Prevents tifffile default 'tifffile.py'
        auto_args['metadata'] = None   # Prevents auto-generated {"shape": [...]}
    
    return auto_args

def _write_thumbnail_ifd(writer: TiffWriter, thumbnail_image: np.ndarray, dng_tags: MetadataTags, subifds_count: int = 1) -> None:
    """Write thumbnail IFD exactly as in write_dng function."""
    # Prepare thumbnail specific tags

    _PREVIEWCOLORSPACE_SRGB = 1
    dng_tags.add_tag("PreviewColorSpace", _PREVIEWCOLORSPACE_SRGB)

    # Write Thumbnail to SubIFD 0
    thumb_ifd_args = {
        "photometric": "rgb",  # Interprets data as RGB
        "planarconfig": 1,  # Standard for RGB: 1 = CONTIG
        "compression": "jpeg",  # JPEG compression for thumbnail
        "compressionargs": {"level": 90},  # JPEG quality (0-100, higher is better)
        "extratags": dng_tags,
        "subfiletype": 1,  # Reduced resolution image (standard for DNG previews)
        "subifds": int(subifds_count),
    }
    
    # Add auto-generated tag handling (thumbnail is IFD0)
    auto_args = _prepare_auto_args(dng_tags, is_ifd0=True)
    thumb_ifd_args.update(auto_args)
    
    # set datasize to max uncompressed size to avoid writing strips
    datasize = thumbnail_image.shape[0] * thumbnail_image.shape[1] * 3
    writer.write(
        thumbnail_image,  # Use the thumbnail_image directly
        **thumb_ifd_args,
        rowsperstrip=datasize,
    )


def _prepare_ifd_args(metadata: MetadataTags, compression, include_ifd0_tags: bool, **base_args) -> dict:
    """Prepare TiffWriter IFD args, extracting tags that have kwargs equivalents.
    
    Extracts Software, XResolution, YResolution, ResolutionUnit from metadata,
    converts to TiffWriter kwargs format, removes them from metadata, and
    merges with the provided base args.
    
    Args:
        metadata: MetadataTags instance to use as extratags (modified in-place)
        compression: COMPRESSION enum value for this IFD
        include_ifd0_tags: True if writing to IFD0, False if writing to SubIFD
        **base_args: Base IFD args (photometric, subfiletype, etc.)
        
    Returns:
        Dict with complete IFD args including extratags and extracted kwargs
    """
    # Start with base args
    ifd_args = dict(base_args)
    ifd_args["compression"] = compression
    
    # Handle auto-generated tags
    auto_args = _prepare_auto_args(metadata, is_ifd0=include_ifd0_tags)
    ifd_args.update(auto_args)
    
    # Extract Resolution (need both X and Y)
    x_res = metadata.get_tag('XResolution') if 'XResolution' in metadata else None
    y_res = metadata.get_tag('YResolution') if 'YResolution' in metadata else None
    
    if x_res is not None and y_res is not None:
        # Convert rational tuples to float if needed
        if isinstance(x_res, tuple):
            x_res = x_res[0] / x_res[1] if x_res[1] else x_res[0]
        if isinstance(y_res, tuple):
            y_res = y_res[0] / y_res[1] if y_res[1] else y_res[0]
        ifd_args['resolution'] = (x_res, y_res)
        metadata.remove_tag('XResolution')
        metadata.remove_tag('YResolution')
    
    # Extract ResolutionUnit
    if 'ResolutionUnit' in metadata:
        ifd_args['resolutionunit'] = metadata.get_tag('ResolutionUnit')
        metadata.remove_tag('ResolutionUnit')
    
    ifd_args["extratags"] = metadata

    return ifd_args

def _prepare_ifd0_tags(metadata: Optional[MetadataTags], has_jxl: bool) -> MetadataTags:
    """Create DNG metadata tags with defaults and version info.
    
    Tag names are from tifffile.py TiffTagRegistry.
    Tag types are from tifffile.py DATA_DTYPES.
    
    Args:
        metadata: Optional user-supplied metadata to include
        has_jxl: Whether JXL compression is being used (affects backward version)
        
    Returns:
        MetadataTags object with complete DNG metadata
    """
    dng_tags = MetadataTags()

    # Use metadata if provided, otherwise create empty metadata
    dng_tags.extend(metadata)
    
    # Add required tags if not already set
    if "Orientation" not in dng_tags:
        _ORIENTATION_HORIZONTAL = 1
        
        dng_tags.add_tag("Orientation", _ORIENTATION_HORIZONTAL)
    
    # Default to sRGB color space with proper matrices for passthrough
    if "ColorMatrix1" not in dng_tags and "ForwardMatrix1" not in dng_tags:
        # ColorMatrix1: XYZ D50 → Camera (sRGB)
        # Needed for correct camera_white computation
        # Computed as: ProPhoto→sRGB @ XYZ_D50→ProPhoto
        # ProPhoto→sRGB is computed in raw_render.py as: XYZ_D65→sRGB @ D50→D65 @ ProPhoto→XYZ_D50
        d50_to_d65 = raw_render.compute_bradford_adaptation(raw_render.D50_xy, raw_render.D65_xy)
        prophoto_to_srgb = raw_render.XYZ_D65_TO_SRGB @ d50_to_d65 @ raw_render.PROPHOTO_RGB_TO_XYZ_D50
        xyz_d50_to_srgb = prophoto_to_srgb @ raw_render.XYZ_D50_TO_PROPHOTO_RGB
        dng_tags.add_tag("ColorMatrix1", xyz_d50_to_srgb)

        # ForwardMatrix1: Camera (sRGB) → PCS (XYZ D50)
        # Provides direct mapping and bypasses ColorMatrix1 scaling
        # Computed as: D65→D50 @ sRGB→XYZ_D65
        d65_to_d50 = raw_render.compute_bradford_adaptation(raw_render.D65_xy, raw_render.D50_xy)
        forward_matrix1 = d65_to_d50 @ raw_render.SRGB_TO_XYZ_D65
        dng_tags.add_tag("ForwardMatrix1", forward_matrix1)
    
    if "CalibrationIlluminant1" not in dng_tags:
        dng_tags.add_tag("CalibrationIlluminant1", 23)  # 23 = D50
    
    if "AsShotWhiteXY" not in dng_tags and "AsShotNeutral" not in dng_tags:
        # D50 white point in xy chromaticity coordinates
        dng_tags.add_tag("AsShotWhiteXY", [0.34567, 0.35850])
    
    if "AnalogBalance" not in dng_tags:
        # Neutral analog balance
        dng_tags.add_tag("AnalogBalance", [1.0, 1.0, 1.0])
    
    def _version_bytes(major: int, minor: int, patch: int, build: int) -> bytes:
        return bytes([major, minor, patch, build])

    # Set DNGVersion and DNGBackwardVersion
    default_ver = _version_bytes(1, 7, 1, 0)
    
    # Set DNGVersion if not present
    if "DNGVersion" not in dng_tags:
        dng_tags.add_tag("DNGVersion", default_ver)
        default_back = _version_bytes(1, 7, 1, 0) if has_jxl else _version_bytes(1, 4, 0, 0)
    else:
        default_back = dng_tags.get_tag("DNGVersion")
    
    # Set DNGBackwardVersion if not present
    if "DNGBackwardVersion" not in dng_tags:
        dng_tags.add_tag("DNGBackwardVersion", default_back)
        
    return dng_tags


# =============================================================================
# DNG Tag Copy Allowlist
# =============================================================================
# Used when copying tags from source DNG files. User-provided tags bypass this.

# Tags TiffWriter manages automatically - never copy as extratags
_TIFFWRITER_MANAGED_TAGS = {
    'NewSubfileType', 'SubfileType', 'ImageWidth', 'ImageLength',
    'BitsPerSample', 'Compression', 'PhotometricInterpretation',
    'ImageDescription', 'StripOffsets', 'SamplesPerPixel', 'SampleFormat',
    'RowsPerStrip', 'StripByteCounts', 'XResolution', 'YResolution',
    'PlanarConfiguration', 'ResolutionUnit', 'Software',
    'TileWidth', 'TileLength', 'TileOffsets', 'TileByteCounts',
    'SubIFDs', 'ExifTag', 'GPSTag', 'InteroperabilityTag',
}

# JPEG thumbnail tags - not valid for raw data IFDs
_JPEG_THUMBNAIL_TAGS = {
    'YCbCrCoefficients', 'YCbCrSubSampling', 'YCbCrPositioning',
    'ReferenceBlackWhite', 'JPEGInterchangeFormat',
    'JPEGInterchangeFormatLength', 'JPEGTables', 'JPEGRestartInterval',
    'JPEGLosslessPredictors', 'JPEGPointTransforms',
    'JPEGQTables', 'JPEGDCTables', 'JPEGACTables',
}

# Digest tags only valid for main IFD (computed from main raw data)
_DIGEST_TAGS = {
    'NewRawImageDigest', 'RawImageDigest', 'OriginalRawFileDigest',
}

# Tags invalidated when decompressing (digests become stale)
_COMPRESSION_INVALIDATED_TAGS = _DIGEST_TAGS | {
    'CacheVersion', 'JXLDistance', 'JXLEffort', 'JXLDecodeSpeed',
}

# Tags only valid for preview IFDs, not standalone DNGs
_PREVIEW_ONLY_TAGS = {
    'PreviewApplicationName', 'PreviewApplicationVersion',
    'PreviewSettingsName', 'PreviewSettingsDigest',
    'PreviewColorSpace', 'PreviewDateTime', 'RawToPreviewGain',
}

# Base allowlist = registry keys minus structural/thumbnail/preview tags
_DNG_METADATA_ALLOWLIST = (
    set(TIFF_TAG_TYPE_REGISTRY.keys())
    - _TIFFWRITER_MANAGED_TAGS
    - _JPEG_THUMBNAIL_TAGS
    - _PREVIEW_ONLY_TAGS
)


def _get_dng_copy_tags(exclude_compression: bool) -> set[str]:
    """Return set of tag names allowed when copying from source files.
    
    This filters tags read from source DNG files.
    
    Args:
        exclude_compression: True when decompressing (invalidates digests)
        
    Returns:
        Set of allowed tag names
    """
    tags = _DNG_METADATA_ALLOWLIST.copy()
    if exclude_compression:
        tags -= _COMPRESSION_INVALIDATED_TAGS
    return tags


def _filter_metadata_tags(
    tags: MetadataTags,
    *,
    allowed_names: set[str],
) -> MetadataTags:
    filtered = MetadataTags()
    for code, dtype, count, value, _ in tags:
        _, tag_name, _ = resolve_tag(int(code))
        if tag_name is not None and tag_name in allowed_names:
            filtered.add_raw_tag(int(code), int(dtype), int(count), value)
    return filtered


@dataclass(frozen=True)
class IfdSpec:
    data: Union[np.ndarray, "DngPage"]
    bits_per_pixel: Optional[int] = None  # If None for ndarray data, inferred from dtype (uint8->8, uint16->16)
    photometric: Optional[str] = None
    cfa_pattern: str = "RGGB"
    jxl_distance: Optional[float] = None
    jxl_effort: Optional[int] = None
    decompress: bool = False
    page_tags: Optional[MetadataTags] = None  # Page-level tags (ActiveArea, BlackLevel, etc.)
    inherit_ifd0_tags_from_source: bool = True  # Copy IFD0 tags from source DngPage
    inherit_page_tags_from_source: bool = True  # Copy page tags from source DngPage


def write_dng(
    destination_file: Union[Path, io.BytesIO],
    *,
    ifd0_tags: Optional[MetadataTags] = None,
    subifds: List[IfdSpec],
    preview_image: Optional[np.ndarray] = None,
    skip_tags: Optional[set[str]] = None,
) -> None:
    """Write raw data to a DNG file using tifffile.

    Args:
        destination_file: Path or io.BytesIO object where to save the DNG file.
        ifd0_tags: Optional user-supplied MetadataTags (common/camera tags) to
            apply to IFD0.
        subifds: Ordered list of images to write. If preview_image is provided,
            all entries are written as SubIFDs under IFD0. If preview_image is
            not provided, the first entry becomes IFD0 and the rest become
            SubIFDs. Each IfdSpec can specify inherit_ifd0_tags_from_source
            and inherit_page_tags_from_source to control tag inheritance.
        preview_image: Optional preview/thumbnail image
        skip_tags: Optional set of tag names to skip when copying from source
            DNG pages (advanced; used to strip problematic or invalid tags when
            `subifds` contain `DngPage` objects).
    """

    if not subifds:
        raise ValueError("subifds must contain at least one image")

    def _tifffile_supports_tiled_page_copy(page: "DngPage") -> bool:
        if not page.is_tiled:
            return True
        tile_h, tile_w = page.tilelength, page.tilewidth
        return (
            tile_h <= page.imagelength
            and tile_w <= page.imagewidth
            and tile_h % 16 == 0
            and tile_w % 16 == 0
        )

    def _effective_decompress(spec: IfdSpec) -> bool:
        if spec.decompress:
            return True
        if isinstance(spec.data, DngPage) and spec.data.is_tiled and not _tifffile_supports_tiled_page_copy(spec.data):
            return True
        return False

    def _has_jxl_ifd(spec: IfdSpec) -> bool:
        if isinstance(spec.data, DngPage):
            if _effective_decompress(spec):
                return False
            return spec.data.compression in (COMPRESSION.JPEGXL, COMPRESSION.JPEGXL_DNG)
        return spec.jxl_distance is not None

    has_jxl = any(_has_jxl_ifd(s) for s in subifds)

    source_ifd0_tags = MetadataTags()

    # Convenience: if caller passes DngPage(s) with inherit_ifd0_tags_from_source=True,
    # copy IFD0-level tags from the source file's IFD0 and apply ifd0_tags as overrides.
    for spec in subifds:
        if isinstance(spec.data, DngPage) and spec.inherit_ifd0_tags_from_source:
            source_ifd0_tags = spec.data.get_ifd0_tags()
            break

    merged_ifd0_tags = source_ifd0_tags
    merged_ifd0_tags.extend(ifd0_tags)

    if preview_image is not None:
        exclude_compression_ifd0 = True
    else:
        first_spec = subifds[0]
        exclude_compression_ifd0 = isinstance(first_spec.data, DngPage) and _effective_decompress(first_spec)

    allowed_ifd0_tags = _get_dng_copy_tags(exclude_compression=exclude_compression_ifd0)
    # CFA tags belong to specific image IFDs, not file-level IFD0 metadata
    allowed_ifd0_tags -= CFA_TAGS
    if skip_tags:
        allowed_ifd0_tags -= skip_tags

    final_ifd0_tags = _filter_metadata_tags(merged_ifd0_tags, allowed_names=allowed_ifd0_tags)
    ifd0_tags = final_ifd0_tags

    prepared_ifd0_tags = _prepare_ifd0_tags(ifd0_tags, has_jxl=has_jxl)
    _ensure_float_tags_be(prepared_ifd0_tags)

    if isinstance(destination_file, Path):
        logger.debug(f"Writing DNG to {destination_file}")
    else:
        logger.debug("Writing DNG to in-memory buffer")

    def _write_page_ifd(
        writer: TiffWriter,
        page: "DngPage",
        *,
        include_ifd0_tags: bool,
        subifds_count: int,
        subfiletype: int,
        page_tags: Optional[MetadataTags] = None,
        inherit_page_tags_from_source: bool = True,
    ) -> None:
        # Get allowlist of tags to copy from source files (bounded to registry)
        # Always exclude digest tags 
        allowed_tags = _get_dng_copy_tags(exclude_compression=False)
        allowed_tags -= _DIGEST_TAGS
        if skip_tags:
            allowed_tags -= skip_tags
        # Strip CFA tags when writing LinearRaw
        if page.is_linear_raw:
            allowed_tags -= CFA_TAGS

        # Get page tags with inheritance logic
        if inherit_page_tags_from_source:
            page_tags_unfiltered = page.get_page_tags()
            page_tags_unfiltered.extend(page_tags)
        else:
            page_tags_unfiltered = page_tags if page_tags is not None else MetadataTags()
        
        # Combine page tags and IFD0 tags
        if include_ifd0_tags:
            page_tags_unfiltered.extend(prepared_ifd0_tags.copy())
        
        # Filter all tags at once 
        raw_ifd_tags = _filter_metadata_tags(
            page_tags_unfiltered,
            allowed_names=allowed_tags,
        )
        _ensure_float_tags_be(raw_ifd_tags)

        # prepare the args to tif writer
        prepare_kwargs = {
            "subfiletype": int(subfiletype),
            "photometric": page.photometric,
        }
        if subifds_count:
            prepare_kwargs["subifds"] = int(subifds_count)

        raw_ifd_args = _prepare_ifd_args(raw_ifd_tags, page.compression, include_ifd0_tags, **prepare_kwargs)

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
                tile_valid = (
                    tile_shape[0] <= page.imagelength
                    and tile_shape[1] <= page.imagewidth
                    and tile_shape[0] % 16 == 0
                    and tile_shape[1] % 16 == 0
                )
                if not tile_valid:
                    raise ValueError(
                        f"Cannot copy tiled page: tile {tile_shape} not supported by tifffile "
                        f"(must be <= image size and multiple of 16). Use decompress=True."
                    )
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

    def _write_ifd_from_spec(
        writer: TiffWriter,
        spec: IfdSpec,
        *,
        include_ifd0_tags: bool,
        subifds_count: int,
        subfiletype: int,
    ) -> None:
        if isinstance(spec.data, DngPage) and not _effective_decompress(spec):
            # copy page data as is (no decompress)
            _write_page_ifd(
                writer,
                spec.data,
                include_ifd0_tags=include_ifd0_tags,
                subifds_count=subifds_count,
                subfiletype=subfiletype,
                page_tags=spec.page_tags,
                inherit_page_tags_from_source=spec.inherit_page_tags_from_source,
            )
            return

        if isinstance(spec.data, DngPage):
            # copy page data with a decompress
            if spec.data.is_tiled and not _tifffile_supports_tiled_page_copy(spec.data):
                tile_shape = (spec.data.tilelength, spec.data.tilewidth)
                logger.warning(
                    "Falling back to decoded DNG write path for tiled page with unsupported tile %s "
                    "(must be <= image size and multiple of 16 for tifffile direct copy).",
                    tile_shape,
                )
            page = spec.data
            bits_per_pixel = int(page.bitspersample)
            
            # Extract page tags with inheritance logic
            if spec.inherit_page_tags_from_source:
                page_tags_unfiltered = page.get_page_tags()
                page_tags_unfiltered.extend(spec.page_tags)
            else:
                page_tags_unfiltered = spec.page_tags
            
            if page.is_cfa:
                cfa_result = page.get_cfa()
                if cfa_result is None:
                    raise ValueError("Failed to extract CFA data from page")
                raw_data, cfa_pattern = cfa_result
                photometric = "cfa"
            elif page.is_linear_raw:
                raw_data = page.get_linear_raw()
                if raw_data is None:
                    raise ValueError("Failed to extract LINEAR_RAW data from page")
                cfa_pattern = spec.cfa_pattern
                photometric = "linear_raw"
            else:
                raise ValueError(
                    f"Page must be CFA or LINEAR_RAW, got photometric={page.photometric_name}"
                )
            jxl_distance = None
            jxl_effort = None
        else:
            if spec.photometric not in ("cfa", "linear_raw"):
                raise ValueError(
                    f"Unsupported photometric: {spec.photometric}. Must be 'cfa' or 'linear_raw'"
                )
            if spec.bits_per_pixel is None:
                if spec.data.dtype == np.uint8:
                    bits_per_pixel = 8
                elif spec.data.dtype == np.uint16:
                    bits_per_pixel = 16
                elif spec.data.dtype == np.float16:
                    bits_per_pixel = 16
                elif spec.data.dtype == np.float32:
                    bits_per_pixel = 32
                else:
                    raise ValueError(
                        f"Unsupported dtype {spec.data.dtype}. Supported: uint8, uint16, float16, float32"
                    )
            else:
                bits_per_pixel = int(spec.bits_per_pixel)
            raw_data = spec.data
            photometric = spec.photometric
            cfa_pattern = spec.cfa_pattern
            jxl_distance = spec.jxl_distance
            jxl_effort = spec.jxl_effort
            page_tags_unfiltered = spec.page_tags

        # Validate input shape based on photometric
        if photometric == "cfa":
            if raw_data.ndim != 2:
                raise ValueError(
                    f"Expected 2D raw_data (H, W) for photometric='cfa', got shape {raw_data.shape}"
                )
            samples_per_pixel = 1
        else:
            if raw_data.ndim != 3 or raw_data.shape[-1] != 3:
                raise ValueError(
                    f"Expected 3D raw_data (H, W, 3) for photometric='linear_raw', got shape {raw_data.shape}"
                )
            samples_per_pixel = 3

        # Ensure integer data is uint16/uint8 for tifffile (skip float types)
        if raw_data.dtype not in (np.float16, np.float32):
            if bits_per_pixel > 8 and raw_data.dtype != np.uint16:
                bits_per_pixel = 16
                raw_data = raw_data.astype(np.uint16)
            elif bits_per_pixel <= 8 and raw_data.dtype != np.uint8:
                bits_per_pixel = 8
                raw_data = raw_data.astype(np.uint8)

        raw_datasize = int(
            raw_data.shape[0] * raw_data.shape[1] * samples_per_pixel * bits_per_pixel / 8
        )

        # Filter page tags before preparing IFD
        if page_tags_unfiltered is not None:
            allowed_tags = _get_dng_copy_tags(exclude_compression=True)
            if skip_tags:
                allowed_tags -= skip_tags
            # Strip CFA tags when writing LinearRaw
            if photometric == "linear_raw":
                allowed_tags -= CFA_TAGS
            page_tags = _filter_metadata_tags(
                page_tags_unfiltered,
                allowed_names=allowed_tags,
            )
        else:
            page_tags = None

        # Build tag set
        raw_ifd_tags = page_tags.copy() if page_tags else MetadataTags()
        if include_ifd0_tags:
            raw_ifd_tags.extend(prepared_ifd0_tags.copy())
        if photometric == "cfa":
            raw_ifd_tags.add_tag("CFAPattern", cfa_pattern)
            raw_ifd_tags.add_tag("CFARepeatPatternDim", (2, 2))
            raw_ifd_tags.add_tag("CFAPlaneColor", bytes([0, 1, 2]))
        
        _ensure_float_tags_be(raw_ifd_tags)

        # Prepare IFD args        
        prepare_kwargs = {"subfiletype": int(subfiletype), "photometric": photometric}
        if photometric == "linear_raw":
            prepare_kwargs["planarconfig"] = 1
        if subifds_count:
            prepare_kwargs["subifds"] = int(subifds_count)

        # Write IFD
        if jxl_distance is not None:
            if not (0.0 <= jxl_distance <= 15.0):
                logger.warning(f"JXL distance {jxl_distance} is outside the typical range [0.0, 15.0].")
            actual_effort = jxl_effort if jxl_effort is not None else 5

            raw_ifd_tags.add_tag("JXLDistance", jxl_distance)
            raw_ifd_tags.add_tag("JXLEffort", actual_effort)
            if photometric == "cfa":
                raw_data = swizzle_cfa_data(raw_data)
                raw_ifd_tags.add_tag("ColumnInterleaveFactor", 2)
                raw_ifd_tags.add_tag("RowInterleaveFactor", 2)
            
            raw_ifd_args = _prepare_ifd_args(raw_ifd_tags, COMPRESSION.JPEGXL_DNG, include_ifd0_tags, **prepare_kwargs)

            encoded_bytes = imagecodecs.jpegxl_encode(
                raw_data, distance=jxl_distance, effort=actual_effort
            )
            def encoded_data_iterator():
                yield encoded_bytes

            writer.write(
                data=encoded_data_iterator(),
                shape=raw_data.shape,
                dtype=raw_data.dtype,
                bitspersample=bits_per_pixel,
                rowsperstrip=raw_datasize,
                **raw_ifd_args,
            )
        else:
            # Uncompressed path
            raw_ifd_args = _prepare_ifd_args(raw_ifd_tags, COMPRESSION.NONE, include_ifd0_tags, **prepare_kwargs)

            writer.write(raw_data, rowsperstrip=raw_datasize, **raw_ifd_args)

    try:
        with TiffWriter(destination_file, bigtiff=False, byteorder='>') as tif:

            if preview_image is not None:
                _write_thumbnail_ifd(
                    tif,
                    preview_image,
                    prepared_ifd0_tags.copy(),
                    subifds_count=len(subifds),
                )

                for index, spec in enumerate(subifds):
                    _write_ifd_from_spec(
                        tif,
                        spec,
                        include_ifd0_tags=False,
                        subifds_count=0,
                        subfiletype=0 if index == 0 else 1,
                    )

            else:
                # No preview: first image becomes IFD0

                first_spec, rest_specs = subifds[0], subifds[1:]
                _write_ifd_from_spec(
                    tif,
                    first_spec,
                    include_ifd0_tags=True,
                    subifds_count=len(rest_specs),
                    subfiletype=0,
                )

                for index, spec in enumerate(rest_specs):
                    _write_ifd_from_spec(
                        tif,
                        spec,
                        include_ifd0_tags=False,
                        subifds_count=0,
                        subfiletype=1,
                    )

        if isinstance(destination_file, Path):
            logger.debug(f"Successfully wrote DNG file to {destination_file}")
        else:
            logger.debug("Successfully wrote DNG file to in-memory buffer")

    except Exception as e:
        logger.error(f"Error saving DNG file with TiffWriter: {e}")
        raise


def write_dng_from_array(
    destination_file: Union[Path, io.BytesIO],
    data: np.ndarray,
    *,
    ifd0_tags: Optional[MetadataTags] = None,
    photometric: str = "cfa",
    cfa_pattern: str = "RGGB",
    bits_per_pixel: Optional[int] = None,
    jxl_distance: Optional[float] = None,
    jxl_effort: Optional[int] = None,
    preview_image: Optional[np.ndarray] = None,
) -> None:
    write_dng(
        destination_file=destination_file,
        ifd0_tags=ifd0_tags,
        subifds=[
            IfdSpec(
                data=data,
                bits_per_pixel=bits_per_pixel,
                photometric=photometric,
                cfa_pattern=cfa_pattern,
                jxl_distance=jxl_distance,
                jxl_effort=jxl_effort,
            )
        ],
        preview_image=preview_image,
    )


def write_dng_from_page(
    page: "DngPage",
    destination_file: Union[Path, io.BytesIO],
    metadata: Optional[MetadataTags] = None,
    preview_image: Optional[np.ndarray] = None,
    skip_tags: Optional[set[str]] = None,
    decompress: bool = False,
) -> None:
    """Write DNG file from an existing DngPage.
    
    By default, copies compressed raw data directly without decompression,
    preserving the original compression. Set decompress=True to decode and
    re-encode the data (slower but works with any tile configuration).
    
    Args:
        page: DngPage containing the raw data to copy
        destination_file: Path or io.BytesIO object where to save the DNG file
        metadata: Optional user-supplied MetadataTags to override source tags
        preview_image: Optional preview/thumbnail image
        skip_tags: Optional set of tag names to skip when copying (e.g., for debugging)
        decompress: If True, decode raw data and write uncompressed. Use when
            direct copy fails due to unsupported tile configurations.
    
    Raises:
        ValueError: If decompress=True and page is not CFA or LINEAR_RAW
        ValueError: If decompress=False and tile configuration is unsupported
    """
    # IFD structure (same as write_dng):
    # - If preview_image exists: IFD0 = preview (JPEG thumbnail), SubIFD0 = raw image
    # - If no preview_image: IFD0 = raw image (main image)
    
    write_dng(
        destination_file=destination_file,
        ifd0_tags=metadata,
        subifds=[IfdSpec(data=page, decompress=decompress)],
        preview_image=preview_image,
        skip_tags=skip_tags,
    )

def _generate_preview(
    page: DngPage, max_dimension: int, demosaic_algorithm: str = "OPENCV_EA"
) -> np.ndarray:
    """Generate preview image from DNG page.
    
    Args:
        page: DNG page to render
        max_dimension: Maximum dimension for preview (downscale by powers of 2)
        demosaic_algorithm: Algorithm to use for demosaicing
    
    Returns:
        RGB uint8 array suitable for JPEG thumbnail
    """
    import cv2
    
    rendered = page.render_raw(
        output_dtype=np.uint8, demosaic_algorithm=demosaic_algorithm
    )
    if rendered is None:
        raise RuntimeError("Failed to render page for preview generation")
    
    height, width = rendered.shape[:2]
    max_dim = max(height, width)
    
    if max_dim <= max_dimension:
        return rendered
    
    scale = 1
    while max_dim / (scale * 2) > max_dimension:
        scale *= 2
    
    new_width = width // scale
    new_height = height // scale
    
    downscaled = cv2.resize(
        rendered, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    
    return downscaled


# CFA-specific tags that must be stripped when writing LinearRaw
CFA_TAGS = {
    "CFAPattern",
    "CFARepeatPatternDim",
    "CFAPlaneColor",
    "CFALayout",
    "BayerGreenSplit",
}

# Tags applied during stage1 and stage2 processing that must be stripped
# when extracting processed data via get_camera_rgb()
STAGE1_STAGE2_TAGS = {
    # Stage1 tags (CFA-specific)
    *CFA_TAGS,
    # Interleave factors (applied during CFA JXL encoding)
    "ColumnInterleaveFactor",
    "RowInterleaveFactor",
    # Stage2 tags
    "BlackLevel",
    "BlackLevelRepeatDim",
    "BlackLevelDeltaH",
    "BlackLevelDeltaV",
    "WhiteLevel",
    "LinearizationTable",
    "ActiveArea",
    "OpcodeList2",
    # Digest tags (invalid after transformation)
    "NewRawImageDigest",
    "RawDataUniqueID",
}


def copy_dng(
    source_file: Union[str, Path],
    destination_file: Union[str, Path, io.BytesIO],
    *,
    scale: float = 1.0,
    demosaic: bool = False,
    demosaic_algorithm: str = "OPENCV_EA",
    strip_tags: Optional[set[str]] = None,
    generate_preview: bool = False,
    preview_max_dimension: int = 1024,
    jxl_distance: Optional[float] = None,
    jxl_effort: Optional[int] = None,
    ifd0_tags: Optional[MetadataTags] = None,
) -> None:
    """Copy a DNG file with optional transformations.
    
    Args:
        source_file: Path to source DNG file
        destination_file: Path to destination DNG file, or io.BytesIO for in-memory output
        scale: Scale factor for image (default: 1.0). If != 1.0, forces conversion to LINEAR_RAW
        demosaic: If True, convert CFA to LINEAR_RAW
        demosaic_algorithm: Demosaic algorithm to use (default: RCD)
        strip_tags: Optional set of tag names to strip from output
        generate_preview: If True, generate preview/thumbnail
        preview_max_dimension: Maximum dimension for preview (default: 1024)
        jxl_distance: JXL compression distance (0=lossless, >0=lossy). None=uncompressed.
        jxl_effort: JXL compression effort (1-9). Default 5 if jxl_distance is set.
        ifd0_tags: Optional MetadataTags to override/add to IFD0 (e.g., UniqueCameraModel)
    
    Raises:
        ValueError: If input is invalid
        RuntimeError: If DNG processing fails
    """
    source_path = Path(source_file)
    is_stream_output = isinstance(destination_file, io.BytesIO)
    dest_path = destination_file if is_stream_output else Path(destination_file)
    
    if not source_path.exists():
        raise ValueError(f"Source file does not exist: {source_path}")
    
    if is_stream_output:
        logger.info(f"Copying DNG: {source_path} -> <stream>")
    else:
        logger.info(f"Copying DNG: {source_path} -> {dest_path}")
    
    dng_file = DngFile(source_path)
    page = dng_file.get_main_page()
    
    if page is None:
        raise RuntimeError(f"No main page found in DNG file: {source_path}")
    
    if not (page.is_cfa or page.is_linear_raw):
        raise ValueError(
            f"Main page must be CFA or LINEAR_RAW, got photometric={page.photometric_name}"
        )
    
    # Determine if we need to transform the image
    # Only demosaic if the input is actually CFA
    needs_demosaic = demosaic and page.is_cfa
    needs_jxl = jxl_distance is not None
    needs_transform = needs_demosaic or scale != 1.0 or needs_jxl
    
    if needs_transform:
        # Extract processed camera RGB data
        logger.info(f"Extracting camera RGB (demosaic={demosaic}, scale={scale})")
        raw_data = page.get_camera_rgb(demosaic_algorithm)
        if raw_data is None:
            raise RuntimeError("Failed to extract camera RGB")
        
        # Strip all stage1/stage2 tags since they've been applied
        strip_tags = (strip_tags or set()) | STAGE1_STAGE2_TAGS

        # Apply scaling if needed
        if scale != 1.0:
            # When scaling, apply crop first, then scale
            crop_origin = page.get_tag("DefaultCropOrigin")
            crop_size = page.get_tag("DefaultCropSize")
            
            if crop_origin is not None and crop_size is not None:
                logger.info(f"Applying crop: origin={crop_origin}, size={crop_size}")
                crop_x = int(crop_origin[0])
                crop_y = int(crop_origin[1])
                crop_w = int(crop_size[0])
                crop_h = int(crop_size[1])
                raw_data = raw_data[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            
            logger.info(f"Applying scale: {scale}")
            import cv2
            h, w = raw_data.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            raw_data = cv2.resize(raw_data, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # If scaling, also strip crop tags since we've applied the crop
            strip_tags = strip_tags | {"DefaultCropOrigin", "DefaultCropSize"}
        
        # Convert float32 [0, 1] to uint16 for DNG storage
        raw_data = raw_render.convert_dtype(raw_data, np.uint16)
        
        # Create IfdSpec with numpy array
        ifd_spec = IfdSpec(
            data=raw_data,
            bits_per_pixel=16,
            photometric="linear_raw",
            cfa_pattern=None,
            jxl_distance=jxl_distance,
            jxl_effort=jxl_effort,
            page_tags=page.get_page_tags(),
        )
    else:
        # No transformation - use page copy API to preserve everything
        logger.info("Using page copy (no transformation)")
        ifd_spec = IfdSpec(
            data=page,
            inherit_ifd0_tags_from_source=True,
            inherit_page_tags_from_source=True,
        )
    
    # Extract IFD0 tags and merge with user overrides
    merged_ifd0_tags = page.get_ifd0_tags()
    merged_ifd0_tags.extend(ifd0_tags)
    
    if generate_preview:
        # Two-pass approach: write to memory first, generate preview from that,
        # then write final DNG with preview
        logger.info("Generating preview from transformed image (two-pass)")
        
        # First pass: write transformed DNG to memory
        temp_buffer = io.BytesIO()
        write_dng(
            destination_file=temp_buffer,
            ifd0_tags=merged_ifd0_tags,
            subifds=[ifd_spec],
            preview_image=None,
            skip_tags=strip_tags,
        )
        
        # Load the in-memory DNG and generate preview from it
        temp_buffer.seek(0)
        temp_dng = DngFile(temp_buffer)
        temp_page = temp_dng.get_main_page()
        if temp_page is None:
            raise RuntimeError("Failed to load temporary DNG for preview generation")
        
        logger.info(f"Generating preview (max dimension: {preview_max_dimension})")
        preview_image = _generate_preview(
            temp_page, preview_max_dimension, demosaic_algorithm
        )
        
        # Second pass: use temp_page for image data (efficiency), but original
        # page_tags to avoid flattened SubIFD tags from temp DNG's IFD0
        write_dng(
            destination_file=dest_path,
            ifd0_tags=merged_ifd0_tags,
            subifds=[IfdSpec(
                data=temp_page,
                page_tags=page.get_page_tags(),
                inherit_ifd0_tags_from_source=False,
                inherit_page_tags_from_source=False,
            )],
            preview_image=preview_image,
            skip_tags=strip_tags,
        )
    else:
        write_dng(
            destination_file=dest_path,
            ifd0_tags=merged_ifd0_tags,
            subifds=[ifd_spec],
            skip_tags=strip_tags,
        )
    
    if is_stream_output:
        logger.info("Successfully wrote DNG to stream")
    else:
        logger.info(f"Successfully wrote DNG to {dest_path}")


def decode_dng(
    file: Union[str, Path, IO[bytes], DngFile, DngPage],
    output_dtype: type = np.uint16,
    demosaic_algorithm: str = "OPENCV_EA",
    use_coreimage_if_available: bool = False,
    use_xmp: bool = True,
    rendering_params: dict = None,
    strict: bool = True,
) -> np.ndarray:
    """
    Decode a DNG file or page to a numpy array.
    
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
            Note: Not supported for DngPage input
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
        RGB image array with shape (height, width, 3) and specified dtype
    """
    # Handle DngPage input
    if isinstance(file, DngPage):
        page = file
        
        # Warn if Core Image was requested but not available for DngPage
        # TODO: when copy_dng function is done we can create a DngFile with requested subifd as main page
        if use_coreimage_if_available:
            logger.warning("Core Image not supported for DngPage rendering it does not have a subifd option")
        
        # Check if page is raw or preview
        if page.is_cfa or page.is_linear_raw:
            # Raw page - render with parameters
            result = page.render_raw(
                output_dtype=output_dtype,
                use_xmp=use_xmp,
                rendering_params=rendering_params,
                strict=strict,
            )
        else:
            # Preview page - validate no rendering params
            if rendering_params:
                raise ValueError("Rendering parameters not allowed for preview pages (RGB/YCBCR)")
            
            # Decode preview page
            result = page.get_preview_rgb(output_dtype=output_dtype)
        
        if result is None:
            raise RuntimeError("Failed to decode DNG page")
        
        return result
    
    # Try Core Image path if requested
    if use_coreimage_if_available:
        try:
            from ._dngio_coreimage import core_image_available, decode_dng_coreimage

            if core_image_available:
                return decode_dng_coreimage(
                    file=file,
                    use_xmp=use_xmp,
                    output_dtype=output_dtype,
                    rendering_params=rendering_params,
                )

            logger.warning(
                "Core Image requested but not available; falling back to Python pipeline."
            )
        except ImportError:
            logger.warning(
                "Core Image requested but PyObjC/Core Image dependencies are not installed; "
                "falling back to Python pipeline."
            )
    
    # Python SDK pipeline
    # Create or use DngFile
    dng_file = file if isinstance(file, DngFile) else DngFile(file)

    result = dng_file.render_raw(
        output_dtype=output_dtype,
        demosaic_algorithm=demosaic_algorithm,
        use_xmp=use_xmp,
        rendering_params=rendering_params,
    )
    if result is None:
        raise RuntimeError(f"No main image page found in DNG file: {file}")
    
    return result
