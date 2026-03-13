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

logger = logging.getLogger(__name__)

class RawStageSelector(str, Enum):
    RAW = "raw"
    LINEARIZED = "linearized"
    LINEARIZED_PLUS_OPS = "linearized_plus_ops"

# Import metadata classes from tiff_metadata module
from .tiff_metadata import (
    MetadataTags,
    TagSpec,
    TIFF_DTYPE_TO_STR,
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
        tag = self.tags.get(254)  # NewSubfileType tag ID
        if tag is None:
            return False
        return tag.value == self.SF_MAIN_IMAGE
    
    @property
    def is_preview(self) -> bool:
        """True if this page is a preview image."""
        tag = self.tags.get(254)  # NewSubfileType tag ID
        if tag is None:
            return False
        return tag.value in (self.SF_PREVIEW_IMAGE, self.SF_ALT_PREVIEW_IMAGE)
    
    @property
    def xmp(self) -> XmpMetadata:
        """XMP metadata from this page (or parent if not found locally)."""
        xmp_string = self.get_tag('XMP', str)
        if xmp_string is None:
            xmp_string = ''
        return XmpMetadata(xmp_string)
    
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
        
        effective_type = return_type or get_native_type(raw_tag.dtype, raw_tag.count)
        shape_spec = None
        if registry_spec and registry_spec.shape and registry_spec.count == raw_tag.count:
            shape_spec = TagSpec(TIFF_DTYPE_TO_STR.get(raw_tag.dtype, 'B'), raw_tag.count, registry_spec.shape)
        
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
        # Local import to avoid a module-level circular import between dngio/raw_render.
        from . import raw_render

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

    def get_camera_rgb(self, demosaic_algorithm: str = "RCD") -> Optional[np.ndarray]:
        """Extract the camera-RGB intermediate used for the final color pipeline.

        This corresponds to the `rgb_camera` input passed into
        `raw_render._render_camera_rgb(...)` during `render()`.

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

        from . import raw_render

        rgb_camera = raw_render.demosaic(
            cfa_normalized, cfa_pattern, algorithm=demosaic_algorithm
        )
        rgb_camera = np.clip(rgb_camera, 0.0, 1.0).astype(np.float32, copy=False)
        return rgb_camera
    
    def render(
        self,
        output_dtype: type = np.uint16,
        demosaic_algorithm: str = "RCD",
        strict: bool = True,
    ) -> "np.ndarray | None":
        from . import raw_render

        try:
            unsupported = raw_render.validate_dng_tags(self, strict=strict)
            if unsupported and not strict:
                logger.warning(
                    f"DNG contains unsupported tags (processing anyway): {', '.join(unsupported)}"
                )

            rgb_camera = self.get_camera_rgb(demosaic_algorithm=demosaic_algorithm)
            if rgb_camera is None:
                logger.error("Failed to extract camera RGB from DNG")
                return None

            result = raw_render._render_camera_rgb(
                page=self,
                rgb_camera=rgb_camera,
                output_dtype=output_dtype,
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
        import io
        from pathlib import Path
        
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
            
            # Recursively process sub-pages
            if tiff_page.pages:
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

    def get_cfa(
        self, stage: RawStageSelector = RawStageSelector.RAW
    ) -> Optional[tuple[np.ndarray, str]]:
        """Get CFA data from the main page.

        This is a convenience wrapper around `get_main_page()` +
        `DngPage.get_cfa(stage=...)`.

        Args:
            stage: Which stage of the raw pipeline to return.

        Returns:
            Tuple of (raw_cfa_array, cfa_pattern_string), or None if no suitable
            page is found.
        """
        page = self.get_main_page()
        if page is None or not page.is_cfa:
            return None
        return page.get_cfa(stage=stage)

    def get_linear_raw(
        self, stage: RawStageSelector = RawStageSelector.RAW
    ) -> Optional[np.ndarray]:
        """Get linear raw data from the main page.

        This is a convenience wrapper around `get_main_page()` +
        `DngPage.get_linear_raw(stage=...)`.

        Args:
            stage: Which stage of the raw pipeline to return.

        Returns:
            Linear raw array, or None if no suitable page is found.
        """
        page = self.get_main_page()
        if page is None or not page.is_linear_raw:
            return None
        return page.get_linear_raw(stage=stage)

    def get_camera_rgb(self, demosaic_algorithm: str = "RCD") -> Optional[np.ndarray]:
        """Get the camera-RGB intermediate from the main page.

        Convenience wrapper around `get_main_page()` + `DngPage.get_camera_rgb(...)`.

        Args:
            demosaic_algorithm: Demosaic algorithm to use when the main page is CFA.

        Returns:
            Camera RGB array (H, W, 3) float32 in [0, 1], or None if no suitable
            main page is found.
        """
        page = self.get_main_page()
        if page is None:
            return None
        return page.get_camera_rgb(demosaic_algorithm=demosaic_algorithm)

    def render(
        self,
        output_dtype: type = np.uint16,
        demosaic_algorithm: str = "RCD",
        strict: bool = True,
    ) -> "np.ndarray | None":
        """Render the main page to RGB.

        Convenience wrapper around `get_main_page()` + `DngPage.render(...)`.
        """
        page = self.get_main_page()
        if page is None:
            return None
        return page.render(
            output_dtype=output_dtype,
            demosaic_algorithm=demosaic_algorithm,
            strict=strict,
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


def _write_thumbnail_ifd(writer: TiffWriter, thumbnail_image: np.ndarray, dng_tags: MetadataTags) -> None:
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
        "subifds": 1,  # Has main image as subifd
    }
    # set datasize to max uncompressed size to avoid writing strips
    datasize = thumbnail_image.shape[0] * thumbnail_image.shape[1] * 3
    writer.write(
        thumbnail_image,  # Use the thumbnail_image directly
        **thumb_ifd_args,
        rowsperstrip=datasize,
    )


def _prepare_ifd_args(metadata: MetadataTags, **base_args) -> dict:
    """Prepare TiffWriter IFD args, extracting tags that have kwargs equivalents.
    
    Extracts Software, XResolution, YResolution, ResolutionUnit from metadata,
    converts to TiffWriter kwargs format, removes them from metadata, and
    merges with the provided base args.
    
    Args:
        metadata: MetadataTags instance to use as extratags (modified in-place)
        **base_args: Base IFD args (photometric, compression, etc.)
        
    Returns:
        Dict with complete IFD args including extratags and extracted kwargs
    """
    # Start with base args
    ifd_args = dict(base_args)
    
    # Extract Software
    if 'Software' in metadata:
        ifd_args['software'] = metadata.get_tag('Software')
        metadata.remove_tag('Software')
    
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
    
    # Set remaining metadata as extratags (after extracting kwargs equivalents)
    ifd_args['extratags'] = metadata
    
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
    if metadata is not None:
        dng_tags.extend(metadata)
    
    # Add required tags if not already set
    if "Orientation" not in dng_tags:
        _ORIENTATION_HORIZONTAL = 1
        
        dng_tags.add_tag("Orientation", _ORIENTATION_HORIZONTAL)
    
    if "ColorMatrix1" not in dng_tags:
        identity_matrix = np.identity(3, dtype=np.float64)
        dng_tags.add_tag("ColorMatrix1", identity_matrix)
    
    if "CalibrationIlluminant1" not in dng_tags:
        dng_tags.add_tag("CalibrationIlluminant1", 0)  # 0 = Unknown

    dng_tags.add_tag("DNGVersion", bytes([1, 7, 1, 0]))
    if not has_jxl:
        # need latest version for CFA compression but lots of old software can't handle it
        dng_tags.add_tag("DNGBackwardVersion", bytes([1, 4, 0, 0]))
    else:
        dng_tags.add_tag("DNGBackwardVersion", bytes([1, 7, 1, 0]))
        
    return dng_tags


def write_dng(
    raw_data: np.ndarray,
    destination_file: Union[Path, io.BytesIO],
    bits_per_pixel: int,
    photometric: str = "cfa",
    cfa_pattern: str = "RGGB",
    metadata: Optional[MetadataTags] = None,
    jxl_distance: Optional[float] = None,
    jxl_effort: Optional[int] = None,
    preview_image: Optional[np.ndarray] = None
) -> None:
    """Write raw data to a DNG file using tifffile.

    Args:
        raw_data: Raw image data as numpy array. Shape depends on photometric:
                  - "cfa": 2D array (H, W)
                  - "linear_raw": 3D array (H, W, 3)
        destination_file: Path or io.BytesIO object where to save the DNG file.
        bits_per_pixel: Number of bits per pixel (e.g. 12, 14, 16)
        photometric: Photometric interpretation, either "cfa" or "linear_raw"
        cfa_pattern: CFA pattern string, e.g., 'RGGB'. Only used when
                     photometric="cfa".
        jxl_distance: JPEG XL Butteraugli distance. Lower is higher quality.
                     Default: None (no JXL compression).
        jxl_effort: JPEG XL compression effort (1-9). Higher is more
                    compression/slower. Only used if jxl_distance is also
                    specified. Default: None (codec default).
        preview_image: Optional preview/thumbnail image
        metadata: Optional user-supplied MetadataTags to override defaults
    """
    if photometric not in ("cfa", "linear_raw"):
        raise ValueError(
            f"Unsupported photometric: {photometric}. Must be 'cfa' or 'linear_raw'"
        )

    if isinstance(destination_file, Path):
        logger.debug(f"Writing DNG ({photometric}) to {destination_file}")
    else:
        logger.debug(f"Writing DNG ({photometric}) to in-memory buffer")

    # Validate input shape based on photometric
    if photometric == "cfa":
        if raw_data.ndim != 2:
            raise ValueError(
                f"Expected 2D raw_data (H, W) for photometric='cfa', "
                f"got shape {raw_data.shape}"
            )
        samples_per_pixel = 1
    else:  # linear_raw
        if raw_data.ndim != 3 or raw_data.shape[-1] != 3:
            raise ValueError(
                f"Expected 3D raw_data (H, W, 3) for photometric='linear_raw', "
                f"got shape {raw_data.shape}"
            )
        samples_per_pixel = 3

    # Ensure data is uint16 for tifffile when bits_per_pixel > 8
    if bits_per_pixel > 8 and raw_data.dtype != np.uint16:
        bits_per_pixel = 16
        processed_raw_data = raw_data.astype(np.uint16)
    elif bits_per_pixel <= 8 and raw_data.dtype != np.uint8:
        bits_per_pixel = 8
        processed_raw_data = raw_data.astype(np.uint8)
    else:
        processed_raw_data = raw_data

    # IFD structure:
    # - If preview_image exists: IFD0 = preview (JPEG thumbnail), SubIFD0 = raw image
    # - If no preview_image: IFD0 = raw image (main image)
    ifd0_tags = _prepare_ifd0_tags(metadata, has_jxl=jxl_distance is not None)
    _ensure_float_tags_be(ifd0_tags)

    try:
        with TiffWriter(destination_file, bigtiff=False, byteorder='>') as tif:
            if preview_image is not None:
                _write_thumbnail_ifd(tif, preview_image, ifd0_tags)

            # Prepare raw image IFD tags
            raw_ifd_tags = MetadataTags()

            # Add CFA-specific tags
            if photometric == "cfa":
                raw_ifd_tags.add_tag("CFAPattern", cfa_pattern)
                raw_ifd_tags.add_tag("CFARepeatPatternDim", (2, 2))
                raw_ifd_tags.add_tag("CFAPlaneColor", bytes([0, 1, 2]))

            # Handle JXL compression
            if jxl_distance is not None:
                if not (0.0 <= jxl_distance <= 15.0):
                    logger.warning(
                        f"JXL distance {jxl_distance} is outside the "
                        f"typical range [0.0, 15.0]."
                    )
                compression_type = "JPEGXL_DNG"
                actual_effort = jxl_effort if jxl_effort is not None else 5
                compressionargs = {
                    "distance": jxl_distance,
                    "effort": actual_effort
                }
                logger.debug(
                    f"Writing DNG with JXL compression, "
                    f"distance: {jxl_distance}, effort: {actual_effort}"
                )

                raw_ifd_tags.add_tag("JXLDistance", jxl_distance)
                raw_ifd_tags.add_tag("JXLEffort", actual_effort)

                # CFA data needs swizzling for JXL compression
                if photometric == "cfa":
                    processed_raw_data = swizzle_cfa_data(processed_raw_data)
                    raw_ifd_tags.add_tag("ColumnInterleaveFactor", 2)
                    raw_ifd_tags.add_tag("RowInterleaveFactor", 2)
            else:
                compression_type = COMPRESSION.NONE
                compressionargs = {}

            _ensure_float_tags_be(raw_ifd_tags)

            # If no preview, raw IFD becomes IFD0 and needs ifd0_tags
            if preview_image is None:
                raw_ifd_tags.extend(ifd0_tags)

            # Prepare raw image IFD arguments (extracts Software/Resolution from metadata)
            raw_ifd_args = _prepare_ifd_args(
                raw_ifd_tags,
                subfiletype=0,
                photometric=photometric,
                compression=compression_type,
                compressionargs=compressionargs,
            )

            # Add photometric-specific IFD args
            if photometric == "cfa":
                raw_ifd_args["subifds"] = 0
            else:  # linear_raw
                raw_ifd_args["planarconfig"] = 1  # CONTIG

            # Calculate rowsperstrip
            raw_datasize = int(
                processed_raw_data.shape[0]
                * processed_raw_data.shape[1]
                * samples_per_pixel
                * bits_per_pixel
                / 8
            )
            tif.write(processed_raw_data, **raw_ifd_args, rowsperstrip=raw_datasize)

        if isinstance(destination_file, Path):
            logger.debug(f"Successfully wrote DNG file to {destination_file}")
        else:
            logger.debug("Successfully wrote DNG file to in-memory buffer")

    except Exception as e:
        logger.error(f"Error saving DNG file with TiffWriter: {e}")
        raise

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

# Tags invalidated when decompressing (digests become stale)
_COMPRESSION_INVALIDATED_TAGS = {
    'NewRawImageDigest', 'RawImageDigest', 'OriginalRawFileDigest',
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

def _get_dng_copy_tags(exclude_compression: bool = False) -> set:
    """Return set of tag names allowed when copying from source files.
    
    This filters tags read from source DNG files. User-provided tags
    (e.g., metadata) should bypass this filter entirely.
    
    Args:
        exclude_compression: True when decompressing (invalidates digests)
        
    Returns:
        Set of allowed tag names
    """
    tags = _DNG_METADATA_ALLOWLIST.copy()
    if exclude_compression:
        tags -= _COMPRESSION_INVALIDATED_TAGS
    return tags


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
    
    # Get allowlist of tags to copy from source files (bounded to registry)
    # Exclude compression tags when decompressing; preview tags always excluded
    allowed_tags = _get_dng_copy_tags(exclude_compression=decompress)
    if skip_tags:
        allowed_tags -= skip_tags
    
    # Build ifd0_tags from source IFD0 (color calibration, camera info, etc.)
    ifd0_tags = MetadataTags()
    if page.ifd0 is not None:
        for tag in page.ifd0.tags.values():
            if tag.name in allowed_tags:
                ifd0_tags.add_raw_tag(tag.name, tag.dtype, tag.count, tag.value)
    
    # Apply user-supplied metadata overrides to ifd0_tags
    ifd0_tags.extend(metadata)
    _ensure_float_tags_be(ifd0_tags)
    
    # Build raw_ifd_tags from page-specific tags (CFAPattern, compression, etc.)
    raw_ifd_tags = MetadataTags()
    for tag in page.tags.values():
        if tag.name in allowed_tags:
            raw_ifd_tags.add_raw_tag(tag.name, tag.dtype, tag.count, tag.value)
    _ensure_float_tags_be(raw_ifd_tags)

    if isinstance(destination_file, Path):
        logger.debug(f"Writing DNG from DngPage to {destination_file}")
    else:
        logger.debug("Writing DNG from DngPage to in-memory buffer")

    if decompress:
        # Decode and re-encode path - works with any tile configuration
        bits_per_pixel = page.bitspersample
        
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
            cfa_pattern = None
            photometric = "linear_raw"
        else:
            raise ValueError(
                f"Page must be CFA or LINEAR_RAW, got photometric={page.photometric_name}"
            )
        
        # For decompress path, merge tags for write_dng (it handles IFD structure internally)
        merged_tags = ifd0_tags.copy()
        merged_tags.extend(raw_ifd_tags)
        
        write_dng(
            raw_data=raw_data,
            destination_file=destination_file,
            bits_per_pixel=bits_per_pixel,
            cfa_pattern=cfa_pattern,
            photometric=photometric,
            metadata=merged_tags,
            preview_image=preview_image,
        )
    else:
        # Direct copy path - preserves original compression
        is_uncompressed = page.compression == COMPRESSION.NONE
        try:
            with TiffWriter(destination_file, bigtiff=False, byteorder='>') as tif:

                if preview_image is not None:
                    _write_thumbnail_ifd(tif, preview_image, ifd0_tags)
                else:
                    # No preview: raw IFD becomes IFD0 and needs ifd0_tags
                    raw_ifd_tags.extend(ifd0_tags)

                # For uncompressed data, use numpy array so tifffile handles byte order
                # For compressed data, use raw bytes iterator
                if is_uncompressed:
                    raw_data = page.asarray()
                    logger.debug(f"Read uncompressed data: {raw_data.shape} {raw_data.dtype}")
                else:
                    fh = page.parent.filehandle
                    compressed_segments = list(fh.read_segments(
                        page.dataoffsets,
                        page.databytecounts,
                        sort=True
                    ))

                    def compressed_data_iterator():
                        try:
                            for segment_data, index in compressed_segments:
                                yield segment_data
                        except GeneratorExit:
                            return

                    logger.debug(f"Read {len(compressed_segments)} compressed segments from page")

                # Prepare raw image IFD arguments (extracts Software/Resolution)
                raw_ifd_args = _prepare_ifd_args(
                    raw_ifd_tags,
                    subfiletype=0,
                    photometric=page.photometric,
                    subifds=0,
                    compression=page.compression,
                )

                # Determine shape based on samples per pixel
                samples_per_pixel = page.samplesperpixel if hasattr(page, 'samplesperpixel') else 1
                if samples_per_pixel > 1:
                    write_shape = (page.imagelength, page.imagewidth, samples_per_pixel)
                else:
                    write_shape = (page.imagelength, page.imagewidth)
                
                if is_uncompressed:
                    # Uncompressed: pass numpy array, tifffile handles byte order
                    tif.write(
                        data=raw_data,
                        bitspersample=page.bitspersample,
                        **raw_ifd_args,
                    )
                    logger.debug(f"Successfully wrote uncompressed raw data")
                elif page.is_tiled:
                    # Compressed tiled: use iterator
                    tile_shape = (page.tilelength, page.tilewidth)
                    tile_valid = (
                        tile_shape[0] <= page.imagelength and 
                        tile_shape[1] <= page.imagewidth and
                        tile_shape[0] % 16 == 0 and 
                        tile_shape[1] % 16 == 0
                    )
                    if not tile_valid:
                        raise ValueError(
                            f"Cannot copy tiled page: tile {tile_shape} not supported by tifffile "
                            f"(must be <= image size and multiple of 16). Use decompress=True."
                        )
                    tif.write(
                        data=compressed_data_iterator(),
                        shape=write_shape,
                        dtype=page.dtype,
                        bitspersample=page.bitspersample,
                        tile=tile_shape,
                        **raw_ifd_args,
                    )
                    logger.debug(f"Successfully copied tiled compressed data ({sum(page.databytecounts)} bytes)")
                else:
                    # Compressed stripped: use iterator
                    raw_datasize = page.imagelength * page.imagewidth * samples_per_pixel * (page.bitspersample // 8)
                    tif.write(
                        data=compressed_data_iterator(),
                        shape=write_shape,
                        dtype=page.dtype,
                        bitspersample=page.bitspersample,
                        rowsperstrip=raw_datasize,
                        **raw_ifd_args,
                    )
                    logger.debug(f"Successfully copied stripped compressed data ({sum(page.databytecounts)} bytes)")
            
            if isinstance(destination_file, Path):
                logger.debug(f"Successfully wrote DNG file to {destination_file}")
            else:
                logger.debug("Successfully wrote DNG file to in-memory buffer")
            
        except Exception as e:
            logger.error(f"Error writing DNG from DngPage: {e}")
            raise

def decode_dng(
    file: Union[str, Path, IO[bytes], DngFile],
    output_dtype: type = np.uint16,
    demosaic_algorithm: str = "RCD",
    strict: bool = True,
    use_coreimage_if_available: bool = False,
    use_xmp: bool = True,
    **processing_params,
) -> np.ndarray:
    """
    Decode a DNG file to a numpy array.
    
    By default uses the Python SDK pipeline. When use_coreimage_if_available=True,
    uses macOS Core Image pipeline if available (supports XMP and processing params).
    
    Args:
        file: Path to DNG file, file-like object containing DNG data, or DngFile instance
        output_dtype: Output numpy data type (np.uint8, np.uint16, np.float16, np.float32)
        demosaic_algorithm: Demosaic algorithm for Python pipeline - "RCD" (default), "VNG", etc.
        strict: If True, raise error on unsupported DNG tags (Python pipeline only)
        use_coreimage_if_available: If True, use Core Image pipeline when available on macOS
        use_xmp: Whether to read XMP metadata for processing defaults (Core Image only)
        **processing_params: Core Image processing parameters:
            - temperature: Color temperature in Kelvin
            - tint: Tint adjustment
            - exposure: Exposure adjustment in EV
            - tone_curve: SplineCurve for tone mapping
            - noise_reduction: Noise reduction amount
            - orientation: EXIF orientation code
    
    Returns:
        RGB image array with shape (height, width, 3) and specified dtype
    """
    # Try Core Image path if requested
    if use_coreimage_if_available:
        try:
            from ._dngio_coreimage import core_image_available, decode_dng_coreimage

            if core_image_available:
                return decode_dng_coreimage(
                    file=file,
                    use_xmp=use_xmp,
                    output_dtype=output_dtype,
                    **processing_params,
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

    result = dng_file.render(
        output_dtype=output_dtype,
        demosaic_algorithm=demosaic_algorithm,
        strict=strict,
    )
    if result is None:
        raise RuntimeError(f"No main image page found in DNG file: {file}")
    
    return result
