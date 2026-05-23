# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

"""DNG file format support and utilities.

This module provides functionality for reading, writing, and processing DNG files.
"""
from __future__ import annotations

import io
import logging
import numpy as np

from dataclasses import dataclass, replace
from datetime import datetime
from enum import auto, Enum, IntEnum
from pathlib import Path
from typing import Any, IO, TypeAlias

from .deps import (
    cv2_proxy as cv2,
    imagecodecs_proxy as imagecodecs,
    tifffile_proxy as tifffile,
)


# Package imports
from . import raw_render
from .compress import compress_ifd, deswizzle_cfa_data
from .raw_render import DemosaicAlgorithm
from .common import PerfTimer, get_active_timer, scoped_perf_timer
from .tiff_metadata import (
    MetadataTags,
    Orientation,
    SubFileType,
    TIFF_TAG_TYPE_REGISTRY,
    XmpMetadata,
    convert_tag_value,
    resolve_tag,
    filter_tags_by_ifd_category,
)

# tifffile followups:
# - Copying compressed tiled pages is not always possible (e.g. tile size / alignment constraints) and can require a decode + re-encode fallback, which currently emits a warning.

logger = logging.getLogger(__name__)


# =============================================================================
# Type Aliases
# =============================================================================

# Common file path types
PathLike: TypeAlias = str | Path | IO[bytes]

# DNG file/page input types (using forward references)
DngInput: TypeAlias = str | Path | IO[bytes] | "DngFile" | "DngPage"


# =============================================================================
# Constants
# =============================================================================


class PyramidFilter(Enum):
    """Filter types for pyramid generation."""
    LANCZOS8 = auto()
    LANCZOS4 = auto()
    CATMULL_ROM = auto()


# =============================================================================
# Core DNG Classes
# =============================================================================

class DngPage(tifffile.TiffPage):
    """TiffPage subclass with DNG-specific functionality.
    
    Provides convenient access to DNG tags with automatic translation,
    parent IFD tag inheritance, and raw data extraction methods.
    
    Inherits all TiffPage attributes and methods. Created by "upgrading"
    an existing TiffPage instance.
    """
    
    def __new__(cls, *args, **kwargs):
        """Create DngPage instance without calling TiffPage.__init__."""
        return object.__new__(cls)
    
    def __init__(self, tiff_page: tifffile.TiffPage):
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
    def photometric_name(self) -> str | None:
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

    def get_rendered_size(
        self, apply_orientation: bool = True, rendering_params: dict | None = None
    ) -> tuple[int, int]:
        """Get the final dimensions after DefaultCrop is applied.
        
        Args:
            apply_orientation: If True, swap width/height for 90° rotations
            rendering_params: Optional rendering parameters dict. If provided
                and contains 'orientation', that value will be used instead
                of the metadata orientation.
        
        Returns:
            Tuple of (width, height) after crop is applied.
            If DefaultCropSize is not present, returns (imagewidth, imagelength).
        """
        crop_size = self.get_tag("DefaultCropSize")
        if crop_size is not None:
            w, h = int(crop_size[0]), int(crop_size[1])
        else:
            w, h = self.imagewidth or 0, self.imagelength or 0
        
        # Swap dimensions for 90° rotations if requested
        if apply_orientation:            
            # Use orientation from rendering_params if provided, otherwise from metadata
            orientation = self.ifd0.get_tag("Orientation")
            if rendering_params and 'orientation' in rendering_params:
                orientation = rendering_params['orientation']

            # Orientation is an IFD0 tag
            if orientation in (Orientation.ROTATE_90_CW, Orientation.ROTATE_270_CW):
                w, h = h, w
        
        return w, h

    def get_xmp(self) -> XmpMetadata | None:
        """Return XMP metadata as an `XmpMetadata` object."""
        xmp = self.get_tag("XMP")
        return xmp

    def _get_tag_object(self, tag: str | int) -> tuple | None:
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
        tag: str | int,
        return_type: type | None = None,
    ) -> Any | None:
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

    def get_raw_tag(self, tag: str | int) -> Any | None:
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
    
    def get_time_from_tags(self, time_type: str = "original") -> datetime | None:
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
    
    def _decode_segmented(self, decode_func) -> np.ndarray:
        """Decode segmented compressed image data with error handling.
        
        Handles both tiled images (2D grid of tiles) and strip-based images
        (vertical stack of strips).
        
        Args:
            decode_func: Function to decode segment data (e.g., imagecodecs.jpegxl_decode).
                        Invalid segments that fail to decode are filled with zeros.
        
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
            # Multiple segments - decode each and assemble
            img_width = self.imagewidth
            img_height = self.imagelength
            samples = self.samplesperpixel or 1
            
            # Will be set from first successful decode
            output = None
            dtype = None
            
            # Handle tiled vs strip-based layouts
            if self.is_tiled:
                # Tiled layout: 2D grid of tiles
                tile_width = self.tilewidth
                tile_height = self.tilelength
                tiles_x = (img_width + tile_width - 1) // tile_width
                
                for i, (tile_data, _) in enumerate(segments):
                    try:
                        tile = decode_func(tile_data)
                        
                        # Create output array on first successful decode
                        if output is None:
                            dtype = tile.dtype
                            if samples == 1:
                                output = np.zeros((img_height, img_width), dtype=dtype)
                            else:
                                output = np.zeros((img_height, img_width, samples), dtype=dtype)
                    except Exception as e:
                        logger.warning(f"Failed to decode tile {i} ({type(e).__name__}): {e}, filling with zeros")
                        continue
                    
                    ty = (i // tiles_x) * tile_height
                    tx = (i % tiles_x) * tile_width
                    
                    # Handle edge tiles that may be smaller
                    th = min(tile_height, img_height - ty)
                    tw = min(tile_width, img_width - tx)
                    
                    output[ty:ty+th, tx:tx+tw] = tile[:th, :tw]
            else:
                # Strip-based layout: vertical stack of strips
                y_offset = 0
                for i, (strip_data, _) in enumerate(segments):
                    try:
                        strip = decode_func(strip_data)
                        
                        # Create output array on first successful decode
                        if output is None:
                            dtype = strip.dtype
                            if samples == 1:
                                output = np.zeros((img_height, img_width), dtype=dtype)
                            else:
                                output = np.zeros((img_height, img_width, samples), dtype=dtype)
                    except Exception as e:
                        logger.warning(f"Failed to decode strip {i} ({type(e).__name__}): {e}, filling with zeros")
                        continue
                    
                    # Handle last strip which may be shorter
                    sh = min(strip.shape[0], img_height - y_offset)
                    output[y_offset:y_offset+sh, :] = strip[:sh, :]
                    y_offset += sh
            
            # If all segments failed, raise error
            if output is None:
                raise ValueError(
                    f"No valid segments found in image. "
                    f"All {len(segments)} segments failed decoding."
                )
            
            return output
    
    def _decode_raw(self) -> np.ndarray:
        """Decode raw image data based on compression type."""
        if self.compression in (
            tifffile.COMPRESSION.JPEGXL,
            tifffile.COMPRESSION.JPEGXL_DNG,
        ):
            return self._decode_segmented(imagecodecs.jpegxl_decode)
        elif self.compression == tifffile.COMPRESSION.JPEG:
            if self.is_cfa:
                def decode_tile(tile_data: bytes) -> np.ndarray:
                    tile = imagecodecs.jpeg_decode(
                        tile_data,
                        bitspersample=self.bitspersample
                    )
                    # JPEG CFA format: (height, width/2, 2) -> (height, width)
                    if tile.ndim == 3 and tile.shape[2] == 2:
                        h, w_half, _ = tile.shape
                        tile = tile.reshape(h, w_half * 2)
                    return tile
            else:
                def decode_tile(tile_data: bytes) -> np.ndarray:
                    return imagecodecs.jpeg_decode(
                        tile_data,
                        bitspersample=self.bitspersample,
                        colorspace=2,  # RGB - bypass YCbCr conversion
                        outcolorspace=2
                    )
            return self._decode_segmented(decode_tile)
        else:
            return self.asarray()

    def get_cfa(self) -> tuple[np.ndarray, str] | None:
        """Extract CFA data and pattern from this page.

        Returns:
            Tuple of (cfa_array, cfa_pattern_str) or None if not a CFA page.
            cfa_pattern_str is e.g., 'RGGB', 'BGGR'.
        """
        if not self.is_cfa:
            return None

        timer = get_active_timer()
        timer.start_step("decode_cfa (c++)")
        raw_cfa = self._decode_raw()
        timer.end_step()

        if raw_cfa is None:
            return None

        col_interleave = self.get_tag("ColumnInterleaveFactor")
        row_interleave = self.get_tag("RowInterleaveFactor")
        if col_interleave == 2 and row_interleave == 2:
            timer.start_step("deswizzle_cfa")
            raw_cfa = deswizzle_cfa_data(raw_cfa)
            timer.end_step()

        cfa_pattern = self.get_tag("CFAPattern", str) or "RGGB"

        return raw_cfa, cfa_pattern
    
    def get_linear_raw(self) -> np.ndarray | None:
        """Extract LINEAR_RAW data from this page.

        Returns:
            Raw linear data array or None if not a LINEAR_RAW page.
        """
        if not self.is_linear_raw:
            return None

        timer = get_active_timer()
        timer.start_step("decode_linear_raw (c++)")
        raw_linear = self._decode_raw()
        timer.end_step()

        return raw_linear

    def get_camera_rgb_raw(
        self, 
        demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.OPENCV_EA
        ) -> np.ndarray | None:
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
                f"get_camera_rgb_raw() requires CFA or LINEAR_RAW page, "
                f"got {self.photometric_name}"
            )

        # Get raw data
        cfa_pattern = None
        if self.is_linear_raw:
            data = self.get_linear_raw()
            if data is None:
                return None
        else:
            cfa_result = self.get_cfa()
            if cfa_result is None:
                return None
            data, cfa_pattern = cfa_result

        return raw_render._raw_to_camera_rgb(
            self, data, self.photometric_name, cfa_pattern, demosaic_algorithm
        )
    
    def decode_to_rgb(
        self,
        output_dtype: type = np.uint8
    ) -> np.ndarray | None:
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
        
        # Apply orientation rotation
        orientation = self.get_tag("Orientation")
        if orientation is not None:
            image = raw_render.apply_tiff_orientation(image, orientation)
        
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
        
        unsupported = raw_render.validate_dng_tags(self, strict=strict)
        if unsupported and not strict:
            logger.warning(
                f"DNG contains unsupported tags (processing anyway): {', '.join(unsupported)}"
            )

        with scoped_perf_timer("render_raw_page", logger):

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

class DngFile(tifffile.TiffFile):

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
    
    def write_to(self, destination: str | Path | io.IOBase) -> None:
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
    def ifd0(self) -> DngPage | None:
        """Return IFD0 as a DngPage, or None if no pages exist."""
        return DngPage(self.pages[0]) if self.pages else None

    def get_flattened_pages(self) -> list[DngPage]:
        """Get all pages as DngPage instances.
        
        Returns:
            List of DngPage objects in flattened order. Tag inheritance
            falls back to IFD0 via TiffPage.parent.
        """
        def build_recursive(pages_list: list | None) -> list[DngPage]:
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
    
    def get_main_page(self) -> DngPage | None:
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

    def _find_optimal_raw_page(self, scale: float) -> DngPage | None:
        """Find the optimal raw page for the given scale factor.
        
        Searches all flattened pages for CFA or LINEAR_RAW pages and returns
        the page with the smallest max dimension that is still >= target dimension.
        
        Args:
            scale: Scaling factor (e.g., 0.5 for half size)
            
        Returns:
            Optimal raw page for rendering, or None if no raw pages are found.
        """
        # Get main page to calculate target dimension
        main_page = self.get_main_page()
        if main_page is None:
            return None
        
        # Calculate target max dimension using cropped dimensions
        main_w, main_h = main_page.get_rendered_size(apply_orientation=False)
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
            page_w, page_h = page.get_rendered_size(apply_orientation=False)
            max_dim = max(page_w, page_h)
            if max_dim >= target_max_dim:
                candidates.append((page, max_dim))
        
        if candidates:
            # Use the smallest page that still meets the target
            optimal_page = min(candidates, key=lambda x: x[1])[0]
        else:
            # No page meets target - use main page
            optimal_page = main_page
        
        return optimal_page

    def get_ifd0_tags(self, convert_exif: bool = True) -> MetadataTags:
        """Return a copy of IFD0 tags as a MetadataTags object.
        
        Args:
            convert_exif: If True (default), convert ExifTag dictionary to individual TIFF tags.
        """
        return self.ifd0.get_ifd0_tags(convert_exif=convert_exif) if self.ifd0 else MetadataTags()

    def get_tag(
        self,
        tag: str | int,
        return_type: type | None = None,
    ) -> Any | None:
        """See `DngPage.get_tag`."""
        return self.ifd0.get_tag(tag, return_type=return_type) if self.ifd0 else None

    def get_xmp(self) -> XmpMetadata | None:
        """See `DngPage.get_xmp`."""
        return self.ifd0.get_xmp() if self.ifd0 else None

    def get_time_from_tags(self, time_type: str = "original") -> datetime | None:
        """See `DngPage.get_time_from_tags`."""
        return self.ifd0.get_time_from_tags(time_type=time_type) if self.ifd0 else None
    
    def get_rendered_size(
        self, apply_orientation: bool = True, rendering_params: dict | None = None
    ) -> tuple[int, int] | None:
        """See `DngPage.get_rendered_size`."""
        main_page = self.get_main_page()
        return main_page.get_rendered_size(
            apply_orientation=apply_orientation, rendering_params=rendering_params
        ) if main_page else None

    def _forward_main_page(self, method_name: str, *args, require=None, **kwargs):
        page = self.get_main_page()
        if page is None:
            return None
        if require is not None and not require(page):
            return None
        method = getattr(page, method_name)
        return method(*args, **kwargs)
        
    def get_cfa(self) -> tuple[np.ndarray, str] | None:
        """See `DngPage.get_cfa`."""
        return self._forward_main_page("get_cfa")

    def get_linear_raw(self) -> np.ndarray | None:
        """See `DngPage.get_linear_raw`."""
        return self._forward_main_page("get_linear_raw")

    def get_camera_rgb_raw(
        self,
        demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.OPENCV_EA,
    ) -> np.ndarray | None:
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
        scale: float | None = None,
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
            # Find optimal page for scaling
            render_page = self._find_optimal_raw_page(scale)
            if render_page is None:
                return None
            
            # Calculate target dimensions by scaling main page dimensions
            main_w, main_h = main_page.get_rendered_size(apply_orientation=False)
            target_w = int(main_w * scale)
            target_h = int(main_h * scale)
        
        # Validate DNG tags
        unsupported = raw_render.validate_dng_tags(render_page, strict=strict)
        if unsupported and not strict:
            logger.warning(
                f"DNG contains unsupported tags (processing anyway): {', '.join(unsupported)}"
            )
        
        _timer = PerfTimer("render_raw_file")

        try:
            # Get camera RGB raw from render page
            rgb_camera = render_page.get_camera_rgb_raw(demosaic_algorithm=demosaic_algorithm)
            if rgb_camera is None:
                return None
            
            # Apply resize if needed (when scaling and dimensions don't match)
            scale_needed = False
            if target_w is not None and target_h is not None:
                render_w, render_h = render_page.get_rendered_size(apply_orientation=False)
                if render_w != target_w or render_h != target_h:
                    # If upscaling, defer to post-render for better performance
                    if render_w < target_w and render_h < target_h:
                        scale_needed = True
                    else:
                        # Downscaling: do it now before rendering
                        _timer.start_step("pre_scale (opencv)")
                        rgb_camera = cv2.resize(
                            rgb_camera, (target_w, target_h), interpolation=cv2.INTER_AREA
                        )
                        _timer.end_step()
            
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
            del rgb_camera
            
            # Post-render upscaling if needed
            if scale_needed:            
                # Check if orientation swaps dimensions (same logic as get_rendered_size)
                orientation = self.ifd0.get_tag("Orientation")
                if rendering_params and 'orientation' in rendering_params:
                    orientation = rendering_params['orientation']

                # Swap dimensions for 90° rotations
                final_w, final_h = target_w, target_h
                if orientation in (Orientation.ROTATE_90_CW, Orientation.ROTATE_270_CW):
                    final_w, final_h = final_h, final_w

                _timer.start_step("post_scale (opencv)")
                rgb_image = cv2.resize(
                    rgb_image, (final_w, final_h), interpolation=cv2.INTER_LINEAR
                )
                _timer.end_step()
            
            return rgb_image
        finally:
            _timer.close()
            _timer.log_report(logger)

    def get_preview_rgb(
        self,
        output_dtype: type = np.uint8,
    ) -> np.ndarray | None:
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


def _prepare_ifd_args(
    metadata: MetadataTags,
    compression,
    is_ifd0: bool,
    subfiletype: int,
    photometric: str,
    subifds_count: int = 0,
    compressionargs: dict | None = None,
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
        ifd_args['software'] = metadata.get_tag('Software') or "muimg"
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
        tags.add_tag("Orientation", Orientation.HORIZONTAL)
    
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


# ExtraCameraProfiles is currently in _TIFFWRITER_MANAGED_TAGS to strip it during copy operations
# since we don't support reading or writing it yet. It possibly could be in a simlar 
# way that exif and gps are suported.
#
# ExtraCameraProfiles structure (from DNG spec):
# - Tag 50933 (0xC6F5) contains offset(s) to IFD(s) containing camera profile data
# - Each profile IFD contains tags like ProfileName, ColorMatrix, etc.
# - Similar to how ExifTag points to an Exif IFD
#
# To read ExtraCameraProfiles:
# - Use tifffile's internal IFD reading functions
#    - Follows offset pointers to read tag values
#    - Returns dict of tag name -> value
#
# To add ExtraCameraProfiles support in the future:
# 1. Determine profile IFD structure/format from DNG spec or sample files
# 2. Write read_profile_ifd() function similar to read_exif_ifd():
#    def read_profile_ifd(fh, byteorder, dtype, count, offsetsize):
#        return read_tags(fh, byteorder, offsetsize, TIFF.PROFILE_TAGS, maxifds=1)[0]
# 3. Add to TIFF.TAG_READERS: 50933: read_profile_ifd
# 4. Define TIFF.PROFILE_TAGS registry with profile IFD tag names
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
    'ExtraCameraProfiles',
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
    'ColumnInterleaveFactor', 'RowInterleaveFactor',  # CFA swizzle tags will be added back in if JXL encoding
    'Software' # if we are not doing a pure copy of the image bits then replace software with muimg
}

def _filter_metadata_tags(
    tags: MetadataTags,
    exclude_names: set[str] | None = None,
) -> None:
    """Filter tags in-place by removing tags in the exclude set.
    
    Args:
        tags: MetadataTags to filter (modified in-place)
        exclude_names: Set of tag names to remove (None = no filtering)
    """
    for tag_name in (exclude_names or set()):
        tags.remove_tag(tag_name)


@dataclass(slots=True)
class PageEncoding:
    """TIFF page encoding specification.
    
    Groups compression type, codec-specific arguments, and tile/strip layout.
    
    Args:
        compression: Compression type (None = tifffile.COMPRESSION.NONE)
        compression_args: Codec-specific args. For JXL, defaults to 
            {'distance': 0.0, 'effort': 4} (lossless) if not specified.
        tile_size: Tile dimensions (height, width). Mutually exclusive with rows_per_strip.
        rows_per_strip: Rows per strip. Mutually exclusive with tile_size.
        
    Notes:
        - If both tile_size and rows_per_strip are None, uses single-strip layout (full image).
        - tile_size and rows_per_strip cannot both be specified.
    """
    compression: tifffile.COMPRESSION | None = None
    compression_args: dict | None = None
    tile_size: tuple[int, int] | None = None
    rows_per_strip: int | None = None
    
    def get_compression(self) -> tuple[tifffile.COMPRESSION, dict | None]:
        """Get compression type and args with defaults applied.
        
        Returns:
            Tuple of (compression, compression_args). Compression defaults to
            tifffile.COMPRESSION.NONE if None. For JXL compression, args default to
            {'distance': 0.0, 'effort': 4} (lossless) if not specified.
        """
        comp = self.compression if self.compression is not None else tifffile.COMPRESSION.NONE
        args = self.compression_args
        
        # Apply JXL defaults if no args specified
        if comp in (tifffile.COMPRESSION.JPEGXL, tifffile.COMPRESSION.JPEGXL_DNG) and args is None:
            args = {'distance': 0.0, 'effort': 4}
        
        return (comp, args)

@dataclass(slots=True)
class IfdPageSpec:
    """Specification for writing a DNG page from a source DNG file.
    
    Args:
        page: Source DNG page to write
        subfiletype: NewSubFileType value (0=main, 1=preview)
        transcode_encoding: PageEncoding for transcode mode. None = COPY mode
            (preserve source compression). PageEncoding(...) = TRANSCODE mode
            (decompress and re-encode with specified compression).
        extratags: Additional metadata tags to add
        strip_tags: Tag names to remove from source
        copy_page_tags: Whether to copy tags from source page
    """
    page: DngPage
    subfiletype: int = 0
    transcode_encoding: PageEncoding | None = None
    extratags: MetadataTags | None = None
    strip_tags: set[str] | None = None
    copy_page_tags: bool = True
    
    def requires_transcode(self) -> bool:
        """Check if this spec requires transcoding (decompression + recompression)."""
        if self.transcode_encoding is not None:
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
    
    def get_compression(self) -> tuple[tifffile.COMPRESSION, dict | None]:
        """Get compression and args for this spec.
        
        Returns:
            Tuple of (compression, compression_args). For COPY mode, returns
            (page.compression, None). For TRANSCODE mode, returns encoding's
            compression with defaults applied.
        """
        return (self.transcode_encoding.get_compression() 
                if self.transcode_encoding 
                else (self.page.compression, None))


@dataclass(slots=True)
class IfdDataSpec:
    """Specification for writing a DNG IFD from raw array data.
    
    Args:
        data: Raw image data array
        photometric: Photometric interpretation ("CFA", "LINEAR_RAW", "RGB", "YCBCR")
        subfiletype: NewSubFileType value (0=main, 1=preview)
        cfa_pattern: CFA pattern (only used for photometric="CFA")
        encoding: PageEncoding for compression. None means no compression (tifffile.COMPRESSION.NONE).
        extratags: Additional metadata tags to add
        bits_per_sample: Bits per sample (e.g., 10, 12, 14 for raw data). None means
            infer from dtype (uint8→8, uint16→16, float16→16, float32→32). Use this to
            specify non-standard bit depths like 10-bit or 12-bit data stored in uint16 arrays.
    """
    data: np.ndarray
    photometric: str
    subfiletype: int = 0
    cfa_pattern: str | None = None
    encoding: PageEncoding | None = None
    extratags: MetadataTags | None = None
    bits_per_sample: int | None = None
    
    def requires_transcode(self) -> bool:
        """Check if this spec requires transcoding (always False for array data)."""
        return False
    
    def get_compression(self) -> tuple[tifffile.COMPRESSION, dict | None]:
        """Get compression and args for this spec.
        
        Returns:
            Tuple of (compression, compression_args). If encoding is None,
            returns (tifffile.COMPRESSION.NONE, None). Otherwise returns encoding's
            compression with defaults applied.
        """
        return self.encoding.get_compression() if self.encoding else (tifffile.COMPRESSION.NONE, None)


# IFD specification type alias (defined after the classes)
IfdSpec: TypeAlias = IfdPageSpec | IfdDataSpec


def write_dng(
    destination_file: str | Path | io.BytesIO,
    *,
    ifd0_spec: IfdSpec,
    subifds: list[IfdSpec] = None,
    num_compression_workers: int = 1,
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
        num_compression_workers: Number of parallel compression workers (default: 1).
    """
    def _write_page_ifd(
        writer: tifffile.TiffWriter,
        page: "DngPage",
        *,
        raw_ifd_args: dict,
    ) -> None:
        # IFD args are already prepared by caller
        
        logger.debug(f"_write_page_ifd: compression={page.compression}, is_tiled={page.is_tiled}, shape={page.shape}")
        
        # For uncompressed data, use asarray() to handle byte order conversion
        # For compressed data, copy raw bytes (compression is byte-order independent)
        if page.compression == tifffile.COMPRESSION.NONE:
            data = page.asarray()
        else:
            # Read raw compressed segments
            segments = list(
                page.parent.filehandle.read_segments(page.dataoffsets, page.databytecounts))
            segment_bytes = [seg_data for seg_data, _ in segments]
            data = iter(segment_bytes)
            logger.debug(f"Extracted {len(segment_bytes)} compressed segment bytes")
            
            # Set tile or rowsperstrip parameter for compressed data
            if page.is_tiled:
                raw_ifd_args['tile'] = (page.tilelength, page.tilewidth)
            else:
                raw_ifd_args['rowsperstrip'] = (np.prod(page.shape) * page.bitspersample + 7) // 8
        
        writer.write(
            data=data,
            shape=page.shape,
            dtype=page.dtype,
            bitspersample=page.bitspersample,
            **raw_ifd_args,
        )

        segment_type = "tiled" if page.is_tiled else "stripped"
        compression_type = "uncompressed" if page.compression == tifffile.COMPRESSION.NONE else "compressed"
        logger.debug(
            f"Successfully copied {segment_type} {compression_type} data ({sum(page.databytecounts)} bytes)"
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
        writer: tifffile.TiffWriter,
        spec: IfdSpec,
        *,
        is_ifd0: bool = False,
        needs_v1_7_1: bool,
        main_spec: IfdSpec | None = None,
        num_compression_workers: int = 1,
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
                if is_ifd0:
                    # For IFD0, also merge IFD0-specific tags, if page was not ifd0 then remove
                    # the compression tags
                    extratags = spec.page.get_ifd0_tags()
                    if not spec.page.is_ifd0:
                        _filter_metadata_tags(extratags, exclude_names=_COMPRESSION_INVALIDATED_TAGS)
                extratags |= spec.page.get_page_tags()
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
                    comp, _ = main_spec.get_compression()
                    has_compression = comp != tifffile.COMPRESSION.NONE
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
        # Get compression and args from spec
        compression, compression_args = spec.get_compression()
        
        # Normalize compression type and args (skip for COPY mode)
        if not (isinstance(spec, IfdPageSpec) and not spec.requires_transcode()):
            if compression in (tifffile.COMPRESSION.JPEGXL, tifffile.COMPRESSION.JPEGXL_DNG):
                # Normalize JXL compression variants to JPEGXL_DNG
                compression = tifffile.COMPRESSION.JPEGXL_DNG
            elif compression == tifffile.COMPRESSION.JPEG:
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
        
        # Extract BitsPerSample before stripping (tifffile manages this tag)
        meta_bps = ifd_tags.get_tag("BitsPerSample")
        
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
            if spec.transcode_encoding is None and spec.page.is_tiled:
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
                uncomp_data, _ = cfa_result
            elif photometric == "LINEAR_RAW":
                uncomp_data = spec.page.get_linear_raw()
                if uncomp_data is None:
                    raise ValueError("Failed to extract LINEAR_RAW data from page")
            else:
                # Non-raw photometric (RGB, YCBCR, etc.) - use asarray like decode() does
                try:
                    uncomp_data = spec.page.asarray()
                except Exception as e:
                    raise ValueError(f"Failed to extract data from page (photometric={photometric}): {e}")
        else:
            # IfdDataSpec: data from array
            if spec.data.dtype not in (np.uint8, np.uint16, np.float16, np.float32):
                raise ValueError(
                    f"Unsupported dtype {spec.data.dtype}. Supported: uint8, uint16, float16, float32"
                )
            
            # Resolve bits_per_sample: spec field > metadata tag > dtype
            if spec.bits_per_sample is None and meta_bps is None:
                bits_per_sample = spec.data.dtype.itemsize * 8
            else:
                if spec.bits_per_sample is not None:
                    bits_per_sample = spec.bits_per_sample
                else:
                    bits_per_sample = int(
                        meta_bps.flat[0] if isinstance(meta_bps, np.ndarray) else meta_bps)

                # Validate bits_per_sample against dtype
                dtype_bits = spec.data.dtype.itemsize * 8
                if spec.data.dtype in (np.float16, np.float32):
                    if bits_per_sample != dtype_bits:
                        raise ValueError(
                            f"bits_per_sample={bits_per_sample} incompatible with float dtype "
                            f"{spec.data.dtype} (must be {dtype_bits})"
                        )
                else:
                    if bits_per_sample > dtype_bits:
                        raise ValueError(
                            f"bits_per_sample={bits_per_sample} exceeds dtype {spec.data.dtype} "
                            f"capacity ({dtype_bits} bits)"
                        )
                    if bits_per_sample < 8:
                        raise ValueError(f"bits_per_sample must be >= 8, got {bits_per_sample}")
                        
            # Add CFA tags for array data with CFA photometric
            if photometric == "CFA":
                # Resolve CFA pattern: spec > metadata tag > default RGGB
                cfa_pattern = (
                    spec.cfa_pattern
                    or ifd_tags.get_tag("CFAPattern")
                    or "RGGB"
                )
                ifd_tags.add_tag("CFAPattern", cfa_pattern)
                if ifd_tags.get_tag("CFARepeatPatternDim") is None:
                    ifd_tags.add_tag("CFARepeatPatternDim", (2, 2))
                if ifd_tags.get_tag("CFAPlaneColor") is None:
                    ifd_tags.add_tag("CFAPlaneColor", bytes([0, 1, 2]))

            uncomp_data = spec.data

        # Get encoding object for layout parameters
        encoding = spec.transcode_encoding if isinstance(spec, IfdPageSpec) else spec.encoding
        
        # Validate encoding parameters and determine layout
        if encoding:
            # Validate mutual exclusivity
            if encoding.tile_size is not None and encoding.rows_per_strip is not None:
                raise ValueError(
                    "tile_size and rows_per_strip are mutually exclusive. "
                    "Specify one or neither (for single-strip layout)."
                )
            
            # Validate tile_size dimensions
            if encoding.tile_size is not None:
                tile_h, tile_w = encoding.tile_size
                if tile_h <= 0 or tile_w <= 0:
                    raise ValueError(
                        f"tile_size dimensions must be positive, got {encoding.tile_size}"
                    )
            
            # Validate rows_per_strip
            if encoding.rows_per_strip is not None and encoding.rows_per_strip <= 0:
                raise ValueError(
                    f"rows_per_strip must be positive, got {encoding.rows_per_strip}"
                )
            
            # Add layout args to ifd_args for writer.write()
            if encoding.tile_size is not None:
                ifd_args['tile'] = encoding.tile_size
            elif encoding.rows_per_strip is not None:
                ifd_args['rowsperstrip'] = encoding.rows_per_strip
            else:
                ifd_args['rowsperstrip'] = (np.prod(uncomp_data.shape) * bits_per_sample + 7) // 8   

            
            # Segment and compress data
            dng_photometric_types = photometric in ("CFA", "LINEAR_RAW")
            
            if (compression in (tifffile.COMPRESSION.JPEGXL, tifffile.COMPRESSION.JPEGXL_DNG, tifffile.COMPRESSION.JPEG) 
                and dng_photometric_types) or compression == tifffile.COMPRESSION.NONE:
                
                # Add compression-specific tags before encoding
                if compression in (tifffile.COMPRESSION.JPEGXL, tifffile.COMPRESSION.JPEGXL_DNG):
                    jxl_distance = compression_args.get('distance', 0.0) if compression_args else 0.0
                    jxl_effort = compression_args.get('effort', 5) if compression_args else 5
                    
                    if not (0.0 <= jxl_distance <= 15.0):
                        logger.warning(f"JXL distance {jxl_distance} is outside the typical range [0.0, 15.0].")

                    ifd_tags.add_tag("JXLDistance", jxl_distance)
                    ifd_tags.add_tag("JXLEffort", jxl_effort)
                    
                    if photometric == "CFA":
                        ifd_tags.add_tag("ColumnInterleaveFactor", 2)
                        ifd_tags.add_tag("RowInterleaveFactor", 2)
                
                # Use manual segmentation and compression
                encoded_segments = compress_ifd(
                    uncomp_data,
                    compression,
                    compression_args,
                    bits_per_sample,
                    photometric,
                    writer.tiff.byteorder,
                    encoding.tile_size,
                    encoding.rows_per_strip,
                    num_compression_workers
                )
                write_data = iter(encoded_segments)
            else:
                write_data = uncomp_data
        else:
            write_data = uncomp_data
            ifd_args['rowsperstrip'] = (np.prod(uncomp_data.shape) * bits_per_sample + 7) // 8
        
        # Prepare tags for write (convert arrays to target byte order)
        ifd_args["extratags"] = _prepare_tags_for_write(ifd_tags, writer.tiff.byteorder)

        writer.write(
            write_data,
            shape=uncomp_data.shape,
            dtype=uncomp_data.dtype,
            bitspersample=bits_per_sample,
            **ifd_args,
        )

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
    needs_v1_7_1 = any(
        s.get_compression()[0] in (tifffile.COMPRESSION.JPEGXL, tifffile.COMPRESSION.JPEGXL_DNG) 
        for s in all_specs
    )

    try:
        with tifffile.TiffWriter(destination_file, bigtiff=False, byteorder='<') as tif:

            # Write IFD0
            _write_ifd_from_spec(
                tif,
                ifd0_spec,
                is_ifd0=True,
                needs_v1_7_1=needs_v1_7_1,
                main_spec=main_spec,
                num_compression_workers=num_compression_workers
            )

            # Write subifds
            for spec in subifds:
                _write_ifd_from_spec(
                    tif,
                    spec,
                    needs_v1_7_1=needs_v1_7_1,
                    main_spec=main_spec,
                    num_compression_workers=num_compression_workers
                )

        if isinstance(destination_file, Path):
            logger.debug(f"Successfully wrote DNG file to {destination_file}")
        else:
            logger.debug("Successfully wrote DNG file to in-memory buffer")

    except Exception as e:
        logger.error(f"Error saving DNG file with TiffWriter ({type(e).__name__}): {e}")
        raise


def create_dng(
    *,
    ifd0_spec: IfdSpec,
    subifds: list[IfdSpec] = None,
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


class PreviewScale(IntEnum):
    """Preview scale factors as pyramid levels.
    
    Values represent the pyramid level (each level is 1/2 the previous):
    - FULL = 0 (full resolution)
    - HALF = 1 (1/2 resolution)
    - QUARTER = 2 (1/4 resolution)
    - EIGHTH = 3 (1/8 resolution)
    """
    FULL = 0
    HALF = 1
    QUARTER = 2
    EIGHTH = 3


@dataclass(slots=True)
class PreviewParams:
    """Parameters for preview generation in DNG files.
    
    If this object is provided (not None), a preview will be generated.
    If None, no preview is generated.
    
    Attributes:
        scale: Preview scale factor (default: QUARTER = 1/4 size)
        compression: Compression type for preview (default: JPEG)
        compression_args: Arguments for compression (default: {'level': 90} for JPEG)
        rendering_params: Override rendering params (Temperature, Tint, etc.)
        use_xmp: Use XMP metadata for preview rendering (default: True)
    """
    scale: PreviewScale = PreviewScale.QUARTER
    compression: tifffile.COMPRESSION | None = None
    compression_args: dict | None = None
    rendering_params: dict | None = None
    use_xmp: bool = True


@dataclass(slots=True)
class PyramidParams:
    """Parameters for pyramid level generation in DNG files.
    
    Attributes:
        levels: Number of pyramid levels to generate (0=none)
        encoding: PageEncoding for pyramid levels (None = no compression)
        extratags: Additional metadata tags to add to each pyramid level
        filter: Filter type for pyramid downscaling (default: CATMULL_ROM)
    """
    levels: int = 0
    encoding: PageEncoding | None = None
    extratags: MetadataTags | None = None
    filter: PyramidFilter = PyramidFilter.CATMULL_ROM


def _generate_pyramid(
    image: np.ndarray,
    num_levels: int,
    filter: PyramidFilter = PyramidFilter.CATMULL_ROM,
) -> list[np.ndarray]:
    """Generate image pyramid levels using configurable downsampling filter.
    
    Creates a pyramid where each level is exactly 1/2 x 1/2 of the previous level,
    using a separable filter for high-quality downsampling.
    
    Args:
        image: Input image (any dtype, 2D or 3D)
        num_levels: Maximum number of pyramid levels to generate (including level 0)
        filter: Filter type for downsampling (LANCZOS8, LANCZOS4, CATMULL_ROM)
    
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
    timer = get_active_timer()

    def make_lanczos_kernel(a: int) -> np.ndarray:
        """Generate a Lanczos kernel for 2:1 downsampling."""
        positions = np.arange(-a + 0.5, a, 1.0)
        kernel = np.sinc(positions) * np.sinc(positions / a)
        kernel = kernel / kernel.sum()
        return kernel.astype(np.float32)
    
    # Select kernel based on filter type
    if filter == PyramidFilter.CATMULL_ROM:
        kernel = np.array([-0.0625, 0.5625, 0.5625, -0.0625], dtype=np.float32)
        anchor = (1, 1)
    elif filter == PyramidFilter.LANCZOS4:
        kernel = make_lanczos_kernel(a=2)  # 4-tap
        anchor = (1, 1)
    else:  # LANCZOS8 (default)
        kernel = make_lanczos_kernel(a=4)  # 8-tap
        anchor = (3, 3)
    
    levels = [image]
    current = image
    
    while len(levels) < num_levels:
        h, w = current.shape[:2]
        if min(h, w) <= 16:
            break
        next_h, next_w = (h + 1) // 2, (w + 1) // 2
        if min(next_h, next_w) <= 16:
            break
        
        timer.start_step(f"pyramid_level_{len(levels)}_filter_{filter.name}")
        filtered = cv2.sepFilter2D(
            current, -1, kernel, kernel, anchor=anchor, borderType=cv2.BORDER_REFLECT_101)
        downsampled = filtered[::2, ::2]
        timer.end_step()
        
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


def _write_dng_with_params(
    destination_file: str | Path | io.BytesIO,
    main_spec: IfdSpec,
    *,
    ifd0_tags: MetadataTags,
    scale: float = 1.0,
    demosaic: bool = False,
    demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.DNGSDK_BILINEAR,
    preview: PreviewParams | None = None,
    pyramid: PyramidParams | None = None,
    num_compression_workers: int = 1,
) -> None:
    """Write DNG with transforms, preview, and pyramid from a page spec.

    Extracts camera RGB from main_spec, applies scaling/demosaic, generates
    pyramid levels, and writes the final DNG with optional preview.

    Args:
        destination_file: Destination path or io.BytesIO
        main_spec: Source IFD spec (IfdPageSpec or IfdDataSpec).
        ifd0_tags: IFD0-level metadata tags.
        scale: Scale factor (default: 1.0)
        demosaic: If True, convert CFA to LINEAR_RAW
        demosaic_algorithm: Demosaic algorithm to use
        preview: Preview generation parameters.
        pyramid: Pyramid generation parameters.
        num_compression_workers: Number of parallel compression workers.
    """
    timer = get_active_timer()

    # Extract camera RGB (always needed if we didn't take fast return path)
    timer.start_step("extract_camera_rgb")
    logger.info(f"Extracting camera RGB (demosaic={demosaic}, scale={scale})")
    cfa_pattern = None
    if isinstance(main_spec, IfdPageSpec):
        if main_spec.page.is_linear_raw:
            raw_data = main_spec.page.get_linear_raw()
        else:
            cfa_result = main_spec.page.get_cfa()
            if cfa_result is None:
                raise RuntimeError("Failed to extract CFA data")
            raw_data, cfa_pattern = cfa_result
        if raw_data is None:
            raise RuntimeError("Failed to extract raw data")
        
        photometric = main_spec.page.photometric_name
    else:
        raw_data = main_spec.data
        photometric = main_spec.photometric
        if photometric == "CFA":
            cfa_pattern = (
                main_spec.cfa_pattern
                or (main_spec.extratags.get_tag("CFAPattern")
                    if main_spec.extratags else None)
                or "RGGB"
            )

    camera_rgb = raw_render._raw_to_camera_rgb(
        main_spec.extratags, raw_data, photometric, cfa_pattern, demosaic_algorithm
    )
    timer.end_step()
    
    # if we are transforming then the main_spec is always a DataSpec with the transformed data
    if ((demosaic and photometric == "CFA") or scale != 1.0):

        # Apply scaling if needed
        if scale != 1.0:
            timer.start_step("apply_scaling")
            logger.info(f"Applying scale: {scale}")
            h, w = camera_rgb.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            camera_rgb = cv2.resize(camera_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            timer.end_step()

        if isinstance(main_spec, IfdPageSpec):
            if main_spec.transcode_encoding:
                main_encoding = main_spec.transcode_encoding
            else:
                main_encoding = PageEncoding(
                    compression=main_spec.page.compression,
                    compression_args=None
                )
        else:
            main_encoding = main_spec.encoding

        # Strip stage/digest tags since we are writing a new linear_raw DNG
        tags_to_strip = STAGE1_STAGE2_TAGS | STAGE3_TAGS | _DIGEST_TAGS
        _filter_metadata_tags(ifd0_tags, exclude_names=tags_to_strip)
        _filter_metadata_tags(main_spec.extratags, exclude_names=tags_to_strip)

        # change main_spec 
        main_spec = IfdDataSpec(
            data=raw_render.convert_dtype(camera_rgb, np.uint16),
            photometric="LINEAR_RAW",
            subfiletype=SubFileType.MAIN_IMAGE,
            encoding=main_encoding,
            extratags=main_spec.extratags
        )

    # generate preview tags for muimg engine
    preview_tags = MetadataTags()
    preview_tags.add_tag("PreviewApplicationName", "muimg")
    preview_tags.add_tag("PreviewApplicationVersion", "1.0.0")

    # Compute pyramid levels needed 
    num_pyramid_levels = 1  # Level 0 is always the original
    preview_level_idx = 0  # Default to level 0 if no preview
    
    # Calculate levels needed for preview
    if preview:
        # if preview it is ifd0, also add in preview tags
        ifd0_tags |= preview_tags

        # Direct mapping: FULL=0, HALF=1, QUARTER=2, EIGHTH=3
        preview_level_idx = preview.scale
        num_pyramid_levels = preview_level_idx + 1
    
    # Take max with requested pyramid levels
    if pyramid:
        num_pyramid_levels = max(num_pyramid_levels, pyramid.levels + 1)
    
    # Generate pyramid - use CATMULL_ROM for faster preview generation
    timer.start_step("generate_pyramid")
    filter_type = pyramid.filter if pyramid else PyramidFilter.CATMULL_ROM
    pyramid_images = _generate_pyramid(camera_rgb, num_pyramid_levels, filter=filter_type)
    logger.info(f"Generated {len(pyramid_images)} pyramid levels (including original)")
    timer.end_step()
    
    # Build pyramid level specs (levels 1+)
    pyramid_specs = []
    if pyramid and pyramid.levels > 0:

        # create pyramid tags
        pyramid_tags = preview_tags | (pyramid.extratags if pyramid else None)

        for level_idx in range(1, len(pyramid_images)):
            pyramid_spec = IfdDataSpec(
                data=raw_render.convert_dtype(pyramid_images[level_idx], np.uint16),
                photometric="LINEAR_RAW",
                subfiletype=SubFileType.PREVIEW_IMAGE,
                encoding=pyramid.encoding,
                extratags=pyramid_tags,
            )
            pyramid_specs.append(pyramid_spec)
    
    # Generate rendered preview if requested
    if not preview:
        # No preview: IFD0 = main, SubIFD0+ = pyramid
        timer.start_step("write_dng_no_preview")
        write_dng(
            destination_file=destination_file,
            ifd0_spec=main_spec,
            subifds=pyramid_specs,
            num_compression_workers=num_compression_workers
        )
        timer.end_step()
    else:            
        logger.info(f"Rendering preview from pyramid level {preview_level_idx} ({pyramid_images[preview_level_idx].shape[:2]})")
        
        # Create copy of ifd0_tags with Orientation equal to HORIZONTAL
        # since the Orientation tag is persisted across the copy
        ifd0_tags_no_orientation = ifd0_tags.copy()
        ifd0_tags_no_orientation.add_tag("Orientation", Orientation.HORIZONTAL)
        
        # Render with color transforms
        timer.start_step("render_preview")
        rendered_preview = raw_render._render_camera_rgb(
            ifd0_tags=ifd0_tags_no_orientation,
            raw_ifd_tags=main_spec.extratags,
            rgb_camera=pyramid_images[preview_level_idx],
            output_dtype=np.uint8,
            rendering_params=preview.rendering_params,
            use_xmp=preview.use_xmp,
        )
        timer.end_step()
        
        # Create preview spec for IFD0
        preview_encoding = PageEncoding(
            compression=preview.compression,
            compression_args=preview.compression_args
        ) if preview.compression else None
        preview_spec = IfdDataSpec(
            data=rendered_preview,
            photometric="RGB",
            subfiletype=SubFileType.PREVIEW_IMAGE,
            encoding=preview_encoding,
            extratags=ifd0_tags,
        )
        
        # Write: IFD0 = preview, SubIFD0 = main, SubIFD1+ = pyramid
        timer.start_step("write_dng_with_preview")
        write_dng(
            destination_file=destination_file,
            ifd0_spec=preview_spec,
            subifds=[main_spec] + pyramid_specs,
            num_compression_workers=num_compression_workers
        )
        timer.end_step()
    
    if isinstance(destination_file, io.BytesIO):
        logger.info("Successfully wrote DNG to stream")
    else:
        logger.info(f"Successfully wrote DNG to {destination_file}")


def write_dng_from_array(
    destination_file: str | Path | io.BytesIO,
    data_spec: IfdDataSpec,
    *,
    scale: float = 1.0,
    demosaic: bool = False,
    demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.DNGSDK_BILINEAR,
    preview: PreviewParams | None = None,
    pyramid: PyramidParams | None = None,
    num_compression_workers: int = 1,
) -> None:
    """Write raw array data to a DNG file with optional preview and pyramid generation.
    
    Args:
        destination_file: Path or io.BytesIO object where to save the DNG file
        data_spec: IfdDataSpec containing raw image data and metadata
        preview: PreviewParams for preview generation (None = no preview)
        pyramid: PyramidParams for pyramid generation (None = no pyramid)
        num_compression_workers: Number of parallel compression workers (default: 1)
    """

    with scoped_perf_timer("write_dng_from_array", logger) as timer:
        # fast path - only one ifd requested and no data transformations
        if not (preview or pyramid or scale != 1.0 or (demosaic and data_spec.photometric == "CFA")):
            write_dng(
                destination_file,
                ifd0_spec=data_spec,
                num_compression_workers=num_compression_workers,
            )
            return

        # filter metadata tags into those that belong in ifd0 and those in main page
        ifd0_tags = MetadataTags()
        main_page_tags = MetadataTags()
        if data_spec.extratags:
            ifd0_tags |= data_spec.extratags
            ifd0_tags = filter_tags_by_ifd_category(
                ifd0_tags, ["any", "dng_ifd0", "ifd0", "exif", "dng_profile"])
            _filter_metadata_tags(ifd0_tags, exclude_names=_TIFFWRITER_MANAGED_TAGS)

            main_page_tags |= data_spec.extratags
            main_page_categories = ["any", "dng_raw"]
            if data_spec.photometric == "CFA" and not demosaic:
                main_page_categories += ["dng_raw:cfa"]
            main_page_tags = filter_tags_by_ifd_category(main_page_tags, main_page_categories)
            if not preview:
                main_page_tags = ifd0_tags | main_page_tags # if duplicates main page takes precedence

        # update data_spec with filtered main page tags
        updated_spec = replace(data_spec, extratags=main_page_tags)

        _write_dng_with_params(
            destination_file=destination_file,
            main_spec=updated_spec,
            ifd0_tags=ifd0_tags,
            scale=scale,
            demosaic=demosaic,
            demosaic_algorithm=demosaic_algorithm,
            preview=preview,
            pyramid=pyramid,
            num_compression_workers=num_compression_workers,
        )


def create_dng_from_array(
    data_spec: IfdDataSpec,
    *,
    preview: PreviewParams | None = None,
    pyramid: PyramidParams | None = None,
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
        >>> preview = PreviewParams(scale=PreviewScale.QUARTER)  # Presence means generate preview
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


def write_dng_from_page(
    destination_file: str | Path | io.BytesIO,
    page: IfdPageSpec | DngPage,
    *,
    scale: float = 1.0,
    demosaic: bool = False,
    demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.DNGSDK_BILINEAR,
    preview: PreviewParams | None = None,
    pyramid: PyramidParams | None = None,
    copy_ifd0_tags: bool = True,
    ifd0_extratags: MetadataTags | None = None,
    ifd0_strip_tags: set[str] | None = None,
    num_compression_workers: int = 1,
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
        num_compression_workers: Number of parallel compression workers (default: 1)
    
    Raises:
        ValueError: If input is invalid
        RuntimeError: If DNG processing fails
    """
    
    with scoped_perf_timer("write_dng_from_page", logger) as timer:

        # Ensure we have an IfdPageSpec
        source_page_spec = page if isinstance(page, IfdPageSpec) else IfdPageSpec(page=page)
        
        if not (source_page_spec.page.is_cfa or source_page_spec.page.is_linear_raw):
            raise ValueError(
                f"Page must be CFA or LINEAR_RAW, got photometric={source_page_spec.page.photometric_name}"
            )

        # do we change the main page pixels?
        main_needs_scale_or_demosaic = ((demosaic and source_page_spec.page.is_cfa) or scale != 1.0)

        # TODO: consider stripping compression invalidated and tiff tags from user supplied extra tags

        # handle ifd0 tags
        ifd0_tags = MetadataTags()
        if copy_ifd0_tags:
            ifd0_tags = source_page_spec.page.get_ifd0_tags()
            ifd0_tags = filter_tags_by_ifd_category(
                ifd0_tags, ["any", "dng_ifd0", "ifd0", "exif", "dng_profile"])
            _filter_metadata_tags(ifd0_tags, exclude_names=_TIFFWRITER_MANAGED_TAGS)

        _filter_metadata_tags(
            ifd0_tags, exclude_names=(ifd0_strip_tags or set()) | _COMPRESSION_INVALIDATED_TAGS)
        ifd0_tags |= ifd0_extratags

        # handle main page tags
        main_page_tags = MetadataTags()
        if source_page_spec.copy_page_tags:
            main_page_tags |= source_page_spec.page.get_page_tags()

            main_page_categories = ["any"]
            if source_page_spec.page.is_linear_raw or main_needs_scale_or_demosaic:
                main_page_categories += ["dng_raw"]
            elif source_page_spec.page.is_cfa:
                main_page_categories += ["dng_raw", "dng_raw:cfa"]
            main_page_tags = filter_tags_by_ifd_category(main_page_tags, main_page_categories)
            _filter_metadata_tags(main_page_tags, exclude_names=source_page_spec.strip_tags)
        if not preview:
            main_page_tags = ifd0_tags | main_page_tags # if duplicates main page takes precedence
        main_page_tags |= source_page_spec.extratags    

        # create a new main page spec that is our incoming spec with correct subfiletype and tags
        main_spec = replace(
            source_page_spec, 
            subfiletype=SubFileType.MAIN_IMAGE, 
            extratags=main_page_tags,
            copy_page_tags=False)

        if not main_needs_scale_or_demosaic:            
            # Fast path: no preview/pyramid - write and return immediately
            if preview is None and not (pyramid and pyramid.levels > 0):
                logger.info("Using fast path for main spec (no extra params given)")

                timer.start_step("write_dng_fast_path")
                write_dng(
                    destination_file=destination_file,
                    ifd0_spec=main_spec,
                    num_compression_workers=num_compression_workers
                )
                timer.end_step()
                if isinstance(destination_file, io.BytesIO):
                    logger.info("Successfully wrote DNG to stream")
                else:
                    logger.info(f"Successfully wrote DNG to {destination_file}")
                return

        _write_dng_with_params(
            destination_file=destination_file,
            main_spec=main_spec,
            ifd0_tags=ifd0_tags,
            scale=scale,
            demosaic=demosaic,
            demosaic_algorithm=demosaic_algorithm,
            preview=preview,
            pyramid=pyramid,
            num_compression_workers=num_compression_workers,
        )


def create_dng_from_page(
    page: IfdPageSpec | DngPage,
    *,
    scale: float = 1.0,
    demosaic: bool = False,
    demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.OPENCV_EA,
    preview: PreviewParams | None = None,
    pyramid: PyramidParams | None = None,
    copy_ifd0_tags: bool = True,
    ifd0_extratags: MetadataTags | None = None,
    ifd0_strip_tags: set[str] | None = None,
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
        >>> preview = PreviewParams(scale=PreviewScale.QUARTER, compression=tifffile.COMPRESSION.JPEG)
        >>> pyramid = PyramidParams(levels=2, encoding=PageEncoding(compression=tifffile.COMPRESSION.JPEGXL_DNG))
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
    file: str | Path | IO[bytes] | DngFile | DngPage,
    output_dtype: type = np.uint16,
    demosaic_algorithm: DemosaicAlgorithm = DemosaicAlgorithm.OPENCV_EA,
    use_coreimage_if_available: bool = False,
    use_xmp: bool = True,
    rendering_params: dict = None,
    strict: bool = True,
    scale: float = 1.0,
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
            - 'ToneCurvePV2012': Main tone curve as CubicSpline or list of (x,y) points
            - 'ToneCurvePV2012Red': Red channel tone curve
            - 'ToneCurvePV2012Green': Green channel tone curve
            - 'ToneCurvePV2012Blue': Blue channel tone curve
            - 'crlcp:PerspectiveModel': Lens correction profile
            - 'orientation': TIFF orientation value (1-8, int) to override EXIF orientation
            - 'highlight_preserving_exposure': Use highlight preservation (Python pipeline only)
        scale: Scaling factor for output resolution (e.g., 0.5 for half size). For Core Image
            with DngPage input, scales the raw array before processing. For Python SDK, selects
            optimal pyramid level and applies final scaling.
    
    Returns:
        Tuple of (image, metadata):
            - image: RGB image array with shape (height, width, 3) and specified dtype
            - metadata: MetadataTags containing IFD0 tags
    """
    # Try Core Image path if requested
    if use_coreimage_if_available:
        try:
            from ._dngio_coreimage import core_image_available, decode_dng_coreimage

            if core_image_available:
                # For Core Image, if scaling is needed, create a new DngFile from the main page with scale applied
                if isinstance(file, DngPage):
                    dng_file = create_dng_from_page(file, scale=scale)
                else:
                    # Create or use DngFile
                    dng_file = file if isinstance(file, DngFile) else DngFile(file)
                    
                    # If scaling is needed, create a new DngFile from the main page with scale applied
                    if scale != 1.0:
                        main_page = dng_file.get_main_page()
                        if main_page is None:
                            raise ValueError("No main page found in DNG file")
                        dng_file = create_dng_from_page(main_page, scale=scale)
                
                # Extract metadata
                metadata = dng_file.get_ifd0_tags()
                
                image = decode_dng_coreimage(
                    file=dng_file,
                    use_xmp=use_xmp,
                    output_dtype=output_dtype,
                    rendering_params=rendering_params,
                )
                # Image has been rotated during rendering, set Orientation to HORIZONTAL (normal)
                metadata.add_tag("Orientation", Orientation.HORIZONTAL)
                return image, metadata

            logger.warning(
                "Core Image requested but not supported on this platform; falling back to Python pipeline."
            )
        except Exception as e:
            logger.warning(
                f"Core Image processing failed ({type(e).__name__}): {e}; falling back to Python pipeline."
            )
    
    # Python SDK pipeline
    # For DngPage input, convert to DngFile without scaling (render_raw will handle it)
    if isinstance(file, DngPage):
        dng_file = create_dng_from_page(file, scale=1.0)
    else:
        # Create or use DngFile
        dng_file = file if isinstance(file, DngFile) else DngFile(file)
    
    # Extract metadata
    metadata = dng_file.get_ifd0_tags()
    
    # Always pass scale to render_raw for Python pipeline
    result = dng_file.render_raw(
        output_dtype=output_dtype,
        demosaic_algorithm=demosaic_algorithm,
        use_xmp=use_xmp,
        rendering_params=rendering_params,
        strict=strict,
        scale=scale,
    )
    if result is None:
        raise RuntimeError(f"No main image page found in DNG file: {file}")
    
    # Image has been rotated during rendering, set Orientation to HORIZONTAL (normal)
    metadata.add_tag("Orientation", Orientation.HORIZONTAL)
    return result, metadata
