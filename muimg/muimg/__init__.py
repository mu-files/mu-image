"""
DNG and raw image processing utilities.

This module provides functionality for working with DNG files and raw image data.
"""

from .tiff_metadata import (
    BAYER_PATTERN_MAP,
    INVERSE_BAYER_PATTERN_MAP,
    MetadataTags,
    XmpMetadata,
    translate_dng_tag,
)
from .dngio import (
    DngFile,
    DngPage,
    cfa_from_dng,
    cfa_from_rgb_planes,
    rgb_planes_from_cfa,
    rgb_planes_from_dng,
    write_dng,
    write_dng_linearraw,
    write_dng_from_page,
    decode_raw,
    convert_raw,
    convert_raw_to_stream,
)
from .color_mac import process_raw_core_image
from .csv import CsvOrderedWriter, CsvReader, CsvWriter
from .processing import ProcessingPipeline
from .imgio import decode_image
from .videoio import SequenceEncodePipeline
from .color import (
    SplineCurve,
    UnsupportedDNGTagError,
    interp_center,
    interp_center_green,
    fix_hot_pixels,
    linear_raw_from_cfa,
    linear_raw_from_dng,
    process_raw,
)

__all__ = [
    'BAYER_PATTERN_MAP',
    'INVERSE_BAYER_PATTERN_MAP',
    'CsvOrderedWriter',
    'CsvReader',
    'CsvWriter',
    'DngFile',
    'DngPage',
    'MetadataTags',
    'XmpMetadata',
    'translate_dng_tag',
    'UnsupportedDNGTagError',
    'cfa_from_dng',
    'cfa_from_rgb_planes',
    'convert_raw',
    'convert_raw_to_stream',
    'decode_image',
    'decode_raw',
    'fix_hot_pixels',
    'interp_center',
    'interp_center_green',
    'linear_raw_from_cfa',
    'linear_raw_from_dng',
    'process_raw',
    'process_raw_core_image',
    'ProcessingPipeline',
    'SequenceEncodePipeline',
    'rgb_planes_from_cfa',
    'rgb_planes_from_dng',
    'SplineCurve',
    'write_dng',
    'write_dng_linearraw',
    'write_dng_from_page',
]
