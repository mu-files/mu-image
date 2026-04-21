"""TIFF/DNG metadata support classes.

This module provides classes for creating and parsing TIFF/DNG metadata tags.
"""
from __future__ import annotations

import logging
import numpy as np

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from fractions import Fraction
from tifffile import PHOTOMETRIC, TIFF
from typing import Optional, Union, Dict, Any, Type, List, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Tag Type Registry
# =============================================================================
# Maps tag names to TagSpec with:
#   dtype: TIFF data type string(s) ('s'=ascii, 'H'=ushort, 'I'=ulong, '2I'=urational, 
#          '2i'=srational, 'B'=byte, 'f'=float, 'd'=double). Can be a list for type inference.
#   count: Expected count, or None for variable length
#   shape: Target shape for reshaping arrays (e.g., (3, 3) for matrices), or None
#   dng_ifd: IFD location category ("dng_raw", "dng_ifd0", "dng_profile", "any", etc.)
#
# This registry enables auto-conversion: clients provide friendly Python types
# (float, datetime, str, np.ndarray) and we convert to appropriate TIFF format.
#
# For tags that accept multiple types (per DNG SDK), dtype can be a list.
# Type inference order: int types first, then float/rational types.
# Example: ["I", "2I"] means int→LONG, float→RATIONAL

class TiffType(int, Enum):
    """TIFF data type codes with metadata.
    
    Provides readable type names with metadata access via properties.
    
    Usage:
        TiffType.RATIONAL == 5  # True (int comparison)
        TiffType.RATIONAL.dtype_str  # '2I'
    """
    
    def __new__(cls, code: int, dtype_str: str):
        obj = int.__new__(cls, code)
        obj._value_ = code
        obj.dtype_str = dtype_str
        return obj
    
    # Type definitions with (code, dtype_str)
    BYTE = (1, 'B')           # Unsigned 8-bit integer
    ASCII = (2, 's')          # 8-bit byte containing 7-bit ASCII
    SHORT = (3, 'H')          # Unsigned 16-bit integer
    LONG = (4, 'I')           # Unsigned 32-bit integer
    RATIONAL = (5, '2I')      # Two LONGs: numerator, denominator
    SBYTE = (6, 'b')          # Signed 8-bit integer
    UNDEFINED = (7, 'B')      # 8-bit byte (uninterpreted data)
    SSHORT = (8, 'h')         # Signed 16-bit integer
    SLONG = (9, 'i')          # Signed 32-bit integer
    SRATIONAL = (10, '2i')    # Two SLONGs: numerator, denominator
    FLOAT = (11, 'f')         # 32-bit IEEE floating point
    DOUBLE = (12, 'd')        # 64-bit IEEE floating point
    LONG8 = (16, 'Q')         # Unsigned 64-bit integer
    SLONG8 = (17, 'q')        # Signed 64-bit integer

# Derived lookups for convenience
TIFFTYPE_TO_DTYPE_STR = {t.value: t.dtype_str for t in TiffType}

# Multi-type dtype lists for tags that accept multiple types
# Use these constants in TagSpec definitions to ensure consistency
TIFFTYPE_SHORT_OR_LONG = [TiffType.SHORT, TiffType.LONG]        # SHORT or LONG (prefer LONG for ints)
TIFFTYPE_ASCII_OR_BYTE = [TiffType.ASCII, TiffType.BYTE]        # ASCII or BYTE (ASCII for str, BYTE for bytes/UTF-8)
TIFFTYPE_INT_OR_RATIONAL = [TiffType.SHORT, TiffType.LONG, TiffType.RATIONAL]   # SHORT, LONG, or RATIONAL (RATIONAL for floats, LONG for ints)

# =============================================================================
# CFA Pattern Constants
# =============================================================================
# Mapping between CFA pattern strings and byte codes (0=R, 1=G, 2=B)
CFA_PATTERN_TO_CODES: dict[str, tuple[int, ...]] = {
    "RGGB": (0, 1, 1, 2),
    "BGGR": (2, 1, 1, 0),
    "GRBG": (1, 0, 2, 1),
    "GBRG": (1, 2, 0, 1),
}
CFA_CODES_TO_PATTERN: dict[tuple[int, ...], str] = {v: k for k, v in CFA_PATTERN_TO_CODES.items()}

# =============================================================================
# IFD Location Categories
# =============================================================================
# Specifies which IFD type a tag should appear in (per DNG specification):
# - "any": Can appear in any IFD (default for basic TIFF structure tags)
# - "dng_ifd0": Must only be in IFD 0 (main metadata, not in SubIFDs)
# - "exif": Can be in IFD 0 or EXIF IFD (not in preview/thumbnail SubIFDs)
# - "dng_raw": Must be in IFD containing raw image data (CFA or LINEAR_RAW)
# - "dng_raw:cfa": A raw IFD with PhotometricInterpretation = CFA
# - "dng_profile": Can be in IFD 0 or Camera Profile IFDs
# - "dng_preview": Must be in Preview NewSubFileType

@dataclass
class TagSpec:
    """Specification for a TIFF/DNG tag.
    
    Attributes:
        dtype: TIFF data type(s). Can be a single TiffType or list for type inference.
               When a list, int values use first matching int type, floats use first
               matching float/rational type. Examples: TiffType.BYTE, TiffType.RATIONAL, TiffType.UNDEFINED.
        count: Expected count, or None for variable length.
        shape: Target shape for array tags (e.g., (3, 3) for matrices). If specified,
               the returned np.ndarray will be reshaped to this shape.
        dng_ifd: IFD type where this tag should appear (see IFD Location Categories above).
    """
    dtype: Union[TiffType, List[TiffType]]  # Single type or list for inference
    count: Optional[int] = None
    shape: Optional[Tuple[int, ...]] = None
    dng_ifd: str = "any"
    
    def get_dtype_for_value(self, value: Any) -> TiffType:
        """Select appropriate dtype based on value type.
        
        For single-type specs, returns that type.
        For multi-type specs, selects based on value type using explicit case handling.
        """
        if isinstance(self.dtype, TiffType):
            return self.dtype
        
        # Handle multi-type lists explicitly
        if self.dtype == TIFFTYPE_SHORT_OR_LONG:
            # SHORT or LONG: always use LONG for simplicity
            return TiffType.LONG
        elif self.dtype == TIFFTYPE_ASCII_OR_BYTE:
            # ASCII or BYTE: ASCII for str, BYTE for bytes/UTF-8
            return TiffType.ASCII if isinstance(value, str) else TiffType.BYTE
        elif self.dtype == TIFFTYPE_INT_OR_RATIONAL:
            # SHORT, LONG, or RATIONAL: RATIONAL for float or pre-formatted rational tuples
            # Check for rational tuple (even-length flat tuple of ints: (num, denom) or (n1, d1, n2, d2, ...))
            if (isinstance(value, tuple) and len(value) % 2 == 0 and len(value) >= 2 and
                all(isinstance(x, (int, np.integer)) for x in value)):
                return TiffType.RATIONAL
            # Check for float scalar
            if isinstance(value, (float, np.floating)):
                return TiffType.RATIONAL
            # Check for float array/list
            if isinstance(value, (list, tuple)) and len(value) > 0:
                if isinstance(value[0], (float, np.floating)):
                    return TiffType.RATIONAL
            if hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.floating):
                return TiffType.RATIONAL
            return TiffType.LONG
        else:
            raise ValueError(f"Unknown multi-type dtype list: {self.dtype}. "
                           f"Add explicit handling or use a defined constant.")
    
def get_native_type(dtype: Union[int, str, list], count: Optional[int]) -> Optional[type]:
    """Determine the most appropriate Python type for a TIFF tag.
    
    Args:
        dtype: TIFF dtype code (int, e.g., 1, 2, 5) or dtype string (e.g., 'H', 'I', '2I')
        count: Tag count, or None for variable length
        
    Returns:
        Appropriate Python type:
        - str for string types
        - np.ndarray for multi-value tags (count > 1 or variable length numeric)
        - float for RATIONAL types (2I, 2i) with count=1
        - int for integer types with count=1
        - bytes for variable-length byte arrays
    """
    # Normalize dtype to string
    if isinstance(dtype, int):
        dtype_str = TIFFTYPE_TO_DTYPE_STR.get(dtype)
        if dtype_str is None:
            return None
    elif isinstance(dtype, list):
        dtype_str = dtype[0]  # Use first dtype for multi-type tags
    else:
        dtype_str = dtype
    # String types -> str
    if dtype_str == 's':
        return str
    
    # Multi-value tags (fixed count > 1) -> ndarray
    if count is not None and count > 1:
        return np.ndarray
    
    # Variable-length numeric tags -> ndarray (safer default for processing)
    if count is None and dtype_str not in ('s', 'B'):
        return np.ndarray
    
    # RATIONAL types with count=1 -> float
    if dtype_str in ('2I', '2i') and count == 1:
        return float
    
    # Integer types with count=1 -> int
    if dtype_str in ('B', 'b', 'H', 'h', 'I', 'i', 'Q', 'q') and count == 1:
        return int
    
    # Float types with count=1 -> float
    if dtype_str in ('f', 'd') and count == 1:
        return float
    
    # Byte arrays -> bytes
    if dtype_str == 'B':
        return bytes
    
    return None


# Comprehensive TIFF/DNG/EXIF tag registry, sorted by tag code
# Types verified against DNG SDK source and tifffile registry
# For multi-type tags, first type is for integers, second for floats
TIFF_TAG_TYPE_REGISTRY: Dict[str, TagSpec] = {
    "ProcessingSoftware": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 11
    "NewSubfileType": TagSpec(TiffType.LONG, 1, dng_ifd="any"),  # 254
    "SubfileType": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 255
    "ImageWidth": TagSpec(TIFFTYPE_SHORT_OR_LONG, 1, dng_ifd="any"),  # 256
    "ImageLength": TagSpec(TIFFTYPE_SHORT_OR_LONG, 1, dng_ifd="any"),  # 257
    "BitsPerSample": TagSpec(TiffType.SHORT, None, dng_ifd="any"),  # 258
    "Compression": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 259
    "PhotometricInterpretation": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 262
    "Thresholding": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 263
    "CellWidth": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 264
    "CellLength": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 265
    "FillOrder": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 266
    "DocumentName": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 269
    "ImageDescription": TagSpec(TiffType.ASCII, None, dng_ifd="dng_ifd0"),  # 270
    "Make": TagSpec(TiffType.ASCII, None, dng_ifd="dng_ifd0"),  # 271
    "Model": TagSpec(TiffType.ASCII, None, dng_ifd="dng_ifd0"),  # 272
    "StripOffsets": TagSpec(TIFFTYPE_SHORT_OR_LONG, None, dng_ifd="any"),  # 273
    "Orientation": TagSpec(TiffType.SHORT, 1, dng_ifd="dng_ifd0"),  # 274
    "SamplesPerPixel": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 277
    "RowsPerStrip": TagSpec(TIFFTYPE_SHORT_OR_LONG, 1, dng_ifd="any"),  # 278
    "StripByteCounts": TagSpec(TIFFTYPE_SHORT_OR_LONG, None, dng_ifd="any"),  # 279
    "MinSampleValue": TagSpec(TiffType.SHORT, None, dng_ifd="any"),  # 280
    "MaxSampleValue": TagSpec(TiffType.SHORT, None, dng_ifd="any"),  # 281
    "XResolution": TagSpec(TiffType.RATIONAL, 1, dng_ifd="any"),  # 282
    "YResolution": TagSpec(TiffType.RATIONAL, 1, dng_ifd="any"),  # 283
    "PlanarConfiguration": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 284
    "PageName": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 285
    "XPosition": TagSpec(TiffType.RATIONAL, 1, dng_ifd="any"),  # 286
    "YPosition": TagSpec(TiffType.RATIONAL, 1, dng_ifd="any"),  # 287
    "FreeOffsets": TagSpec(TiffType.LONG, None, dng_ifd="any"),  # 288
    "FreeByteCounts": TagSpec(TiffType.LONG, None, dng_ifd="any"),  # 289
    "GrayResponseUnit": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 290
    "GrayResponseCurve": TagSpec(TiffType.SHORT, None, dng_ifd="any"),  # 291
    "T4Options": TagSpec(TiffType.LONG, 1, dng_ifd="any"),  # 292
    "T6Options": TagSpec(TiffType.LONG, 1, dng_ifd="any"),  # 293
    "ResolutionUnit": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 296
    "PageNumber": TagSpec(TiffType.SHORT, 2, dng_ifd="any"),  # 297
    "TransferFunction": TagSpec(TiffType.SHORT, None, dng_ifd="any"),  # 301
    "Software": TagSpec(TiffType.ASCII, None, dng_ifd="dng_ifd0"),  # 305
    "DateTime": TagSpec(TiffType.ASCII, 20, dng_ifd="dng_ifd0"),  # 306
    "Artist": TagSpec(TiffType.ASCII, None, dng_ifd="dng_ifd0"),  # 315
    "HostComputer": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 316
    "Predictor": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 317
    "WhitePoint": TagSpec(TiffType.RATIONAL, 2, dng_ifd="any"),  # 318
    "PrimaryChromaticities": TagSpec(TiffType.RATIONAL, 6, dng_ifd="any"),  # 319
    "ColorMap": TagSpec(TiffType.SHORT, None, dng_ifd="any"),  # 320
    "HalftoneHints": TagSpec(TiffType.SHORT, 2, dng_ifd="any"),  # 321
    "TileWidth": TagSpec(TIFFTYPE_SHORT_OR_LONG, 1, dng_ifd="any"),  # 322
    "TileLength": TagSpec(TIFFTYPE_SHORT_OR_LONG, 1, dng_ifd="any"),  # 323
    "TileOffsets": TagSpec(TiffType.LONG, None, dng_ifd="any"),  # 324
    "TileByteCounts": TagSpec(TIFFTYPE_SHORT_OR_LONG, None, dng_ifd="any"),  # 325
    "SubIFDs": TagSpec(TiffType.LONG, None, dng_ifd="any"),  # 330
    "InkSet": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 332
    "InkNames": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 333
    "NumberOfInks": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 334
    "DotRange": TagSpec(TiffType.SHORT, None, dng_ifd="any"),  # 336
    "TargetPrinter": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 337
    "ExtraSamples": TagSpec(TiffType.SHORT, None, dng_ifd="any"),  # 338
    "SampleFormat": TagSpec(TiffType.SHORT, None, dng_ifd="any"),  # 339
    "SMinSampleValue": TagSpec(TiffType.DOUBLE, None, dng_ifd="any"),  # 340
    "SMaxSampleValue": TagSpec(TiffType.DOUBLE, None, dng_ifd="any"),  # 341
    "TransferRange": TagSpec(TiffType.SHORT, 6, dng_ifd="any"),  # 342
    "ClipPath": TagSpec(TiffType.BYTE, None, dng_ifd="any"),  # 343
    "XClipPathUnits": TagSpec(TiffType.LONG, 1, dng_ifd="any"),  # 344
    "YClipPathUnits": TagSpec(TiffType.LONG, 1, dng_ifd="any"),  # 345
    "Indexed": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 346
    "JPEGTables": TagSpec(TiffType.BYTE, None, dng_ifd="any"),  # 347
    "OPIProxy": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 351
    "VersionYear": TagSpec(TiffType.BYTE, 4, dng_ifd="any"),  # 404
    "Decode": TagSpec(TiffType.SRATIONAL, None, dng_ifd="any"),  # 433
    "DefaultImageColor": TagSpec(TiffType.SHORT, None, dng_ifd="any"),  # 434
    "JPEGInterchangeFormat": TagSpec(TiffType.LONG, 1, dng_ifd="any"),  # 513
    "JPEGInterchangeFormatLength": TagSpec(TiffType.LONG, 1, dng_ifd="any"),  # 514
    "JPEGRestartInterval": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 515
    "JPEGLosslessPredictors": TagSpec(TiffType.SHORT, None, dng_ifd="any"),  # 517
    "JPEGPointTransforms": TagSpec(TiffType.SHORT, None, dng_ifd="any"),  # 518
    "JPEGQTables": TagSpec(TiffType.LONG, None, dng_ifd="any"),  # 519
    "JPEGDCTables": TagSpec(TiffType.LONG, None, dng_ifd="any"),  # 520
    "JPEGACTables": TagSpec(TiffType.LONG, None, dng_ifd="any"),  # 521
    "YCbCrCoefficients": TagSpec(TiffType.RATIONAL, 3, dng_ifd="any"),  # 529
    "YCbCrSubSampling": TagSpec(TiffType.SHORT, 2, dng_ifd="any"),  # 530
    "YCbCrPositioning": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 531
    "ReferenceBlackWhite": TagSpec(TiffType.RATIONAL, 6, dng_ifd="any"),  # 532
    "StripRowCounts": TagSpec(TiffType.LONG, None, dng_ifd="any"),  # 559
    "XMP": TagSpec(TiffType.BYTE, None, dng_ifd="any"),  # 700
    "ICCProfileDescriptor": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 770
    "Rating": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 18246
    "RatingPercent": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 18249
    "PrintFlags": TagSpec(TiffType.BYTE, None, dng_ifd="any"),  # 20485
    "PrintFlagsVersion": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 20486
    "PrintFlagsCrop": TagSpec(TiffType.LONG, 1, dng_ifd="any"),  # 20487
    "PrintFlagsBleedWidth": TagSpec(TiffType.LONG, 1, dng_ifd="any"),  # 20488
    "PrintFlagsBleedWidthScale": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 20489
    "InteroperabilityIndex": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 20545
    "InteroperabilityVersion": TagSpec(TiffType.BYTE, 4, dng_ifd="any"),  # 20546
    "FrameDelay": TagSpec(TiffType.LONG, None, dng_ifd="any"),  # 20736
    "LoopCount": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 20737
    "VignettingCorrParams": TagSpec(TiffType.SRATIONAL, None, dng_ifd="any"),  # 28722
    "ChromaticAberrationCorrParams": TagSpec(TiffType.SRATIONAL, None, dng_ifd="any"),  # 28725
    "DistortionCorrParams": TagSpec(TiffType.SRATIONAL, None, dng_ifd="any"),  # 28727
    
    # Private tags >= 32768
    "ImageID": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 32781
    "Matteing": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 32995
    "DataType": TagSpec(TiffType.SHORT, None, dng_ifd="any"),  # 32996
    "ImageDepth": TagSpec(TIFFTYPE_SHORT_OR_LONG, 1, dng_ifd="any"),  # 32997
    "TileDepth": TagSpec(TIFFTYPE_SHORT_OR_LONG, 1, dng_ifd="any"),  # 32998
    "Model2": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 33405
    "CFARepeatPatternDim": TagSpec(TiffType.SHORT, 2, dng_ifd="dng_raw:cfa"),  # 33421
    "CFAPattern": TagSpec(TiffType.BYTE, None, dng_ifd="dng_raw:cfa"),  # 33422
    "BatteryLevel": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 33423
    "Copyright": TagSpec(TiffType.ASCII, None, dng_ifd="dng_ifd0"),  # 33432
    "ExposureTime": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 33434
    "FNumber": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 33437
    "ModelPixelScaleTag": TagSpec(TiffType.DOUBLE, 3, dng_ifd="dng_ifd0"),  # 33550
    "IPTCNAA": TagSpec(TiffType.BYTE, None, dng_ifd="any"),  # 33723
    "ModelTiepointTag": TagSpec(TiffType.DOUBLE, None, dng_ifd="dng_ifd0"),  # 33922
    "ModelTransformationTag": TagSpec(TiffType.DOUBLE, 16, dng_ifd="dng_ifd0"),  # 34264
    "WB_GRGBLevels": TagSpec(TiffType.RATIONAL, 4, dng_ifd="any"),  # 34306
    "ImageResources": TagSpec(TiffType.BYTE, None, dng_ifd="any"),  # 34377
    "ExifTag": TagSpec(TiffType.LONG, 1, dng_ifd="dng_ifd0"),  # 34665
    "InterColorProfile": TagSpec(TiffType.BYTE, None, dng_ifd="any"),  # 34675
    "GeoKeyDirectoryTag": TagSpec(TiffType.SHORT, None, dng_ifd="dng_ifd0"),  # 34735
    "GeoDoubleParamsTag": TagSpec(TiffType.DOUBLE, None, dng_ifd="dng_ifd0"),  # 34736
    "GeoAsciiParamsTag": TagSpec(TiffType.ASCII, None, dng_ifd="dng_ifd0"),  # 34737
    "ExposureProgram": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 34850
    "SpectralSensitivity": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 34852
    "GPSTag": TagSpec(TiffType.LONG, 1, dng_ifd="dng_ifd0"),  # 34853
    "ISOSpeedRatings": TagSpec(TiffType.SHORT, None, dng_ifd="exif"),  # 34855
    "OECF": TagSpec(TiffType.BYTE, None, dng_ifd="exif"),  # 34856
    "Interlace": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 34857
    "TimeZoneOffset": TagSpec(TiffType.SSHORT, None, dng_ifd="exif"),  # 34858
    "SelfTimerMode": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 34859
    "SensitivityType": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 34864
    "StandardOutputSensitivity": TagSpec(TiffType.LONG, 1, dng_ifd="exif"),  # 34865
    "RecommendedExposureIndex": TagSpec(TiffType.LONG, 1, dng_ifd="exif"),  # 34866
    "ISOSpeed": TagSpec(TiffType.LONG, 1, dng_ifd="exif"),  # 34867
    "ISOSpeedLatitudeyyy": TagSpec(TiffType.LONG, 1, dng_ifd="exif"),  # 34868
    "ISOSpeedLatitudezzz": TagSpec(TiffType.LONG, 1, dng_ifd="exif"),  # 34869
    "ExifVersion": TagSpec(TiffType.BYTE, 4, dng_ifd="exif"),  # 36864
    "DateTimeOriginal": TagSpec(TiffType.ASCII, 20, dng_ifd="exif"),  # 36867
    "DateTimeDigitized": TagSpec(TiffType.ASCII, 20, dng_ifd="exif"),  # 36868
    "OffsetTime": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 36880
    "OffsetTimeOriginal": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 36881
    "OffsetTimeDigitized": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 36882
    "ComponentsConfiguration": TagSpec(TiffType.BYTE, 4, dng_ifd="exif"),  # 37121
    "CompressedBitsPerPixel": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 37122
    "ShutterSpeedValue": TagSpec(TiffType.SRATIONAL, 1, dng_ifd="exif"),  # 37377
    "ApertureValue": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 37378
    "BrightnessValue": TagSpec(TiffType.SRATIONAL, 1, dng_ifd="exif"),  # 37379
    "ExposureBiasValue": TagSpec(TiffType.SRATIONAL, 1, dng_ifd="exif"),  # 37380
    "MaxApertureValue": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 37381
    "SubjectDistance": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 37382
    "MeteringMode": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 37383
    "LightSource": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 37384
    "Flash": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 37385
    "FocalLength": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 37386
    "FlashEnergy": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 37387
    "SpatialFrequencyResponse": TagSpec(TiffType.BYTE, None, dng_ifd="exif"),  # 37388
    "Noise": TagSpec(TiffType.BYTE, None, dng_ifd="exif"),  # 37389
    "FocalPlaneXResolution": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 37390
    "FocalPlaneYResolution": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 37391
    "FocalPlaneResolutionUnit": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 37392
    "ImageNumber": TagSpec(TIFFTYPE_SHORT_OR_LONG, 1, dng_ifd="exif"),  # 37393
    "SecurityClassification": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 37394
    "ImageHistory": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 37395
    "SubjectLocation": TagSpec(TiffType.SHORT, 2, dng_ifd="exif"),  # 37396
    "ExposureIndex": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 37397
    "TIFFEPStandardID": TagSpec(TiffType.BYTE, 4, dng_ifd="exif"),  # 37398
    "SensingMethod": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 37399
    "MakerNote": TagSpec(TiffType.BYTE, None, dng_ifd="exif"),  # 37500
    "UserComment": TagSpec(TiffType.BYTE, None, dng_ifd="exif"),  # 37510
    "SubsecTime": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 37520
    "SubsecTimeOriginal": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 37521
    "SubsecTimeDigitized": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 37522
    "ImageSourceData": TagSpec(TiffType.BYTE, None, dng_ifd="exif"),  # 37724
    "Temperature": TagSpec(TiffType.SRATIONAL, 1, dng_ifd="exif"),  # 37888
    "Humidity": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 37889
    "Pressure": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 37890
    "WaterDepth": TagSpec(TiffType.SRATIONAL, 1, dng_ifd="exif"),  # 37891
    "Acceleration": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 37892
    "CameraElevationAngle": TagSpec(TiffType.SRATIONAL, 1, dng_ifd="exif"),  # 37893
    "FlashpixVersion": TagSpec(TiffType.BYTE, 4, dng_ifd="exif"),  # 40960
    "ColorSpace": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 40961
    "PixelXDimension": TagSpec(TIFFTYPE_SHORT_OR_LONG, 1, dng_ifd="exif"),  # 40962
    "PixelYDimension": TagSpec(TIFFTYPE_SHORT_OR_LONG, 1, dng_ifd="exif"),  # 40963
    "RelatedSoundFile": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 40964
    "InteroperabilityTag": TagSpec(TiffType.LONG, 1, dng_ifd="exif"),  # 40965
    "TIFF-EPStandardID": TagSpec(TiffType.BYTE, 4, dng_ifd="exif"),  # 41494
    "FileSource": TagSpec(TiffType.UNDEFINED, 1, dng_ifd="exif"),  # 41728 - UNDEFINED type
    "SceneType": TagSpec(TiffType.UNDEFINED, 1, dng_ifd="exif"),  # 41729 - UNDEFINED type
    "CustomRendered": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 41985
    "ExposureMode": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 41986
    "WhiteBalance": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 41987
    "DigitalZoomRatio": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 41988
    "FocalLengthIn35mmFilm": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 41989
    "SceneCaptureType": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 41990
    "GainControl": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 41991
    "Contrast": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 41992
    "Saturation": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 41993
    "Sharpness": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 41994
    "DeviceSettingDescription": TagSpec(TiffType.BYTE, None, dng_ifd="exif"),  # 41995
    "SubjectDistanceRange": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 41996
    "ImageUniqueID": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 42016
    "CameraOwnerName": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 42032
    "BodySerialNumber": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 42033
    "LensSpecification": TagSpec(TiffType.RATIONAL, 4, dng_ifd="exif"),  # 42034
    "LensMake": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 42035
    "LensModel": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 42036
    "LensSerialNumber": TagSpec(TiffType.ASCII, None, dng_ifd="exif"),  # 42037
    "CompositeImage": TagSpec(TiffType.SHORT, 1, dng_ifd="exif"),  # 42080
    "SourceImageNumberCompositeImage": TagSpec(TiffType.SHORT, None, dng_ifd="exif"),  # 42081
    "SourceExposureTimesCompositeImage": TagSpec(TiffType.BYTE, None, dng_ifd="exif"),  # 42082
    "Gamma": TagSpec(TiffType.RATIONAL, 1, dng_ifd="exif"),  # 42240
    "PixelFormat": TagSpec(TiffType.BYTE, 16, dng_ifd="any"),  # 48129
    "ImageType": TagSpec(TiffType.SHORT, 1, dng_ifd="any"),  # 48132
    "OriginalFileName": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 50547
    "DNGVersion": TagSpec(TiffType.BYTE, 4, dng_ifd="dng_ifd0"),  # 50706
    "DNGBackwardVersion": TagSpec(TiffType.BYTE, 4, dng_ifd="dng_ifd0"),  # 50707
    "UniqueCameraModel": TagSpec(TiffType.ASCII, None, dng_ifd="dng_ifd0"),  # 50708
    "LocalizedCameraModel": TagSpec(TIFFTYPE_ASCII_OR_BYTE, None, dng_ifd="dng_ifd0"),  # 50709
    "CFAPlaneColor": TagSpec(TiffType.BYTE, None, dng_ifd="dng_raw:cfa"),  # 50710
    "CFALayout": TagSpec(TiffType.SHORT, 1, dng_ifd="dng_raw:cfa"),  # 50711
    "LinearizationTable": TagSpec(TiffType.SHORT, None, dng_ifd="dng_raw"),  # 50712
    "BlackLevelRepeatDim": TagSpec(TiffType.SHORT, 2, dng_ifd="dng_raw"),  # 50713
    "BlackLevel": TagSpec(TIFFTYPE_INT_OR_RATIONAL, None, dng_ifd="dng_raw"),  # 50714
    "BlackLevelDeltaH": TagSpec(TiffType.SRATIONAL, None, dng_ifd="dng_raw"),  # 50715
    "BlackLevelDeltaV": TagSpec(TiffType.SRATIONAL, None, dng_ifd="dng_raw"),  # 50716
    "WhiteLevel": TagSpec(TIFFTYPE_SHORT_OR_LONG, None, dng_ifd="dng_raw"),  # 50717
    "DefaultScale": TagSpec(TiffType.RATIONAL, 2, dng_ifd="dng_raw"),  # 50718
    "DefaultCropOrigin": TagSpec(TIFFTYPE_INT_OR_RATIONAL, 2, dng_ifd="dng_raw"),  # 50719
    "DefaultCropSize": TagSpec(TIFFTYPE_INT_OR_RATIONAL, 2, dng_ifd="dng_raw"),  # 50720
    "ColorMatrix1": TagSpec(TiffType.SRATIONAL, 9, (3, 3), dng_ifd="dng_profile"),  # 50721 - 3x3 matrix
    "ColorMatrix2": TagSpec(TiffType.SRATIONAL, 9, (3, 3), dng_ifd="dng_profile"),  # 50722 - 3x3 matrix
    "CameraCalibration1": TagSpec(TiffType.SRATIONAL, 9, (3, 3), dng_ifd="dng_ifd0"),  # 50723 - 3x3 matrix
    "CameraCalibration2": TagSpec(TiffType.SRATIONAL, 9, (3, 3), dng_ifd="dng_ifd0"),  # 50724 - 3x3 matrix
    "ReductionMatrix1": TagSpec(TiffType.SRATIONAL, None, dng_ifd="dng_profile"),  # 50725
    "ReductionMatrix2": TagSpec(TiffType.SRATIONAL, None, dng_ifd="dng_profile"),  # 50726
    "AnalogBalance": TagSpec(TiffType.RATIONAL, None, dng_ifd="dng_ifd0"),  # 50727
    "AsShotNeutral": TagSpec(TiffType.RATIONAL, None, dng_ifd="dng_ifd0"),  # 50728
    "AsShotWhiteXY": TagSpec(TiffType.RATIONAL, 2, dng_ifd="dng_ifd0"),  # 50729
    "BaselineExposure": TagSpec(TiffType.SRATIONAL, 1, dng_ifd="dng_ifd0"),  # 50730
    "BaselineNoise": TagSpec(TiffType.RATIONAL, 1, dng_ifd="dng_ifd0"),  # 50731
    "BaselineSharpness": TagSpec(TiffType.RATIONAL, 1, dng_ifd="dng_ifd0"),  # 50732
    "BayerGreenSplit": TagSpec(TiffType.LONG, 1, dng_ifd="dng_raw:cfa"),  # 50733
    "LinearResponseLimit": TagSpec(TiffType.RATIONAL, 1, dng_ifd="dng_ifd0"),  # 50734
    "CameraSerialNumber": TagSpec(TiffType.ASCII, None, dng_ifd="dng_ifd0"),  # 50735
    "LensInfo": TagSpec(TiffType.RATIONAL, 4, dng_ifd="dng_ifd0"),  # 50736
    "ChromaBlurRadius": TagSpec(TiffType.RATIONAL, 1, dng_ifd="dng_raw"),  # 50737
    "AntiAliasStrength": TagSpec(TiffType.RATIONAL, 1, dng_ifd="dng_raw"),  # 50738
    "ShadowScale": TagSpec(TiffType.RATIONAL, 1, dng_ifd="dng_ifd0"),  # 50739
    "DNGPrivateData": TagSpec(TiffType.BYTE, None, dng_ifd="dng_ifd0"),  # 50740
    "MakerNoteSafety": TagSpec(TiffType.SHORT, 1, dng_ifd="dng_ifd0"),  # 50741
    "RawImageSegmentation": TagSpec(TiffType.SHORT, 3, dng_ifd="any"),  # 50752
    "CalibrationIlluminant1": TagSpec(TiffType.SHORT, 1, dng_ifd="dng_profile"),  # 50778
    "CalibrationIlluminant2": TagSpec(TiffType.SHORT, 1, dng_ifd="dng_profile"),  # 50779
    "BestQualityScale": TagSpec(TiffType.RATIONAL, 1, dng_ifd="dng_raw"),  # 50780
    "RawDataUniqueID": TagSpec(TiffType.BYTE, 16, dng_ifd="dng_ifd0"),  # 50781
    "OriginalRawFileName": TagSpec(TIFFTYPE_ASCII_OR_BYTE, None, dng_ifd="dng_ifd0"),  # 50827
    "OriginalRawFileData": TagSpec(TiffType.BYTE, None, dng_ifd="dng_ifd0"),  # 50828
    "ActiveArea": TagSpec(TIFFTYPE_SHORT_OR_LONG, 4, dng_ifd="dng_raw"),  # 50829
    "MaskedAreas": TagSpec(TIFFTYPE_SHORT_OR_LONG, None, dng_ifd="dng_raw"),  # 50830
    "AsShotICCProfile": TagSpec(TiffType.BYTE, None, dng_ifd="dng_ifd0"),  # 50831
    "AsShotPreProfileMatrix": TagSpec(TiffType.SRATIONAL, None, dng_ifd="dng_ifd0"),  # 50832
    "CurrentICCProfile": TagSpec(TiffType.BYTE, None, dng_ifd="dng_ifd0"),  # 50833
    "CurrentPreProfileMatrix": TagSpec(TiffType.SRATIONAL, None, dng_ifd="dng_ifd0"),  # 50834
    "ColorimetricReference": TagSpec(TiffType.SHORT, 1, dng_ifd="dng_ifd0"),  # 50879
    "CameraCalibrationSignature": TagSpec(TIFFTYPE_ASCII_OR_BYTE, None, dng_ifd="dng_ifd0"),  # 50931
    "ProfileCalibrationSignature": TagSpec(TIFFTYPE_ASCII_OR_BYTE, None, dng_ifd="dng_profile"),  # 50932
    "ProfileIFD": TagSpec(TiffType.LONG, None, dng_ifd="dng_ifd0"),  # 50933 - variable count (can have multiple profiles)
    "AsShotProfileName": TagSpec(TIFFTYPE_ASCII_OR_BYTE, None, dng_ifd="dng_ifd0"),  # 50934
    "NoiseReductionApplied": TagSpec(TiffType.RATIONAL, 1, dng_ifd="dng_raw"),  # 50935
    "ProfileName": TagSpec(TIFFTYPE_ASCII_OR_BYTE, None, dng_ifd="dng_profile"),  # 50936
    "ProfileHueSatMapDims": TagSpec(TiffType.LONG, None, dng_ifd="dng_profile"),  # 50937
    "ProfileHueSatMapData1": TagSpec(TiffType.FLOAT, None, dng_ifd="dng_profile"),  # 50938
    "ProfileHueSatMapData2": TagSpec(TiffType.FLOAT, None, dng_ifd="dng_profile"),  # 50939
    "ProfileToneCurve": TagSpec(TiffType.FLOAT, None, dng_ifd="dng_profile"),  # 50940
    "ProfileEmbedPolicy": TagSpec(TiffType.LONG, 1, dng_ifd="dng_profile"),  # 50941
    "ProfileCopyright": TagSpec(TIFFTYPE_ASCII_OR_BYTE, None, dng_ifd="dng_profile"),  # 50942
    "ForwardMatrix1": TagSpec(TiffType.SRATIONAL, 9, (3, 3), dng_ifd="dng_profile"),  # 50964 - 3x3 matrix
    "ForwardMatrix2": TagSpec(TiffType.SRATIONAL, 9, (3, 3), dng_ifd="dng_profile"),  # 50965 - 3x3 matrix
    "PreviewApplicationName": TagSpec(TiffType.ASCII, None, dng_ifd="dng_preview"),  # 50966
    "PreviewApplicationVersion": TagSpec(TiffType.ASCII, None, dng_ifd="dng_preview"),  # 50967
    "PreviewSettingsName": TagSpec(TiffType.ASCII, None, dng_ifd="dng_preview"),  # 50968
    "PreviewSettingsDigest": TagSpec(TiffType.BYTE, 16, dng_ifd="dng_preview"),  # 50969
    "PreviewColorSpace": TagSpec(TiffType.LONG, 1, dng_ifd="dng_preview"),  # 50970
    "PreviewDateTime": TagSpec(TiffType.ASCII, None, dng_ifd="dng_preview"),  # 50971
    "RawImageDigest": TagSpec(TiffType.BYTE, 16, dng_ifd="dng_ifd0"),  # 50972
    "OriginalRawFileDigest": TagSpec(TiffType.BYTE, 16, dng_ifd="dng_ifd0"),  # 50973
    "SubTileBlockSize": TagSpec(TiffType.SHORT, 2, dng_ifd="dng_raw"),  # 50974
    "RowInterleaveFactor": TagSpec(TiffType.SHORT, 1, dng_ifd="dng_raw"),  # 50975
    "ProfileLookTableDims": TagSpec(TiffType.LONG, None, dng_ifd="dng_profile"),  # 50981
    "ProfileLookTableData": TagSpec(TiffType.FLOAT, None, dng_ifd="dng_profile"),  # 50982
    "OpcodeList1": TagSpec(TiffType.BYTE, None, dng_ifd="dng_raw"),  # 51008
    "OpcodeList2": TagSpec(TiffType.BYTE, None, dng_ifd="dng_raw"),  # 51009
    "OpcodeList3": TagSpec(TiffType.BYTE, None, dng_ifd="dng_raw"),  # 51022
    # NoiseProfile: DNG spec says Raw IFD, but Adobe SDK writes to both main and raw IFDs
    # for legacy compatibility (dng_image_writer.cpp:8544). SDK reads from both locations.
    "NoiseProfile": TagSpec(TiffType.DOUBLE, None, dng_ifd="any"),  # 51041
    "TimeCodes": TagSpec(TiffType.BYTE, None, dng_ifd="any"),  # 51043
    "FrameRate": TagSpec(TiffType.RATIONAL, 1, dng_ifd="any"),  # 51044
    "TStop": TagSpec(TiffType.RATIONAL, None, dng_ifd="any"),  # 51058
    "ReelName": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 51081
    "OriginalDefaultFinalSize": TagSpec(TIFFTYPE_SHORT_OR_LONG, 2, dng_ifd="any"),  # 51089
    "OriginalBestQualitySize": TagSpec(TIFFTYPE_SHORT_OR_LONG, 2, dng_ifd="any"),  # 51090
    "OriginalDefaultCropSize": TagSpec(TIFFTYPE_INT_OR_RATIONAL, 2, dng_ifd="any"),  # 51091
    "CameraLabel": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 51105
    "ProfileHueSatMapEncoding": TagSpec(TiffType.LONG, 1, dng_ifd="dng_profile"),  # 51107
    "ProfileLookTableEncoding": TagSpec(TiffType.LONG, 1, dng_ifd="dng_profile"),  # 51108
    "BaselineExposureOffset": TagSpec(TiffType.SRATIONAL, 1, dng_ifd="dng_ifd0"),  # 51109
    "DefaultBlackRender": TagSpec(TiffType.LONG, 1, dng_ifd="dng_ifd0"),  # 51110
    "NewRawImageDigest": TagSpec(TiffType.BYTE, 16, dng_ifd="dng_ifd0"),  # 51111
    "RawToPreviewGain": TagSpec(TiffType.DOUBLE, 1, dng_ifd="dng_preview"),  # 51112
    "CacheBlob": TagSpec(TiffType.BYTE, None, dng_ifd="dng_preview"),  # 51113
    "CacheVersion": TagSpec(TiffType.LONG, 1, dng_ifd="dng_preview"),  # 51114
    "DefaultUserCrop": TagSpec(TiffType.RATIONAL, 4, dng_ifd="dng_raw"),  # 51125
    "DepthFormat": TagSpec(TiffType.SHORT, 1, dng_ifd="dng_ifd0"),  # 51177
    "DepthNear": TagSpec(TiffType.RATIONAL, 1, dng_ifd="dng_ifd0"),  # 51178
    "DepthFar": TagSpec(TiffType.RATIONAL, 1, dng_ifd="dng_ifd0"),  # 51179
    "DepthUnits": TagSpec(TiffType.SHORT, 1, dng_ifd="dng_ifd0"),  # 51180
    "DepthMeasureType": TagSpec(TiffType.SHORT, 1, dng_ifd="dng_ifd0"),  # 51181
    "EnhanceParams": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 51182
    "ProfileGainTableMap": TagSpec(TiffType.BYTE, None, dng_ifd="dng_raw"),  # 52525
    "SemanticName": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 52526
    "SemanticInstanceID": TagSpec(TiffType.ASCII, None, dng_ifd="any"),  # 52528
    "CalibrationIlluminant3": TagSpec(TiffType.SHORT, 1, dng_ifd="dng_profile"),  # 52529
    "CameraCalibration3": TagSpec(TiffType.SRATIONAL, 9, (3, 3), dng_ifd="dng_ifd0"),  # 52530 - 3x3 matrix
    "ColorMatrix3": TagSpec(TiffType.SRATIONAL, 9, (3, 3), dng_ifd="dng_profile"),  # 52531 - 3x3 matrix
    "ForwardMatrix3": TagSpec(TiffType.SRATIONAL, 9, (3, 3), dng_ifd="dng_profile"),  # 52532 - 3x3 matrix
    "IlluminantData1": TagSpec(TiffType.BYTE, None, dng_ifd="dng_profile"),  # 52533
    "IlluminantData2": TagSpec(TiffType.BYTE, None, dng_ifd="dng_profile"),  # 52534
    "IlluminantData3": TagSpec(TiffType.BYTE, None, dng_ifd="dng_profile"),  # 52535
    "MaskSubArea": TagSpec(TiffType.LONG, 4, dng_ifd="any"),  # 52536
    "ProfileHueSatMapData3": TagSpec(TiffType.FLOAT, None, dng_ifd="dng_profile"),  # 52537
    "ReductionMatrix3": TagSpec(TiffType.SRATIONAL, None, dng_ifd="dng_profile"),  # 52538
    "RGBTables": TagSpec(TiffType.BYTE, None, dng_ifd="dng_profile"),  # 52543
    "ProfileGainTableMap2": TagSpec(TiffType.BYTE, None, dng_ifd="dng_profile"),  # 52544
    "ColumnInterleaveFactor": TagSpec(TiffType.SHORT, 1, dng_ifd="dng_raw"),  # 52547
    "ImageSequenceInfo": TagSpec(TiffType.BYTE, None, dng_ifd="dng_ifd0"),  # 52548
    "ProfileToneMethod": TagSpec(TiffType.LONG, 1, dng_ifd="dng_profile"),  # 52549 (not in tifffile)
    "ImageStats": TagSpec(TiffType.BYTE, None, dng_ifd="dng_raw"),  # 52550
    "ProfileDynamicRange": TagSpec(TiffType.DOUBLE, 1, dng_ifd="dng_profile"),  # 52551
    "ProfileGroupName": TagSpec(TIFFTYPE_ASCII_OR_BYTE, None, dng_ifd="dng_profile"),  # 52552
    "JXLDistance": TagSpec(TiffType.FLOAT, 1, dng_ifd="any"),  # 52553
    "JXLEffort": TagSpec(TiffType.LONG, 1, dng_ifd="any"),  # 52554
    "JXLDecodeSpeed": TagSpec(TiffType.LONG, 1, dng_ifd="any"),  # 52555
    "Padding": TagSpec(TiffType.BYTE, None, dng_ifd="any"),  # 59932
    "OffsetSchema": TagSpec(TiffType.SLONG, 1, dng_ifd="any"),  # 59933
}

# =============================================================================
# Local TIFF Tags Registry
# =============================================================================
# Extends tifffile's TIFF.TAGS with DNG-specific tags not in tifffile.
# Use LOCAL_TIFF_TAGS instead of TIFF.TAGS throughout the codebase.

class LocalTiffTags:
    """Bidirectional TIFF tag lookup extending tifffile's registry."""
    
    # Tags not in tifffile or that need code override
    # tifffile has duplicate names for TIFF-EP vs EXIF tags; we keep both with
    # the lower code as primary and the EXIF (41xxx) variant with "-Exif" suffix
    _EXTRA_TAGS = {
        # TIFF-EP tags (primary, used by DNG)
        33422: "CFAPattern",
        37387: "FlashEnergy",
        37388: "SpatialFrequencyResponse",
        37389: "Noise",
        37390: "FocalPlaneXResolution",
        37391: "FocalPlaneYResolution",
        37392: "FocalPlaneResolutionUnit",
        37393: "ImageNumber",
        37394: "SecurityClassification",
        37395: "ImageHistory",
        37396: "SubjectLocation",
        37399: "SensingMethod",
        # EXIF equivalents (suffixed)
        41483: "FlashEnergy-Exif",
        41484: "SpatialFrequencyResponse-Exif",
        41485: "Noise-Exif",
        41486: "FocalPlaneXResolution-Exif",
        41487: "FocalPlaneYResolution-Exif",
        41488: "FocalPlaneResolutionUnit-Exif",
        41489: "ImageNumber-Exif",
        41490: "SecurityClassification-Exif",
        41491: "ImageHistory-Exif",
        41492: "SubjectLocation-Exif",
        41495: "SensingMethod-Exif",
        41730: "CFAPattern-Exif",
        # Other duplicates
        256: "ImageWidth",
        48256: "ImageWidth-Extended",
        347: "JPEGTables",
        437: "JPEGTables-Alt",
        # DNG-specific not in tifffile
        52549: "ProfileToneMethod",
        # Corrected tags (tifffile has wrong code)
        52535: "IlluminantData3",  # tifffile incorrectly has 53535
    }
    
    def __init__(self):
        self._by_name: Dict[str, int] = {}
        self._by_code: Dict[int, str] = {}
        
        # Add our extra/corrected tags first (these take precedence)
        for code, name in self._EXTRA_TAGS.items():
            self._by_name[name] = code
            self._by_code[code] = name
        
        # Add from tifffile's registry, but don't overwrite our corrections
        for code, name in TIFF.TAGS.items():
            if code not in self._by_code:
                self._by_code[code] = name
            if name not in self._by_name:
                self._by_name[name] = code
    
    def __contains__(self, key) -> bool:
        if isinstance(key, str):
            return key in self._by_name
        return key in self._by_code
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self._by_name[key]
        return self._by_code[key]
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

LOCAL_TIFF_TAGS = LocalTiffTags()


def resolve_tag(tag: Union[str, int]) -> Tuple[Optional[int], Optional[str], Optional[TagSpec]]:
    """Resolve a tag name or code to (tag_id, tag_name, spec).
    
    Args:
        tag: Either a numeric tag code (int) or tag name string
        
    Returns:
        Tuple of (tag_id, tag_name, spec) where any may be None if not found.
        For unknown numeric tags, tag_name will be str(tag_id) as fallback.
    """
    if isinstance(tag, int):
        tag_id = tag
        tag_name = LOCAL_TIFF_TAGS.get(tag_id)
        # Fallback to string representation for unknown numeric tags
        if tag_name is None:
            tag_name = str(tag_id)
    else:
        tag_name = tag
        tag_id = LOCAL_TIFF_TAGS.get(tag_name)
    
    spec = TIFF_TAG_TYPE_REGISTRY.get(tag_name) if tag_name else None
    return tag_id, tag_name, spec


def _decode_tag_value(
    tag_name: str,
    tag_value: Any,
    tag_dtype: int,
    spec: Optional[TagSpec] = None, 
    return_type: Optional[type] = None,
) -> Any:
    """Decode a raw TIFF value to a Python type (TIFF → Python).
    
    Internal function - use get_tag() or convert_tag_value() instead.
    
    Pure conversion function - takes raw value and dtype, returns converted value.
    
    Args:
        tag_name: Tag name (e.g., 'CFAPattern', 'ColorMatrix1')
        tag_value: Raw value from tifffile (tuple for rationals, bytes, int, etc.)
        tag_dtype: TIFF dtype code (5=RATIONAL, 10=SRATIONAL, etc.)
        spec: Optional TagSpec for validation and reshaping (e.g., shape=(3,3) for matrices)
        return_type: Target type for conversion (float, np.ndarray, str, None=raw)
    
    Returns:
        Converted value, or raw value if no conversion requested/possible.
    """
    # Validate tag dtype against spec if available
    if spec is not None:
        file_dtype_str = TIFFTYPE_TO_DTYPE_STR.get(tag_dtype)
        expected_dtypes = spec.dtype if isinstance(spec.dtype, list) else [spec.dtype]
        if file_dtype_str and file_dtype_str not in expected_dtypes:
            name_str = f"Tag '{tag_name}'" if tag_name else "Tag"
            logger.warning(
                f"{name_str} has dtype {file_dtype_str} (code {tag_dtype}) "
                f"but spec expects {expected_dtypes}"
            )
    
    # Helper to decode and strip null terminators from string-like values
    def _decode_string(val):
        if isinstance(val, bytes):
            return val.decode('utf-8', errors='replace').rstrip('\x00')
        elif isinstance(val, np.ndarray):
            return val.tobytes().decode('utf-8', errors='replace').rstrip('\x00')
        elif isinstance(val, str):
            return val.rstrip('\x00')
        return str(val)
    
    # If no return_type specified, return raw value
    # ASCII strings: strip null terminators to provide clean Python strings
    # (encode_tag_value will add them back when writing to TIFF)
    if return_type is None:
        # ASCII strings - remove TIFF null terminator
        if tag_dtype == TiffType.ASCII:
            return _decode_string(tag_value)
        return tag_value
    
    # Handle string conversion
    if return_type is str:
        return _decode_string(tag_value)
    
    # Handle datetime conversion (EXIF datetime strings → datetime)
    if return_type is datetime:
        if isinstance(tag_value, bytes):
            tag_value = tag_value.decode('utf-8', errors='replace')
        dt_str = str(tag_value).strip('\x00').strip()
        if not dt_str:
            return None
        # Parse common datetime formats
        formats = [
            '%Y:%m:%d %H:%M:%S',      # EXIF standard
            '%Y-%m-%d %H:%M:%S',      # ISO with space
            '%Y-%m-%dT%H:%M:%S',      # ISO with T
        ]
        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        logger.warning(f"Unable to parse datetime string: {dt_str}")
        return None
    
    # Handle float conversion (rationals → float)
    if return_type is float:
        if tag_dtype in (5, 10) and isinstance(tag_value, tuple) and len(tag_value) >= 2:
            # Single rational: num/denom
            if len(tag_value) == 2:
                return tag_value[0] / tag_value[1] if tag_value[1] != 0 else 0.0
            # Multiple rationals: return first as float
            return tag_value[0] / tag_value[1] if tag_value[1] != 0 else 0.0
        try:
            return float(tag_value)
        except (TypeError, ValueError):
            return tag_value
    
    # Handle np.ndarray conversion (matrices, arrays)
    if return_type is np.ndarray:
        # Convert RATIONAL/SRATIONAL to float array
        if tag_dtype in (5, 10) and isinstance(tag_value, tuple) and len(tag_value) % 2 == 0:
            floats = [
                tag_value[i] / tag_value[i+1] if tag_value[i+1] != 0 else 0.0
                for i in range(0, len(tag_value), 2)
            ]
            arr = np.array(floats)
            # Reshape if spec defines a shape
            if spec is not None and spec.shape is not None:
                return arr.reshape(spec.shape)
            return arr
        # Already array-like
        if isinstance(tag_value, np.ndarray):
            if spec is not None and spec.shape is not None:
                return tag_value.reshape(spec.shape)
            return tag_value
        arr = np.array(tag_value)
        if spec is not None and spec.shape is not None:
            return arr.reshape(spec.shape)
        return arr
    
    # Generic type conversion attempt
    try:
        return return_type(tag_value)
    except (TypeError, ValueError):
        return tag_value

def convert_tag_value(
    tag_name: str,
    tag_value: Any,
    tag_dtype: int,
    tag_count: int,
    registry_spec: Optional[TagSpec] = None,
    return_type: Optional[type] = None,
) -> Any:
    """Convert a tag value with special formatting and type conversion.
    
    This is the high-level conversion function that:
    1. Checks for special formatting (XMP, DNGVersion, etc.)
    2. Determines effective return type (auto if None)
    3. Creates shape spec if registry spec matches count
    4. Calls decode_tag_value for standard conversion
    
    Args:
        tag_name: Tag name (e.g., 'CFAPattern', 'ColorMatrix1')
        tag_value: Raw or normalized value
        tag_dtype: TIFF dtype code (5=RATIONAL, 10=SRATIONAL, etc.)
        tag_count: Number of values in the tag
        registry_spec: Optional registry TagSpec for shape info
        return_type: Target type for conversion (float, np.ndarray, str, None=auto)
    
    Returns:
        Converted value with special formatting applied if applicable.
    """
    # Special formatting applies when return_type is None (auto) or matches the special type
    # XMP: return XmpMetadata object
    if tag_name == "XMP" and (return_type is None or return_type is XmpMetadata):
        xmp_string = _decode_tag_value(tag_name, tag_value, tag_dtype, None, str)
        if xmp_string is None:
            xmp_string = ""
        return XmpMetadata(xmp_string)
    
    # DNG Version tags: return 4-tuple (major, minor, patch, build)
    if tag_name in ("DNGVersion", "DNGBackwardVersion") and (return_type is None or return_type is tuple):
        version_bytes = _decode_tag_value(tag_name, tag_value, tag_dtype, None, bytes)
        if version_bytes is None:
            return None
        # Pad to 4 bytes if needed (null bytes may be stripped)
        padded = version_bytes + b'\x00' * (4 - len(version_bytes))
        return (padded[0], padded[1], padded[2], padded[3])
    
    # CFAPattern: return friendly pattern string (e.g., "RGGB")
    if tag_name == "CFAPattern" and (return_type is None or return_type is str):
        pattern_bytes = _decode_tag_value(tag_name, tag_value, tag_dtype, None, bytes)
        if pattern_bytes is None:
            return None
        return CFA_CODES_TO_PATTERN.get(tuple(pattern_bytes), str(pattern_bytes))
    
    # PhotometricInterpretation: convert enum to readable name
    if tag_name == "PhotometricInterpretation" and (return_type is None or return_type is str):
        photometric_names = {
            PHOTOMETRIC.CFA: "CFA",
            PHOTOMETRIC.LINEAR_RAW: "LINEAR_RAW",
        }
        return photometric_names.get(tag_value, str(tag_value))
    
    # Validate count against registry spec
    if registry_spec and registry_spec.count is not None and tag_count != registry_spec.count:
        name_str = f"Tag '{tag_name}'" if tag_name else "Tag"
        logger.warning(
            f"{name_str} has count {tag_count} but spec expects {registry_spec.count}"
        )
    
    # Determine effective return type (auto-convert if None)
    effective_type = return_type or get_native_type(tag_dtype, tag_count)
    
    # Use registry spec shape only if count matches tag
    shape_spec = None
    if registry_spec and registry_spec.shape and registry_spec.count == tag_count:
        shape_spec = TagSpec(
            TIFFTYPE_TO_DTYPE_STR.get(tag_dtype, 'B'), tag_count, registry_spec.shape
        )
    
    return _decode_tag_value(tag_name, tag_value, tag_dtype, shape_spec, effective_type)


def encode_tag_value(tag_name: str, value: Any, spec: TagSpec) -> tuple:
    """Encode a Python value to TIFF format based on tag spec (Python → TIFF).
    
    Args:
        tag_name: Tag name (e.g., 'CFAPattern', 'ColorMatrix1')
        value: Python value (float, int, str, datetime, np.ndarray, bytes, etc.)
        spec: TagSpec defining the expected TIFF type (may support multiple types)
        
    Returns:
        Tuple of (dtype, count, converted_value) ready for add_tag
    """
    # Special case: CFAPattern accepts pattern key strings
    if tag_name == "CFAPattern" and isinstance(value, str):
        if value in CFA_PATTERN_TO_CODES:
            value = bytes(CFA_PATTERN_TO_CODES[value])
        else:
            logger.warning(f"Unknown CFAPattern '{value}', using default 'RGGB'")
            value = bytes(CFA_PATTERN_TO_CODES["RGGB"])
    
    # UTF-8 string tags: tags with [ASCII, BYTE] dtype support UTF-8 encoding
    # Includes: LocalizedCameraModel, ProfileName, ProfileCopyright, etc.
    if isinstance(value, str) and isinstance(spec.dtype, list) and TiffType.BYTE in spec.dtype:
        value = value.encode("utf-8") + b"\x00"
    
    # Select appropriate dtype based on value type (handles multi-type TagSpecs)
    dtype = spec.get_dtype_for_value(value)
    
    # === String handling (ASCII) ===
    if dtype == TiffType.ASCII:
        if hasattr(value, 'strftime'):
            # Convert datetime-like objects to TIFF format
            value = value.strftime("%Y:%m:%d %H:%M:%S")
        if isinstance(value, bytes):
            value = value.decode('utf-8', errors='replace')
        value = str(value)
        if not value.endswith('\x00'):
            value = value + '\x00'
        return (dtype, len(value), value)
    
    # === Byte array handling (BYTE, UNDEFINED) ===
    if dtype in (TiffType.BYTE, TiffType.UNDEFINED):
        if isinstance(value, bytes):
            return (dtype, len(value), value)
        if isinstance(value, (list, tuple)):
            value = bytes(value)
            return (dtype, len(value), value)
        if hasattr(value, '__array__'):
            arr = np.asarray(value)
            value = arr.tobytes()
            return (dtype, len(value), value)
        # Single byte
        return (dtype, 1, bytes([int(value)]))
    
    # === Rational handling (RATIONAL, SRATIONAL) ===
    if dtype in (TiffType.RATIONAL, TiffType.SRATIONAL):
        max_denom = 10000
        
        # Check if value is already in rational format by comparing with spec.count
        # For EXIF tags: spec.count=1 with 2-element tuple means pre-formatted (num, denom)
        # For arrays: spec.count=N with 2N-element tuple means N pre-formatted rationals
        if isinstance(value, tuple) and len(value) % 2 == 0:
            expected_count = spec.count if spec.count is not None else len(value) // 2
            # If tuple length matches expected rational count, treat as pre-formatted
            if len(value) == expected_count * 2:
                return (dtype, expected_count, value)
        
        # Handle array-like objects (numpy, pandas, xarray, etc.)
        if hasattr(value, '__array__'):
            flat = np.asarray(value).flatten()
            rationals = []
            for v in flat:
                frac = Fraction(float(v)).limit_denominator(max_denom)
                rationals.extend([frac.numerator, frac.denominator])
            return (dtype, len(flat), tuple(rationals))
        
        # Handle lists/tuples of floats - convert each to rational
        if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], (int, float)):
            rationals = []
            for v in value:
                frac = Fraction(float(v)).limit_denominator(max_denom)
                rationals.extend([frac.numerator, frac.denominator])
            return (dtype, len(value), tuple(rationals))
        
        # Handle single float/int
        if isinstance(value, (int, float)):
            frac = Fraction(float(value)).limit_denominator(max_denom)
            return (dtype, 1, (frac.numerator, frac.denominator))
    
    # === Integer handling (SHORT, LONG, SSHORT, SLONG, LONG8, SLONG8) ===
    if dtype in (TiffType.SHORT, TiffType.LONG, TiffType.SSHORT, TiffType.SLONG, TiffType.LONG8, TiffType.SLONG8):
        if isinstance(value, (list, tuple)) or hasattr(value, '__array__'):
            arr = np.asarray(value).flatten().tolist() if hasattr(value, '__array__') else list(value)
            return (dtype, len(arr), tuple(arr) if len(arr) > 1 else arr[0])
        return (dtype, 1, int(value))
    
    # === Float handling (FLOAT, DOUBLE) ===
    if dtype in (TiffType.FLOAT, TiffType.DOUBLE):
        if isinstance(value, (list, tuple)) or hasattr(value, '__array__'):
            arr = np.asarray(value).flatten().tolist() if hasattr(value, '__array__') else list(value)
            return (dtype, len(arr), tuple(arr) if len(arr) > 1 else arr[0])
        return (dtype, 1, float(value))
    
    # Fallback: return as-is with count from spec
    count = spec.count if spec.count is not None else 1
    return (dtype, count, value)

def normalize_array_to_target_byteorder(value, target_byteorder: str) -> Any:
    """Normalize array byte order to target byte order.
    
    For numpy arrays, source byte order is determined from the array's dtype.
    
    Args:
        value: Array value (ndarray, scalar, string, etc.)
        target_byteorder: Target byte order ('>' big-endian, '<' little-endian, '=' system)
        
    Returns:
        Value converted to target byte order
    """
    import sys
    system_byteorder = '<' if sys.byteorder == 'little' else '>'

    # Resolve '=' to actual system byte order
    if target_byteorder == '=':
        target_byteorder = system_byteorder
    
    # Multi-byte typed arrays - always convert to explicit byte order
    if isinstance(value, np.ndarray) and value.dtype.byteorder not in ('|', target_byteorder):
        # Convert to target byte order with explicit marker
        # This handles both actual byte swapping and normalizing '=' to explicit byte order
        base_dtype = value.dtype.str[1:]  # Remove byte order prefix
        target_dtype = f'{target_byteorder}{base_dtype}'
        return value.astype(target_dtype)
    
    return value

def get_cfa_pattern_codes(pattern: str) -> tuple:
    """Convert CFA pattern string to code tuple.
    
    Args:
        pattern: CFA pattern string ('RGGB', 'BGGR', 'GRBG', 'GBRG')
        
    Returns:
        Tuple of 4 codes (0=R, 1=G, 2=B), e.g., (0, 1, 1, 2) for RGGB.
        Returns (0, 1, 1, 2) as default if pattern not recognized.
    """
    return CFA_PATTERN_TO_CODES.get(pattern, (0, 1, 1, 2))


def filter_tags_by_ifd_category(tags: 'MetadataTags', include_categories: List[str]) -> 'MetadataTags':
    """Filter tags to only those belonging to specified IFD categories.
    
    Args:
        tags: Input tags to filter
        include_categories: List of IFD categories to keep (e.g., ["any", "dng_ifd0", "exif"])
        
    Returns:
        New MetadataTags with only tags matching the specified categories
    """
    from . import tiff_metadata as tm
    
    filtered = tm.MetadataTags()
    
    for code, dtype, count, value, _ in tags:
        _, tag_name, spec = resolve_tag(int(code))
        if tag_name is None or spec is None:
            # Unknown tag - preserve it (user may have custom/proprietary tags)
            filtered.add_raw_tag(int(code), int(dtype), int(count), value)
            continue
        
        # Check if tag's dng_ifd category is in the include list
        if spec.dng_ifd in include_categories:
            filtered.add_raw_tag(int(code), int(dtype), int(count), value)
    
    return filtered

def _get_time_impl(
    tag_source: Any,
    time_type: str = "original"
) -> Optional[datetime]:
    """Internal implementation for extracting datetime from EXIF time tags.
    
    Args:
        tag_source: Any object with a get_tag(name, return_type) method
        time_type: Which set of tags to read:
            - "original" (default): DateTimeOriginal, SubsecTimeOriginal, OffsetTimeOriginal
            - "digitized": DateTimeDigitized, SubsecTimeDigitized, OffsetTimeDigitized
            - "modified": DateTime, SubsecTime, OffsetTime
            - "preview": PreviewDateTime (no subsec/offset tags)
            
    Returns:
        datetime object with microseconds and timezone if available, or None if
        the datetime tag is not present or cannot be parsed.
    """
    # Define tag name mapping (same as add_time_tags)
    tag_names = {
        "original": ("DateTimeOriginal", "SubsecTimeOriginal", "OffsetTimeOriginal"),
        "digitized": ("DateTimeDigitized", "SubsecTimeDigitized", "OffsetTimeDigitized"),
        "modified": ("DateTime", "SubsecTime", "OffsetTime"),
        "preview": ("PreviewDateTime", None, None),
    }
    
    if time_type not in tag_names:
        raise ValueError(f"time_type must be one of {list(tag_names.keys())}, got '{time_type}'")
    
    datetime_tag, subsec_tag, offset_tag = tag_names[time_type]
    
    # Get the main datetime value (request datetime conversion)
    dt_obj = tag_source.get_tag(datetime_tag, datetime)
    if dt_obj is None:
        return None
    
    # EXIF DateTime format has no subseconds, so add from SubsecTime* tag
    if subsec_tag:
        subsec_str = tag_source.get_tag(subsec_tag)  # Returns str (dtype "s")
        if subsec_str:
            subsec_str = str(subsec_str).strip('\x00').strip()
            if subsec_str:
                try:
                    # SubsecTime is typically milliseconds (3 digits)
                    # Pad or truncate to 6 digits for microseconds
                    subsec_str = subsec_str.ljust(6, '0')[:6]
                    microseconds = int(subsec_str)
                    dt_obj = dt_obj.replace(microsecond=microseconds)
                except (ValueError, AttributeError):
                    pass
    
    # Get timezone offset if available
    if offset_tag:
        offset_str = tag_source.get_tag(offset_tag)  # Returns str (dtype "s")
        if offset_str:
            offset_str = str(offset_str).strip('\x00').strip()
            # Parse offset like "+08:00" or "-05:00"
            if len(offset_str) >= 5:
                try:
                    sign = 1 if offset_str[0] == '+' else -1
                    hours = int(offset_str[1:3])
                    minutes = int(offset_str[4:6]) if len(offset_str) >= 6 else 0
                    total_seconds = sign * (hours * 3600 + minutes * 60)
                    
                    from datetime import timezone, timedelta
                    tz = timezone(timedelta(seconds=total_seconds))
                    dt_obj = dt_obj.replace(tzinfo=tz)
                except (ValueError, IndexError):
                    pass  # Keep naive datetime if offset parsing fails
    
    return dt_obj


def _convert_exif_dict_to_tags(tags: MetadataTags) -> None:
    """Convert ExifTag dictionary to individual TIFF tags.
    
    Since TiffWriter cannot write ExifIFD structures, this function converts
    EXIF tags from the dictionary format (as read by tifffile) to individual
    TIFF tags that can be written as regular tags. The ExifTag dict itself
    remains in the tags instance.
    
    Args:
        tags: MetadataTags instance containing ExifTag to convert.
              Individual EXIF tags are added. Existing tags are not overwritten.
        
    Example:
        tags = MetadataTags()
        tags.add_raw_tag('ExifTag', ...)  # Contains {"ExposureTime": [1, 400], ...}
        convert_exif_dict_to_tags(tags)
        # tags now contains ExposureTime and FNumber as regular TIFF tags
        # plus the original ExifTag dict
    """
    exif_dict = tags.get_tag('ExifTag', dict)
    if not exif_dict:
        return
    
    for tag_name, value in exif_dict.items():
        # Skip ExifVersion - it's already converted to string by tifffile and not useful as TIFF tag
        if tag_name == 'ExifVersion':
            continue
        
        # Check if tag has a type spec and is not already present
        if tag_name in TIFF_TAG_TYPE_REGISTRY and tag_name not in tags:
            try:
                tags.add_tag(tag_name, value)
            except (ValueError, TypeError) as e:
                # Value format doesn't match tag spec
                logger.debug(f"Skipping EXIF tag '{tag_name}': {e}")

# helper class to convert create a list of tags for tifffile.TiffWriter
class MetadataTags:
    
    @dataclass
    class StoredTag:
        code: int
        dtype: int  # TIFF dtype code
        count: int
        value: Any
    
    def __init__(self):
        self._tags: Dict[int, MetadataTags.StoredTag] = {}

    def __iter__(self):
        """Iterate over tags, sorted by tag code.
        
        Yields tuples in format expected by TiffWriter: (code, dtype, count, value, writeonce)
        """
        for code in sorted(self._tags.keys()):
            tag = self._tags[code]
            yield (tag.code, tag.dtype, tag.count, tag.value, False)
    
    def __len__(self):
        """Return the number of tags."""
        return len(self._tags)
    
    def __contains__(self, tag: Union[int, str]) -> bool:
        """Check if a tag exists by code (int) or name (str)."""
        tag_id, _, _ = resolve_tag(tag)
        if tag_id is None:
            return False
        return tag_id in self._tags

    def remove_tag(self, tag: Union[int, str]) -> bool:
        """Remove a tag by code (int) or name (str).
        
        Args:
            tag: Tag code or name to remove
            
        Returns:
            True if tag was removed, False if it didn't exist
        """
        tag_id, _, _ = resolve_tag(tag)
        if tag_id is None:
            return False
        if tag_id in self._tags:
            del self._tags[tag_id]
            return True
        return False

    def copy(self, convert_exif: bool = True) -> MetadataTags:
        """Create a deep copy of this MetadataTags instance.
        
        Args:
            convert_exif: If True (default), convert ExifTag dictionary to individual TIFF tags. 
        
        Returns:
            New MetadataTags instance with copied tags.
        """
        import copy
        new_instance = MetadataTags()
        # Deep copy the tags dict to avoid shared mutable objects
        new_instance._tags = copy.deepcopy(self._tags)
        
        # Convert EXIF dict to individual tags if requested
        if convert_exif:
            _convert_exif_dict_to_tags(new_instance)
    
        return new_instance

    def add_tag(self, tag_name: str, value: Any) -> None:
        """Add a tag with automatic type conversion based on registry.
        
        The tag type and format are looked up in TIFF_TAG_TYPE_REGISTRY and 
        the value is automatically converted to the appropriate TIFF format.
        
        Args:
            tag_name: Name of the TIFF/DNG tag (e.g., 'ExposureTime', 'ColorMatrix1')
            value: Python value - will be auto-converted based on tag type:
                   - float/int → rational for ExposureTime, matrices, etc.
                   - datetime → TIFF date string for DateTime tags
                   - str → null-terminated ASCII
                   - np.ndarray → flattened rational array for matrices
                   - bytes → raw byte array
                   
        Special cases:
            - CFAPattern: accepts pattern key string ('RGGB', 'BGGR', 'GRBG', 'GBRG')
              which is auto-converted to the appropriate byte pattern
        
        Raises:
            KeyError: If tag_name is not in TIFF_TAG_TYPE_REGISTRY
            
        Example:
            tags.add_tag("ExposureTime", 0.01)  # Auto-converts to rational
            tags.add_tag("ColorMatrix1", np.eye(3))  # Auto-converts 3x3 to rationals
            tags.add_tag("DateTimeOriginal", datetime.now())  # Auto-formats
            tags.add_tag("Make", "Canon")  # Auto null-terminates
            tags.add_tag("CFAPattern", "RGGB")  # Auto-converts to bytes
        """
        tag_id, _, spec = resolve_tag(tag_name)
        if tag_id is None or spec is None:
            raise KeyError(f"Tag '{tag_name}' not in TIFF_TAG_TYPE_REGISTRY. "
                          f"Use add_raw_tag() for tags not in registry.")
        
        dtype, count, converted_value = encode_tag_value(tag_name, value, spec)
        self.add_raw_tag(tag_id, dtype, count, converted_value)

    def add_raw_tag(
        self, 
        name_or_code: Union[str, int], 
        dtype: Union[TiffType, int], 
        count: int, 
        value: Any
    ) -> None:
        """Add a tag with explicit type specification.
        
        Values are automatically normalized to system byte order for internal storage.
        For numpy arrays, source byte order is determined from the array's dtype.
        
        Args:
            name_or_code: Tag name string or numeric code
            dtype: TIFF data type (TiffType enum or int code, e.g., TiffType.RATIONAL or 5)
            count: Number of values
            value: The tag value (numpy arrays will be normalized to system byte order)
        """
        tag_code, _, _ = resolve_tag(name_or_code)
        if tag_code is None:
            raise KeyError(f"Tag '{name_or_code}' not found in LOCAL_TIFF_TAGS.")

        # TiffType is int, Enum, so int() works for both TiffType and plain int
        tag_dtype = int(dtype)

        # Normalize value to system byte order for internal storage
        normalized_value = normalize_array_to_target_byteorder(value, '=')

        self._tags[tag_code] = self.StoredTag(code=tag_code, dtype=tag_dtype, count=count, value=normalized_value)

    def add_time_tags(
        self,
        time_value: Union[str, datetime, Any],
        time_type: str = "original",
        timezone: Optional[str] = None
    ) -> None:
        """Add datetime, subsecond, and timezone offset tags.
        
        Parses a datetime value and sets the appropriate EXIF time tags:
        - DateTime/DateTimeOriginal/DateTimeDigitized (format: YYYY:MM:DD HH:MM:SS)
        - SubsecTime/SubsecTimeOriginal/SubsecTimeDigitized (milliseconds)
        - OffsetTime/OffsetTimeOriginal/OffsetTimeDigitized (timezone offset)
        
        Args:
            time_value: Datetime value - can be:
                - datetime object (with or without microseconds)
                - string in various formats (ISO, EXIF, etc.)
                - any object with strftime method
            time_type: Which set of tags to use:
                - "original" (default): DateTimeOriginal, SubsecTimeOriginal, OffsetTimeOriginal
                - "digitized": DateTimeDigitized, SubsecTimeDigitized, OffsetTimeDigitized
                - "modified": DateTime, SubsecTime, OffsetTime
                - "preview": PreviewDateTime (no subsec/offset tags)
            timezone: Optional timezone string (e.g., "America/Los_Angeles", "UTC")
                     Used to set OffsetTime* tag. If time_value is timezone-aware,
                     that timezone is used instead.
                     
        Example:
            tags.add_time_tags(datetime.now(), "original", "America/Los_Angeles")
            tags.add_time_tags("2024-03-06T15:30:00.123", "digitized")
        """
        # Define tag name mapping
        tag_names = {
            "original": ("DateTimeOriginal", "SubsecTimeOriginal", "OffsetTimeOriginal"),
            "digitized": ("DateTimeDigitized", "SubsecTimeDigitized", "OffsetTimeDigitized"),
            "modified": ("DateTime", "SubsecTime", "OffsetTime"),
            "preview": ("PreviewDateTime", None, None),
        }
        
        if time_type not in tag_names:
            raise ValueError(f"time_type must be one of {list(tag_names.keys())}, got '{time_type}'")
        
        datetime_tag, subsec_tag, offset_tag = tag_names[time_type]
        
        # Parse string to datetime if needed
        dt_obj = None
        subsec_str = None
        
        if isinstance(time_value, str):
            # Try common datetime formats
            formats = [
                '%Y:%m:%d %H:%M:%S.%f',  # EXIF with microseconds
                '%Y:%m:%d %H:%M:%S',      # EXIF standard
                '%Y-%m-%d %H:%M:%S.%f',   # ISO with space and microseconds
                '%Y-%m-%d %H:%M:%S',      # ISO with space
                '%Y-%m-%dT%H:%M:%S.%f',   # ISO with T and microseconds
                '%Y-%m-%dT%H:%M:%S',      # ISO with T
                '%Y-%m-%dT%H:%M:%S.%f%z', # ISO with timezone
                '%Y-%m-%dT%H:%M:%S%z',    # ISO with timezone, no microseconds
            ]
            for fmt in formats:
                try:
                    dt_obj = datetime.strptime(time_value, fmt)
                    break
                except ValueError:
                    continue
            else:
                logger.warning(f"Unable to parse time string: {time_value}")
                return
        elif hasattr(time_value, 'strftime'):
            dt_obj = time_value
        else:
            logger.warning(f"time_value must be string or datetime-like, got {type(time_value)}")
            return
        
        # Extract subseconds (milliseconds, 3 digits)
        if hasattr(dt_obj, 'microsecond') and dt_obj.microsecond > 0:
            subsec_str = f"{dt_obj.microsecond // 1000:03d}"
        
        # Format datetime string (EXIF format, exactly 19 chars + null = 20)
        datetime_str = dt_obj.strftime('%Y:%m:%d %H:%M:%S')
        self.add_tag(datetime_tag, datetime_str)
        
        # Add subseconds if available and tag exists for this time_type
        if subsec_str and subsec_tag:
            self.add_tag(subsec_tag, subsec_str)
        
        # Handle timezone offset
        if offset_tag:
            offset_str = None
            
            # Check if dt_obj has timezone info
            if hasattr(dt_obj, 'tzinfo') and dt_obj.tzinfo is not None:
                try:
                    offset = dt_obj.utcoffset()
                    if offset is not None:
                        total_seconds = int(offset.total_seconds())
                        hours, remainder = divmod(abs(total_seconds), 3600)
                        minutes = remainder // 60
                        sign = '+' if total_seconds >= 0 else '-'
                        offset_str = f"{sign}{hours:02d}:{minutes:02d}"
                except Exception:
                    pass
            
            # Use provided timezone if no offset from dt_obj
            if offset_str is None and timezone:
                try:
                    from zoneinfo import ZoneInfo
                    tz = ZoneInfo(timezone)
                    # Get offset for the given datetime
                    if hasattr(dt_obj, 'replace'):
                        aware_dt = dt_obj.replace(tzinfo=tz)
                        offset = aware_dt.utcoffset()
                        if offset is not None:
                            total_seconds = int(offset.total_seconds())
                            hours, remainder = divmod(abs(total_seconds), 3600)
                            minutes = remainder // 60
                            sign = '+' if total_seconds >= 0 else '-'
                            offset_str = f"{sign}{hours:02d}:{minutes:02d}"
                except Exception as e:
                    logger.warning(f"Invalid timezone '{timezone}': {e}")
            
            if offset_str:
                self.add_tag(offset_tag, offset_str)

    def add_xmp(self, xmp: 'XmpMetadata') -> None:
        """Add XMP metadata to the DNG file.
        
        Args:
            xmp: XmpMetadata instance to serialize and write to XMP tag.
        
        Example:
           from .xmp import XmpMetadata
            xmp = XmpMetadata.from_attributes({
                'crs:Temperature': '3900',
                'crs:Tint': '0',
                'crs:Exposure2012': '-0.5',
                'tiff:Orientation': '1'
            })
            tags.add_xmp(xmp)
        """
        xmp_bytes = xmp_metadata_to_packet(xmp)
        self.add_tag("XMP", xmp_bytes)
        logger.debug(f"Added XMP metadata with {len(xmp._attributes)} properties")

    def extend(self, other: Optional[MetadataTags]) -> None:
        """Add all tags from another MetadataTags instance.
        
        Args:
            other: MetadataTags instance to add tags from
            
        Raises:
            TypeError: If other is not a MetadataTags instance
        """
        if other is None:
            return
        if not isinstance(other, MetadataTags):
            raise TypeError(f"Expected MetadataTags instance, got {type(other).__name__}")
        # Dict update - other's tags override existing
        for _, tag in other._tags.items():
            self.add_raw_tag(tag.code, tag.dtype, tag.count, tag.value)

    def __or__(self, other: Optional[MetadataTags]) -> MetadataTags:
        """Combine two MetadataTags instances using the | operator.
        """
        result = self.copy()
        result.extend(other)
        return result

    def __ior__(self, other: Optional[MetadataTags]) -> MetadataTags:
        """In-place union using the |= operator.
        """
        self.extend(other)
        return self

    def get_xmp(self) -> Optional['XmpMetadata']:
        """Return XMP metadata as an `XmpMetadata` object."""
        xmp = self.get_tag("XMP")
        return xmp

    def _get_tag_info(
        self, tag: Union[str, int]
    ) -> Optional[tuple]:
        """Internal helper to resolve tag and get stored tag object.
        
        Args:
            tag: Either a numeric tag code (int) or tag name string
        
        Returns:
            Tuple of (tag_id, tag_name, registry_spec, stored_tag) or None if not found.
        """
        tag_id, tag_name, spec = resolve_tag(tag)
        
        if tag_id is None:
            logger.warning(f"Tag '{tag}' not found in LOCAL_TIFF_TAGS.")
            return None
        
        if tag_id not in self._tags:
            return None
        
        return (tag_id, tag_name, spec, self._tags[tag_id])

    def get_tag(
        self, 
        tag: Union[str, int], 
        return_type: Optional[type] = None
    ) -> Optional[Any]:
        """Get a tag value with type conversion.
        
        Args:
            tag: Either a numeric tag code (int) or tag name string
            return_type: Controls type conversion:
                - None (default): auto-convert to native type from TagSpec
                - specific type (str, float, list, etc.): convert to that type
        
        Returns:
            Tag value (converted based on return_type), or None if not found.
            
        See also:
            get_raw_tag: Returns raw tag value without any conversion.
        """
        tag_info = self._get_tag_info(tag)
        if tag_info is None:
            return None
        
        _, tag_name, spec, t = tag_info
        return convert_tag_value(tag_name, t.value, t.dtype, t.count, spec, return_type)

    def get_raw_tag(self, tag: Union[str, int]) -> Optional[Any]:
        """Get raw tag value without any type conversion.
        
        Args:
            tag: Either a numeric tag code (int) or tag name string
        
        Returns:
            Raw tag value as stored, or None if not found.
            
        See also:
            get_tag: Returns tag value with automatic or specified type conversion.
        """
        tag_info = self._get_tag_info(tag)
        if tag_info is None:
            return None
        
        return tag_info[3].value
    
    def get_time_from_tags(self, time_type: str = "original") -> Optional[datetime]:
        """Extract datetime from EXIF time tags with subseconds and timezone.
        
        This is the inverse of add_time_tags(). Reads the datetime, subsecond,
        and timezone offset tags and combines them into a single datetime object.
        
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
            capture_time = tags.get_time("original")
            if capture_time:
                print(f"Captured at {capture_time}")
        """
        return _get_time_impl(self, time_type)


class XmpMetadata:
    """Encapsulates XMP metadata parsing and querying for DNG files.
    
    Read-only by convention: use factory methods to create new instances.
    """
    
    def __init__(self, xmp_string: str = ""):
        """Initialize XmpMetadata from an XMP string.
        
        Args:
            xmp_string: Raw XMP metadata string from DNG file (default: empty)
        """
        self._xmp_string = xmp_string
        self._attributes = self._parse(xmp_string)
    
    @classmethod
    def from_attributes(cls, attributes: Dict[str, Any]) -> 'XmpMetadata':
        """Create XmpMetadata from a dict of fully-qualified XMP attributes.
        
        Args:
            attributes: Dict mapping XMP property names to values.
                       Keys should be fully qualified (e.g., 'crs:Temperature', 'dc:subject').
                       Values can be:
                       - str for scalar properties
                       - list[str] for rdf:Bag properties (e.g., dc:subject)
                       - list[tuple[float, float]] for tone curves
        
        Returns:
            New XmpMetadata instance
        """
        instance = cls.__new__(cls)
        instance._attributes = dict(attributes)
        return instance
    
    def merged(self, other: Union['XmpMetadata', Dict[str, Any]]) -> 'XmpMetadata':
        """Create a new XmpMetadata with attributes from this instance and other.
        
        Args:
            other: XmpMetadata instance or dict of attributes to merge.
                  Attributes in 'other' override attributes in 'self'.
        
        Returns:
            New XmpMetadata instance with merged attributes
        """
        merged_attrs = dict(self._attributes)
        if isinstance(other, XmpMetadata):
            merged_attrs.update(other._attributes)
        else:
            merged_attrs.update(other)
        return XmpMetadata.from_attributes(merged_attrs)
    
    def _parse(self, xmp_data: str) -> Dict[str, Any]:
        """Parse all XMP attributes and sequences from the XMP metadata into a dictionary.
        
        Returns:
            Dictionary mapping attribute names to values.

            - Scalar attributes map to strings (e.g., 'crs:Temperature': '3900').
            - rdf:Bag maps to list[str] (e.g., 'dc:subject': ['key=val', ...]).
            - rdf:Seq maps to structured lists.
              - For ToneCurve* sequences: list[tuple[float, float]] normalized to 0..1.
              - For other sequences: list[str] of raw rdf:li text.
        """
        if not xmp_data:
            return {}

        import re
        from defusedxml import ElementTree as DET

        attributes: Dict[str, Any] = {}

        rdf_uri = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

        uri_to_prefix = {
            rdf_uri: "rdf",
            "http://ns.adobe.com/camera-raw-settings/1.0/": "crs",
            "http://ns.adobe.com/camera-raw-embedded-lens-profile/1.0/": "crlcp",
            "http://ns.adobe.com/photoshop/1.0/": "photoshop",
            "http://ns.adobe.com/photoshop/1.0/camera-profile": "stCamera",
            "http://purl.org/dc/elements/1.1/": "dc",
            "http://ns.adobe.com/tiff/1.0/": "tiff",
            "http://ns.adobe.com/xap/1.0/": "xmp",
            "adobe:ns:meta/": "x",
        }

        def _qname_to_prefixed_name(qname: str) -> Optional[str]:
            if not qname:
                return None
            if qname[0] != "{":
                return qname
            try:
                uri, local = qname[1:].split("}", 1)
            except ValueError:
                return None

            prefix = uri_to_prefix.get(uri)
            if prefix is None:
                return None
            return f"{prefix}:{local}"

        def _parse_seq_li_values(prop_name: str, li_values: list[str]) -> Any:
            if "ToneCurve" in prop_name:
                points: list[tuple[float, float]] = []
                for li_value in li_values:
                    coords = [coord.strip() for coord in li_value.split(",")]
                    if len(coords) != 2:
                        continue
                    try:
                        x_norm = float(coords[0]) / 255.0
                        y_norm = float(coords[1]) / 255.0
                    except ValueError:
                        continue
                    points.append((x_norm, y_norm))
                return points

            # Default: keep raw rdf:li text as a list
            return li_values

        try:
            xmp_text = xmp_data
            if isinstance(xmp_text, bytes):
                xmp_text = xmp_text.decode("utf-8", errors="replace")
            xmp_text = xmp_text.lstrip("\ufeff")
            xmp_text = re.sub(r"<\?xpacket[^>]*\?>", "", xmp_text)
            root = DET.fromstring(xmp_text)
        except Exception as e:
            logger.debug(f"Failed to parse XMP XML: {e}")
            return {}

        # Only parse top-level Description elements, not nested ones (e.g., inside crs:Look)
        # Use ./ instead of .// to avoid recursive search
        rdf_root = root.find(f".//{{{rdf_uri}}}RDF")
        if rdf_root is None:
            logger.debug("No rdf:RDF element found in XMP")
            return {}
        
        descriptions = rdf_root.findall(f"./{{{rdf_uri}}}Description")
        for desc in descriptions:
            for attr_qname, attr_value in desc.attrib.items():
                if attr_qname.startswith("{http://www.w3.org/2000/xmlns/}"):
                    continue

                attr_name = _qname_to_prefixed_name(attr_qname)
                if attr_name is None:
                    continue
                attributes[attr_name] = str(attr_value)

            for prop_elem in list(desc):
                prop_name = _qname_to_prefixed_name(prop_elem.tag)
                if prop_name is None:
                    continue

                seq = prop_elem.find(f".//{{{rdf_uri}}}Seq")
                bag = prop_elem.find(f".//{{{rdf_uri}}}Bag")

                container = seq if seq is not None else bag
                if container is None:
                    continue

                li_elems = container.findall(f"{{{rdf_uri}}}li")
                li_values = [li.text.strip() for li in li_elems if li.text and li.text.strip()]
                if not li_values:
                    continue

                if seq is not None:
                    attributes[prop_name] = _parse_seq_li_values(prop_name, li_values)
                else:
                    attributes[prop_name] = li_values

        logger.debug(f"Parsed {len(attributes)} XMP attributes")
        return attributes
    
    def xpath_query(self, element_name: str, namespace_uri: str) -> Optional[Dict[str, str]]:
        """Query XMP for elements by namespace URI and local name.
        
        Searches the entire XMP tree for elements matching the given namespace and name,
        and returns the first match's attributes as a dict.
        
        Args:
            element_name: Local element name (e.g., 'PerspectiveModel')
            namespace_uri: Full namespace URI (e.g., 'http://ns.adobe.com/camera-raw-embedded-lens-profile/1.0/')
        
        Returns:
            Dict of attribute name -> value for the first matching element, or None if not found.
            Attribute names are returned with their namespace prefix (e.g., 'stCamera:Version').
        
        Example:
            pm = xmp.xpath_query('PerspectiveModel', 'http://ns.adobe.com/camera-raw-embedded-lens-profile/1.0/')
            if pm:
                k1 = pm.get('stCamera:RadialDistortParam1')
        """
        result = self.xpath_query_with_parent(element_name, namespace_uri, include_parent=False)
        return result[0] if result else None
    
    def xpath_query_with_parent(self, element_name: str, namespace_uri: str, include_parent: bool = True) -> Optional[tuple]:
        """Query XMP for elements and optionally include direct parent element attributes.
        
        General-purpose method that finds an element and optionally returns its parent's attributes.
        Uses ElementTree's parent map to find the immediate parent element.
        
        Args:
            element_name: Local element name (e.g., 'PerspectiveModel')
            namespace_uri: Full namespace URI
            include_parent: If True, includes parent element's attributes in result
        
        Returns:
            Tuple of (element_dict, parent_dict) or None if not found.
            parent_dict will be None if include_parent is False or parent has no attributes.
        
        Example:
            elem, parent = xmp.xpath_query_with_parent('PerspectiveModel', 'http://...', include_parent=True)
            if parent:
                focal_length = parent.get('stCamera:FocalLength')
        """
        if not hasattr(self, '_xmp_string') or not self._xmp_string:
            return None
        
        try:
            import re
            from defusedxml import ElementTree as DET
            
            # Parse XMP
            xmp_text = self._xmp_string
            if isinstance(xmp_text, bytes):
                xmp_text = xmp_text.decode("utf-8", errors="replace")
            xmp_text = xmp_text.lstrip("\ufeff")
            xmp_text = re.sub(r"<\?xpacket[^>]*\?>", "", xmp_text)
            root = DET.fromstring(xmp_text)
            
            # Build namespace prefix map
            uri_to_prefix = {
                "http://ns.adobe.com/camera-raw-settings/1.0/": "crs",
                "http://ns.adobe.com/camera-raw-embedded-lens-profile/1.0/": "crlcp",
                "http://ns.adobe.com/photoshop/1.0/": "photoshop",
                "http://ns.adobe.com/photoshop/1.0/camera-profile": "stCamera",
                "http://purl.org/dc/elements/1.1/": "dc",
                "http://ns.adobe.com/tiff/1.0/": "tiff",
                "http://ns.adobe.com/xap/1.0/": "xmp",
            }
            
            def extract_attributes(elem):
                """Extract attributes from an element with namespace prefixes."""
                result = {}
                for attr_qname, attr_value in elem.attrib.items():
                    if attr_qname.startswith("{http://www.w3.org/2000/xmlns/}"):
                        continue
                    
                    # Convert qualified name to prefixed name
                    if attr_qname[0] == "{":
                        try:
                            uri, local = attr_qname[1:].split("}", 1)
                            prefix = uri_to_prefix.get(uri)
                            attr_name = f"{prefix}:{local}" if prefix else local
                        except ValueError:
                            attr_name = attr_qname
                    else:
                        attr_name = attr_qname
                    
                    result[attr_name] = str(attr_value)
                return result
            
            # Find all matching elements
            search_path = f".//{{{namespace_uri}}}{element_name}"
            elements = root.findall(search_path)
            if not elements:
                return None
            
            # Use first match
            elem = elements[0]
            
            # If element has no direct attributes, check for child rdf:Description
            if len(elem.attrib) == 0:
                rdf_uri = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                desc_children = elem.findall(f"{{{rdf_uri}}}Description")
                if desc_children:
                    elem = desc_children[0]
            
            # Extract element attributes
            elem_attrs = extract_attributes(elem)
            
            # Extract parent attributes if requested
            parent_attrs = None
            if include_parent:
                # Build parent map to find parent element
                parent_map = {c: p for p in root.iter() for c in p}
                parent_elem = parent_map.get(elements[0])  # Get parent of original found element
                
                if parent_elem is not None and len(parent_elem.attrib) > 0:
                    parent_attrs = extract_attributes(parent_elem)
            
            return (elem_attrs, parent_attrs) if elem_attrs or parent_attrs else None
            
        except Exception as e:
            logger.debug(f"XPath query failed for {element_name}: {e}")
            return None
    
    def get_root_prop(self, prop: str, return_type: Optional[Type] = None) -> Optional[Any]:
        """Get a root-level XMP property value with optional type conversion.
        
        Only works with properties at the root Description level. For deeply nested
        properties (e.g., crlcp:PerspectiveModel), use xpath_query() instead.
        
        Args:
            prop: Property name. If no namespace prefix, 'crs:' is automatically prepended.
                 Examples: 'Temperature' -> 'crs:Temperature', 'tiff:Orientation' -> 'tiff:Orientation'
            return_type: Optional type to convert the value to (e.g., float, int)
        
        Returns:
            The property value, optionally converted to return_type. None if not found.
        """
        # Auto-prepend 'crs:' if no namespace specified
        if ':' not in prop:
            prop = f'crs:{prop}'
        
        value = self._attributes.get(prop)
        if value is None:
            return None
        
        if return_type is None:
            return value

        if return_type is list:
            if isinstance(value, list):
                return value
            return [value]
        
        # Try to convert using the type's constructor
        try:
            return return_type(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert XMP property '{prop}' value '{value}' to type {return_type}: {e}")
            return None
    
    def get_formatted_string(self, strip_whitespace: bool = True, filter_blank_lines: bool = True) -> Optional[str]:
        """Get the raw XMP string with optional formatting.
        
        Args:
            strip_whitespace: If True, strip trailing whitespace from each line
            filter_blank_lines: If True, remove blank lines (except xpacket lines)
        
        Returns:
            Formatted XMP string, or None if no XMP data available
        """
        if not hasattr(self, '_xmp_string') or not self._xmp_string:
            return None
        
        xmp_str = self._xmp_string
        if isinstance(xmp_str, bytes):
            xmp_str = xmp_str.decode('utf-8', errors='replace')
        
        if not strip_whitespace and not filter_blank_lines:
            return xmp_str
        
        lines = xmp_str.splitlines()
        
        if strip_whitespace:
            lines = [line.rstrip() for line in lines]
        
        if filter_blank_lines:
            lines = [line for line in lines if line.strip() or line.startswith('<?xpacket')]
        
        return '\n'.join(lines)


# =============================================================================
# XMP Codec Functions
# =============================================================================

def xmp_packet_to_metadata(packet: Union[bytes, str]) -> XmpMetadata:
    """Parse an XMP packet (bytes or string) into XmpMetadata.
    
    Args:
        packet: Raw XMP packet from TIFF XMP tag (bytes or string)
    
    Returns:
        XmpMetadata instance
    """
    if isinstance(packet, bytes):
        packet = packet.decode('utf-8', errors='replace')
    return XmpMetadata(packet)


def xmp_metadata_to_packet(xmp: XmpMetadata) -> bytes:
    """Serialize XmpMetadata to an XMP packet (bytes).
    
    Args:
        xmp: XmpMetadata instance to serialize
    
    Returns:
        XMP packet as UTF-8 encoded bytes
    """
    # Build XMP XML structure
    xmp_lines = [
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>',
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 5.4.0">',
        '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
        '    <rdf:Description rdf:about=""',
    ]
    
    # Namespace declarations
    namespaces = {
        'crs': 'http://ns.adobe.com/camera-raw-settings/1.0/',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'tiff': 'http://ns.adobe.com/tiff/1.0/',
        'xmp': 'http://ns.adobe.com/xap/1.0/',
    }
    
    for prefix, uri in namespaces.items():
        xmp_lines.append(f'        xmlns:{prefix}="{uri}"')
    
    # Separate scalar attributes from structured (Bag/Seq)
    scalar_attrs = {}
    structured_props = {}
    
    for key, value in xmp._attributes.items():
        if isinstance(value, list):
            structured_props[key] = value
        else:
            scalar_attrs[key] = value
    
    # Add scalar attributes to Description element
    if scalar_attrs:
        for key, value in scalar_attrs.items():
            # Escape XML special characters
            escaped_value = str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
            xmp_lines.append(f'        {key}="{escaped_value}"')
    
    xmp_lines.append('    >')
    
    # Add structured properties (Bag/Seq)
    for key, value in structured_props.items():
        if not value:
            continue
        
        # Determine if this is a Seq or Bag
        # ToneCurve properties use Seq, dc:subject uses Bag
        is_seq = 'ToneCurve' in key or isinstance(value[0], tuple)
        container_type = 'Seq' if is_seq else 'Bag'
        
        xmp_lines.append(f'      <{key}>')
        xmp_lines.append(f'        <rdf:{container_type}>')
        
        for item in value:
            if isinstance(item, tuple) and len(item) == 2:
                # Tone curve point: convert normalized 0-1 floats back to 0-255 integers
                x_int = int(round(item[0] * 255.0))
                y_int = int(round(item[1] * 255.0))
                xmp_lines.append(f'          <rdf:li>{x_int}, {y_int}</rdf:li>')
            else:
                # String item
                escaped_item = str(item).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                xmp_lines.append(f'          <rdf:li>{escaped_item}</rdf:li>')
        
        xmp_lines.append(f'        </rdf:{container_type}>')
        xmp_lines.append(f'      </{key}>')
    
    xmp_lines.extend([
        '    </rdf:Description>',
        '  </rdf:RDF>',
        '</x:xmpmeta>',
        '<?xpacket end="w"?>',
    ])
    
    xmp_string = '\n'.join(xmp_lines)
    return xmp_string.encode('utf-8')

