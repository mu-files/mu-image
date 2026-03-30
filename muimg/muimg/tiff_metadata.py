"""TIFF/DNG metadata support classes.

This module provides classes for creating and parsing TIFF/DNG metadata tags.
"""
from __future__ import annotations

import logging
import numpy as np

from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from tifffile import PHOTOMETRIC, TIFF
from typing import Optional, Union, Dict, Any, Type, List, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Tag Type Registry
# =============================================================================
# Maps tag names to (dtype, count) where:
#   dtype: TIFF data type string ('s'=ascii, 'H'=ushort, 'I'=ulong, '2I'=urational, 
#          '2i'=srational, 'B'=byte, 'f'=float, 'd'=double)
#   count: Expected count, or None for variable length
#
# This registry enables auto-conversion: clients provide friendly Python types
# (float, datetime, str, np.ndarray) and we convert to appropriate TIFF format.
#
# For tags that accept multiple types (per DNG SDK), dtype can be a list.
# Type inference order: int types first, then float/rational types.
# Example: ["I", "2I"] means int→LONG, float→RATIONAL

# =============================================================================
# TIFF Data Type Information
# =============================================================================
# Unified table mapping tifffile dtype codes to dtype strings and categories.
# - code: tifffile numeric dtype code
# - dtype_str: our dtype string used in TagSpec  
# - category: 'int' or 'float' for type inference when TagSpec has multiple dtypes

@dataclass
class TiffDtype:
    """TIFF data type specification."""
    code: int
    dtype_str: str
    category: str  # 'int' or 'float'
    name: str


TIFF_DTYPES = {
    1:  TiffDtype(1,  'B',  'int',   'BYTE'),
    2:  TiffDtype(2,  's',  'int',   'ASCII'),
    3:  TiffDtype(3,  'H',  'int',   'SHORT'),
    4:  TiffDtype(4,  'I',  'int',   'LONG'),
    5:  TiffDtype(5,  '2I', 'float', 'RATIONAL'),
    6:  TiffDtype(6,  'b',  'int',   'SBYTE'),
    7:  TiffDtype(7,  'B',  'int',   'UNDEFINED'),
    8:  TiffDtype(8,  'h',  'int',   'SSHORT'),
    9:  TiffDtype(9,  'i',  'int',   'SLONG'),
    10: TiffDtype(10, '2i', 'float', 'SRATIONAL'),
    11: TiffDtype(11, 'f',  'float', 'FLOAT'),
    12: TiffDtype(12, 'd',  'float', 'DOUBLE'),
    16: TiffDtype(16, 'Q',  'int',   'LONG8'),
    17: TiffDtype(17, 'q',  'int',   'SLONG8'),
}

# Derived lookups for convenience
TIFF_DTYPE_TO_STR = {dt.code: dt.dtype_str for dt in TIFF_DTYPES.values()}
DTYPE_CATEGORY = {dt.dtype_str: dt.category for dt in TIFF_DTYPES.values()}

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
# - "ifd0": Must only be in IFD 0 (main metadata, not in SubIFDs)
# - "exif": Can be in IFD 0 or EXIF IFD (not in preview/thumbnail SubIFDs)
# - "raw": Must be in IFD containing raw image data (CFA or LINEAR_RAW)
# - "raw:cfa": A raw IFD with PhotometricInterpretation = CFA
# - "profile": Can be in IFD 0 or Camera Profile IFDs


@dataclass
class TagSpec:
    """Specification for a TIFF/DNG tag.
    
    Attributes:
        dtype: TIFF data type(s). Can be a single string or list for type inference.
               When a list, int values use first matching int type, floats use first
               matching float/rational type. Signedness is encoded in dtype ('2i' vs '2I').
        count: Expected count, or None for variable length.
        shape: Target shape for array tags (e.g., (3, 3) for matrices). If specified,
               the returned np.ndarray will be reshaped to this shape.
        ifd_location: IFD type where this tag should appear (see IFD Location Categories above).
    """
    dtype: Union[str, List[str]]  # Single type or list for inference
    count: Optional[int] = None
    shape: Optional[Tuple[int, ...]] = None
    ifd_location: str = "any"  # Required: "raw", "main", "exif", "raw:cfa", "profile", or "any"
    
    @staticmethod
    def _get_value_category(value: Any) -> Optional[str]:
        """Determine the category ('int' or 'float') of a Python value."""
        # Scalar checks
        if isinstance(value, (int, np.integer)):
            return 'int'
        if isinstance(value, (float, np.floating)):
            return 'float'
        
        # Array checks
        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.integer):
                return 'int'
            if np.issubdtype(value.dtype, np.floating):
                return 'float'
        
        # List/tuple: check first element
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return TagSpec._get_value_category(value[0])
        
        return None
    
    def get_dtype_for_value(self, value: Any) -> str:
        """Select appropriate dtype based on value type."""
        if isinstance(self.dtype, str):
            return self.dtype
        
        category = self._get_value_category(value)
        if category:
            for dt in self.dtype:
                if DTYPE_CATEGORY.get(dt) == category:
                    return dt
        
        return self.dtype[0]  # Fallback to first
    
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
        dtype_str = TIFF_DTYPE_TO_STR.get(dtype)
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
    "ProcessingSoftware": TagSpec("s", None, ifd_location="any"),  # 11
    "NewSubfileType": TagSpec("I", 1, ifd_location="any"),  # 254
    "SubfileType": TagSpec("H", 1, ifd_location="any"),  # 255
    "ImageWidth": TagSpec(["H", "I"], 1, ifd_location="any"),  # 256
    "ImageLength": TagSpec(["H", "I"], 1, ifd_location="any"),  # 257
    "BitsPerSample": TagSpec("H", None, ifd_location="any"),  # 258
    "Compression": TagSpec("H", 1, ifd_location="any"),  # 259
    "PhotometricInterpretation": TagSpec("H", 1, ifd_location="any"),  # 262
    "Thresholding": TagSpec("H", 1, ifd_location="any"),  # 263
    "CellWidth": TagSpec("H", 1, ifd_location="any"),  # 264
    "CellLength": TagSpec("H", 1, ifd_location="any"),  # 265
    "FillOrder": TagSpec("H", 1, ifd_location="any"),  # 266
    "DocumentName": TagSpec("s", None, ifd_location="any"),  # 269
    "ImageDescription": TagSpec("s", None, ifd_location="ifd0"),  # 270
    "Make": TagSpec("s", None, ifd_location="ifd0"),  # 271
    "Model": TagSpec("s", None, ifd_location="ifd0"),  # 272
    "StripOffsets": TagSpec(["H", "I"], None, ifd_location="any"),  # 273
    "Orientation": TagSpec("H", 1, ifd_location="ifd0"),  # 274
    "SamplesPerPixel": TagSpec("H", 1, ifd_location="any"),  # 277
    "RowsPerStrip": TagSpec(["H", "I"], 1, ifd_location="any"),  # 278
    "StripByteCounts": TagSpec(["H", "I"], None, ifd_location="any"),  # 279
    "MinSampleValue": TagSpec("H", None, ifd_location="any"),  # 280
    "MaxSampleValue": TagSpec("H", None, ifd_location="any"),  # 281
    "XResolution": TagSpec("2I", 1, ifd_location="any"),  # 282
    "YResolution": TagSpec("2I", 1, ifd_location="any"),  # 283
    "PlanarConfiguration": TagSpec("H", 1, ifd_location="any"),  # 284
    "PageName": TagSpec("s", None, ifd_location="any"),  # 285
    "XPosition": TagSpec("2I", 1, ifd_location="any"),  # 286
    "YPosition": TagSpec("2I", 1, ifd_location="any"),  # 287
    "FreeOffsets": TagSpec("I", None, ifd_location="any"),  # 288
    "FreeByteCounts": TagSpec("I", None, ifd_location="any"),  # 289
    "GrayResponseUnit": TagSpec("H", 1, ifd_location="any"),  # 290
    "GrayResponseCurve": TagSpec("H", None, ifd_location="any"),  # 291
    "T4Options": TagSpec("I", 1, ifd_location="any"),  # 292
    "T6Options": TagSpec("I", 1, ifd_location="any"),  # 293
    "ResolutionUnit": TagSpec("H", 1, ifd_location="any"),  # 296
    "PageNumber": TagSpec("H", 2, ifd_location="any"),  # 297
    "TransferFunction": TagSpec("H", None, ifd_location="any"),  # 301
    "Software": TagSpec("s", None, ifd_location="ifd0"),  # 305
    "DateTime": TagSpec("s", 20, ifd_location="ifd0"),  # 306
    "Artist": TagSpec("s", None, ifd_location="ifd0"),  # 315
    "HostComputer": TagSpec("s", None, ifd_location="any"),  # 316
    "Predictor": TagSpec("H", 1, ifd_location="any"),  # 317
    "WhitePoint": TagSpec("2I", 2, ifd_location="any"),  # 318
    "PrimaryChromaticities": TagSpec("2I", 6, ifd_location="any"),  # 319
    "ColorMap": TagSpec("H", None, ifd_location="any"),  # 320
    "HalftoneHints": TagSpec("H", 2, ifd_location="any"),  # 321
    "TileWidth": TagSpec(["H", "I"], 1, ifd_location="any"),  # 322
    "TileLength": TagSpec(["H", "I"], 1, ifd_location="any"),  # 323
    "TileOffsets": TagSpec("I", None, ifd_location="any"),  # 324
    "TileByteCounts": TagSpec(["H", "I"], None, ifd_location="any"),  # 325
    "SubIFDs": TagSpec("I", None, ifd_location="any"),  # 330
    "InkSet": TagSpec("H", 1, ifd_location="any"),  # 332
    "InkNames": TagSpec("s", None, ifd_location="any"),  # 333
    "NumberOfInks": TagSpec("H", 1, ifd_location="any"),  # 334
    "DotRange": TagSpec("H", None, ifd_location="any"),  # 336
    "TargetPrinter": TagSpec("s", None, ifd_location="any"),  # 337
    "ExtraSamples": TagSpec("H", None, ifd_location="any"),  # 338
    "SampleFormat": TagSpec("H", None, ifd_location="any"),  # 339
    "SMinSampleValue": TagSpec("d", None, ifd_location="any"),  # 340
    "SMaxSampleValue": TagSpec("d", None, ifd_location="any"),  # 341
    "TransferRange": TagSpec("H", 6, ifd_location="any"),  # 342
    "ClipPath": TagSpec("B", None, ifd_location="any"),  # 343
    "XClipPathUnits": TagSpec("I", 1, ifd_location="any"),  # 344
    "YClipPathUnits": TagSpec("I", 1, ifd_location="any"),  # 345
    "Indexed": TagSpec("H", 1, ifd_location="any"),  # 346
    "JPEGTables": TagSpec("B", None, ifd_location="any"),  # 347
    "OPIProxy": TagSpec("H", 1, ifd_location="any"),  # 351
    "VersionYear": TagSpec("B", 4, ifd_location="any"),  # 404
    "Decode": TagSpec("2i", None, ifd_location="any"),  # 433
    "DefaultImageColor": TagSpec("H", None, ifd_location="any"),  # 434
    "JPEGInterchangeFormat": TagSpec("I", 1, ifd_location="any"),  # 513
    "JPEGInterchangeFormatLength": TagSpec("I", 1, ifd_location="any"),  # 514
    "JPEGRestartInterval": TagSpec("H", 1, ifd_location="any"),  # 515
    "JPEGLosslessPredictors": TagSpec("H", None, ifd_location="any"),  # 517
    "JPEGPointTransforms": TagSpec("H", None, ifd_location="any"),  # 518
    "JPEGQTables": TagSpec("I", None, ifd_location="any"),  # 519
    "JPEGDCTables": TagSpec("I", None, ifd_location="any"),  # 520
    "JPEGACTables": TagSpec("I", None, ifd_location="any"),  # 521
    "YCbCrCoefficients": TagSpec("2I", 3, ifd_location="any"),  # 529
    "YCbCrSubSampling": TagSpec("H", 2, ifd_location="any"),  # 530
    "YCbCrPositioning": TagSpec("H", 1, ifd_location="any"),  # 531
    "ReferenceBlackWhite": TagSpec("2I", 6, ifd_location="any"),  # 532
    "StripRowCounts": TagSpec("I", None, ifd_location="any"),  # 559
    "XMP": TagSpec("B", None, ifd_location="any"),  # 700
    "ICCProfileDescriptor": TagSpec("s", None, ifd_location="any"),  # 770
    "Rating": TagSpec("H", 1, ifd_location="any"),  # 18246
    "RatingPercent": TagSpec("H", 1, ifd_location="any"),  # 18249
    "PrintFlags": TagSpec("B", None, ifd_location="any"),  # 20485
    "PrintFlagsVersion": TagSpec("H", 1, ifd_location="any"),  # 20486
    "PrintFlagsCrop": TagSpec("I", 1, ifd_location="any"),  # 20487
    "PrintFlagsBleedWidth": TagSpec("I", 1, ifd_location="any"),  # 20488
    "PrintFlagsBleedWidthScale": TagSpec("H", 1, ifd_location="any"),  # 20489
    "InteroperabilityIndex": TagSpec("s", None, ifd_location="any"),  # 20545
    "InteroperabilityVersion": TagSpec("B", 4, ifd_location="any"),  # 20546
    "FrameDelay": TagSpec("I", None, ifd_location="any"),  # 20736
    "LoopCount": TagSpec("H", 1, ifd_location="any"),  # 20737
    "VignettingCorrParams": TagSpec("2i", None, ifd_location="any"),  # 28722
    "ChromaticAberrationCorrParams": TagSpec("2i", None, ifd_location="any"),  # 28725
    "DistortionCorrParams": TagSpec("2i", None, ifd_location="any"),  # 28727
    
    # Private tags >= 32768
    "ImageID": TagSpec("s", None, ifd_location="any"),  # 32781
    "Matteing": TagSpec("H", 1, ifd_location="any"),  # 32995
    "DataType": TagSpec("H", None, ifd_location="any"),  # 32996
    "ImageDepth": TagSpec(["H", "I"], 1, ifd_location="any"),  # 32997
    "TileDepth": TagSpec(["H", "I"], 1, ifd_location="any"),  # 32998
    "Model2": TagSpec("s", None, ifd_location="any"),  # 33405
    "CFARepeatPatternDim": TagSpec("H", 2, ifd_location="raw:cfa"),  # 33421
    "CFAPattern": TagSpec("B", None, ifd_location="raw:cfa"),  # 33422
    "BatteryLevel": TagSpec("2I", 1, ifd_location="exif"),  # 33423
    "Copyright": TagSpec("s", None, ifd_location="ifd0"),  # 33432
    "ExposureTime": TagSpec("2I", 1, ifd_location="exif"),  # 33434
    "FNumber": TagSpec("2I", 1, ifd_location="exif"),  # 33437
    "ModelPixelScaleTag": TagSpec("d", 3, ifd_location="ifd0"),  # 33550
    "IPTCNAA": TagSpec("B", None, ifd_location="any"),  # 33723
    "ModelTiepointTag": TagSpec("d", None, ifd_location="ifd0"),  # 33922
    "ModelTransformationTag": TagSpec("d", 16, ifd_location="ifd0"),  # 34264
    "WB_GRGBLevels": TagSpec("2I", 4, ifd_location="any"),  # 34306
    "ImageResources": TagSpec("B", None, ifd_location="any"),  # 34377
    "ExifTag": TagSpec("I", 1, ifd_location="ifd0"),  # 34665
    "InterColorProfile": TagSpec("B", None, ifd_location="any"),  # 34675
    "GeoKeyDirectoryTag": TagSpec("H", None, ifd_location="ifd0"),  # 34735
    "GeoDoubleParamsTag": TagSpec("d", None, ifd_location="ifd0"),  # 34736
    "GeoAsciiParamsTag": TagSpec("s", None, ifd_location="ifd0"),  # 34737
    "ExposureProgram": TagSpec("H", 1, ifd_location="exif"),  # 34850
    "SpectralSensitivity": TagSpec("s", None, ifd_location="exif"),  # 34852
    "GPSTag": TagSpec("I", 1, ifd_location="ifd0"),  # 34853
    "ISOSpeedRatings": TagSpec("H", None, ifd_location="exif"),  # 34855
    "OECF": TagSpec("B", None, ifd_location="exif"),  # 34856
    "Interlace": TagSpec("H", 1, ifd_location="exif"),  # 34857
    "TimeZoneOffset": TagSpec("h", None, ifd_location="exif"),  # 34858
    "SelfTimerMode": TagSpec("H", 1, ifd_location="exif"),  # 34859
    "SensitivityType": TagSpec("H", 1, ifd_location="exif"),  # 34864
    "StandardOutputSensitivity": TagSpec("I", 1, ifd_location="exif"),  # 34865
    "RecommendedExposureIndex": TagSpec("I", 1, ifd_location="exif"),  # 34866
    "ISOSpeed": TagSpec("I", 1, ifd_location="exif"),  # 34867
    "ISOSpeedLatitudeyyy": TagSpec("I", 1, ifd_location="exif"),  # 34868
    "ISOSpeedLatitudezzz": TagSpec("I", 1, ifd_location="exif"),  # 34869
    "ExifVersion": TagSpec("B", 4, ifd_location="exif"),  # 36864
    "DateTimeOriginal": TagSpec("s", 20, ifd_location="exif"),  # 36867
    "DateTimeDigitized": TagSpec("s", 20, ifd_location="exif"),  # 36868
    "OffsetTime": TagSpec("s", None, ifd_location="exif"),  # 36880
    "OffsetTimeOriginal": TagSpec("s", None, ifd_location="exif"),  # 36881
    "OffsetTimeDigitized": TagSpec("s", None, ifd_location="exif"),  # 36882
    "ComponentsConfiguration": TagSpec("B", 4, ifd_location="exif"),  # 37121
    "CompressedBitsPerPixel": TagSpec("2I", 1, ifd_location="exif"),  # 37122
    "ShutterSpeedValue": TagSpec("2i", 1, ifd_location="exif"),  # 37377
    "ApertureValue": TagSpec("2I", 1, ifd_location="exif"),  # 37378
    "BrightnessValue": TagSpec("2i", 1, ifd_location="exif"),  # 37379
    "ExposureBiasValue": TagSpec("2i", 1, ifd_location="exif"),  # 37380
    "MaxApertureValue": TagSpec("2I", 1, ifd_location="exif"),  # 37381
    "SubjectDistance": TagSpec("2I", 1, ifd_location="exif"),  # 37382
    "MeteringMode": TagSpec("H", 1, ifd_location="exif"),  # 37383
    "LightSource": TagSpec("H", 1, ifd_location="exif"),  # 37384
    "Flash": TagSpec("H", 1, ifd_location="exif"),  # 37385
    "FocalLength": TagSpec("2I", 1, ifd_location="exif"),  # 37386
    "FlashEnergy": TagSpec("2I", 1, ifd_location="exif"),  # 37387
    "SpatialFrequencyResponse": TagSpec("B", None, ifd_location="exif"),  # 37388
    "Noise": TagSpec("B", None, ifd_location="exif"),  # 37389
    "FocalPlaneXResolution": TagSpec("2I", 1, ifd_location="exif"),  # 37390
    "FocalPlaneYResolution": TagSpec("2I", 1, ifd_location="exif"),  # 37391
    "FocalPlaneResolutionUnit": TagSpec("H", 1, ifd_location="exif"),  # 37392
    "ImageNumber": TagSpec(["H", "I"], 1, ifd_location="exif"),  # 37393
    "SecurityClassification": TagSpec("s", None, ifd_location="exif"),  # 37394
    "ImageHistory": TagSpec("s", None, ifd_location="exif"),  # 37395
    "SubjectLocation": TagSpec("H", 2, ifd_location="exif"),  # 37396
    "ExposureIndex": TagSpec("2I", 1, ifd_location="exif"),  # 37397
    "TIFFEPStandardID": TagSpec("B", 4, ifd_location="exif"),  # 37398
    "SensingMethod": TagSpec("H", 1, ifd_location="exif"),  # 37399
    "MakerNote": TagSpec("B", None, ifd_location="exif"),  # 37500
    "UserComment": TagSpec("B", None, ifd_location="exif"),  # 37510
    "SubsecTime": TagSpec("s", None, ifd_location="exif"),  # 37520
    "SubsecTimeOriginal": TagSpec("s", None, ifd_location="exif"),  # 37521
    "SubsecTimeDigitized": TagSpec("s", None, ifd_location="exif"),  # 37522
    "ImageSourceData": TagSpec("B", None, ifd_location="exif"),  # 37724
    "Temperature": TagSpec("2i", 1, ifd_location="exif"),  # 37888
    "Humidity": TagSpec("2I", 1, ifd_location="exif"),  # 37889
    "Pressure": TagSpec("2I", 1, ifd_location="exif"),  # 37890
    "WaterDepth": TagSpec("2i", 1, ifd_location="exif"),  # 37891
    "Acceleration": TagSpec("2I", 1, ifd_location="exif"),  # 37892
    "CameraElevationAngle": TagSpec("2i", 1, ifd_location="exif"),  # 37893
    "FlashpixVersion": TagSpec("B", 4, ifd_location="exif"),  # 40960
    "ColorSpace": TagSpec("H", 1, ifd_location="exif"),  # 40961
    "PixelXDimension": TagSpec(["H", "I"], 1, ifd_location="exif"),  # 40962
    "PixelYDimension": TagSpec(["H", "I"], 1, ifd_location="exif"),  # 40963
    "RelatedSoundFile": TagSpec("s", None, ifd_location="exif"),  # 40964
    "InteroperabilityTag": TagSpec("I", 1, ifd_location="exif"),  # 40965
    "TIFF-EPStandardID": TagSpec("B", 4, ifd_location="exif"),  # 41494
    "FileSource": TagSpec("B", 1, ifd_location="exif"),  # 41728
    "SceneType": TagSpec("B", 1, ifd_location="exif"),  # 41729
    "CustomRendered": TagSpec("H", 1, ifd_location="exif"),  # 41985
    "ExposureMode": TagSpec("H", 1, ifd_location="exif"),  # 41986
    "WhiteBalance": TagSpec("H", 1, ifd_location="exif"),  # 41987
    "DigitalZoomRatio": TagSpec("2I", 1, ifd_location="exif"),  # 41988
    "FocalLengthIn35mmFilm": TagSpec("H", 1, ifd_location="exif"),  # 41989
    "SceneCaptureType": TagSpec("H", 1, ifd_location="exif"),  # 41990
    "GainControl": TagSpec("H", 1, ifd_location="exif"),  # 41991
    "Contrast": TagSpec("H", 1, ifd_location="exif"),  # 41992
    "Saturation": TagSpec("H", 1, ifd_location="exif"),  # 41993
    "Sharpness": TagSpec("H", 1, ifd_location="exif"),  # 41994
    "DeviceSettingDescription": TagSpec("B", None, ifd_location="exif"),  # 41995
    "SubjectDistanceRange": TagSpec("H", 1, ifd_location="exif"),  # 41996
    "ImageUniqueID": TagSpec("s", None, ifd_location="exif"),  # 42016
    "CameraOwnerName": TagSpec("s", None, ifd_location="exif"),  # 42032
    "BodySerialNumber": TagSpec("s", None, ifd_location="exif"),  # 42033
    "LensSpecification": TagSpec("2I", 4, ifd_location="exif"),  # 42034
    "LensMake": TagSpec("s", None, ifd_location="exif"),  # 42035
    "LensModel": TagSpec("s", None, ifd_location="exif"),  # 42036
    "LensSerialNumber": TagSpec("s", None, ifd_location="exif"),  # 42037
    "CompositeImage": TagSpec("H", 1, ifd_location="exif"),  # 42080
    "SourceImageNumberCompositeImage": TagSpec("H", None, ifd_location="exif"),  # 42081
    "SourceExposureTimesCompositeImage": TagSpec("B", None, ifd_location="exif"),  # 42082
    "Gamma": TagSpec("2I", 1, ifd_location="exif"),  # 42240
    "PixelFormat": TagSpec("B", 16, ifd_location="any"),  # 48129
    "ImageType": TagSpec("H", 1, ifd_location="any"),  # 48132
    "OriginalFileName": TagSpec("s", None, ifd_location="any"),  # 50547
    "DNGVersion": TagSpec("B", 4, ifd_location="ifd0"),  # 50706
    "DNGBackwardVersion": TagSpec("B", 4, ifd_location="ifd0"),  # 50707
    "UniqueCameraModel": TagSpec("s", None, ifd_location="ifd0"),  # 50708
    "LocalizedCameraModel": TagSpec(["s", "B"], None, ifd_location="ifd0"),  # 50709
    "CFAPlaneColor": TagSpec("B", None, ifd_location="raw:cfa"),  # 50710
    "CFALayout": TagSpec("H", 1, ifd_location="raw:cfa"),  # 50711
    "LinearizationTable": TagSpec("H", None, ifd_location="raw"),  # 50712
    "BlackLevelRepeatDim": TagSpec("H", 2, ifd_location="raw"),  # 50713
    "BlackLevel": TagSpec(["H", "I", "2I"], None, ifd_location="raw"),  # 50714
    "BlackLevelDeltaH": TagSpec("2i", None, ifd_location="raw"),  # 50715
    "BlackLevelDeltaV": TagSpec("2i", None, ifd_location="raw"),  # 50716
    "WhiteLevel": TagSpec(["H", "I"], None, ifd_location="raw"),  # 50717
    "DefaultScale": TagSpec("2I", 2, ifd_location="raw"),  # 50718
    "DefaultCropOrigin": TagSpec(["H", "I", "2I"], 2, ifd_location="raw"),  # 50719
    "DefaultCropSize": TagSpec(["H", "I", "2I"], 2, ifd_location="raw"),  # 50720
    "ColorMatrix1": TagSpec("2i", 9, (3, 3), ifd_location="profile"),  # 50721 - 3x3 matrix
    "ColorMatrix2": TagSpec("2i", 9, (3, 3), ifd_location="profile"),  # 50722 - 3x3 matrix
    "CameraCalibration1": TagSpec("2i", 9, (3, 3), ifd_location="ifd0"),  # 50723 - 3x3 matrix
    "CameraCalibration2": TagSpec("2i", 9, (3, 3), ifd_location="ifd0"),  # 50724 - 3x3 matrix
    "ReductionMatrix1": TagSpec("2i", None, ifd_location="profile"),  # 50725
    "ReductionMatrix2": TagSpec("2i", None, ifd_location="profile"),  # 50726
    "AnalogBalance": TagSpec("2I", None, ifd_location="ifd0"),  # 50727
    "AsShotNeutral": TagSpec("2I", None, ifd_location="ifd0"),  # 50728
    "AsShotWhiteXY": TagSpec("2I", 2, ifd_location="ifd0"),  # 50729
    "BaselineExposure": TagSpec("2i", 1, ifd_location="ifd0"),  # 50730
    "BaselineNoise": TagSpec("2I", 1, ifd_location="ifd0"),  # 50731
    "BaselineSharpness": TagSpec("2I", 1, ifd_location="ifd0"),  # 50732
    "BayerGreenSplit": TagSpec("I", 1, ifd_location="raw:cfa"),  # 50733
    "LinearResponseLimit": TagSpec("2I", 1, ifd_location="ifd0"),  # 50734
    "CameraSerialNumber": TagSpec("s", None, ifd_location="ifd0"),  # 50735
    "LensInfo": TagSpec("2I", 4, ifd_location="ifd0"),  # 50736
    "ChromaBlurRadius": TagSpec("2I", 1, ifd_location="raw"),  # 50737
    "AntiAliasStrength": TagSpec("2I", 1, ifd_location="raw"),  # 50738
    "ShadowScale": TagSpec("2I", 1, ifd_location="ifd0"),  # 50739
    "DNGPrivateData": TagSpec("B", None, ifd_location="ifd0"),  # 50740
    "MakerNoteSafety": TagSpec("H", 1, ifd_location="ifd0"),  # 50741
    "RawImageSegmentation": TagSpec("H", 3, ifd_location="any"),  # 50752
    "CalibrationIlluminant1": TagSpec("H", 1, ifd_location="profile"),  # 50778
    "CalibrationIlluminant2": TagSpec("H", 1, ifd_location="profile"),  # 50779
    "BestQualityScale": TagSpec("2I", 1, ifd_location="raw"),  # 50780
    "RawDataUniqueID": TagSpec("B", 16, ifd_location="ifd0"),  # 50781
    "OriginalRawFileName": TagSpec(["s", "B"], None, ifd_location="ifd0"),  # 50827
    "OriginalRawFileData": TagSpec("B", None, ifd_location="ifd0"),  # 50828
    "ActiveArea": TagSpec(["H", "I"], 4, ifd_location="raw"),  # 50829
    "MaskedAreas": TagSpec(["H", "I"], None, ifd_location="raw"),  # 50830
    "AsShotICCProfile": TagSpec("B", None, ifd_location="ifd0"),  # 50831
    "AsShotPreProfileMatrix": TagSpec("2i", None, ifd_location="ifd0"),  # 50832
    "CurrentICCProfile": TagSpec("B", None, ifd_location="ifd0"),  # 50833
    "CurrentPreProfileMatrix": TagSpec("2i", None, ifd_location="ifd0"),  # 50834
    "ColorimetricReference": TagSpec("H", 1, ifd_location="ifd0"),  # 50879
    "CameraCalibrationSignature": TagSpec(["s", "B"], None, ifd_location="ifd0"),  # 50931
    "ProfileCalibrationSignature": TagSpec(["s", "B"], None, ifd_location="profile"),  # 50932
    "ProfileIFD": TagSpec("I", None, ifd_location="any"),  # 50933 - variable count (can have multiple profiles)
    "AsShotProfileName": TagSpec(["s", "B"], None, ifd_location="ifd0"),  # 50934
    "NoiseReductionApplied": TagSpec("2I", 1, ifd_location="raw"),  # 50935
    "ProfileName": TagSpec(["s", "B"], None, ifd_location="profile"),  # 50936
    "ProfileHueSatMapDims": TagSpec("I", None, ifd_location="profile"),  # 50937
    "ProfileHueSatMapData1": TagSpec("f", None, ifd_location="profile"),  # 50938
    "ProfileHueSatMapData2": TagSpec("f", None, ifd_location="profile"),  # 50939
    "ProfileToneCurve": TagSpec("f", None, ifd_location="profile"),  # 50940
    "ProfileEmbedPolicy": TagSpec("I", 1, ifd_location="profile"),  # 50941
    "ProfileCopyright": TagSpec(["s", "B"], None, ifd_location="profile"),  # 50942
    "ForwardMatrix1": TagSpec("2i", 9, (3, 3), ifd_location="profile"),  # 50964 - 3x3 matrix
    "ForwardMatrix2": TagSpec("2i", 9, (3, 3), ifd_location="profile"),  # 50965 - 3x3 matrix
    "PreviewApplicationName": TagSpec("s", None, ifd_location="any"),  # 50966
    "PreviewApplicationVersion": TagSpec("s", None, ifd_location="any"),  # 50967
    "PreviewSettingsName": TagSpec("s", None, ifd_location="any"),  # 50968
    "PreviewSettingsDigest": TagSpec("B", 16, ifd_location="any"),  # 50969
    "PreviewColorSpace": TagSpec("I", 1, ifd_location="any"),  # 50970
    "PreviewDateTime": TagSpec("s", None, ifd_location="any"),  # 50971
    "RawImageDigest": TagSpec("B", 16, ifd_location="ifd0"),  # 50972
    "OriginalRawFileDigest": TagSpec("B", 16, ifd_location="ifd0"),  # 50973
    "SubTileBlockSize": TagSpec("H", 2, ifd_location="raw"),  # 50974
    "RowInterleaveFactor": TagSpec("H", 1, ifd_location="raw"),  # 50975
    "ProfileLookTableDims": TagSpec("I", None, ifd_location="profile"),  # 50981
    "ProfileLookTableData": TagSpec("f", None, ifd_location="profile"),  # 50982
    "OpcodeList1": TagSpec("B", None, ifd_location="raw"),  # 51008
    "OpcodeList2": TagSpec("B", None, ifd_location="raw"),  # 51009
    "OpcodeList3": TagSpec("B", None, ifd_location="raw"),  # 51022
    # NoiseProfile: DNG spec says Raw IFD, but Adobe SDK writes to both main and raw IFDs
    # for legacy compatibility (dng_image_writer.cpp:8544). SDK reads from both locations.
    "NoiseProfile": TagSpec("d", None, ifd_location="raw"),  # 51041
    "TimeCodes": TagSpec("B", None, ifd_location="any"),  # 51043
    "FrameRate": TagSpec("2I", 1, ifd_location="any"),  # 51044
    "TStop": TagSpec("2I", None, ifd_location="any"),  # 51058
    "ReelName": TagSpec("s", None, ifd_location="any"),  # 51081
    "OriginalDefaultFinalSize": TagSpec(["H", "I"], 2, ifd_location="any"),  # 51089
    "OriginalBestQualitySize": TagSpec(["H", "I"], 2, ifd_location="any"),  # 51090
    "OriginalDefaultCropSize": TagSpec(["H", "I", "2I"], 2, ifd_location="any"),  # 51091
    "CameraLabel": TagSpec("s", None, ifd_location="any"),  # 51105
    "ProfileHueSatMapEncoding": TagSpec("I", 1, ifd_location="profile"),  # 51107
    "ProfileLookTableEncoding": TagSpec("I", 1, ifd_location="profile"),  # 51108
    "BaselineExposureOffset": TagSpec("2i", 1, ifd_location="ifd0"),  # 51109
    "DefaultBlackRender": TagSpec("I", 1, ifd_location="ifd0"),  # 51110
    "NewRawImageDigest": TagSpec("B", 16, ifd_location="ifd0"),  # 51111
    "RawToPreviewGain": TagSpec("d", 1, ifd_location="raw"),  # 51112
    "DefaultUserCrop": TagSpec("2I", 4, ifd_location="raw"),  # 51125
    "DepthFormat": TagSpec("H", 1, ifd_location="any"),  # 51177
    "DepthNear": TagSpec("2I", 1, ifd_location="any"),  # 51178
    "DepthFar": TagSpec("2I", 1, ifd_location="any"),  # 51179
    "DepthUnits": TagSpec("H", 1, ifd_location="any"),  # 51180
    "DepthMeasureType": TagSpec("H", 1, ifd_location="any"),  # 51181
    "EnhanceParams": TagSpec("s", None, ifd_location="any"),  # 51182
    "ProfileGainTableMap": TagSpec("B", None, ifd_location="raw"),  # 52525
    "SemanticName": TagSpec("s", None, ifd_location="any"),  # 52526
    "SemanticInstanceID": TagSpec("s", None, ifd_location="any"),  # 52528
    "CalibrationIlluminant3": TagSpec("H", 1, ifd_location="profile"),  # 52529
    "CameraCalibration3": TagSpec("2i", 9, (3, 3), ifd_location="ifd0"),  # 52530 - 3x3 matrix
    "ColorMatrix3": TagSpec("2i", 9, (3, 3), ifd_location="profile"),  # 52531 - 3x3 matrix
    "ForwardMatrix3": TagSpec("2i", 9, (3, 3), ifd_location="profile"),  # 52532 - 3x3 matrix
    "IlluminantData1": TagSpec("B", None, ifd_location="profile"),  # 52533
    "IlluminantData2": TagSpec("B", None, ifd_location="profile"),  # 52534
    "MaskSubArea": TagSpec("I", 4, ifd_location="any"),  # 52536
    "ProfileHueSatMapData3": TagSpec("f", None, ifd_location="profile"),  # 52537
    "ReductionMatrix3": TagSpec("2i", None, ifd_location="profile"),  # 52538
    "RGBTables": TagSpec("B", None, ifd_location="profile"),  # 52543
    "ProfileGainTableMap2": TagSpec("B", None, ifd_location="any"),  # 52544
    "ColumnInterleaveFactor": TagSpec("H", 1, ifd_location="raw"),  # 52547
    "ImageSequenceInfo": TagSpec("B", None, ifd_location="any"),  # 52548
    "ProfileToneMethod": TagSpec("I", 1, ifd_location="profile"),  # 52549 (not in tifffile)
    "ImageStats": TagSpec("B", None, ifd_location="raw"),  # 52550
    "ProfileDynamicRange": TagSpec("d", 1, ifd_location="profile"),  # 52551
    "ProfileGroupName": TagSpec(["s", "B"], None, ifd_location="profile"),  # 52552
    "JXLDistance": TagSpec("f", 1, ifd_location="any"),  # 52553
    "JXLEffort": TagSpec("I", 1, ifd_location="any"),  # 52554
    "JXLDecodeSpeed": TagSpec("I", 1, ifd_location="any"),  # 52555
    "IlluminantData3": TagSpec("B", None, ifd_location="profile"),  # 53535
    "Padding": TagSpec("B", None, ifd_location="any"),  # 59932
    "OffsetSchema": TagSpec("i", 1, ifd_location="any"),  # 59933
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
    }
    
    def __init__(self):
        self._by_name: Dict[str, int] = {}
        self._by_code: Dict[int, str] = {}
        
        # Copy from tifffile's registry
        for code, name in TIFF.TAGS.items():
            self._by_name[name] = code
            self._by_code[code] = name
        
        # Add our extra tags
        for code, name in self._EXTRA_TAGS.items():
            self._by_name[name] = code
            self._by_code[code] = name
    
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
    
    # UTF-8 string tags: tags with ["s", "B"] dtype support UTF-8 encoding
    # Includes: LocalizedCameraModel, ProfileName, ProfileCopyright, etc.
    if isinstance(value, str) and isinstance(spec.dtype, list) and "B" in spec.dtype:
        value = value.encode("utf-8") + b"\x00"
    
    # Select appropriate dtype based on value type (handles multi-type TagSpecs)
    dtype = spec.get_dtype_for_value(value)
    
    # === String handling ===
    if dtype == 's':
        if hasattr(value, 'strftime'):
            # Convert datetime-like objects to TIFF format
            value = value.strftime("%Y:%m:%d %H:%M:%S")
        if isinstance(value, bytes):
            value = value.decode('utf-8', errors='replace')
        value = str(value)
        if not value.endswith('\x00'):
            value = value + '\x00'
        return (dtype, len(value), value)
    
    # === Byte array handling ===
    if dtype == 'B':
        if isinstance(value, bytes):
            return (dtype, len(value), value)
        if isinstance(value, (list, tuple)):
            value = bytes(value)
            return (dtype, len(value), value)
        if hasattr(value, '__array__'):
            value = np.asarray(value).tobytes()
            return (dtype, len(value), value)
        # Single byte
        return (dtype, 1, bytes([int(value)]))
    
    # === Rational handling (2I or 2i) ===
    if dtype in ('2I', '2i'):
        max_denom = 10000
        
        # Handle array-like objects (numpy, pandas, xarray, etc.)
        if hasattr(value, '__array__'):
            flat = np.asarray(value).flatten()
            rationals = []
            for v in flat:
                frac = Fraction(float(v)).limit_denominator(max_denom)
                rationals.extend([frac.numerator, frac.denominator])
            return (dtype, len(flat), tuple(rationals))
        
        # Handle lists/tuples of floats
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
        
        # Already in rational tuple format (num, denom, num, denom, ...)
        if isinstance(value, tuple) and len(value) % 2 == 0:
            return (dtype, len(value) // 2, value)
    
    # === Integer handling ===
    if dtype in ('H', 'I', 'h', 'i', 'Q', 'q'):
        if isinstance(value, (list, tuple)) or hasattr(value, '__array__'):
            arr = np.asarray(value).flatten().tolist() if hasattr(value, '__array__') else list(value)
            return (dtype, len(arr), tuple(arr) if len(arr) > 1 else arr[0])
        return (dtype, 1, int(value))
    
    # === Float handling ===
    if dtype in ('f', 'd'):
        if isinstance(value, (list, tuple)) or hasattr(value, '__array__'):
            arr = np.asarray(value).flatten().tolist() if hasattr(value, '__array__') else list(value)
            return (dtype, len(arr), tuple(arr) if len(arr) > 1 else arr[0])
        return (dtype, 1, float(value))
    
    # Fallback: return as-is with count from spec
    count = spec.count if spec.count is not None else 1
    return (dtype, count, value)

def get_cfa_pattern_codes(pattern: str) -> tuple:
    """Convert CFA pattern string to code tuple.
    
    Args:
        pattern: CFA pattern string ('RGGB', 'BGGR', 'GRBG', 'GBRG')
        
    Returns:
        Tuple of 4 codes (0=R, 1=G, 2=B), e.g., (0, 1, 1, 2) for RGGB.
        Returns (0, 1, 1, 2) as default if pattern not recognized.
    """
    return CFA_PATTERN_TO_CODES.get(pattern, (0, 1, 1, 2))


def resolve_tag(tag: Union[str, int]) -> Tuple[Optional[int], Optional[str], Optional[TagSpec]]:
    """Resolve a tag name or code to (tag_id, tag_name, spec).
    
    Args:
        tag: Either a numeric tag code (int) or tag name string
        
    Returns:
        Tuple of (tag_id, tag_name, spec) where any may be None if not found.
    """
    if isinstance(tag, int):
        tag_id = tag
        tag_name = LOCAL_TIFF_TAGS.get(tag_id)
    else:
        tag_name = tag
        tag_id = LOCAL_TIFF_TAGS.get(tag_name)
    
    spec = TIFF_TAG_TYPE_REGISTRY.get(tag_name) if tag_name else None
    return tag_id, tag_name, spec


def special_tag_format(tag_name: str, raw_value: Any, dtype: int, return_type: Optional[type]) -> Optional[Any]:
    """Handle special formatting for specific tags.
    
    Some tags require special handling beyond standard TIFF type conversion:
    - XMP: Returns XmpMetadata object for easy manipulation
    - DNGVersion/DNGBackwardVersion: Returns 4-tuple (major, minor, patch, build) for easy comparison
    
    Args:
        tag_name: Name of the tag
        raw_value: Raw value from TIFF tag
        dtype: TIFF dtype code
        return_type: Requested return type (None for auto)
    
    Returns:
        Formatted value if this is a special tag, None otherwise (use normal decoding)
    """
    # Only apply special formatting when return_type is None (auto)
    if return_type is not None:
        return None
    
    # XMP: return XmpMetadata object
    if tag_name == "XMP":
        xmp_string = decode_tag_value(tag_name, raw_value, dtype, None, str)
        if xmp_string is None:
            xmp_string = ""
        return XmpMetadata(xmp_string)
    
    # DNG Version tags: return 4-tuple (major, minor, patch, build)
    if tag_name in ("DNGVersion", "DNGBackwardVersion"):
        # Decode as bytes first
        version_bytes = decode_tag_value(tag_name, raw_value, dtype, None, bytes)
        if version_bytes is None:
            return None
        # Pad to 4 bytes if needed (null bytes may be stripped)
        padded = version_bytes + b'\x00' * (4 - len(version_bytes))
        return (padded[0], padded[1], padded[2], padded[3])
    
    return None


def decode_tag_value(
    tag_name: str,
    tag_value: Any,
    tag_dtype: int,
    spec: Optional[TagSpec] = None, 
    return_type: Optional[type] = None,
) -> Any:
    """Decode a raw TIFF value to a Python type (TIFF → Python).
    
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
        file_dtype_str = TIFF_DTYPE_TO_STR.get(tag_dtype)
        expected_dtypes = spec.dtype if isinstance(spec.dtype, list) else [spec.dtype]
        if file_dtype_str and file_dtype_str not in expected_dtypes:
            name_str = f"Tag '{tag_name}'" if tag_name else "Tag"
            logger.warning(
                f"{name_str} has dtype {file_dtype_str} (code {tag_dtype}) "
                f"but spec expects {expected_dtypes}"
            )
    
    # If no return_type specified, return raw value
    if return_type is None:
        return tag_value
    
    # Handle string conversion
    if return_type is str:
        # CFAPattern: bytes → pattern string (e.g., "RGGB")
        if tag_name == "CFAPattern" and isinstance(tag_value, bytes):
            return CFA_CODES_TO_PATTERN.get(tuple(tag_value), str(tag_value))
        # PhotometricInterpretation: enum → name string
        if tag_name == "PhotometricInterpretation":
            photometric_names = {
                PHOTOMETRIC.CFA: "CFA",
                PHOTOMETRIC.LINEAR_RAW: "LINEAR_RAW",
            }
            return photometric_names.get(tag_value, str(tag_value))
        # Handle bytes and numpy arrays (e.g., XMP)
        if isinstance(tag_value, bytes):
            return tag_value.decode('utf-8', errors='replace').rstrip('\x00')
        if isinstance(tag_value, np.ndarray):
            # Numpy array of bytes - extract and decode
            return tag_value.tobytes().decode('utf-8', errors='replace').rstrip('\x00')
        return str(tag_value)
    
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
        if isinstance(tag, str):
            tag = LOCAL_TIFF_TAGS.get(tag, tag)
        return tag in self._tags

    def remove_tag(self, tag: Union[int, str]) -> bool:
        """Remove a tag by code (int) or name (str).
        
        Args:
            tag: Tag code or name to remove
            
        Returns:
            True if tag was removed, False if it didn't exist
        """
        if isinstance(tag, str):
            tag = LOCAL_TIFF_TAGS.get(tag, tag)
        if tag in self._tags:
            del self._tags[tag]
            return True
        return False

    def copy(self) -> MetadataTags:
        """Create a deep copy of this MetadataTags instance.
        
        Returns:
            New MetadataTags instance with copied tags.
        """
        import copy
        new_instance = MetadataTags()
        # Deep copy the tags dict to avoid shared mutable objects
        new_instance._tags = copy.deepcopy(self._tags)
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
        dtype: Union[str, int], 
        count: int, 
        value: Any
    ) -> None:
        """Add a tag with explicit type specification.
        
        Args:
            name_or_code: Tag name string or numeric code
            dtype: TIFF data type string ('s', 'H', 'I', '2I', etc.) or int
            count: Number of values
            value: The tag value
        """
        if isinstance(name_or_code, str):
            tag_code = LOCAL_TIFF_TAGS[name_or_code]
        else:
            tag_code = name_or_code

        # Handle dtype parameter - can be string key or DATATYPE enum value
        if isinstance(dtype, str):
            tag_dtype = TIFF.DATA_DTYPES[dtype]
        else:
            tag_dtype = dtype

        self._tags[tag_code] = self.StoredTag(code=tag_code, dtype=tag_dtype, count=count, value=value)

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
        """Add all tags from another MetadataTags instance."""
        if other is None:
            return
        if not isinstance(other, MetadataTags):
            raise TypeError(f"Expected MetadataTags instance, got {type(other).__name__}")
        # Dict update - other's tags override existing
        for code, tag in other._tags.items():
            self.add_raw_tag(tag.code, tag.dtype, tag.count, tag.value)

    def get_xmp(self) -> Optional['XmpMetadata']:
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
        
        # Dict lookup
        if tag_id in self._tags:
            t = self._tags[tag_id]
            
            # Check for special formatting (XMP, DNGVersion, etc.)
            special_value = special_tag_format(tag_name, t.value, t.dtype, return_type)
            if special_value is not None:
                return special_value
            
            effective_type = return_type or get_native_type(t.dtype, t.count)
            
            # Use registry spec shape only if count matches tag
            shape_spec = None
            if registry_spec and registry_spec.shape and registry_spec.count == t.count:
                shape_spec = TagSpec(TIFF_DTYPE_TO_STR.get(t.dtype, 'B'), t.count, registry_spec.shape)
            
            return decode_tag_value(tag_name, t.value, t.dtype, shape_spec, effective_type)
        
        return None

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
        
        # Dict lookup
        if tag_id in self._tags:
            return self._tags[tag_id].value
        
        return None
    
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
