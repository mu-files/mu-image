"""TIFF/DNG metadata support classes.

This module provides classes for creating and parsing TIFF/DNG metadata tags.
"""
import logging
import numpy as np

from tifffile import PHOTOMETRIC, TIFF
from typing import Optional, Union, Dict, Any, Type

logger = logging.getLogger(__name__)

BAYER_PATTERN_MAP = {
    "RGGB": (0, 1, 1, 2),  # R G / G B
    "BGGR": (2, 1, 1, 0),  # B G / G R
    "GRBG": (1, 0, 2, 1),  # G R / B G
    "GBRG": (1, 2, 0, 1),  # G B / R G
}

# Inverse mapping from 2x2 CFA codes to string key
INVERSE_BAYER_PATTERN_MAP = {v: k for k, v in BAYER_PATTERN_MAP.items()}

# Reverse lookup for CFAPattern bytes to string
_BAYER_PATTERN_BYTES_TO_STR = {bytes(v): k for k, v in BAYER_PATTERN_MAP.items()}

# Matrix tag names that require special conversion
_MATRIX_TAG_NAMES = {
    "ColorMatrix1", "ColorMatrix2", "ColorMatrix3",
    "ForwardMatrix1", "ForwardMatrix2", "ForwardMatrix3",
    "CameraCalibration1", "CameraCalibration2", "CameraCalibration3",
}


def translate_dng_tag(page_tag) -> Any:
    """Translate a raw TIFF tag to a usable Python type.
    
    Args:
        page_tag: A tifffile TiffTag object from page.tags[tag_id].
                  Has attributes: name (str), value (Any), dtype (int).
                  dtype codes: 5=RATIONAL, 10=SRATIONAL.
    
    Returns:
        Translated value: matrices as np.ndarray, CFAPattern as string,
        PhotometricInterpretation as string, rationals as float arrays.
    """
    tag_name = page_tag.name
    tag_value = page_tag.value
    tag_dtype = page_tag.dtype

    # Handle special cases first
    if tag_name == "CFAPattern":
        if isinstance(tag_value, bytes):
            tag_value = _BAYER_PATTERN_BYTES_TO_STR.get(tag_value, tag_value)
    elif tag_name == "PhotometricInterpretation":
        if tag_value == PHOTOMETRIC.CFA:
            tag_value = "CFA"
        elif tag_value == PHOTOMETRIC.LINEAR_RAW:
            tag_value = "LINEAR_RAW"
    elif tag_name in _MATRIX_TAG_NAMES:
        tag_value = _rational_tuple_to_matrix(tag_value)
    # Auto-convert RATIONAL (5) and SRATIONAL (10) types to float arrays
    elif tag_dtype in (5, 10):
        if isinstance(tag_value, tuple) and len(tag_value) % 2 == 0:
            tag_value = np.array([
                tag_value[i] / tag_value[i+1] if tag_value[i+1] != 0 else 0.0
                for i in range(0, len(tag_value), 2)
            ])

    return tag_value


def _rational_tuple_to_matrix(rational_tuple: tuple) -> np.ndarray:
    """Convert a tuple of rational pairs to a 3x3 numpy matrix.
    
    Args:
        rational_tuple: Flat tuple of (num, denom, num, denom, ...) pairs,
                       typically 18 values for a 3x3 matrix.
    
    Returns:
        3x3 numpy array of floats.
    """
    floats = [
        rational_tuple[i] / rational_tuple[i+1] if rational_tuple[i+1] != 0 else 0.0
        for i in range(0, len(rational_tuple), 2)
    ]
    return np.array(floats).reshape(3, 3)


# helper class to convert create a list of tags for tifffile.TiffWriter
class MetadataTags:

    '''
    TIFF.DATA_DTYPES (2nd argument to add_tag) used in tifffile.py below have following mapping:
        'B': unsigned byte
        's': ascii string
        'H': unsigned short
        'I': unsigned long
        '2I': unsigned rational
        'b': signed byte
        'h': signed short
        'i': signed long
        '2i': signed rational
        'f': float
        'd': double
    '''

    @staticmethod
    def _matrix_to_rational_tuple(matrix: np.ndarray, denominator: int = 10000) -> tuple:
        """Converts a NumPy float matrix to a flat tuple of (numerator, denominator) pairs."""
        # Flatten the matrix and use the common rational conversion helper
        flat_array = matrix.flatten()
        rational_list = MetadataTags.float_array_to_rationals(flat_array, max_denominator=denominator)
        return tuple(rational_list)

    @staticmethod
    def float_array_to_rationals(float_array, max_denominator: int = 10000):
        """Convert a list/array of floats to TIFF rational format using Fraction for precision."""
        from fractions import Fraction
        
        rationals = []
        for val in float_array:
            frac = Fraction(val).limit_denominator(max_denominator)
            rationals.extend([frac.numerator, frac.denominator])
        return rationals

    def __init__(self):
        self._tags = []
        self._xmp_data = None  # Store XMP dict for retrieval

    def add_tag(self, tag):
        tag_code = -1

        if isinstance(tag[0], str):
            tag_code = TIFF.TAGS[tag[0]]
        elif isinstance(tag[0], int):
            tag_code = tag[0]

        # Handle dtype parameter - can be string key or DATATYPE enum value
        if isinstance(tag[1], str):
            tag_dtype = TIFF.DATA_DTYPES[tag[1]]
        else:
            # Assume it's already a DATATYPE enum value
            tag_dtype = tag[1]

        tag_formatted_contents = (tag_code, tag_dtype, tag[2], tag[3], False)
        
        # Check for duplicates and overwrite if one is found, else append
        for i, existing_tag in enumerate(self._tags):
            if existing_tag[0] == tag_code:
                self._tags[i] = tag_formatted_contents
                return

        self._tags.append(tag_formatted_contents)

    def add_string_tag(self, tag_name_str, string_value):
        """Helper to add a standard ASCII string tag with null termination."""
        string_value_with_null = string_value + "\x00"
        length = len(string_value_with_null)
        self.add_tag((tag_name_str, "s", length, string_value_with_null))

    def extend(self, other: "MetadataTags") -> None:
        """Add all tags from another MetadataTags instance."""
        if not isinstance(other, MetadataTags):
            raise TypeError(f"Expected MetadataTags instance, got {type(other).__name__}")
        # Use add_tag to ensure proper duplicate handling instead of direct list extension
        for tag_tuple in other._tags:
            # tag_tuple format: (tag_code, tag_dtype, count, value, writeonce)
            # Convert to add_tag format: (tag_name_or_code, dtype, count, value)
            self.add_tag((tag_tuple[0], tag_tuple[1], tag_tuple[2], tag_tuple[3]))
    
    def copy(self) -> "MetadataTags":
        """Create a deep copy of this MetadataTags instance.
        
        Returns:
            New MetadataTags instance with copied tags and XMP data.
        """
        import copy
        new_instance = MetadataTags()
        # Deep copy the tags list to avoid shared mutable objects
        new_instance._tags = copy.deepcopy(self._tags)
        # Copy XMP data if present
        if self._xmp_data is not None:
            new_instance._xmp_data = self._xmp_data.copy()
        return new_instance

    def add_cfa_pattern_tag(self, cfa_pattern_key: str):
        """Helper to add the CFAPattern tag using the class's Bayer pattern map."""
        pattern_tuple = BAYER_PATTERN_MAP.get(cfa_pattern_key, BAYER_PATTERN_MAP["RGGB"])
        pattern_bytes = bytes(pattern_tuple)
        self.add_tag(("CFAPattern", "B", 4, pattern_bytes))

    def add_matrix_as_rational_tag(
        self,
        tag_name_str: str,
        float_matrix_np: np.ndarray,
        denominator: int = 10000,
    ):
        """Converts a float matrix to rationals and adds it as a tag."""
        flat_tuple_values = MetadataTags._matrix_to_rational_tuple(float_matrix_np, denominator)
        self.add_tag((tag_name_str, "2i", 9, flat_tuple_values))

    def add_float_array_as_rational_tag(
        self,
        tag_name_str: str,
        float_array,
        max_denominator: int = 10000,
    ):
        """Converts a float array to rationals and adds it as a tag."""
        rational_list = MetadataTags.float_array_to_rationals(float_array, max_denominator)
        count = len(float_array)
        self.add_tag((tag_name_str, "2I", count, tuple(rational_list)))

    def __iter__(self):
        """Iterate over tags, sorted by tag code."""
        self._tags.sort(key=lambda x: x[0])
        return iter(self._tags)
    
    def __len__(self):
        """Return the number of tags."""
        return len(self._tags)
    
    def __contains__(self, tag: Union[int, str]) -> bool:
        """Check if a tag exists by code (int) or name (str)."""
        if isinstance(tag, str):
            tag = TIFF.TAGS.get(tag, tag)
        return any(t[0] == tag for t in self._tags)
    
    def get_xmp(self) -> Optional[Dict[str, Union[str, int, float]]]:
        """Get a copy of the XMP data dictionary that was added via add_xmp().
        
        Returns:
            Copy of XMP properties dictionary, or None if add_xmp() was never called.
            Returns a copy to prevent inadvertent modification of internal state.
        """
        return self._xmp_data.copy() if self._xmp_data is not None else None

    def add_xmp(self, xmp_data: Dict[str, Union[str, int, float]]) -> None:
        """
        Add XMP metadata to this MetadataTags instance.
        
        Args:
            xmp_data: Dictionary of XMP properties to add. Keys can include namespace 
                     prefixes (e.g., 'crs:Temperature', 'tiff:Orientation') or will 
                     default to 'crs:' namespace if no prefix is provided.
                     
        Example:
            camera_metadata.add_xmp({
                'Temperature': 5500,
                'Tint': 0,
                'Exposure2012': -0.5,
                'tiff:Orientation': 1
            })
        """
        from datetime import datetime
        
        # Generate timestamp in ISO format with timezone
        now = datetime.now()
        iso_date = now.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Try to format timezone as -07:00 instead of -0700
        try:
            iso_date_tz = now.astimezone().strftime('%Y-%m-%dT%H:%M:%S%z')
            # Insert colon in timezone offset: -0700 -> -07:00
            if len(iso_date_tz) >= 5 and iso_date_tz[-5] in '+-':
                iso_date_tz = iso_date_tz[:-2] + ':' + iso_date_tz[-2:]
        except:
            iso_date_tz = None
        
        # Build XMP properties with namespace handling
        xmp_props = []
        
        # Add standard metadata - minimal required set
        standard_props = {
            'tiff:Orientation': '1',
            'dc:format': 'image/dng',
            'xmp:CreatorTool': 'muimg',
            'xmp:ModifyDate': iso_date,
            'crs:Version': '17.4',
            'crs:ProcessVersion': '15.4',
        }
        
        # Add user-provided data
        sequence_props = {}
        bag_props = {}  # New: for dc:subject style bags
        for key, value in xmp_data.items():
            # Auto-prepend 'crs:' if no namespace specified
            if ':' not in key:
                key = f'crs:{key}'
            
            # Check for bag structure (dc:subject with list of strings)
            if key == 'dc:subject' and isinstance(value, list):
                bag_props[key] = value
            # Check if value has coordinate pairs (list of 2-tuples) - needs <rdf:Seq> structure
            elif hasattr(value, 'points') and isinstance(getattr(value, 'points'), list):
                # SplineCurve object with points attribute (normalized 0-1)
                # Convert to 8-bit for XMP compatibility
                points = getattr(value, 'points')
                sequence_props[key] = [(int(x * 255), int(y * 255)) for x, y in points]
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (tuple, list)) and len(value[0]) == 2:
                # Direct list of 2-tuples
                sequence_props[key] = value
            else:
                standard_props[key] = str(value)
        
        # Format as XMP attributes
        for prop, value in standard_props.items():
            xmp_props.append(f'    {prop}="{value}"')
        
        # Build sequence structures for coordinate pairs
        sequence_xml = ""
        if sequence_props:
            sequence_elements = []
            for prop_name, points in sequence_props.items():
                # Extract namespace and property name for XML element
                if ':' in prop_name:
                    namespace, prop = prop_name.split(':', 1)
                    element_name = f'{namespace}:{prop}'
                else:
                    element_name = prop_name
                
                # Build rdf:li items from coordinate pairs
                sequence_items = []
                for x, y in points:
                    sequence_items.append(f'      <rdf:li>{x}, {y}</rdf:li>')
                
                sequence_elements.append(f'''    <{element_name}>
     <rdf:Seq>
{chr(10).join(sequence_items)}
     </rdf:Seq>
    </{element_name}>''')
            
            sequence_xml = chr(10).join(sequence_elements)

        # Build bag structures for dc:subject style keywords
        bag_xml = ""
        if bag_props:
            bag_elements = []
            for prop_name, items in bag_props.items():
                # Extract namespace and property name for XML element
                if ':' in prop_name:
                    namespace, prop = prop_name.split(':', 1)
                    element_name = f'{namespace}:{prop}'
                else:
                    element_name = prop_name
                
                # Build rdf:li items from list of strings
                bag_items = []
                for item in items:
                    bag_items.append(f'      <rdf:li>{item}</rdf:li>')
                
                bag_elements.append(f'''    <{element_name}>
     <rdf:Bag>
{chr(10).join(bag_items)}
     </rdf:Bag>
    </{element_name}>''')
            
            bag_xml = chr(10).join(bag_elements)

        # Create minimal XMP structure based on Lightroom format
        xmp_content = f'''<?xpacket begin="\\357\\273\\277" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="muimg XMP Core">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:tiff="http://ns.adobe.com/tiff/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
{chr(10).join(xmp_props)}>{sequence_xml}{bag_xml}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''
        
        xmp_bytes = xmp_content.encode('utf-8')
        self.add_tag(("XMP", "B", len(xmp_bytes), xmp_bytes))
        
        # Store the XMP data for later retrieval
        self._xmp_data = xmp_data.copy()
        
        logger.debug(f"Added XMP metadata with {len(xmp_data)} user properties")



class XmpMetadata:
    """Encapsulates XMP metadata parsing and querying for DNG files."""
    
    def __init__(self, xmp_string: str):
        """Initialize XmpMetadata from an XMP string.
        
        Args:
            xmp_string: Raw XMP metadata string from DNG file
        """
        self._attributes = self._parse(xmp_string)
    
    def _parse(self, xmp_data: str) -> Dict[str, str]:
        """Parse all XMP attributes and sequences from the XMP metadata into a dictionary.
        
        Returns:
            Dictionary mapping attribute names to values. Simple attributes map to strings
            (e.g., 'crs:Temperature': '3900'), while sequences map to comma-separated values
            (e.g., 'crs:ToneCurvePV2012': '0,0,56,30,124,125,188,212,255,255')
        """
        if not xmp_data:
            return {}
        
        import re
        
        # Dictionary to store all XMP attributes
        attributes = {}
        
        # Pattern to match XML attributes in the format namespace:attribute="value"
        # This captures attributes like crs:Temperature="3900", tiff:Orientation="1", etc.
        attribute_pattern = r'([a-zA-Z][a-zA-Z0-9]*:[a-zA-Z][a-zA-Z0-9]*)="([^"]*?)"'
        
        # Find all attribute matches
        matches = re.findall(attribute_pattern, xmp_data)
        
        for attr_name, attr_value in matches:
            attributes[attr_name] = attr_value
        

        # Pattern to match rdf:Seq structures like ToneCurvePV2012
        # Use flexible matching that handles any whitespace/newlines between tags
        # Matches: <crs:PropertyName>...<rdf:Seq>...<rdf:li>...</rdf:li>...</rdf:Seq>...</crs:PropertyName>
        # - exclude rdf: namespace
        seq_pattern = r'<((?!rdf:)[a-zA-Z][a-zA-Z0-9]*:[a-zA-Z][a-zA-Z0-9]*?)>.*?<rdf:Seq>(.*?)</rdf:Seq>.*?</\1>'
        seq_matches = re.findall(seq_pattern, xmp_data, re.DOTALL)
        
        logger.debug(f"Found {len(seq_matches)} XMP sequences")
        for seq_name, seq_content in seq_matches:
            # Extract all rdf:li values from the sequence
            li_pattern = r'<rdf:li>([^<]*?)</rdf:li>'
            li_values = re.findall(li_pattern, seq_content)
            
            # Handle different sequence types based on content structure
            processed_values = []
            for li_value in li_values:
                # Split by comma and clean up values
                coords = [coord.strip() for coord in li_value.split(',')]
                
                if len(coords) == 1:
                    # Single value (e.g., ColorVariance: "-50.000000")
                    processed_values.append(coords[0])
                elif len(coords) == 2:
                    # Coordinate pair (e.g., ToneCurve: "0, 0")
                    # Normalize 8-bit values to 0-1 for tone curve properties
                    if 'ToneCurve' in seq_name:
                        x_norm = float(coords[0]) / 255.0
                        y_norm = float(coords[1]) / 255.0
                        processed_values.append(f"({x_norm},{y_norm})")
                    else:
                        processed_values.append(f"({coords[0]},{coords[1]})")
                else:
                    # Multiple values (e.g., PointColors with 19 values)
                    # Store as bracketed list for clarity
                    processed_values.append(f"[{','.join(coords)}]")
            
            attributes[seq_name] = ','.join(processed_values)
        
        # Pattern to match rdf:Bag structures like dc:subject
        # Matches: <dc:subject>...<rdf:Bag>...<rdf:li>...</rdf:li>...</rdf:Bag>...</dc:subject>
        bag_pattern = r'<([a-zA-Z][a-zA-Z0-9]*:[a-zA-Z][a-zA-Z0-9]*?)>.*?<rdf:Bag>(.*?)</rdf:Bag>.*?</\1>'
        bag_matches = re.findall(bag_pattern, xmp_data, re.DOTALL)
        
        logger.debug(f"Found {len(bag_matches)} XMP bags")
        for bag_name, bag_content in bag_matches:
            # Extract all rdf:li values from the bag
            li_pattern = r'<rdf:li>([^<]*?)</rdf:li>'
            li_values = re.findall(li_pattern, bag_content)
            
            # Store as comma-separated values (same as sequences)
            attributes[bag_name] = ','.join(li_values)
        
        logger.debug(f"Parsed {len(attributes)} XMP attributes")
        return attributes
    
    def has_prop(self, prop: str) -> bool:
        """Check if an XMP property exists.
        
        Args:
            prop: Property name. If no namespace prefix, 'crs:' is automatically prepended.
                 Examples: 'Temperature' -> 'crs:Temperature', 'tiff:Orientation' -> 'tiff:Orientation'
        
        Returns:
            True if the property exists in XMP metadata
        """
        # Auto-prepend 'crs:' if no namespace specified
        if ':' not in prop:
            prop = f'crs:{prop}'
        return prop in self._attributes
    
    def get_prop(self, prop: str, return_type: Optional[Type] = None) -> Optional[Any]:
        """Get an XMP property value with optional type conversion.
        
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
        
        # Try to convert using the type's constructor
        try:
            return return_type(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert XMP property '{prop}' value '{value}' to type {return_type}: {e}")
            return None
