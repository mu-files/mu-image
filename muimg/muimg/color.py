import cv2
import numpy as np

from typing import Dict

def process_cfa_raw(image_data: np.ndarray, tags: Dict) -> np.ndarray:
    """
    Processes raw CFA (Color Filter Array) image data.
    (Implementation to be added)
    """

    # Validate image_data
    if image_data.ndim != 2:
        raise ValueError(
            f"image_data must be a 2D array, but got {image_data.ndim} dimensions."
        )
    if image_data.size == 0:
        raise ValueError("image_data must not be empty.")

    # Define Bayer patterns and their corresponding OpenCV conversion codes
    # This dictionary was already present in the user's code.
    # TODO: test the other debayering algorigthms in CV2
    BAYER_PATTERNS_TO_CV2 = {
        "RGGB": cv2.COLOR_BAYER_RG2RGB_EA,
        "BGGR": cv2.COLOR_BAYER_BG2RGB_EA,
        "GRBG": cv2.COLOR_BAYER_GR2RGB_EA,
        "GBRG": cv2.COLOR_BAYER_GB2RGB_EA,
    }

    # Get and validate CFA pattern from tags
    # We expect a key like 'CFAPattern' in the tags dictionary holding a string like "RGGB".
    # This key name might need adjustment based on the actual content of 'tags'.
    cfa_pattern_key = 'CFAPattern' 
    cfa_pattern = tags.get(cfa_pattern_key) 
    if cfa_pattern is None:
        raise ValueError(
            f"CFA pattern string not found in tags. Expected key '{cfa_pattern_key}' in the tags dictionary."
        )
    if not isinstance(cfa_pattern, str):
        raise TypeError(
            f"CFA pattern (from tags['{cfa_pattern_key}']) must be a string, "
            f"but got {type(cfa_pattern).__name__}."
        )
    if cfa_pattern not in BAYER_PATTERNS_TO_CV2:
        raise ValueError(
            f"Invalid CFA pattern: '{cfa_pattern}'. "
            f"Supported patterns are: {list(BAYER_PATTERNS_TO_CV2.keys())}."
        )

    debayer_code = BAYER_PATTERNS_TO_CV2[cfa_pattern]

    #debayer_image = cv2.cvtColor(image_data, debayer_code)
    debayer_image = cv2.demosaicing(image_data, debayer_code)

    color_adapted_image = process_linear_raw(debayer_image, tags)

    return color_adapted_image

def to_linear(linear):
	less = linear <= 0.04045
	linear[less] = linear[less] * (1. / 12.92)
	linear[~less] = np.power((linear[~less] + 0.055) * (1./ 1.055), 2.4)
	return linear

def from_linear(linear):
    srgb = linear.copy()
    less = srgb <= 0.0031308
    srgb[less] = srgb[less] * 12.92
    srgb[~less] = 1.055 * np.power(srgb[~less], 1.0 / 2.4) - 0.055
    return srgb

class BradfordAdaptation:
    """
    Provides matrices and data for Bradford chromatic adaptation.
    The transformation pipeline often involves:
    RGB_out = M_sRGB_from_XYZ @ M_Bradford_inv @ M_diag_cone_response_scaling @ M_Bradford @ M_XYZ_from_CameraRGB @ RGB_in
    where M_diag_cone_response_scaling is diag([X_dst/X_src, Y_dst/Y_src, Z_dst/Z_src])
    after source and destination white points (X_src, Y_src, Z_src) and (X_dst, Y_dst, Z_dst)
    have been transformed into cone response space using M_Bradford.
    """

    # Bradford Cone Primary Matrix (transforms XYZ to LMS-like cone response)
    BRADFORD_MATRIX = np.array([
        [ 0.8951,  0.2664, -0.1614],
        [-0.7502,  1.7135,  0.0367],
        [ 0.0389, -0.0685,  1.0296]
    ], dtype=np.float32)

    # Inverse Bradford Cone Primary Matrix (transforms LMS-like cone response back to XYZ)
    BRADFORD_MATRIX_INV = np.linalg.inv(BRADFORD_MATRIX)

    # Standard Illuminant XYZ values (Y is normalized to 1.0)
    ILLUMINANT_XYZ = {
        "A":   np.array([1.09850, 1.00000, 0.35585], dtype=np.float32),  # Incandescent/Tungsten
        "B":   np.array([0.99072, 1.00000, 0.85223], dtype=np.float32),  # Obsolete, Direct sunlight at noon
        "C":   np.array([0.98074, 1.00000, 1.18232], dtype=np.float32),  # Obsolete, Average/North sky daylight
        "D50": np.array([0.96422, 1.00000, 0.82521], dtype=np.float32),  # Horizon Light. ICC profile PCS.
        "D55": np.array([0.95682, 1.00000, 0.92149], dtype=np.float32),  # Mid-morning/Mid-afternoon Daylight
        "D65": np.array([0.95047, 1.00000, 1.08883], dtype=np.float32),  # Average Daylight (sRGB/Rec.709 white point)
        "D75": np.array([0.94972, 1.00000, 1.22638], dtype=np.float32),  # North sky Daylight
        "E":   np.array([1.00000, 1.00000, 1.00000], dtype=np.float32),  # Equal energy illuminant
        "F1":  np.array([0.92836, 1.00000, 1.03591], dtype=np.float32),  # Daylight Fluorescent
        "F2":  np.array([0.99186, 1.00000, 0.67393], dtype=np.float32),  # Cool White Fluorescent (CWF)
        "F7":  np.array([0.95041, 1.00000, 1.08747], dtype=np.float32),  # Broad-Band Daylight Fluorescent
        "F11": np.array([1.00962, 1.00000, 0.64350], dtype=np.float32)   # Narrow-Band White Fluorescent
    }
    
    @staticmethod
    def get_adaptation_matrix(source_illuminant_name: str, destination_illuminant_name: str) -> np.ndarray:
        """
        Calculates the Bradford chromatic adaptation matrix to transform colors
        from the source illuminant's white point to the destination illuminant's white point.

        Args:
            source_illuminant_name: Name of the source illuminant (e.g., "D65", "A").
            destination_illuminant_name: Name of the destination illuminant (e.g., "D50").

        Returns:
            A 3x3 NumPy array representing the chromatic adaptation matrix.

        Raises:
            ValueError: If source or destination illuminant name is not found in ILLUMINANT_XYZ.
        """
        try:
            source_xyz = BradfordAdaptation.ILLUMINANT_XYZ[source_illuminant_name]
            destination_xyz = BradfordAdaptation.ILLUMINANT_XYZ[destination_illuminant_name]
        except KeyError as e:
            raise ValueError(f"Illuminant name {e} not found in ILLUMINANT_XYZ dictionary.") from e

        # Transform source and destination white points to cone response space
        source_cone_response = BradfordAdaptation.BRADFORD_MATRIX @ source_xyz
        destination_cone_response = BradfordAdaptation.BRADFORD_MATRIX @ destination_xyz

        # Calculate scaling factors in cone response space
        # Using np.divide for safe division, out=np.ones_like to avoid NaN for 0/0.
        scaling_factors = np.divide(destination_cone_response, source_cone_response, 
                                    out=np.ones_like(destination_cone_response), 
                                    where=source_cone_response!=0)

        diagonal_scaling_matrix = np.diag(scaling_factors)

        # Combine matrices: M_adapt = M_bradford_inv @ M_diag_scale @ M_bradford
        adaptation_matrix = BradfordAdaptation.BRADFORD_MATRIX_INV @ diagonal_scaling_matrix @ BradfordAdaptation.BRADFORD_MATRIX
        return adaptation_matrix

def camera_to_rgb_matrix() -> np.ndarray:
    
    # adapted from Adobe's DNG C++ SDK
    # for now just support a single colormatrix

    # in the SDK - XYZ intermediate space is called PCS
    # in dng_render.cpp fCameraToRGB is the final matrix that we are creating here
    # it is composed of RGB_space_from_PCS * CameraToPCS
    #     here RGB_space_from_PCS can be found in second table (since
    #     we are working in D50) on
    #     http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    # 
    
    pass

def process_linear_raw(image_data: np.ndarray, tags: Dict) -> np.ndarray:
    """
    Processes linear raw image data (demosaiced but not color corrected).
    Applies ColorMatrix1 and then an XYZ to sRGB conversion.
    Assumes image_data is in camera native RGB space.
    """

    XYZ_to_sRGB_D65 = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ], dtype=np.float32)

    # compute white balance adjustment
    as_shot = tags.get("AsShotNeutral")
    if as_shot is not None and hasattr(as_shot, '__len__') and len(as_shot) == 6:
        # Minimal parsing for 6-value flattened rational (n1, d1, n2, d2, n3, d3)
        # Assumes valid numeric inputs and non-zero denominators for prototyping.
        as_shot_rgb_gains = np.array([
            as_shot[0] / as_shot[1],
            as_shot[2] / as_shot[3],
            as_shot[4] / as_shot[5]
        ], dtype=np.float32)
        white_balance_matrix = np.eye(3) # np.diag(1.0/as_shot_rgb_gains)
        white_balance_matrix = white_balance_matrix / white_balance_matrix[1,1]
    else:
        # Fallback to identity matrix if AsShotNeutral is missing, not a list/tuple, or not length 3 or 6.
        white_balance_matrix = np.eye(3, dtype=np.float32)

    # color matrix in the DNG is XYZ to camera so invert here
    color_matrix_1 = tags.get("ColorMatrix1")
    camera_to_xyz = np.linalg.inv(color_matrix_1)

    # adapt to SRGB reference white
    adaptation = BradfordAdaptation.get_adaptation_matrix("D55", "D65")

    transform_matrix = XYZ_to_sRGB_D65 @ adaptation @ camera_to_xyz @ white_balance_matrix

    # Normalize Input Data to [0, 1]
    processed_data = image_data.copy() 

    processed_data = processed_data * (1./ 65536.0)
        
    processed_data = processed_data - 512./65536.
    processed_data = np.clip(processed_data, 0, 2.0)
    processed_data = processed_data * 65536. / (65536.-512.)

    original_shape = processed_data.shape
    pixels_flat = processed_data.reshape(-1, 3)  # Shape: (N, 3)

    # Each row in pixels_flat is an (R,G,B) vector.
    # We want to compute M * p.T for each pixel p, which is (p @ M.T).T
    # So, transformed_pixels_flat_T = transform_matrix @ pixels_flat.T
    # transformed_pixels_flat = transformed_pixels_flat_T.T
    # This is equivalent to: transformed_pixels_flat = pixels_flat @ transform_matrix.T
    transformed_pixels_flat = pixels_flat @ transform_matrix.T

    transformed_pixels_flat = from_linear(transformed_pixels_flat)

    transformed_pixels_flat = transformed_pixels_flat * 65536.0

    # Reshape back to original image dimensions (H, W, 3)
    transformed_image = transformed_pixels_flat.reshape(original_shape)

    # Clamp values to the 16-bit range [0, 65535]
    transformed_image_clamped = np.clip(transformed_image, 0, 2**16 - 1)

    # Convert to uint16
    output_image = transformed_image_clamped.astype(np.uint16)

    # Note: This returns linear sRGB, now as uint16. For display, sRGB gamma correction would be needed.


class ToneCurve:
    """
    Represents a tone curve with 5 control points.
    
    The curve uses 8-bit values (0-255) for compatibility with XMP tone curve format,
    but can be normalized to 0-1 range for use with Core Image filters.
    """
    
    def __init__(self, points_or_string=None):
        """
        Initialize ToneCurve.
        
        Args:
            points_or_string: Optional. Can be:
                - None: Initialize with default linear curve (0,0), (255,255)
                - str: Parse from string format "(0,0),(255,255)"
                - list: Use provided points directly
        """
        if points_or_string is None:
            # Default linear curve
            self.points = [
                (0, 0),
                (255, 255)  # Using 255 instead of 256 for proper 8-bit range
            ]
        elif isinstance(points_or_string, str):
            # Parse from string format
            self._parse_from_string(points_or_string)
        elif isinstance(points_or_string, list):
            # Use provided points directly
            self.points = points_or_string
        else:
            raise TypeError(f"ToneCurve constructor expects None, str, or list, got {type(points_or_string)}")
    
    def _parse_from_string(self, string_data: str):
        """Helper method to parse points from string format."""
        import re
        
        # Extract coordinate pairs using regex
        tuple_pattern = r'\((\d+),(\d+)\)'
        matches = re.findall(tuple_pattern, string_data)
        
        if not matches:
            raise ValueError(f"No valid coordinate pairs found in tone curve data: {string_data}")
        
        # Convert string coordinates to integer tuples
        self.points = [(int(x), int(y)) for x, y in matches]
    

    
    def to_normalized(self):
        """
        Convert 8-bit points to normalized 0-1 range for Core Image.
        
        Returns:
            List of (x, y) tuples with values normalized to 0-1 range.
        """
        return [(x / 255.0, y / 255.0) for x, y in self.points]
    
    @classmethod
    def from_scurve(cls, strength: float):
        """
        Create a new ToneCurve instance with S-curve adjustment applied.
        
        Args:
            strength: Float between 0.0 and 1.0, where 0.0 is no adjustment
                     and 1.0 is maximum S-curve effect.
        
        Returns:
            ToneCurve: New instance with S-curve applied.
        """
        curve = cls()
        
        # Clamp strength to valid range
        s = min(max(float(strength), 0.0), 1.0)
        
        # Constants matching the Core Image implementation
        SHADOW_PULL_FACTOR = 0.3
        HIGHLIGHT_PUSH_FACTOR = 0.015
        
        # Convert the Core Image normalized coordinates to 8-bit values
        # Core Image uses 0-1 range, we use 0-255 range
        curve.points = [
            (0, 0),  # Black point stays fixed
            (int(0.53 * 255), int((0.53 - s * SHADOW_PULL_FACTOR) * 255)),  # Shadow adjustment
            (int(0.73 * 255), int(0.73 * 255)),  # Midtone anchor point
            (int(0.90 * 255), int((0.90 + s * HIGHLIGHT_PUSH_FACTOR) * 255)),  # Highlight adjustment  
            (255, 255)  # White point stays fixed
        ]
        
        # Ensure output values stay within valid 8-bit range
        curve.points = [
            (x, min(max(y, 0), 255)) for x, y in curve.points
        ]
        
        return curve
    
    def __str__(self) -> str:
        """
        Convert ToneCurve to string format.
        
        Returns:
            String representation in format "(0,0),(56,30),(124,125),(188,212),(255,255)"
        """
        tuple_strings = [f"({x},{y})" for x, y in self.points]
        return ','.join(tuple_strings)
    
    def __repr__(self):
        return f"ToneCurve(points={self.points})"

