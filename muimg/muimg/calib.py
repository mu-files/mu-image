"""Camera calibration utilities for muimg."""

import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from muimg import DngFile, MetadataTags, write_dng
from muimg.dcp import parse_dcp_file
from muimg.common import setup_logging

logger = logging.getLogger(__name__)


def _calculate_as_shot_neutral(raw_cfa: np.ndarray, cfa_pattern: str, coordinate: Tuple[int, int]) -> Optional[List[float]]:
    """
    Calculate AsShotNeutral by demosaicing CFA data and sampling at given coordinate.
    
    Args:
        raw_cfa: Raw CFA data array
        cfa_pattern: CFA pattern string (e.g., 'RGGB')
        coordinate: (x, y) coordinate to sample
        
    Returns:
        List of 3 normalized RGB values, or None if calculation fails
    """
    try:
        import cv2
        
        # Map CFA patterns to OpenCV demosaicing codes
        BAYER_PATTERNS_TO_CV2 = {
            "RGGB": cv2.COLOR_BAYER_RG2RGB,
            "GRBG": cv2.COLOR_BAYER_GR2RGB,
            "GBRG": cv2.COLOR_BAYER_GB2RGB,
            "BGGR": cv2.COLOR_BAYER_BG2RGB,
        }
        
        if cfa_pattern not in BAYER_PATTERNS_TO_CV2:
            logger.error(f"Unsupported CFA pattern: {cfa_pattern}")
            return None
            
        # Demosaic the CFA data
        debayer_code = BAYER_PATTERNS_TO_CV2[cfa_pattern]
        
        # Ensure data is in the right format for OpenCV
        if raw_cfa.dtype == np.uint16:
            # OpenCV demosaicing works better with 16-bit data
            demosaiced = cv2.demosaicing(raw_cfa, debayer_code)
        else:
            # Convert to uint16 if needed
            raw_16bit = raw_cfa.astype(np.uint16)
            demosaiced = cv2.demosaicing(raw_16bit, debayer_code)
        
        x, y = coordinate
        
        # Check bounds
        if y >= demosaiced.shape[0] or x >= demosaiced.shape[1]:
            logger.error(f"Coordinate ({x}, {y}) is out of bounds for image shape {demosaiced.shape[:2]}")
            return None
            
        # Sample RGB values at the coordinate
        bgr_sample = demosaiced[y, x]  # Note: y is row, x is column, OpenCV returns BGR
        
        # Convert BGR to RGB by swapping R and B channels
        rgb_sample = np.array([bgr_sample[2], bgr_sample[1], bgr_sample[0]])  # BGR -> RGB
        
        # Convert to float and normalize to max value of this sample
        rgb_float = rgb_sample.astype(np.float64)
        max_value = np.max(rgb_float)
        
        if max_value == 0:
            logger.error("RGB sample has no non-zero values")
            return None
            
        # Normalize to [0, 1] range based on sample's max value
        normalized_rgb = rgb_float / max_value
        
        logger.info(f"Raw RGB sample at ({x}, {y}): {rgb_sample}")
        logger.info(f"Max value in sample: {max_value}")
        logger.info(f"Normalized RGB: {normalized_rgb}")
        
        return normalized_rgb.tolist()
        
    except ImportError:
        logger.error("OpenCV (cv2) is required for demosaicing but not available")
        return None
    except Exception as e:
        logger.error(f"Error calculating AsShotNeutral: {e}")
        return None


def _extract_cfa_from_dng(dng_file: DngFile) -> Tuple[np.ndarray, str]:
    """
    Extract raw CFA data and pattern from a DNG file.
    
    Based on the cfa_from_dng function from muallsky.tone_adjust.
    
    Args:
        dng_file: Open DngFile object
        
    Returns:
        Tuple of (raw_cfa_array, cfa_pattern_string)
        
    Raises:
        ValueError: If the DNG file cannot be processed or has no CFA data
    """
    try:
        # 1. Get info about raw pages to find the CFA data
        raw_pages_info = dng_file.get_raw_pages_info()
        if not raw_pages_info:
            raise ValueError("No raw pages found in DNG")

        # 2. Find the first page with CFA photometric interpretation
        cfa_page_details = None
        for page_id_loop, shape, tags_loop in raw_pages_info:
            if tags_loop.get("PhotometricInterpretation") == "CFA":
                cfa_page_details = (page_id_loop, tags_loop)
                break

        if cfa_page_details is None:
            raise ValueError("No page with CFA interpretation found in DNG")

        page_id, tags = cfa_page_details
        
        # 3. Get the CFA pattern from the tags
        cfa_pattern_value = tags.get("CFAPattern")
        if cfa_pattern_value is None:
            raise ValueError(f"Missing CFAPattern tag for page {page_id}")

        # 4. Get the CFA data array
        raw_cfa = dng_file.get_raw_cfa_by_id(page_id)
        if raw_cfa is None:
            raise ValueError(
                f"Failed to retrieve raw CFA data for page {page_id}"
            )

    except Exception as e:
        raise ValueError(f"Error processing DNG file: {e}") from e

    return raw_cfa, cfa_pattern_value


def calib_prep(
    input_dng: Path,
    dcp_file: Optional[Path] = None,
    output_dng: Path = None,
    force: bool = False,
    analog_balance: Optional[List[float]] = None,
    analog_balance_compose: Optional[List[float]] = None,
    neutral_coordinate: Optional[Tuple[int, int]] = None,
    neutral_color: Optional[List[float]] = None
) -> int:
    """
    Prepare calibrated DNG by embedding color matrix from DCP file.
    
    Args:
        input_dng: Path to input DNG file
        dcp_file: Path to DCP file containing color matrix and illuminant (optional)
        output_dng: Path to output DNG file
        force: Whether to overwrite existing output file
        
    Returns:
        Exit code (0 for success, 1 for error)
        
    Note:
        If dcp_file is not provided, uses identity matrix and unknown illuminant.
    """
    try:
        # Check if output file exists and force flag
        if output_dng.exists() and not force:
            logger.error(f"Output file {output_dng} already exists. Use --force to overwrite.")
            return 1
        
        # Parse DCP file to extract color matrix and illuminant, or use defaults
        if dcp_file is not None:
            logger.info(f"Parsing DCP file: {dcp_file}")
            dcp_profile = parse_dcp_file(dcp_file)
            
            # Get primary color matrix and illuminant from DCP
            color_matrix, illuminant = dcp_profile.get_primary_matrix_and_illuminant()
            
            if color_matrix is None or illuminant is None:
                logger.error("No valid color matrix or illuminant found in DCP file")
                return 1
            
            # Normalize matrix by [0][0] element for consistent scaling
            normalization_factor = color_matrix[0][0]
            if normalization_factor != 0:
                color_matrix = color_matrix / normalization_factor
                logger.info(f"Normalized color matrix by factor: {normalization_factor:.6f}")
            else:
                logger.warning("Color matrix [0][0] element is zero, skipping normalization")
            
            # Compose analog balance into color matrix if provided
            if analog_balance_compose is not None:
                logger.info(f"Composing analog balance into color matrix: {analog_balance_compose}")
                # Create diagonal matrix from analog balance
                analog_balance_diag = np.diag(analog_balance_compose)
                logger.info(f"Analog balance diagonal matrix:\n{analog_balance_diag}")
                
                # Compose: diag(analog_balance) * color_matrix
                color_matrix = analog_balance_diag @ color_matrix
                logger.info(f"Composed color matrix:\n{color_matrix}")
            
            logger.info(f"Found color matrix for illuminant: {dcp_profile.get_illuminant_name(illuminant)}")
            logger.info(f"Color matrix shape: {color_matrix.shape}")
            logger.info(f"Final color matrix:\n{color_matrix}")
        else:
            # Use identity matrix and unknown illuminant when no DCP provided
            logger.info("No DCP file provided, using identity matrix and unknown illuminant")
            color_matrix = np.eye(3, dtype=np.float64)  # 3x3 identity matrix
            illuminant = 0  # CALIBRATIONILLUMINANT_UNKNOWN
            dcp_profile = None  # No profile data available
            
            logger.info(f"Using identity color matrix: {color_matrix.shape}")
            logger.info(f"Using unknown illuminant: {illuminant}")
        
        # Load input DNG file and extract CFA data
        logger.info(f"Loading input DNG: {input_dng}")
        with DngFile(input_dng) as dng:
            # Extract CFA data using the same logic as muallsky
            raw_cfa, cfa_pattern = _extract_cfa_from_dng(dng)
            
            # Infer bits per pixel from CFA data type
            if raw_cfa.dtype == np.uint8:
                bits_per_pixel = 8
            elif raw_cfa.dtype == np.uint16:
                bits_per_pixel = 16
            elif raw_cfa.dtype == np.uint32:
                bits_per_pixel = 32
            else:
                # Default fallback, could also check dng.bits_per_pixel
                bits_per_pixel = 16
                logger.warning(f"Unknown CFA dtype {raw_cfa.dtype}, defaulting to {bits_per_pixel}-bit")
            
            logger.info(f"Input DNG: {raw_cfa.shape}, {bits_per_pixel}-bit, {cfa_pattern}")
        
        # Create external metadata with new color matrix and illuminant
        external_metadata = MetadataTags()
        
        # Add color matrix as rational values
        external_metadata.add_matrix_as_rational_tag("ColorMatrix1", color_matrix)
        
        # Add calibration illuminant
        external_metadata.add_tag(("CalibrationIlluminant1", "H", 1, illuminant))
        
        # Add analog balance if provided
        if analog_balance is not None:
            logger.info(f"Adding AnalogBalance: {analog_balance}")
            # Convert float array to rational tuple (numerator, denominator pairs)
            denominator = 10000
            rational_tuple = tuple(
                item for val in analog_balance 
                for item in (int(val * denominator), denominator)
            )
            external_metadata.add_tag(("AnalogBalance", "2I", 3, rational_tuple))
        
        # Handle AsShotNeutral from neutral coordinate or neutral color
        if neutral_coordinate is not None:
            as_shot_neutral = _calculate_as_shot_neutral(raw_cfa, cfa_pattern, neutral_coordinate)
            if as_shot_neutral is not None:
                logger.info(f"Calculated AsShotNeutral: {as_shot_neutral}")
                # Convert to TIFF rational format (numerator, denominator pairs)
                as_shot_rational = []
                for value in as_shot_neutral:
                    # Use a denominator that preserves precision
                    denominator = 1000000
                    numerator = int(value * denominator)
                    as_shot_rational.extend([numerator, denominator])
                
                external_metadata.add_tag(("AsShotNeutral", "2I", 3, as_shot_rational))
        elif neutral_color is not None:
            logger.info(f"Using provided neutral color: {neutral_color}")
            # Convert to TIFF rational format (numerator, denominator pairs)
            as_shot_rational = []
            for value in neutral_color:
                # Use a denominator that preserves precision
                denominator = 1000000
                numerator = int(value * denominator)
                as_shot_rational.extend([numerator, denominator])
            
            external_metadata.add_tag(("AsShotNeutral", "2I", 3, as_shot_rational))
        
        # Add DCP profile information if available
        '''
        if dcp_profile is not None:
            if dcp_profile.profile_description:
                external_metadata.add_string_tag("ProfileDescription", dcp_profile.profile_description)
            
            if dcp_profile.profile_copyright:
                external_metadata.add_string_tag("ProfileCopyright", dcp_profile.profile_copyright)
        '''
        
        # Write output DNG with new color matrix and illuminant
        logger.info(f"Writing calibrated DNG: {output_dng}")
        write_dng(
            raw_data=raw_cfa,
            destination_file=output_dng,
            bits_per_pixel=bits_per_pixel,
            cfa_pattern=cfa_pattern,
            external_camera_profile=external_metadata
        )
        
        logger.info(f"Successfully created calibrated DNG: {output_dng}")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid file format: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error in calib_prep: {e}")
        return 1


def calib_compare(
    dcp_file1: Path,
    dcp_file2: Path
) -> int:
    """
    Compare two DCP files and analyze their color matrices.
    
    Args:
        dcp_file1: Path to first DCP file
        dcp_file2: Path to second DCP file
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        logger.info(f"Comparing DCP files:")
        logger.info(f"  File 1: {dcp_file1}")
        logger.info(f"  File 2: {dcp_file2}")
        
        # Parse both DCP files
        logger.info("\nParsing first DCP file...")
        dcp_profile1 = parse_dcp_file(dcp_file1)
        matrix1, illuminant1 = dcp_profile1.get_primary_matrix_and_illuminant()
        
        if matrix1 is None or illuminant1 is None:
            logger.error("No valid color matrix or illuminant found in first DCP file")
            return 1
            
        # Normalize matrix1 by [0][0] element
        if matrix1[0, 0] != 0:
            norm_factor1 = matrix1[0, 0]
            matrix1 = matrix1 / norm_factor1
            logger.info(f"Normalized matrix1 by factor: {norm_factor1:.6f}")
        else:
            logger.warning("Matrix1 [0][0] element is zero, skipping normalization")
            
        logger.info("\nParsing second DCP file...")
        dcp_profile2 = parse_dcp_file(dcp_file2)
        matrix2, illuminant2 = dcp_profile2.get_primary_matrix_and_illuminant()
        
        if matrix2 is None or illuminant2 is None:
            logger.error("No valid color matrix or illuminant found in second DCP file")
            return 1
            
        # Normalize matrix2 by [0][0] element
        if matrix2[0, 0] != 0:
            norm_factor2 = matrix2[0, 0]
            matrix2 = matrix2 / norm_factor2
            logger.info(f"Normalized matrix2 by factor: {norm_factor2:.6f}")
        else:
            logger.warning("Matrix2 [0][0] element is zero, skipping normalization")

        # Print illuminant information
        logger.info("\n=== ILLUMINANT COMPARISON ===")
        logger.info(f"File 1 illuminant: {dcp_profile1.get_illuminant_name(illuminant1)} ({illuminant1})")
        logger.info(f"File 2 illuminant: {dcp_profile2.get_illuminant_name(illuminant2)} ({illuminant2})")
        
        # Print matrix information
        logger.info("\n=== MATRIX 1 (File 1) ===")
        logger.info(f"Shape: {matrix1.shape}")
        logger.info(f"Matrix:\n{matrix1}")
        
        logger.info("\n=== MATRIX 2 (File 2) ===")
        logger.info(f"Shape: {matrix2.shape}")
        logger.info(f"Matrix:\n{matrix2}")
        
        # the matrices are xyz to rgb so for purposes of below invert them
        matrix1 = np.linalg.inv(matrix1)
        matrix2 = np.linalg.inv(matrix2)

        # Calculate inv(matrix2) * matrix1
        try:
            matrix2_inv = np.linalg.inv(matrix2)
            transform_matrix = matrix2_inv @ matrix1
            
            logger.info("\n=== TRANSFORM MATRIX: inv(Matrix2) * Matrix1 ===")
            logger.info(f"This matrix transforms from File1's color space to File2's color space")
            logger.info(f"Shape: {transform_matrix.shape}")
            logger.info(f"Transform matrix:\n{transform_matrix}")
            
            # Calculate and display some useful metrics
            determinant = np.linalg.det(transform_matrix)
            logger.info(f"\n=== ANALYSIS ===")
            logger.info(f"Transform matrix determinant: {determinant:.6f}")
            
            # Check if matrices are similar (identity-like transform)
            identity_diff = np.linalg.norm(transform_matrix - np.eye(3))
            logger.info(f"Difference from identity: {identity_diff:.6f}")
            
            if identity_diff < 0.1:
                logger.info("✓ Matrices are very similar (small color space difference)")
            elif identity_diff < 0.5:
                logger.info("~ Matrices have moderate differences")
            else:
                logger.info("! Matrices have significant differences")
                
        except np.linalg.LinAlgError as e:
            logger.error(f"Cannot compute matrix inverse: {e}")
            logger.error("Matrix2 may be singular (non-invertible)")
            return 1

        # Find closest diagonal matrix such that matrix1 = matrix2 * diagonal_matrix
        logger.info("\n=== DIAGONAL MATRIX ANALYSIS ===")
        logger.info("Finding closest diagonal matrix D such that Matrix1 = Matrix2 * D")
        
        # Initialize diagonal elements array
        diagonal_elements = np.zeros(matrix1.shape[1])
        
        for i in range(matrix1.shape[1]):
            # Get the i-th column of matrix1 and matrix2
            matrix1_col = matrix1[:, i]
            matrix2_col = matrix2[:, i]
            
            # Calculate the least squares solution for the i-th diagonal element
            if np.linalg.norm(matrix2_col) != 0:
                diagonal_elements[i] = np.dot(matrix1_col, matrix2_col) / np.dot(matrix2_col, matrix2_col)
            else:
                # Handle cases where matrix2_col is zero
                diagonal_elements[i] = 0
                logger.warning(f"Matrix2 column {i} is zero, setting diagonal element to 0")
        
        # Construct the diagonal matrix
        diagonal_elements *= 1.0 / diagonal_elements[1]
        diagonal_matrix = np.diag(diagonal_elements)
        
        logger.info(f"Diagonal matrix D:")
        logger.info(f"Shape: {diagonal_matrix.shape}")
        logger.info(f"Matrix:\n{diagonal_matrix}")
        logger.info(f"Diagonal elements: [{diagonal_elements[0]:.6f}, {diagonal_elements[1]:.6f}, {diagonal_elements[2]:.6f}]")
        
        # Verify the approximation: matrix2 * diagonal_matrix should approximate matrix1
        approximation = matrix2 @ diagonal_matrix
        approximation_error = np.linalg.norm(matrix1 - approximation)
        
        logger.info(f"\n=== DIAGONAL APPROXIMATION QUALITY ===")
        logger.info(f"Matrix2 * D approximation of Matrix1:")
        logger.info(f"Approximation matrix:\n{approximation}")
        logger.info(f"Approximation error (Frobenius norm): {approximation_error:.6f}")
        
        if approximation_error < 0.01:
            logger.info("✓ Excellent diagonal approximation - matrices differ mainly by channel scaling")
        elif approximation_error < 0.1:
            logger.info("~ Good diagonal approximation - matrices have moderate non-diagonal differences")
        else:
            logger.info("! Poor diagonal approximation - matrices have significant structural differences")
            
        # Interpret the diagonal elements
        logger.info(f"\n=== INTERPRETATION ===")
        logger.info(f"Red channel scaling factor: {diagonal_elements[0]:.3f}")
        logger.info(f"Green channel scaling factor: {diagonal_elements[1]:.3f}")
        logger.info(f"Blue channel scaling factor: {diagonal_elements[2]:.3f}")
        
        # Check if it's close to a uniform scaling
        mean_scaling = np.mean(diagonal_elements)
        scaling_variance = np.var(diagonal_elements)
        logger.info(f"Mean scaling factor: {mean_scaling:.3f}")
        logger.info(f"Scaling variance: {scaling_variance:.6f}")
        
        if scaling_variance < 0.01:
            logger.info(f"✓ Nearly uniform scaling (~{mean_scaling:.2f}x across all channels)")
        else:
            logger.info("~ Non-uniform scaling - different channels scaled differently")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in calib_compare: {e}")
        return 1
