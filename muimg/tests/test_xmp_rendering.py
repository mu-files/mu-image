"""Tests for XMP-based rendering against Photoshop reference images.

Validates that XMP metadata (Temperature, Tint, Exposure2012, ToneCurvePV2012)
is correctly parsed and applied during rendering, comparing output against
Photoshop-rendered TIFF files.
"""
import pytest
import numpy as np
from pathlib import Path
import tifffile
import logging

import muimg
from conftest import compute_diff_stats

# Suppress tifffile logging about Photoshop TIFF metadata inconsistencies
logging.getLogger('tifffile').setLevel(logging.CRITICAL)


# Test data directory
XMP_TEST_DIR = Path(__file__).parent.parent.parent.parent / "mu-image-testdata" / "xmptestfiles"

# Output directory for test results
OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_xmp_rendering"


@pytest.fixture(scope="module")
def output_dir():
    """Create output directory for test artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR

# Test files with their expected active XMP properties
# Format: (dng_filename, tiff_filename, expected_active_properties)
# Using LINEAR_RAW color pattern DNGs (300x200, 6x4 color patches)
# These DNGs have DefaultBlackRender=1 to match Photoshop baseline
TEST_CASES = [
    # Color pattern tests (300x200, 6x4 patches)
    (
        "asi676mc.linearraw.uncomp.1ifds.colorpattern.none.dng",
        "asi676mc.linearraw.uncomp.1ifds.colorpattern.none.tif",
        set(),  # No XMP adjustments
    ),
    (
        "asi676mc.linearraw.uncomp.1ifds.colorpattern.exposure.dng",
        "asi676mc.linearraw.uncomp.1ifds.colorpattern.exposure.tif",
        {"exposure"},  # Only exposure adjustment
    ),
    # Full-size tests (4144x2822)
    (
        "asi676mc.linearraw.uncomp.1ifds.none.dng",
        "asi676mc.linearraw.uncomp.1ifds.none.tif",
        set(),  # No XMP adjustments
    ),
    (
        "asi676mc.linearraw.uncomp.1ifds.exposure.dng",
        "asi676mc.linearraw.uncomp.1ifds.exposure.tif",
        {"exposure"},  # Only exposure adjustment
    ),
    # Linear gradient tests (256x4096)
    (
        "linear_gradient_test.exposure.dng",
        "linear_gradient_test.exposure.tif",
        {"exposure"},  # Linear gradient with exposure adjustment
    ),
]


def is_tone_curve_linear(curve: list) -> bool:
    """Check if a tone curve is linear (identity curve).
    
    Args:
        curve: List of (x, y) tuples representing the tone curve
    
    Returns:
        True if the curve is linear [(0,0), (1,1)], False otherwise
    """
    return (len(curve) == 2 and 
            curve[0] == (0.0, 0.0) and 
            curve[1] == (1.0, 1.0))


def is_property_active(xmp: muimg.XmpMetadata, prop_name: str) -> bool:
    """Check if an XMP property is active (present and non-default).
    
    Args:
        xmp: XmpMetadata object
        prop_name: Property name (temperature, tint, exposure, tone_curve)
    
    Returns:
        True if property is active (non-default value)
    """
    if prop_name == "temperature":
        # Temperature is active if WhiteBalance is NOT "As Shot" and Temperature is present
        # "As Shot" means use DNG tags, other values mean use custom XMP temp/tint
        wb = xmp.get_prop("WhiteBalance", str)
        if wb == "As Shot":
            return False
        temp = xmp.get_prop("Temperature", float)
        return temp is not None
    
    elif prop_name == "tint":
        # Tint is active if WhiteBalance is NOT "As Shot" and Tint is present and non-zero
        wb = xmp.get_prop("WhiteBalance", str)
        if wb == "As Shot":
            return False
        tint = xmp.get_prop("Tint", float)
        return tint is not None and abs(tint) > 0.01
    
    elif prop_name == "exposure":
        # Exposure is active if Exposure2012 is present and non-zero
        exposure = xmp.get_prop("Exposure2012", float)
        return exposure is not None and abs(exposure) > 0.01
    
    elif prop_name == "tone_curve":
        # Tone curve is active if ToneCurvePV2012 is present and non-linear
        curve = xmp.get_prop("ToneCurvePV2012", list)
        if curve is None or len(curve) < 2:
            return False
        return not is_tone_curve_linear(curve)
    
    return False


def extract_rendering_params(xmp: muimg.XmpMetadata) -> dict:
    """Extract rendering parameters from XMP, respecting WhiteBalance gating.
    
    Args:
        xmp: XmpMetadata object
    
    Returns:
        Dict of rendering parameters (only active ones)
    """
    params = {}
    
    # Temperature/Tint: only if WhiteBalance is NOT "As Shot"
    # "As Shot" means use DNG tags, other values mean use custom XMP temp/tint
    wb = xmp.get_prop("WhiteBalance", str)
    if wb != "As Shot":
        temp = xmp.get_prop("Temperature", float)
        if temp is not None:
            params["temperature"] = temp
        
        tint = xmp.get_prop("Tint", float)
        if tint is not None:
            params["tint"] = tint
    
    # Exposure: if non-zero
    exposure = xmp.get_prop("Exposure2012", float)
    if exposure is not None and abs(exposure) > 0.01:
        params["exposure"] = exposure
    
    # Tone curve: if non-linear (not just [(0,0), (1,1)])
    curve = xmp.get_prop("ToneCurvePV2012", list)
    if curve is not None and len(curve) >= 2:
        if not is_tone_curve_linear(curve):
            params["tone_curve"] = curve
    
    return params


@pytest.mark.parametrize("dng_name,tif_name,expected_active", TEST_CASES, ids=lambda x: x if isinstance(x, str) else None)
def test_xmp_rendering(dng_name, tif_name, expected_active, output_dir):
    """Test XMP-based rendering against Photoshop reference.
    
    Validates:
    1. Only expected XMP properties are active (non-default)
    2. Rendered output matches Photoshop reference within threshold
    """
    dng_path = XMP_TEST_DIR / dng_name
    tif_path = XMP_TEST_DIR / tif_name
    
    if not dng_path.exists():
        pytest.skip(f"Test file not found: {dng_path}")
    if not tif_path.exists():
        pytest.skip(f"Reference file not found: {tif_path}")
    
    # Load DNG and extract XMP
    # LINEAR_RAW DNGs already have DefaultBlackRender=1 set
    with muimg.DngFile(dng_path) as dng:
        xmp = dng.get_xmp()
        assert xmp is not None, f"No XMP metadata found in {dng_name}"
        
        # Validate which properties are active
        all_props = {"temperature", "tint", "exposure", "tone_curve"}
        active_props = {prop for prop in all_props if is_property_active(xmp, prop)}
        
        # Debug: show XMP values if mismatch
        if active_props != expected_active:
            print(f"\n  XMP Debug for {dng_name}:")
            print(f"    WhiteBalance: {xmp.get_prop('WhiteBalance', str)}")
            print(f"    Temperature: {xmp.get_prop('Temperature', float)}")
            print(f"    Tint: {xmp.get_prop('Tint', float)}")
            print(f"    Exposure2012: {xmp.get_prop('Exposure2012', float)}")
            curve = xmp.get_prop('ToneCurvePV2012', list)
            if curve:
                print(f"    ToneCurvePV2012: {len(curve)} points")
                print(f"      All points: {curve}")
                # Check if linear
                is_linear = all(abs(x - y) < 0.01 for x, y in curve)
                print(f"      Is linear: {is_linear}")
                if not is_linear:
                    print(f"      Non-linear points:")
                    for x, y in curve:
                        if abs(x - y) >= 0.01:
                            print(f"        ({x:.4f}, {y:.4f}) - diff: {abs(x-y):.4f}")
            else:
                print(f"    ToneCurvePV2012: None")
        
        # Check that only expected properties are active
        assert active_props == expected_active, (
            f"Active XMP properties mismatch for {dng_name}:\n"
            f"  Expected: {sorted(expected_active)}\n"
            f"  Got:      {sorted(active_props)}"
        )
        
        # Extract rendering parameters (filtered to exclude defaults)
        render_params = extract_rendering_params(xmp)
        
        # Render with filtered rendering_params, bypassing automatic XMP extraction
        result = dng.render(
            output_dtype=np.uint16,
            demosaic_algorithm="DNGSDK_BILINEAR",
            strict=False,
            use_xmp=False,
            rendering_params=render_params if render_params else None,
        )
        
        assert result is not None, f"Rendering failed for {dng_name}"
    
    # Save rendered output for manual inspection
    output_name = dng_path.stem + "_muimg.tif"
    tifffile.imwrite(str(output_dir / output_name), result)
    
    # Load Photoshop reference
    ref = tifffile.imread(str(tif_path))
    
    # Ensure same shape
    if result.shape != ref.shape:
        pytest.fail(
            f"Shape mismatch for {dng_name}:\n"
            f"  MUIMG:     {result.shape}\n"
            f"  Photoshop: {ref.shape}"
        )
    
    # Compare against reference
    stats = compute_diff_stats(result, ref)
    
    # Threshold: allow up to 3% difference
    # Note: ~2% baseline difference exists between MUIMG and Photoshop
    # due to different demosaic algorithms, color space handling, etc.
    threshold = 2.25
    
    print(f"\n  [XMP] {dng_name}: diff={stats['mean']:.2f}% (threshold={threshold}%)")
    print(f"    Active XMP: {sorted(active_props)}")
    print(f"    Render params: {list(render_params.keys())}")
    
    assert stats["mean"] < threshold, (
        f"XMP rendering diff {stats['mean']:.2f}% > {threshold}% for {dng_name}"
    )

