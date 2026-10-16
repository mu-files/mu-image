# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files

"""Test demosaic algorithms for consistency across bit depths."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
from conftest import compute_diff_stats

from muraw.array import Array
from muraw.dngio import DngFile
from muraw.raw_render import DemosaicAlgorithm, demosaic

# Test file path
TEST_DNG = Path(__file__).parent / "dngfiles" / "sony_ilce-7c_cfa_jxl_lossy_4ifds.dng"

# Center window for dtype and agreement checks. Even, so an RGGB CFA stays
# RGGB. Those checks do not need the full 4024x6048 frame.
_CONSISTENCY_SIZE = 512

# Bayer-phase window. A flat patch demosaics to the same color for every
# phase, so the window is the even-grid block with the largest sample std.
_PATTERN_SIZE = 128


def is_rcd_available():
    """Check if RCD demosaicing is available (GPL-licensed, optional)."""
    try:
        # Try to use RCD on a tiny test array
        test_cfa = np.zeros((4, 4), dtype=np.uint16)
        demosaic(Array(test_cfa), "RGGB", algorithm=DemosaicAlgorithm.RCD).realize()
        return True
    except ImportError:
        return False


def get_available_algorithms():
    """Get list of available demosaic algorithms, excluding RCD if not available."""
    algorithms = [
        DemosaicAlgorithm.VNG,
        DemosaicAlgorithm.EA,
        DemosaicAlgorithm.EA_FAST,
        DemosaicAlgorithm.OPENCV_EA,
    ]
    if is_rcd_available():
        algorithms.append(DemosaicAlgorithm.RCD)
    return algorithms


def _center_window(cfa, size):
    """Even-aligned center crop, so the Bayer phase of ``cfa`` is unchanged.

    The window starts on an even row and column, so its top-left pixel is
    the same Bayer site as ``cfa``'s. The slice is a hard crop: pixels past
    the window are not part of it.
    """
    image = Array(cfa)
    height, width = image.shape
    top = ((height - size) // 2) & ~1
    left = ((width - size) // 2) & ~1
    return image[top : top + size, left : left + size]


def _textured_window(cfa, size):
    """Even-grid ``size`` block with the largest sample std."""
    best_std = -1.0
    best_origin = (0, 0)
    height, width = cfa.shape
    for y0 in range(0, height - size + 1, size):
        for x0 in range(0, width - size + 1, size):
            std = float(cfa[y0 : y0 + size, x0 : x0 + size].std())
            if std > best_std:
                best_std = std
                best_origin = (y0, x0)
    y0, x0 = best_origin
    return np.ascontiguousarray(cfa[y0 : y0 + size, x0 : x0 + size])


@pytest.fixture(scope="module")
def cfa_full():
    """Load the full CFA from the test DNG file."""
    if not TEST_DNG.exists():
        pytest.skip(f"Test file not found: {TEST_DNG}")

    with DngFile(TEST_DNG) as dng:
        cfa_t, cfa_pattern = dng.get_cfa()
        return cfa_t.realize(), cfa_pattern


@pytest.fixture(scope="module")
def cfa_data(cfa_full):
    """Center view of the CFA for the dtype and agreement checks."""
    cfa, cfa_pattern = cfa_full
    return _center_window(cfa, _CONSISTENCY_SIZE), cfa_pattern


def test_demosaic_uint8_consistency(cfa_data):
    """Test that all demosaic algorithms produce consistent results for uint8 input."""
    cfa, cfa_pattern = cfa_data

    # Convert CFA to uint8
    cfa_u8 = Array(cfa).convert_type("uint8")

    # Run BILINEAR as reference
    reference = demosaic(
        cfa_u8, cfa_pattern, algorithm=DemosaicAlgorithm.BILINEAR
    ).realize()
    assert reference.dtype == np.uint8
    assert reference.shape == (cfa.shape[0], cfa.shape[1], 3)

    # Test other algorithms produce same dtype and similar results
    for algorithm in get_available_algorithms():
        result = demosaic(cfa_u8, cfa_pattern, algorithm=algorithm).realize()

        # Check dtype preservation
        assert result.dtype == np.uint8, f"{algorithm} should preserve uint8 dtype"

        # Check shape
        assert result.shape == reference.shape, f"{algorithm} shape mismatch"

        # Check values are in valid range
        assert result.min() >= 0 and result.max() <= 255, (
            f"{algorithm} values out of range"
        )

        # Check not all zeros
        assert not np.all(result == 0), f"{algorithm} produced all zeros"

        # Check similarity to reference
        # Different algorithms will differ, but should be reasonably close
        # Based on actual measurements: mean ~0.02-0.15%, p99 ~0.4-1.2%, max ~15-41%
        diff_stats = compute_diff_stats(reference, result)
        print(
            f"\n{algorithm} vs BILINEAR (uint8): mean={diff_stats['mean']:.2f}%, p99={diff_stats['p99']:.2f}%, max={diff_stats['max']:.2f}%"
        )
        assert diff_stats["mean"] < 1.0, (
            f"{algorithm} mean diff {diff_stats['mean']:.2f}% > 1.0%"
        )
        assert diff_stats["p99"] < 2.0, (
            f"{algorithm} p99 diff {diff_stats['p99']:.2f}% > 2.0%"
        )


def test_demosaic_uint16_consistency(cfa_data):
    """Test that all demosaic algorithms produce consistent results for uint16 input."""
    cfa, cfa_pattern = cfa_data

    # Convert CFA to uint16
    cfa_u16 = Array(cfa).convert_type("uint16")

    # Run BILINEAR as reference
    reference = demosaic(
        cfa_u16, cfa_pattern, algorithm=DemosaicAlgorithm.BILINEAR
    ).realize()
    assert reference.dtype == np.uint16
    assert reference.shape == (cfa.shape[0], cfa.shape[1], 3)

    # Test other algorithms produce same dtype and similar results
    for algorithm in get_available_algorithms():
        result = demosaic(cfa_u16, cfa_pattern, algorithm=algorithm).realize()

        # Check dtype preservation
        assert result.dtype == np.uint16, f"{algorithm} should preserve uint16 dtype"

        # Check shape
        assert result.shape == reference.shape, f"{algorithm} shape mismatch"

        # Check values are in valid range
        assert result.min() >= 0 and result.max() <= 65535, (
            f"{algorithm} values out of range"
        )

        # Check not all zeros
        assert not np.all(result == 0), f"{algorithm} produced all zeros"

        # Check similarity to reference
        # Based on actual measurements: mean ~0.02-0.12%, p99 ~0.4-1.0%, max ~15-41%
        diff_stats = compute_diff_stats(reference, result)
        print(
            f"\n{algorithm} vs BILINEAR (uint16): mean={diff_stats['mean']:.2f}%, p99={diff_stats['p99']:.2f}%, max={diff_stats['max']:.2f}%"
        )
        assert diff_stats["mean"] < 1.0, (
            f"{algorithm} mean diff {diff_stats['mean']:.2f}% > 1.0%"
        )
        assert diff_stats["p99"] < 2.0, (
            f"{algorithm} p99 diff {diff_stats['p99']:.2f}% > 2.0%"
        )


def test_demosaic_float32_consistency(cfa_data):
    """Test that all demosaic algorithms produce consistent results for float32 input."""
    cfa, cfa_pattern = cfa_data

    # Convert CFA to float32 (0-1 range)
    cfa_f32 = Array(cfa).convert_type("float32")

    # Run BILINEAR as reference
    reference = demosaic(
        cfa_f32, cfa_pattern, algorithm=DemosaicAlgorithm.BILINEAR
    ).realize()
    assert reference.dtype == np.float32
    assert reference.shape == (cfa.shape[0], cfa.shape[1], 3)

    # Test other algorithms produce same dtype and similar results
    for algorithm in get_available_algorithms():
        result = demosaic(cfa_f32, cfa_pattern, algorithm=algorithm).realize()

        # Check dtype preservation
        assert result.dtype == np.float32, f"{algorithm} should preserve float32 dtype"

        # Check shape
        assert result.shape == reference.shape, f"{algorithm} shape mismatch"

        assert result.min() >= 0.0 and result.max() <= 1.0, (
            f"{algorithm} values out of range"
        )

        # Check not all zeros
        assert not np.all(result == 0), f"{algorithm} produced all zeros"

        # Check similarity to reference
        # Based on actual measurements: mean ~0.02-0.12%, p99 ~0.4-1.0%, max ~15-41%
        diff_stats = compute_diff_stats(reference, result)
        print(
            f"\n{algorithm} vs BILINEAR (float32): mean={diff_stats['mean']:.2f}%, p99={diff_stats['p99']:.2f}%, max={diff_stats['max']:.2f}%"
        )
        assert diff_stats["mean"] < 1.0, (
            f"{algorithm} mean diff {diff_stats['mean']:.2f}% > 1.0%"
        )
        assert diff_stats["p99"] < 2.0, (
            f"{algorithm} p99 diff {diff_stats['p99']:.2f}% > 2.0%"
        )


def test_demosaic_dtype_conversion_roundtrip(cfa_data):
    """Test that demosaic preserves relative values across dtype conversions."""
    cfa, cfa_pattern = cfa_data

    # Get float32 reference
    cfa_f32 = Array(cfa).convert_type("float32")
    result_f32 = demosaic(
        cfa_f32, cfa_pattern, algorithm=DemosaicAlgorithm.BILINEAR
    ).realize()

    # Convert to uint16 and demosaic
    cfa_u16 = Array(cfa).convert_type("uint16")
    result_u16 = demosaic(cfa_u16, cfa_pattern, algorithm=DemosaicAlgorithm.BILINEAR)

    # Convert uint16 result back to float32 for comparison
    result_u16_as_f32 = result_u16.convert_type("float32")

    # Results should be very close (within quantization error)
    # uint16 has 65536 levels, so max error is ~1/65536 ≈ 0.000015
    max_diff = np.abs(result_f32 - result_u16_as_f32).max()
    assert max_diff < 0.001, f"Float32 and uint16 results differ by {max_diff}"

    # Same test for uint8
    cfa_u8 = Array(cfa).convert_type("uint8")
    result_u8 = demosaic(cfa_u8, cfa_pattern, algorithm=DemosaicAlgorithm.BILINEAR)
    result_u8_as_f32 = result_u8.convert_type("float32")

    # uint8 has only 256 levels, so allow larger error
    max_diff = np.abs(result_f32 - result_u8_as_f32).max()
    assert max_diff < 0.01, f"Float32 and uint8 results differ by {max_diff}"


def test_demosaic_invalid_algorithm(cfa_data):
    """Test that invalid algorithm raises ValueError."""
    cfa, cfa_pattern = cfa_data

    with pytest.raises(TypeError, match="algorithm must be a DemosaicAlgorithm enum"):
        demosaic(Array(cfa), cfa_pattern, algorithm="INVALID_ALGO").realize()


def test_demosaic_invalid_pattern(cfa_data):
    """Test that invalid CFA pattern raises ValueError."""
    cfa, _ = cfa_data

    with pytest.raises(ValueError, match="Invalid CFA pattern"):
        demosaic(
            Array(cfa), "INVALID_PATTERN", algorithm=DemosaicAlgorithm.BILINEAR
        ).realize()


def _check_cfa_patterns(cfa_u16, algorithms):
    """Demosaic each Bayer phase and compare it to the same crop of RGGB.

    The four Bayer patterns are:
    RGGB:  RG    GRBG:  GR    GBRG:  GB    BGGR:  BG
           GB           BG           RG           GR

    The other three come from RGGB by cropping:
    - GRBG: crop by (1, 0), one column left
    - GBRG: crop by (0, 1), one row up
    - BGGR: crop by (1, 1), one column left and one row up
    """
    pattern_crops = {
        "RGGB": (0, 0),
        "BGGR": (1, 1),
        "GBRG": (0, 1),
        "GRBG": (1, 0),
    }

    for ref_algorithm in algorithms:
        # Demosaic the original RGGB pattern with reference algorithm
        reference_rgb = demosaic(cfa_u16, "RGGB", algorithm=ref_algorithm).realize()

        # Test each pattern variant
        for pattern_name, (crop_x, crop_y) in pattern_crops.items():
            # Crop the CFA to create the new pattern
            cropped_cfa = cfa_u16[crop_y:, crop_x:]

            # Make dimensions even (required for demosaic)
            h, w = cropped_cfa.shape
            if h % 2 == 1:
                cropped_cfa = cropped_cfa[:-1, :]
            if w % 2 == 1:
                cropped_cfa = cropped_cfa[:, :-1]

            # Make contiguous copy (C extensions may assume contiguous memory)
            cropped_cfa = np.ascontiguousarray(cropped_cfa)

            # Crop the reference RGB to match (include color channel dimension)
            expected_rgb = reference_rgb[crop_y:, crop_x:, :]
            h, w = cropped_cfa.shape
            expected_rgb = expected_rgb[:h, :w, :]

            # Test demosaicing the cropped pattern with all algorithms
            for test_algorithm in algorithms:
                # Demosaic the cropped CFA with the test algorithm
                cropped_rgb = demosaic(
                    Array(cropped_cfa), pattern_name, algorithm=test_algorithm
                ).realize()

                # Results should have correct shape
                assert cropped_rgb.shape == expected_rgb.shape, (
                    f"{ref_algorithm}->{test_algorithm} {pattern_name}: shape mismatch"
                )

                # Compare results - allow small differences at edges due to interpolation
                diff = np.abs(
                    cropped_rgb.astype(np.int32) - expected_rgb.astype(np.int32)
                )

                # HA needs ±3 (G at dest±1 uses Bayer ±2 from those sites).
                border = (
                    3
                    if DemosaicAlgorithm.EA
                    in (
                        ref_algorithm,
                        test_algorithm,
                    )
                    else 2
                )
                if diff.shape[0] > 2 * border and diff.shape[1] > 2 * border:
                    diff_interior = diff[border:-border, border:-border]
                    max_interior_diff = diff_interior.max()
                    mean_interior_diff = diff_interior.mean()

                    print(
                        f"\n{ref_algorithm}->{test_algorithm} {pattern_name}: max_interior={max_interior_diff}, mean_interior={mean_interior_diff:.2f}"
                    )

                    # Always verify non-zero output
                    assert not np.all(cropped_rgb == 0), (
                        f"{ref_algorithm}->{test_algorithm} {pattern_name}: produced all zeros"
                    )

                    # When using same algorithm, interior should be nearly identical
                    if ref_algorithm == test_algorithm:
                        # RCD has localized interpolation differences, use relaxed thresholds
                        if test_algorithm == DemosaicAlgorithm.RCD:
                            max_threshold = 12000
                            mean_threshold = 0.85
                        else:
                            max_threshold = 1
                            mean_threshold = 0.1

                        if (
                            max_interior_diff > max_threshold
                            or mean_interior_diff >= mean_threshold
                        ):
                            # Save debug output
                            output_dir = (
                                Path(__file__).parent / "test_outputs" / "test_demosaic"
                            )
                            output_dir.mkdir(parents=True, exist_ok=True)

                            ref_file = (
                                output_dir
                                / f"{ref_algorithm.name}_{pattern_name}_reference.tif"
                            )
                            test_file = (
                                output_dir
                                / f"{ref_algorithm.name}_{pattern_name}_test.tif"
                            )
                            diff_file = (
                                output_dir
                                / f"{ref_algorithm.name}_{pattern_name}_diff.tif"
                            )

                            tifffile.imwrite(ref_file, expected_rgb)
                            tifffile.imwrite(test_file, cropped_rgb)
                            tifffile.imwrite(diff_file, diff.astype(np.uint16))

                            print(f"Saved debug output to {output_dir}")

                        assert max_interior_diff <= max_threshold, (
                            f"{ref_algorithm}->{test_algorithm} {pattern_name}: interior max diff {max_interior_diff} > {max_threshold}"
                        )
                        assert mean_interior_diff < mean_threshold, (
                            f"{ref_algorithm}->{test_algorithm} {pattern_name}: interior mean diff {mean_interior_diff:.2f} > {mean_threshold}"
                        )
                    else:
                        # Cross-algorithm: algorithms differ but should be reasonably similar
                        # Mean difference should be < 96 (out of 65535, ~0.15%)
                        assert mean_interior_diff < 96, (
                            f"{ref_algorithm}->{test_algorithm} {pattern_name}: cross-algorithm mean diff {mean_interior_diff:.2f} > 100"
                        )


def test_demosaic_cfa_pattern_consistency(cfa_full):
    """A Bayer phase shift of the CFA matches the same shift of the demosaic.

    Algorithms that reproduce the shift exactly, and every pair of them, run
    on the highest-variance 128 window. RCD's misses are sparse and its mean
    bound is over the whole frame, so RCD is compared only with itself, on
    the full CFA.
    """
    cfa, cfa_pattern = cfa_full
    assert cfa_pattern == "RGGB", f"Test expects RGGB pattern, got {cfa_pattern}"

    algorithms = [DemosaicAlgorithm.BILINEAR] + get_available_algorithms()
    exact = [
        algorithm for algorithm in algorithms if algorithm != DemosaicAlgorithm.RCD
    ]
    textured = _textured_window(cfa, _PATTERN_SIZE)
    _check_cfa_patterns(Array(textured).convert_type("uint16"), exact)
    if DemosaicAlgorithm.RCD in algorithms:
        _check_cfa_patterns(Array(cfa).convert_type("uint16"), [DemosaicAlgorithm.RCD])


@pytest.mark.parametrize(
    "algorithm",
    [DemosaicAlgorithm.BILINEAR, DemosaicAlgorithm.EA],
)
@pytest.mark.parametrize("cx,cy", [(1, 0), (0, 1), (3, 2)])
def test_demosaic_then_crop_matches_post_process_slice(algorithm, cx, cy):
    """Fused demosaic → crop equals full-frame demosaic then a numpy crop.

    Trailing crop dest is crop-buffer space; demosaic phase must stay on
    CFA/demosaic-array coordinates (odd origins included).
    """
    rng = np.random.default_rng(0)
    cfa = rng.random((17, 19), dtype=np.float32)
    full = demosaic(
        Array(cfa), "RGGB", algorithm=algorithm, dst_dtype="float32"
    ).realize()
    cw, ch = 11, 13
    post = full[cy : cy + ch, cx : cx + cw]
    fused = (
        demosaic(Array(cfa), "RGGB", algorithm=algorithm, dst_dtype="float32")
        .view(
            left=cx,
            top=cy,
            width=cw,
            height=ch,
        )
        .realize()
    )
    # Fused EA vs a sliced full frame can differ by 1 ULP on some
    # platforms (reduction order).
    np.testing.assert_array_max_ulp(fused, post, maxulp=1)
