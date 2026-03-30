import math
import pytest

from muimg.raw_render import (
    colortemp_to_uv,
    uvUCS_to_xy,
    xy_to_uvUCS,
    uv_to_colortemp,
    temp_tint_to_xy,
    xy_to_temp_tint,
)


def nearly_equal(a: float, b: float, abs_tol: float) -> bool:
    return abs(a - b) <= abs_tol


@pytest.mark.parametrize("temp", list(range(2000, 15001, 1000)))
def test_color_temp_tint_roundtrip(temp: int):
    """
    Roundtrip test over a grid of temperatures and tints:
      (T, tint) -> (u, v) -> (x, y) -> (u2, v2) -> (T2, tint2)
    and check that T2 ~= T and tint2 ~= tint within tolerances.

    Notes:
    - The conversions are an exact port of Adobe DNG SDK algorithms in the CIE 1960 UCS (u, v) space,
      which is considered "obsolete" but still widely used for CCT computations.
    - Due to piecewise interpolation and floating point arithmetic, allow small tolerances.
    """
    # Tolerances (conservative to avoid false failures from floating point/segmentation boundaries)
    TEMP_ABS_TOL = 0.2       # Kelvin
    TINT_ABS_TOL = 0.002     # Adobe tint units
    
    # Test across range of tints for this temperature
    for tint in range(-80, 81, 20):
        # Forward chain
        u, v = colortemp_to_uv(temp, tint)
        x, y = uvUCS_to_xy(u, v)
        u2, v2 = xy_to_uvUCS(x, y)
        T2, tint2 = uv_to_colortemp(u2, v2)

        assert nearly_equal(T2, float(temp), TEMP_ABS_TOL), (
            f"Temperature roundtrip mismatch [case: temp={temp}, tint={tint}]: "
            f"start={temp}, end={T2:.3f}, diff={abs(T2 - temp):.3f}K (tol={TEMP_ABS_TOL})"
        )
        assert nearly_equal(tint2, float(tint), TINT_ABS_TOL), (
            f"Tint roundtrip mismatch [case: temp={temp}, tint={tint}]: "
            f"start={tint}, end={tint2:.6f}, diff={abs(tint2 - tint):.6f} (tol={TINT_ABS_TOL})"
        )


@pytest.mark.parametrize("temp", [2000, 3200, 5000, 5500, 6500, 10000, 15000])
@pytest.mark.parametrize("tint", [-50, -10, 0, 10, 50])
def test_temp_tint_to_xy_roundtrip(temp: int, tint: int):
    """Verify temp_tint_to_xy and xy_to_temp_tint are inverses."""
    x, y = temp_tint_to_xy(temp, tint)
    temp2, tint2 = xy_to_temp_tint(x, y)
    
    assert nearly_equal(temp2, float(temp), 0.5), (
        f"Temperature roundtrip mismatch: {temp} -> ({x},{y}) -> {temp2}"
    )
    assert nearly_equal(tint2, float(tint), 0.01), (
        f"Tint roundtrip mismatch: {tint} -> ({x},{y}) -> {tint2}"
    )


def test_acr3_curve_basic():
    """Verify get_acr3_curve produces valid tone curve."""
    from muimg.raw_render import get_acr3_curve
    import numpy as np
    
    for num_points in [256, 512, 1024, 4096]:
        curve = get_acr3_curve(num_points)
        
        assert curve.shape == (num_points,), f"Wrong shape: {curve.shape}"
        assert curve.dtype == np.float32, f"Wrong dtype: {curve.dtype}"
        assert curve[0] == 0.0, f"Curve should start at 0: {curve[0]}"
        assert curve[-1] == 1.0, f"Curve should end at 1: {curve[-1]}"
        assert np.all(np.diff(curve) >= 0), "Curve should be monotonic"


def test_bradford_adaptation_basic():
    """Verify compute_bradford_adaptation produces valid adaptation matrices."""
    from muimg.raw_render import compute_bradford_adaptation, D50_xy, D65_xy
    import numpy as np
    
    # D50 -> D65 should produce a non-identity matrix
    adapt = compute_bradford_adaptation(D50_xy, D65_xy)
    assert adapt.shape == (3, 3), f"Wrong shape: {adapt.shape}"
    assert adapt.dtype == np.float64, f"Wrong dtype: {adapt.dtype}"
    
    # Identity adaptation (same src/dst) should give identity matrix
    identity = compute_bradford_adaptation(D50_xy, D50_xy)
    assert np.allclose(identity, np.eye(3), atol=1e-10), "Same src/dst should give identity"
