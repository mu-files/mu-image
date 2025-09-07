import math
import pytest

from muimg.color import (
    colortemp_to_uv,
    uv_to_xy,
    xy_to_uv,
    uv_to_colortemp,
)


def nearly_equal(a: float, b: float, abs_tol: float) -> bool:
    return abs(a - b) <= abs_tol


@pytest.mark.parametrize("temp", list(range(2000, 15001, 1000)))
@pytest.mark.parametrize("tint", list(range(-80, 81, 20)))
def test_color_temp_tint_roundtrip(temp: int, tint: int):
    """
    Roundtrip test over a grid of temperatures and tints:
      (T, tint) -> (u, v) -> (x, y) -> (u2, v2) -> (T2, tint2)
    and check that T2 ~= T and tint2 ~= tint within tolerances.

    Notes:
    - The conversions are an exact port of Adobe DNG SDK algorithms in the CIE 1960 UCS (u, v) space,
      which is considered "obsolete" but still widely used for CCT computations.
    - Due to piecewise interpolation and floating point arithmetic, allow small tolerances.
    """
    # Forward chain
    u, v = colortemp_to_uv(temp, tint)
    x, y = uv_to_xy(u, v)
    u2, v2 = xy_to_uv(x, y)
    T2, tint2 = uv_to_colortemp(u2, v2)

    # Tolerances (conservative to avoid false failures from floating point/segmentation boundaries)
    TEMP_ABS_TOL = 0.2       # Kelvin
    TINT_ABS_TOL = 0.002     # Adobe tint units

    assert nearly_equal(T2, float(temp), TEMP_ABS_TOL), (
        f"Temperature roundtrip mismatch [case: temp={temp}, tint={tint}]: "
        f"start={temp}, end={T2:.3f}, diff={abs(T2 - temp):.3f}K (tol={TEMP_ABS_TOL})"
    )
    assert nearly_equal(tint2, float(tint), TINT_ABS_TOL), (
        f"Tint roundtrip mismatch [case: temp={temp}, tint={tint}]: "
        f"start={tint}, end={tint2:.6f}, diff={abs(tint2 - tint):.6f} (tol={TINT_ABS_TOL})"
    )
