# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files

"""Cubic spline interpolation for tone curves and LUT operations.

This module provides:
- CubicSpline: Cubic spline implementation using only NumPy
- LUT: Lookup table for fast 1D transformations
- ColorSpaceLUT: Cached gamma curve LUTs for color spaces
- ColorSpace: Color space definitions
"""

from __future__ import annotations

import numpy as np
import threading
from enum import Enum, auto


# =============================================================================
# Constants
# =============================================================================

class ColorSpace(Enum):
    """Color space definitions with native gamma encoding.

    Each color space is defined by:
    - Primary chromaticities (red, green, blue)
    - White point (D50 or D65)
    - Gamma encoding (linear, 1.8, 2.2, or sRGB piecewise)

    RGB values (0-5): linear spaces are even (0,2,4) and gamma spaces are odd (1,3,5).
    This allows simple arithmetic to convert between linear and gamma variants.

    Grayscale values (6,7,9): linear=6, gamma 1.8=7, gamma 2.2=9 for monochrome images.
    """
    PROPHOTO_LINEAR = 0
    PROPHOTO_GAMMA = 1           # Gamma 1.8
    ADOBERGB_LINEAR = 2
    ADOBERGB_GAMMA = 3           # Gamma 2.2
    SRGB_LINEAR = 4
    SRGB_GAMMA = 5               # sRGB piecewise gamma
    GRAY_LINEAR = 6              # Monochrome linear (for pattern consistency)
    GRAY_GAMMA_1_8 = 7           # Monochrome gamma 1.8
    GRAY_GAMMA_2_2 = 9           # Monochrome gamma 2.2
    
    def to_linear(self) -> "ColorSpace":
        """Convert gamma space to its linear variant (or return self if already linear)."""
        return ColorSpace(self.value & ~1)  # Clear bit 0
    
    def to_gamma(self) -> "ColorSpace":
        """Convert linear space to its gamma variant (or return self if already gamma)."""
        return ColorSpace(self.value | 1)  # Set bit 0
    
    def is_linear(self) -> bool:
        """Check if this is a linear color space (bit 0 is clear)."""
        return not (self.value & 1)
    
    def is_gamma(self) -> bool:
        """Check if this is a gamma-encoded color space (bit 0 is set)."""
        return bool(self.value & 1)


# =============================================================================
# Cubic Spline Interpolation
# =============================================================================

class CubicSpline:
    """
    Represents a cubic spline curve with control points in normalized 0-1 range.
    
    Uses custom cubic spline interpolation with NumPy. Points are stored as float tuples.
    """
    
    def __init__(self, points_or_string: str | list | None = None, bc_type: str = 'natural'):
        """
        Initialize CubicSpline.
        
        Args:
            points_or_string: Optional. Can be:
                - None: Initialize with default linear curve (0.0, 0.0), (1.0, 1.0)
                - str: Parse from string format "(0.0,0.0),(1.0,1.0)"
                - list: Use provided float points directly
            bc_type: Boundary condition type. Default 'natural'.
                - 'natural': Second derivative = 0 at endpoints
                - ((order, value), (order, value)): Custom derivatives at endpoints
                  where order=1 is first derivative, order=2 is second derivative
        """
        self.bc_type = bc_type
        
        if points_or_string is None:
            # Default linear curve
            self.points = [(0.0, 0.0), (1.0, 1.0)]
        elif isinstance(points_or_string, str):
            # Parse from string format
            self._parse_from_string(points_or_string)
        elif isinstance(points_or_string, list):
            # Use provided points directly (ensure float)
            self.points = [(float(x), float(y)) for x, y in points_or_string]
        else:
            raise TypeError(
                f"CubicSpline constructor expects None, str, or list, "
                f"got {type(points_or_string)}"
            )
        
        # Ensure (0,0) and (1,1) endpoints exist
        # Add (0,0) if not present
        if not any(abs(p[0] - 0.0) < 1e-6 for p in self.points):
            self.points.insert(0, (0.0, 0.0))
        
        # Add (1,1) if not present
        if not any(abs(p[0] - 1.0) < 1e-6 for p in self.points):
            self.points.append((1.0, 1.0))
        
        # Create the spline immediately
        self._create_spline()
    
    def _parse_from_string(self, string_data: str):
        """Helper method to parse points from string format.
        
        Expects normalized 0-1 float format: "(0.0,0.0),(0.5,0.5),(1.0,1.0)"
        """
        import re
        
        # Extract coordinate pairs using regex (supports int or float)
        tuple_pattern = r'\(([0-9.]+),\s*([0-9.]+)\)'
        matches = re.findall(tuple_pattern, string_data)
        
        if not matches:
            raise ValueError(
                f"No valid coordinate pairs found in spline curve data: {string_data}"
            )
        
        # Convert string coordinates to float tuples
        self.points = [(float(x), float(y)) for x, y in matches]
    
    def _create_spline(self):
        """Create cubic spline coefficients using NumPy.
        
        Implements natural cubic spline algorithm:
        1. Build tridiagonal system for second derivatives (M values)
        2. Solve system based on boundary conditions
        3. Store coefficients for piecewise cubic evaluation
        """
        x = np.array([p[0] for p in self.points], dtype=np.float64)
        y = np.array([p[1] for p in self.points], dtype=np.float64)
        n = len(x)
        
        # Ensure monotonically increasing x values
        if not all(x[i] < x[i+1] for i in range(n-1)):
            raise ValueError(
                f"CubicSpline x coordinates must be strictly increasing, but found {x}"
            )
        
        # Compute intervals
        h = np.diff(x)
        
        # Build tridiagonal system for second derivatives (M)
        # A * M = B
        A = np.zeros((n, n), dtype=np.float64)
        B = np.zeros(n, dtype=np.float64)
        
        # Interior points (natural cubic spline equations)
        for i in range(1, n-1):
            A[i, i-1] = h[i-1]
            A[i, i] = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            B[i] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
        
        # Apply boundary conditions
        if self.bc_type == 'natural':
            # Natural boundary: second derivative = 0 at endpoints
            A[0, 0] = 1.0
            A[-1, -1] = 1.0
            B[0] = 0.0
            B[-1] = 0.0
        elif isinstance(self.bc_type, tuple) and len(self.bc_type) == 2:
            # Custom derivative boundaries: ((left_order, left_val), (right_order, right_val))
            left_bc, right_bc = self.bc_type
            
            # Left boundary
            if left_bc[0] == 1:
                # First derivative specified at left
                # S'(x_0) = left_val
                # Using: S'(x_0) = (y_1 - y_0)/h_0 - h_0/3 * (2*M_0 + M_1)
                # Rearrange: 2*h_0*M_0 + h_0*M_1 = 3*((y_1-y_0)/h_0 - left_val)
                A[0, 0] = 2 * h[0]
                A[0, 1] = h[0]
                B[0] = 3 * ((y[1] - y[0]) / h[0] - left_bc[1])
            elif left_bc[0] == 2:
                # Second derivative specified at left
                A[0, 0] = 1.0
                B[0] = left_bc[1]
            else:
                raise ValueError(f"Unsupported left boundary condition order: {left_bc[0]}")
            
            # Right boundary
            if right_bc[0] == 1:
                # First derivative specified at right
                # S'(x_n-1) = right_val
                # Using: S'(x_n-1) = (y_n - y_n-1)/h_n-1 + h_n-1/3 * (M_n-1 + 2*M_n)
                # Rearrange: h_n-1*M_n-1 + 2*h_n-1*M_n = 3*(right_val - (y_n-y_n-1)/h_n-1)
                A[-1, -2] = h[-1]
                A[-1, -1] = 2 * h[-1]
                B[-1] = 3 * (right_bc[1] - (y[-1] - y[-2]) / h[-1])
            elif right_bc[0] == 2:
                # Second derivative specified at right
                A[-1, -1] = 1.0
                B[-1] = right_bc[1]
            else:
                raise ValueError(f"Unsupported right boundary condition order: {right_bc[0]}")
        else:
            raise ValueError(f"Unsupported boundary condition type: {self.bc_type}")
        
        # Solve for M (second derivatives at knots)
        M = np.linalg.solve(A, B)
        
        # Store data for evaluation
        self._x = x
        self._y = y
        self._h = h
        self._M = M
        self._n = n
    
    def __call__(self, x_eval):
        """
        Evaluate the spline at given x value(s) (callable interface).
        
        Args:
            x_eval: Input value (typically 0-1 range), can be scalar or array
            
        Returns:
            Interpolated y value(s)
        """
        # Convert to numpy array for consistent handling
        x_eval = np.atleast_1d(x_eval)
        scalar_input = x_eval.shape == (1,)
        
        result = np.zeros_like(x_eval, dtype=np.float64)
        
        for idx, x_val in enumerate(x_eval):
            # Find interval: x[i] <= x_val < x[i+1]
            # Use searchsorted for efficient binary search
            i = np.searchsorted(self._x[1:], x_val)
            i = min(i, self._n - 2)  # Ensure i is valid interval index
            
            # Compute local coordinate within interval
            dx = x_val - self._x[i]
            h = self._h[i]
            
            # Evaluate cubic polynomial using Hermite form
            # S_i(x) = a + b*dx + c*dx^2 + d*dx^3
            # where dx = x - x_i
            
            # Coefficients from second derivatives
            a = self._y[i]
            b = (self._y[i+1] - self._y[i]) / h - h * (2*self._M[i] + self._M[i+1]) / 3
            c = self._M[i]
            d = (self._M[i+1] - self._M[i]) / (3 * h)
            
            result[idx] = a + b*dx + c*dx*dx + d*dx*dx*dx
        
        return result[0] if scalar_input else result
    
    def resample(self, num_points: int) -> 'CubicSpline':
        """
        Create a new CubicSpline with evenly-spaced control points sampled 
        from this curve.
        
        Args:
            num_points: Number of control points in the new curve
            
        Returns:
            New CubicSpline with resampled control points
        """
        if num_points < 2:
            raise ValueError("resample requires at least 2 points")
        
        # Sample at evenly spaced x values
        x_values = np.linspace(0.0, 1.0, num_points)
        y_values = self(x_values)
        
        # Clip y values to 0-1 range
        y_values = np.clip(y_values, 0.0, 1.0)
        
        new_points = [(float(x), float(y)) for x, y in zip(x_values, y_values)]
        return CubicSpline(new_points)
    
    def __str__(self) -> str:
        """
        Convert CubicSpline to string format.
        
        Returns:
            String representation in format "(0.0,0.0),(0.5,0.5),(1.0,1.0)"
        """
        tuple_strings = [f"({x:.6g},{y:.6g})" for x, y in self.points]
        return ','.join(tuple_strings)
    
    def __repr__(self):
        return f"CubicSpline(points={self.points})"


class LUT:
    """Lookup table for fast 1D transformations.
    
    Stores a 1D array mapping [0, 1] → [0, 1] with linear interpolation.
    Supports arbitrary length and composition operations.
    """
    
    def __init__(self, data: np.ndarray | CubicSpline | str | list | None = None, 
                 size: int = 4096,
                 convert_srgb_gamma_to_linear: bool = False):
        """Initialize LUT.
        
        Args:
            data: LUT data. Can be:
                - np.ndarray: Use directly (must be 1D float32)
                - CubicSpline: Sample to create LUT
                - str/list: Create CubicSpline, then sample
                - None: Identity LUT (linear)
            size: Number of LUT entries (only used if data is not ndarray)
            convert_srgb_gamma_to_linear: If True, convert curve from sRGB gamma 2.2
                encoding to linear encoding. Used for Adobe curves that are defined
                in sRGB gamma space but need to be applied to linear pixels.
        """
        if data is None:
            # Identity LUT
            self.data = np.linspace(0.0, 1.0, size, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        elif isinstance(data, (CubicSpline, str, list)):
            # Build from spline
            if isinstance(data, CubicSpline):
                spline = data
            else:
                spline = CubicSpline(data)
            x = np.linspace(0.0, 1.0, size, dtype=np.float32)
            self.data = np.clip(spline(x), 0.0, 1.0).astype(np.float32)
            
            if convert_srgb_gamma_to_linear:
                # Convert from sRGB gamma space to linear space
                # The curve is defined in sRGB gamma space but needs to be applied to linear pixels
                from muimg.raw_render import ColorSpaceLUT, ColorSpace
                srgb_decode = ColorSpaceLUT(ColorSpace.SRGB_GAMMA, inverse=True, size=size)
                
                # Decode both input and output from sRGB gamma to linear
                linear_input = srgb_decode(x)
                linear_output = srgb_decode(self.data)
                
                # Resample to uniform spacing in linear space
                uniform_linear_x = np.linspace(0.0, 1.0, size, dtype=np.float32)
                self.data = np.interp(uniform_linear_x, linear_input, linear_output).astype(np.float32)
        else:
            raise TypeError(f"Unsupported LUT data type: {type(data)}")
    
    def __call__(self, x: np.ndarray | float) -> np.ndarray | float:
        """Apply LUT with linear interpolation."""
        x = np.asarray(x, dtype=np.float32)
        scalar_input = x.ndim == 0
        x = np.atleast_1d(x)
        
        x = np.clip(x, 0.0, 1.0)
        size = len(self.data)
        indices = x * (size - 1)
        idx_low = np.floor(indices).astype(np.int32)
        idx_high = np.minimum(idx_low + 1, size - 1)
        frac = indices - idx_low
        result = self.data[idx_low] * (1.0 - frac) + self.data[idx_high] * frac
        
        return result[0] if scalar_input else result
    
    def compose_input(self, other: 'LUT' | callable, size: int = None) -> 'LUT':
        """Compose with input remapping: result(x) = self(other(x)).
        
        Equivalent to remap_curve_input(). Creates uniform x samples, applies
        the remap function, then interpolates through this LUT at those positions.
        
        Args:
            other: LUT or function to apply before this LUT
            size: Output LUT size (default: same as self)
        
        Returns:
            New LUT with specified size
        """
        if size is None:
            size = len(self.data)
        
        # Create uniform x samples
        x = np.linspace(0.0, 1.0, size, dtype=np.float32)
        
        # Apply remap function to get positions to sample from
        if isinstance(other, LUT):
            remapped_x = other(x)
        else:
            remapped_x = other(x)
        
        # Interpolate through this LUT at remapped positions
        # Need to use np.interp since remapped_x may not be uniform
        self_x = np.linspace(0.0, 1.0, len(self.data), dtype=np.float32)
        result_data = np.interp(remapped_x, self_x, self.data)
        result_data = np.clip(result_data, 0.0, 1.0).astype(np.float32)
        
        return LUT(result_data)
    
    def compose_output(self, other: 'LUT' | callable, size: int = None) -> 'LUT':
        """Compose with output remapping: result(x) = other(self(x)).
        
        Equivalent to remap_curve_output(). Resamples this LUT to desired size,
        then applies the remap function to the output values.
        
        Args:
            other: LUT or function to apply after this LUT
            size: Output LUT size (default: same as self)
        
        Returns:
            New LUT with specified size
        """
        if size is None:
            size = len(self.data)
        
        # Resample this LUT to desired size if needed
        if len(self.data) == size:
            intermediate = self.data
        else:
            x = np.linspace(0.0, 1.0, size, dtype=np.float32)
            intermediate = self(x)
        
        # Apply remap function to output values
        if isinstance(other, LUT):
            # other is a LUT - interpolate through it
            other_x = np.linspace(0.0, 1.0, len(other.data), dtype=np.float32)
            result_data = np.interp(intermediate, other_x, other.data)
        else:
            # other is a callable function
            result_data = other(intermediate)
        
        result_data = np.clip(result_data, 0.0, 1.0).astype(np.float32)
        return LUT(result_data)
    
    def resample(self, size: int) -> 'LUT':
        """Resample LUT to different size."""
        x = np.linspace(0.0, 1.0, size, dtype=np.float32)
        return LUT(self(x))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __repr__(self):
        return f"LUT(size={len(self.data)})"

    def __array__(self, dtype=None):
        """Allow numpy to coerce LUT to ndarray via np.asarray(lut)."""
        if dtype is not None:
            return self.data.astype(dtype, copy=False)
        return self.data


# Module-level cache for ColorSpaceLUT instances (thread-safe)
_COLORSPACE_LUT_CACHE = {}
_COLORSPACE_LUT_CACHE_LOCK = threading.Lock()


class ColorSpaceLUT(LUT):
    """LUT for color space gamma encoding/decoding.
    
    Instances are cached at module level to avoid recalculating gamma curves.
    Cache is thread-safe using a lock.
    """
    
    def __new__(cls, colorspace: ColorSpace, inverse: bool = False, size: int = 4096):
        """Create or retrieve cached ColorSpaceLUT instance (thread-safe).
        
        Args:
            colorspace: ColorSpace enum (SRGB_GAMMA, PROPHOTO_GAMMA, ADOBERGB_GAMMA)
            inverse: If True, decode (gamma→linear). If False, encode (linear→gamma)
            size: Number of LUT entries
        
        Returns:
            Cached or new ColorSpaceLUT instance
        """
        # Create cache key
        cache_key = (colorspace, inverse, size)
        
        # Thread-safe cache lookup and creation
        with _COLORSPACE_LUT_CACHE_LOCK:
            # Return cached instance if available
            if cache_key in _COLORSPACE_LUT_CACHE:
                return _COLORSPACE_LUT_CACHE[cache_key]
            
            # Create new instance
            instance = super().__new__(cls)
            _COLORSPACE_LUT_CACHE[cache_key] = instance
            return instance
    
    def __init__(self, colorspace: ColorSpace, inverse: bool = False, size: int = 4096):
        """Initialize gamma LUT for a color space.
        
        Note: Due to caching, __init__ may be called multiple times on the same
        instance. We guard against re-initialization.
        """
        # Skip if already initialized (cached instance)
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        x = np.linspace(0.0, 1.0, size, dtype=np.float32)
        
        if colorspace == ColorSpace.SRGB_GAMMA:
            if inverse:
                # sRGB decode (gamma → linear)
                data = np.where(x <= 0.04045, x / 12.92, 
                               np.power((x + 0.055) / 1.055, 2.4))
            else:
                # sRGB encode (linear → gamma)
                data = np.where(x <= 0.0031308, x * 12.92,
                               1.055 * np.power(x, 1.0 / 2.4) - 0.055)
        
        elif colorspace == ColorSpace.PROPHOTO_GAMMA:
            if inverse:
                # ProPhoto decode (gamma → linear)
                data = np.where(x < 16.0 / 512.0, x / 16.0, np.power(x, 1.8))
            else:
                # ProPhoto encode (linear → gamma)
                data = np.where(x < 1.0 / 512.0, x * 16.0, np.power(x, 1.0 / 1.8))
        
        elif colorspace == ColorSpace.ADOBERGB_GAMMA:
            # Adobe RGB uses simple power function (γ = 2.19921875)
            if inverse:
                data = np.power(x, 2.19921875)
            else:
                data = np.power(x, 1.0 / 2.19921875)

        elif colorspace == ColorSpace.GRAY_GAMMA_1_8:
            # Simple gamma 1.8 (no linear segment)
            if inverse:
                data = np.power(x, 1.8)
            else:
                data = np.power(x, 1.0 / 1.8)

        elif colorspace == ColorSpace.GRAY_GAMMA_2_2:
            # Simple gamma 2.2 (no linear segment)
            if inverse:
                data = np.power(x, 2.2)
            else:
                data = np.power(x, 1.0 / 2.2)

        else:
            raise ValueError(f"Unsupported colorspace for LUT: {colorspace}")
        
        super().__init__(data.astype(np.float32))
        self.colorspace = colorspace
        self.inverse = inverse
        self._initialized = True
    
    def __repr__(self):
        return f"ColorSpaceLUT(colorspace={self.colorspace}, inverse={self.inverse}, size={len(self.data)})"
