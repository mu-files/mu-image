# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

"""Cubic spline interpolation for tone curves.

This module provides CubicSpline, a cubic spline implementation using only NumPy.
Supports natural boundary conditions and custom derivative boundaries.
"""

from __future__ import annotations

import numpy as np


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
