# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files

"""DNG opcodes on small synthetic DNGs, read by mu-image, dng_validate and
Core Image, each against a NumPy port of the DNG spec.

mu-image must match the spec. A known difference in dng_validate or Core
Image is a strict xfail that records the measured value. Every case
writes what it measured into a table of where the renderers differ from
the spec. The table goes to ``tests/test_outputs/`` on every run. When a
run measured every case, it also goes to
``reference_renderer_differences.md`` in the private mu-image-engine
checkout next to this repo, under ``docs/staged/``, where the docs wait
until they are published. A partial run, a machine without the reference
renderers, or a machine without that folder leaves the document as it is.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import tifffile
from conftest import TEST_OUTPUTS_DIR
from synthetic_dng import (
    ReferenceTable,
    RendererError,
    bayer_mosaic,
    core_image_available,
    coreimage_render,
    fix_bad_pixels_constant,
    fix_vignette_radial,
    gain_map,
    map_polynomial,
    mu_render,
    mu_stage,
    opcode_list,
    profile_gain_table_map2,
    rgb_ramp,
    rggb,
    sdk_available,
    sdk_render,
    sdk_stage,
    stored_samples,
    write_synthetic_dng,
)
from test_apply_gain_map import _reference as _gain_map_reference
from test_fix_bad_pixels import _reference as _fix_bad_pixels_reference
from test_fix_vignette import _reference as _fix_vignette_reference
from test_map_polynomial import _COEFFICIENTS, _PER_PLANE_SETS, _reference
from tifffile import COMPRESSION

from muraw.dngio import DngFile, IfdPageSpec, PageEncoding, write_dng_from_page
from muraw.raw_render import (
    VIGNETTE_MAP_SPACING,
    parse_opcode_list,
    parse_profile_gain_table_map,
)

# Core Image returns an empty image below 256 x 256 pixels.
HEIGHT = WIDTH = 256
FULL = (0, 0, HEIGHT, WIDTH)
STAGED_DOCS = Path(__file__).parents[3] / "mu-image-engine" / "docs" / "staged"
DOC_PATH = STAGED_DOCS / "reference_renderer_differences.md"

# What each list's stage holds: OpcodeList1 runs on the stored samples,
# OpcodeList2 on the linearized ones, OpcodeList3 after demosaic.
STAGE = {"OpcodeList1": 2, "OpcodeList2": 2, "OpcodeList3": 3}

SDK_STAGE1_SCALING = (
    "At stage 1 (OpcodeList1) on 16-bit data, dng_validate multiplies "
    "coefficient c_j by 65535^(j-1) to map raw codes, but the buffer the "
    "opcode reads was already divided by 65535 when it was copied to float "
    "(`dng_opcode_MapPolynomial::BufferPixelType` and `RefCopyArea16_R32` in "
    "DNG SDK 1.7.1). Every coefficient but c1 comes out wrong: c0 is divided "
    "by 65535, c2 multiplied by 65535."
)
COREIMAGE_NO_MAP_POLYNOMIAL = (
    "Core Image does not apply MapPolynomial, in any opcode list. Its render "
    "is the same with and without the opcode."
)
COREIMAGE_FIX_VIGNETTE = (
    "Core Image applies FixVignetteRadial, but the change it makes to its "
    "render does not follow the change to mu-image's (correlation 0.46 to "
    "0.71). Its render also brightens by about 2% near the optical center, "
    "where the opcode's gain is 1, so it changes more than the opcode asks; "
    "what else it does is not known."
)

# What dng_validate's stage image may differ from the spec by: 32-bit
# samples, and an integer CFA's stage 2, which it writes in 16-bit codes.
SDK_FLOAT_TOLERANCE = 4e-6
SDK_UINT16_TOLERANCE = 1 / 65535


@dataclass(frozen=True)
class MapPolynomial:
    coefficients: tuple
    area: tuple = FULL
    plane: int = 0
    planes: int = 3
    row_pitch: int = 1
    col_pitch: int = 1

    section = "MapPolynomial"
    coreimage_issue = COREIMAGE_NO_MAP_POLYNOMIAL
    sdk_gain_tolerance = 0.0

    def encode(self):
        return map_polynomial(
            self.coefficients,
            self.area,
            self.plane,
            self.planes,
            self.row_pitch,
            self.col_pitch,
        )

    def reference(self, x):
        top, left, bottom, right = self.area
        return _reference(
            x,
            top=top,
            left=left,
            bottom=bottom,
            right=right,
            start_plane=self.plane,
            num_planes=self.planes,
            row_pitch=self.row_pitch,
            col_pitch=self.col_pitch,
            coefficients=list(self.coefficients),
            degree=len(self.coefficients) - 1,
        )

    def mu_reference(self, x):
        """What mu-image computes by design: the spec."""
        return self.reference(x)


@dataclass(frozen=True)
class FixVignetteRadial:
    params: tuple
    center_x: float
    center_y: float

    section = "FixVignetteRadial"
    coreimage_issue = COREIMAGE_FIX_VIGNETTE
    # dng_validate looks the gain up in a table of 65537 uint16 entries
    # with at most 15 fraction bits, indexed by r^2 in 16.16 fixed point
    # (`dng_opcode_FixVignetteRadial::Prepare`, `RefVignetteMask16`). The
    # largest difference measured from the spec, 4.1e-5, times 1.1.
    sdk_gain_tolerance = 4.5e-5

    def encode(self):
        return fix_vignette_radial(self.params, self.center_x, self.center_y)

    def reference(self, x, map_spacing=1):
        return _fix_vignette_reference(
            x,
            params=list(self.params),
            center_x=self.center_x,
            center_y=self.center_y,
            map_spacing=map_spacing,
        )

    def mu_reference(self, x):
        """What mu-image computes by design: the gain interpolated from a
        map at every ``VIGNETTE_MAP_SPACING``-th column and row."""
        return self.reference(x, map_spacing=VIGNETTE_MAP_SPACING)


@dataclass(frozen=True)
class GainMap:
    """A GainMap opcode with a table of ``points`` (points_v, points_h,
    map_planes) gains drawn from ``low`` .. ``high``. ``origin`` and
    ``spacing`` are (vertical, horizontal) fractions of the image; with no
    spacing the map's first and last samples sit at the image's edges."""

    points: tuple
    origin: tuple = (0.0, 0.0)
    spacing: tuple | None = None
    low: float = 0.6
    high: float = 1.8
    seed: int = 7
    area: tuple = FULL
    plane: int = 0
    planes: int = 3
    row_pitch: int = 1
    col_pitch: int = 1

    section = "GainMap"
    coreimage_issue = None
    sdk_gain_tolerance = 0.0

    @property
    def gains(self):
        rng = np.random.default_rng(self.seed)
        return rng.uniform(self.low, self.high, self.points).astype(np.float32)

    def _spacing(self):
        if self.spacing is not None:
            return self.spacing
        return tuple(1.0 / max(points - 1, 1) for points in self.points[:2])

    def encode(self):
        return gain_map(
            self.gains,
            self.area,
            self.origin,
            self._spacing(),
            self.plane,
            self.planes,
            self.row_pitch,
            self.col_pitch,
        )

    def reference(self, x):
        """GainMap as the spec places it: map sample (i, j) at origin +
        spacing * index, in fractions of the image measured from pixel
        centers, and each product capped at 1."""
        height, width = x.shape[:2]
        spacing_v, spacing_h = self._spacing()
        top, left, bottom, right = self.area
        return _gain_map_reference(
            x,
            self.gains,
            map_x0=self.origin[1] * width - 0.5,
            map_y0=self.origin[0] * height - 0.5,
            map_spacing_x=spacing_h * width,
            map_spacing_y=spacing_v * height,
            top=top,
            left=left,
            bottom=bottom,
            right=right,
            start_plane=self.plane,
            num_planes=self.planes,
            row_pitch=self.row_pitch,
            col_pitch=self.col_pitch,
            clamp="max_one",
        )

    def mu_reference(self, x):
        """What mu-image computes by design: the spec."""
        return self.reference(x)


# Where FixBadPixelsConstant cases plant bad samples in a 256 x 256 mosaic:
# a 2 x 2 block, so one sample of each colour whatever the phase; samples
# on and next to each edge, at both parities; and the four corners.
_BAD_SITES = (
    *((row, col) for row in (100, 101) for col in (100, 101)),
    (0, 40),
    (0, 41),
    (1, 60),
    (255, 70),
    (255, 71),
    (254, 90),
    (40, 0),
    (41, 0),
    (60, 1),
    (70, 255),
    (71, 255),
    (90, 254),
    (0, 0),
    (0, 255),
    (255, 0),
    (255, 255),
)
# A 5 x 5 block of bad samples: its centre has no good neighbour of either
# colour, so it keeps the constant.
_BAD_BLOCK = np.s_[200:205, 200:205]

SDK_BAYER_PHASE = (
    "dng_validate reads BayerPhase the other way round from the spec. The "
    "spec gives the colour of the first sample: 0 red, 1 green on a red row, "
    "2 green on a blue row, 3 blue (DNG 1.7.1, page 112). "
    "`dng_opcode_FixBadPixelsConstant::IsGreen` (dng_bad_pixels.h) takes a "
    "site as green when row + col + phase + (phase >> 1) is even, which is "
    "a red or blue site by the spec. A bad red or blue sample therefore "
    "becomes the mean of its diagonal neighbours, which are the other of red "
    "and blue, so it takes the wrong colour. A bad green sample becomes the "
    "mean of the green samples 2 away instead of its diagonal greens: the "
    "right colour, from farther neighbours. With the phase swapped (0 with 1, 2 "
    "with 3) its stage image equals the spec exactly, edges included."
)
COREIMAGE_NO_FIX_BAD_PIXELS = (
    "Core Image does not apply FixBadPixelsConstant. Its render is the same "
    "with and without the opcode and still shows the bad samples, which "
    "change it by up to 0.96 from the same image without them."
)


@dataclass(frozen=True)
class FixBadPixelsConstant:
    """A FixBadPixelsConstant opcode, and the bad samples it fixes:
    ``constant`` at ``_BAD_SITES`` and ``_BAD_BLOCK`` of the stored codes."""

    constant: int
    bayer_phase: int

    section = "FixBadPixelsConstant"
    coreimage_issue = COREIMAGE_NO_FIX_BAD_PIXELS
    sdk_gain_tolerance = 0.0

    def encode(self):
        return fix_bad_pixels_constant(self.constant, self.bayer_phase)

    def plant(self, codes):
        """``codes`` with the bad samples set to the constant."""
        assert not np.any(codes == self.constant), "the image holds the constant"
        codes = codes.copy()
        for row, col in _BAD_SITES:
            codes[row, col] = self.constant
        codes[_BAD_BLOCK] = self.constant
        return codes

    def reference(self, x):
        """The opcode on the stored codes, with the phase as the spec
        defines it. The spec does not say what replaces a neighbour past
        the image; this follows the SDK, which repeats the two outer rows
        and columns."""
        codes = np.round(x * 65535).astype(np.uint16)
        fixed = _fix_bad_pixels_reference(codes, self.constant, self.bayer_phase)
        return fixed.astype(np.float64) / 65535

    def mu_reference(self, x):
        """What mu-image computes by design: the spec."""
        return self.reference(x)


@dataclass(frozen=True)
class Case:
    name: str
    photometric: str
    dtype: str
    opcode_list: str
    opcodes: tuple
    sdk_issue: str | None = None
    shape: tuple = (HEIGHT, WIDTH)
    # A uint16 BlackLevel tag; the stored codes are black_level + value *
    # (65535 - black_level).
    black_level: int = 0
    # What Core Image does on this case, when it is not what it does on
    # every case of the opcode.
    coreimage_issue: str | None = None
    cfa_pattern: str = "RGGB"

    @property
    def stage(self):
        return STAGE[self.opcode_list]

    @property
    def section(self):
        return self.opcodes[0].section

    def stored(self, dtype=None):
        """The samples the DNG stores, in the case's dtype or ``dtype``,
        with the bad samples of a FixBadPixelsConstant opcode."""
        unit = rgb_ramp(*self.shape)
        if self.photometric == "CFA":
            unit = bayer_mosaic(unit, self.cfa_pattern)
        if self.black_level:
            codes = self.black_level + unit * (65535 - self.black_level)
            stored = np.round(codes).astype(np.uint16)
        else:
            stored = stored_samples(unit, dtype or self.dtype)
        for op in self.opcodes:
            if isinstance(op, FixBadPixelsConstant):
                stored = op.plant(stored)
        return stored

    def write(self, path, dtype=None):
        blob = opcode_list(*(op.encode() for op in self.opcodes))
        return self.write_plain(path, dtype, {self.opcode_list: blob})

    def write_plain(self, path, dtype=None, opcode_lists=None):
        """The case's DNG with ``opcode_lists``, none by default."""
        return write_synthetic_dng(
            path,
            self.stored(dtype),
            self.photometric,
            opcode_lists,
            black_level=self.black_level,
            cfa_pattern=self.cfa_pattern,
        )

    def unit_input(self):
        """The linearized samples: a uint16 code less the black level, over
        65535 less the black level."""
        stored = self.stored()
        if stored.dtype == np.uint16:
            black = np.float32(self.black_level)
            return ((stored - black) / (np.float32(65535) - black)).astype(np.float32)
        return stored.astype(np.float32)

    def reference(self, mu=False):
        """The stage image the DNG spec gives, or with ``mu`` the one
        mu-image computes by design. OpcodeList1 output on 16-bit data is
        stored back as 16-bit codes before linearization."""
        x = self.unit_input()
        for op in self.opcodes:
            x = op.mu_reference(x) if mu else op.reference(x)
        if self.opcode_list == "OpcodeList1" and self.dtype == "uint16":
            x = np.clip(np.round(x * 65535), 0, 65535) / 65535
        return x.astype(np.float64)


def _rgb(coefficients, **kwargs):
    return MapPolynomial(tuple(coefficients), **kwargs)


def _cfa(coefficients, **kwargs):
    return MapPolynomial(tuple(coefficients), planes=1, **kwargs)


_DEGREE_3 = _COEFFICIENTS[3]

CASES = [
    *(
        Case(f"degree {degree}", "LINEAR_RAW", "float32", "OpcodeList2", (_rgb(c),))
        for degree, c in _COEFFICIENTS.items()
    ),
    Case(
        "area 33..201 x 41..223, row and column pitch 2",
        "LINEAR_RAW",
        "float32",
        "OpcodeList2",
        (_rgb(_DEGREE_3, area=(33, 41, 201, 223), row_pitch=2, col_pitch=2),),
    ),
    Case(
        "plane 1 only",
        "LINEAR_RAW",
        "float32",
        "OpcodeList2",
        (_rgb(_DEGREE_3, plane=1, planes=1),),
    ),
    Case(
        "one opcode per plane, as cameras write",
        "LINEAR_RAW",
        "float32",
        "OpcodeList2",
        tuple(
            _rgb(_PER_PLANE_SETS[4 * plane : 4 * plane + 4], plane=plane, planes=1)
            for plane in range(3)
        ),
    ),
    Case(
        "negative inputs (x - 0.5, then degree 3)",
        "LINEAR_RAW",
        "float32",
        "OpcodeList2",
        (_rgb([-0.5, 1.0]), _rgb(_DEGREE_3)),
    ),
    Case("degree 3", "LINEAR_RAW", "float32", "OpcodeList3", (_rgb(_DEGREE_3),)),
    Case("degree 3", "LINEAR_RAW", "uint16", "OpcodeList2", (_rgb(_DEGREE_3),)),
    Case("degree 3", "CFA", "uint16", "OpcodeList2", (_cfa(_DEGREE_3),)),
    Case(
        "row and column pitch 2 (red sites)",
        "CFA",
        "uint16",
        "OpcodeList2",
        (_cfa(_DEGREE_3, row_pitch=2, col_pitch=2),),
    ),
    Case(
        "degree 1",
        "CFA",
        "uint16",
        "OpcodeList1",
        (_cfa(_COEFFICIENTS[1]),),
        sdk_issue=SDK_STAGE1_SCALING,
    ),
    Case(
        "degree 3",
        "CFA",
        "uint16",
        "OpcodeList1",
        (_cfa(_DEGREE_3),),
        sdk_issue=SDK_STAGE1_SCALING,
    ),
]

# The vignette cases are 512 x 512: at mu-image's map spacing of 8 the map
# differs from the per-pixel gain by about 3e-4 there, against 1e-3 at
# 256 x 256, so the table's per-pixel column stays readable.
VIGNETTE_SHAPE = (512, 512)
_OFF_CENTER = FixVignetteRadial((0.4, -0.2, 0.3, 0.1, 0.05), 0.3, 0.6)
# Gains up to 3.6 at the farthest corner, so the [0, 1] clamp acts.
_CLAMPED = FixVignetteRadial((1.5, 0.8, 0.0, 0.0, 0.3), 0.3, 0.6)
# Unlike DSC_1437's coefficients: a darkening second term.
_CFA_VIGNETTE = FixVignetteRadial((0.6, -0.3, 0.2), 0.45, 0.4)

CASES += [
    Case(
        "off-center, five terms",
        "LINEAR_RAW",
        "float32",
        "OpcodeList3",
        (_OFF_CENTER,),
        shape=VIGNETTE_SHAPE,
    ),
    Case(
        "gains large enough to clamp",
        "LINEAR_RAW",
        "float32",
        "OpcodeList3",
        (_CLAMPED,),
        shape=VIGNETTE_SHAPE,
    ),
    Case(
        "off-center, five terms",
        "LINEAR_RAW",
        "uint16",
        "OpcodeList2",
        (_OFF_CENTER,),
        shape=VIGNETTE_SHAPE,
    ),
    Case(
        "three terms",
        "CFA",
        "uint16",
        "OpcodeList2",
        (_CFA_VIGNETTE,),
        shape=VIGNETTE_SHAPE,
    ),
    Case(
        "three terms",
        "CFA",
        "uint16",
        "OpcodeList1",
        (_CFA_VIGNETTE,),
        shape=VIGNETTE_SHAPE,
    ),
]


def _bayer_phases(**kwargs):
    """One GainMap per Bayer phase, as cameras write: row and column pitch
    2 from an area offset by the phase, a one-plane map each."""
    return tuple(
        GainMap(
            (5, 6, 1),
            seed=11 + 2 * row + col,
            area=(row, col, HEIGHT, WIDTH),
            planes=1,
            row_pitch=2,
            col_pitch=2,
            **kwargs,
        )
        for row in (0, 1)
        for col in (0, 1)
    )


COREIMAGE_GAIN_MAP_PLANES = (
    "Core Image returns no image for a GainMap that starts at plane 1, or "
    "that has fewer map planes than planes (one map plane for three); it "
    "reads one that starts at plane 0 with as many map planes as planes."
)
COREIMAGE_GAIN_MAP_PER_PLANE = (
    "With a different gain on each of three planes, the change Core Image's "
    "render makes follows mu-image's less closely (correlation 0.84 to 0.90) "
    "than with the same gain on every plane (0.96, measured outside this "
    "table); what it does differently with the planes is not known."
)
COREIMAGE_GAIN_MAP_PITCH = (
    "Core Image applies a GainMap to every sample of its area, whatever the "
    "row and column pitch: its render changes as much with column pitch 2 "
    "as with 1, and on a CFA as much for the red sites alone as for the "
    "whole mosaic."
)
COREIMAGE_GAIN_MAP_LISTS = (
    "Core Image ignores every GainMap of an opcode list that holds more than "
    "one, logging 'Gain Map will be ignored because of Unexpected size of "
    "gainmap parameters'. It therefore ignores the four Bayer phases cameras "
    "write, one GainMap each."
)


def _rgb_gain_map(name, op, **kwargs):
    """A float32 LINEAR_RAW case with ``op`` in OpcodeList2."""
    return Case(name, "LINEAR_RAW", "float32", "OpcodeList2", (op,), **kwargs)


_THREE_PLANES = GainMap((5, 6, 3))

CASES += [
    _rgb_gain_map(
        "one map plane",
        GainMap((5, 6, 1)),
        coreimage_issue=COREIMAGE_GAIN_MAP_PLANES,
    ),
    _rgb_gain_map(
        "three map planes",
        _THREE_PLANES,
        coreimage_issue=COREIMAGE_GAIN_MAP_PER_PLANE,
    ),
    _rgb_gain_map(
        "planes 1 and 2 (G and B), three map planes",
        GainMap((5, 6, 3), plane=1, planes=2),
        coreimage_issue=COREIMAGE_GAIN_MAP_PLANES,
    ),
    _rgb_gain_map(
        "area 10..70 x 17..101",
        GainMap((5, 6, 3), area=(10, 17, 70, 101)),
        coreimage_issue=COREIMAGE_GAIN_MAP_PER_PLANE,
    ),
    _rgb_gain_map(
        "row pitch 2, column pitch 3",
        GainMap((5, 6, 3), row_pitch=2, col_pitch=3),
        coreimage_issue=COREIMAGE_GAIN_MAP_PITCH,
    ),
    _rgb_gain_map(
        "7 x 9 grid, origin 0.1, spacing 0.12, edges held",
        GainMap((7, 9, 3), origin=(0.1, 0.1), spacing=(0.12, 0.12)),
        coreimage_issue=COREIMAGE_GAIN_MAP_PER_PLANE,
    ),
    _rgb_gain_map(
        "gains of 1.2 to 2.5, products capped at 1",
        GainMap((5, 6, 3), low=1.2, high=2.5),
        coreimage_issue=COREIMAGE_GAIN_MAP_PER_PLANE,
    ),
    # The spec caps a product at 1 only; a negative gain gives a negative
    # sample.
    _rgb_gain_map(
        "gains of -0.8 to 1.0, negative products kept",
        GainMap((5, 6, 3), low=-0.8, high=1.0),
        coreimage_issue=COREIMAGE_GAIN_MAP_PER_PLANE,
    ),
    Case(
        "three map planes",
        "LINEAR_RAW",
        "float32",
        "OpcodeList3",
        (_THREE_PLANES,),
        coreimage_issue=COREIMAGE_GAIN_MAP_PER_PLANE,
    ),
    Case(
        "three map planes",
        "LINEAR_RAW",
        "uint16",
        "OpcodeList2",
        (_THREE_PLANES,),
        coreimage_issue=COREIMAGE_GAIN_MAP_PER_PLANE,
    ),
    Case(
        "pitch 1, area 10..70 x 17..101",
        "CFA",
        "uint16",
        "OpcodeList2",
        (GainMap((5, 6, 1), area=(10, 17, 70, 101), planes=1),),
    ),
    *(
        Case(
            "four Bayer phases",
            "CFA",
            "uint16",
            opcode_list_name,
            _bayer_phases(),
            coreimage_issue=COREIMAGE_GAIN_MAP_LISTS,
        )
        for opcode_list_name in ("OpcodeList2", "OpcodeList1")
    ),
    Case(
        "four Bayer phases, black level 1024",
        "CFA",
        "uint16",
        "OpcodeList2",
        _bayer_phases(),
        black_level=1024,
        coreimage_issue=COREIMAGE_GAIN_MAP_LISTS,
    ),
    # The pitch counts from the area's corner, before the image here, so
    # rows and columns 1, 3, 5, … change: the blue sites.
    Case(
        "area from row and column -1 at pitch 2",
        "CFA",
        "uint16",
        "OpcodeList2",
        (
            GainMap(
                (5, 6, 1),
                area=(-1, -1, HEIGHT, WIDTH),
                planes=1,
                row_pitch=2,
                col_pitch=2,
            ),
        ),
        coreimage_issue=COREIMAGE_GAIN_MAP_PITCH,
    ),
]

# The CFA pattern whose first sample has the colour of each Bayer phase.
_PHASE_PATTERNS = {0: "RGGB", 1: "GRBG", 2: "GBRG", 3: "BGGR"}

CASES += [
    *(
        Case(
            f"phase {phase} ({pattern}), constant 0",
            "CFA",
            "uint16",
            "OpcodeList1",
            (FixBadPixelsConstant(0, phase),),
            sdk_issue=SDK_BAYER_PHASE,
            cfa_pattern=pattern,
        )
        for phase, pattern in _PHASE_PATTERNS.items()
    ),
    Case(
        "phase 0 (RGGB), constant 4095",
        "CFA",
        "uint16",
        "OpcodeList1",
        (FixBadPixelsConstant(4095, 0),),
        sdk_issue=SDK_BAYER_PHASE,
    ),
]


@dataclass(frozen=True)
class ProfileGainTableMap2:
    """A ProfileGainTableMap2 tag: a smooth table of ``points``
    (points_v, points_h, table_points) gains of 1 ± 0.3 whose grid covers
    the image, stored as ``data_type``."""

    gamma: float
    data_type: str = "float32"
    points: tuple = (5, 7, 9)
    weights: tuple = (0.3, 0.4, 0.15, 0.05, 0.1)
    seed: int = 5

    section = "ProfileGainTableMap2"
    coreimage_issue = None

    @property
    def gains(self):
        rng = np.random.default_rng(self.seed)
        points_v, points_h, table_points = self.points
        phase = rng.uniform(0, 2 * np.pi, (points_v, points_h, 1))
        position = np.linspace(0, 1, table_points)[None, None, :]
        return (1 + 0.3 * np.sin(2 * np.pi * position + phase)).astype(np.float32)

    def encode(self):
        spacing = tuple(1.0 / (points - 1) for points in self.points[:2])
        return profile_gain_table_map2(
            self.gains, (0.0, 0.0), spacing, self.weights, self.gamma, self.data_type
        )


# The gain table applies inside the render, so its cases compare final
# renders. A BaselineExposure of 0.5 makes the weight's exposure gain
# 2^0.5. The samples are the ramp at half its level: the brightest, times
# a gain of 1.3 and 2^0.5, stays below white, where the two renderers
# clip out-of-range colours differently with or without a gain table.
PGTM_BASELINE_EXPOSURE = 0.5
PGTM_LEVEL = 0.5
# The largest difference measured between mu-image's and dng_validate's
# 16-bit renders, 2.0e-4, times 1.1.
PGTM_TOLERANCE = 2.2e-4
# The renderers' exposure ramp for a positive exposure is the SDK's
# linear one only with this mu-image render parameter.
SDK_RENDERING_PARAMS = {"highlight_preserving_exposure": False}


COREIMAGE_PGTM_GAMMA = (
    "With a gamma of 0.6, the change Core Image's render makes follows "
    "mu-image's less closely (correlation 0.87) than with a gamma of 1, "
    "where it applies the table as mu-image does. mu-image matches "
    "dng_validate on both."
)
COREIMAGE_PGTM_FLOAT16 = (
    "With the table stored as float16, the change Core Image's render makes "
    "hardly follows mu-image's (correlation 0.19), against 0.87 for the same "
    "table stored as float32. mu-image matches dng_validate on both."
)


@dataclass(frozen=True)
class TableCase:
    """A LINEAR_RAW DNG with a ProfileGainTableMap2 tag in IFD0."""

    name: str
    dtype: str
    table: ProfileGainTableMap2
    coreimage_issue: str | None = None
    photometric = "LINEAR_RAW"
    opcode_list = "IFD0"

    @property
    def section(self):
        return self.table.section

    @property
    def opcodes(self):
        return (self.table,)

    def write_plain(self, path, dtype=None, extra_tags=None):
        unit = PGTM_LEVEL * rgb_ramp(HEIGHT, WIDTH)
        return write_synthetic_dng(
            path,
            stored_samples(unit, dtype or self.dtype),
            self.photometric,
            extra_tags={
                "BaselineExposure": PGTM_BASELINE_EXPOSURE,
                **(extra_tags or {}),
            },
        )

    def write(self, path, dtype=None):
        return self.write_plain(
            path, dtype, {"ProfileGainTableMap2": self.table.encode()}
        )


TABLE_CASES = [
    TableCase("gamma 0.6", "float32", ProfileGainTableMap2(0.6), COREIMAGE_PGTM_GAMMA),
    TableCase("gamma 1", "float32", ProfileGainTableMap2(1.0)),
    TableCase(
        "gamma 0.6, float16 table",
        "float32",
        ProfileGainTableMap2(0.6, "float16"),
        COREIMAGE_PGTM_FLOAT16,
    ),
    TableCase("gamma 0.6", "uint16", ProfileGainTableMap2(0.6), COREIMAGE_PGTM_GAMMA),
]


def _label(case):
    return f"{case.photometric} {case.dtype} {case.opcode_list}: {case.name}"


def _sdk_param(case):
    marks = []
    if case.sdk_issue is not None:
        marks.append(pytest.mark.xfail(strict=True, reason=case.sdk_issue))
    return pytest.param(case, id=_label(case), marks=marks)


def _coreimage_issue(case):
    return case.coreimage_issue or case.opcodes[0].coreimage_issue


def _coreimage_param(case):
    issue = _coreimage_issue(case)
    marks = [] if issue is None else [pytest.mark.xfail(strict=True, reason=issue)]
    return pytest.param(case, id=_label(case), marks=marks)


SECTION_FILES = "Reading synthetic DNGs"
COLUMN_MU = "mu-image"
COLUMN_SDK = "dng_validate"
COLUMN_CI = "Core Image"
COLUMN_MU_SDK = "mu-image against dng_validate"


@dataclass(frozen=True)
class FileCase:
    name: str
    photometric: str
    dtype: str
    shape: tuple
    issue: str | None = None


COREIMAGE_FLOAT = "Core Image returns no image for a float32 DNG, LINEAR_RAW or CFA."
COREIMAGE_SMALL = (
    "Core Image returns an empty image for a 96 x 130 DNG, LINEAR_RAW or CFA; "
    "the same data at 256 x 256 renders. The cases therefore use 256 x 256."
)
FILE_CASES = [
    FileCase("uint16 LINEAR_RAW, 256 x 256", "LINEAR_RAW", "uint16", (256, 256)),
    FileCase("uint16 CFA, 256 x 256", "CFA", "uint16", (256, 256)),
    FileCase(
        "uint16 LINEAR_RAW, 96 x 130",
        "LINEAR_RAW",
        "uint16",
        (96, 130),
        COREIMAGE_SMALL,
    ),
    FileCase("uint16 CFA, 96 x 130", "CFA", "uint16", (96, 130), COREIMAGE_SMALL),
    FileCase(
        "float32 LINEAR_RAW, 256 x 256",
        "LINEAR_RAW",
        "float32",
        (256, 256),
        COREIMAGE_FLOAT,
    ),
    FileCase("float32 CFA, 256 x 256", "CFA", "float32", (256, 256), COREIMAGE_FLOAT),
]
SDK_EMPTY_AREA = (
    "dng_validate 1.7.1 rejects an area of all zeros, which the spec reads as "
    "the whole image: `dng_area_spec::GetData` clamps the row and column pitch "
    "to the area's height and width, 0, and then throws because a pitch is "
    "not 1. Camera files write the image bounds instead."
)
EMPTY_AREA_CASE = "dng_validate: MapPolynomial with an area of all zeros"

TABLE = ReferenceTable(
    title="Where dng_validate and Core Image differ from the DNG spec",
    intro=(
        "Generated by `muraw/tests/test_dng_opcodes_synthetic.py`; run that "
        "module to regenerate this file. Each case writes a small synthetic DNG "
        "and compares each renderer with a NumPy port of the DNG spec. "
        "mu-image and dng_validate are compared at the stage right after the "
        "opcode list; Core Image gives only a final render, so its cell says "
        "whether the opcode changes that render the way it changes mu-image's. "
        "A ProfileGainTableMap2 applies inside the render, so its section "
        "compares mu-image's final render with dng_validate's instead of a "
        "NumPy port."
    ),
    sections={
        **{
            section: [COLUMN_MU, COLUMN_SDK, COLUMN_CI]
            for section in dict.fromkeys(case.section for case in CASES)
        },
        ProfileGainTableMap2.section: [COLUMN_MU_SDK, COLUMN_CI],
        SECTION_FILES: ["Result"],
    },
)


@pytest.fixture(scope="module", autouse=True)
def reference_table():
    yield TABLE
    TABLE.write(TEST_OUTPUTS_DIR / "test_dng_opcodes_synthetic" / DOC_PATH.name)
    expected = {section: [] for section in TABLE.sections}
    for case in [*CASES, *TABLE_CASES]:
        expected[case.section].append(_label(case))
    expected[SECTION_FILES] = [f"Core Image: {case.name}" for case in FILE_CASES] + [
        EMPTY_AREA_CASE
    ]
    if TABLE.is_complete(expected) and STAGED_DOCS.is_dir():
        TABLE.write(DOC_PATH)


@pytest.fixture
def out_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("synthetic_dng")


def _max_diff(got, want):
    return float(np.max(np.abs(np.asarray(got, np.float64) - want)))


def _sdk_tolerance(case):
    """dng_validate writes an integer CFA's stage 2 in 16-bit codes. With a
    black level it also rounds the linearized samples to 16-bit codes
    before the opcode list, and each gain scales that half code."""
    sample = SDK_UINT16_TOLERANCE if case.dtype == "uint16" else SDK_FLOAT_TOLERANCE
    if case.black_level:
        largest_gain = max(float(np.max(op.gains)) for op in case.opcodes)
        sample += 0.5 / 65535 * largest_gain
    return sample + sum(op.sdk_gain_tolerance for op in case.opcodes)


@pytest.mark.parametrize("case", [pytest.param(c, id=_label(c)) for c in CASES])
def test_mu_image_matches_the_spec(case, out_dir):
    """mu-image matches the spec as it computes it by design; for
    FixVignetteRadial that is the gain interpolated from its map, and the
    table also records the difference from the per-pixel gain."""
    reference = case.reference(mu=True)
    assert _max_diff(reference, case.unit_input()) > 1e-3, "the case changes nothing"
    path = case.write(out_dir / "case.dng")
    got = mu_stage(path, case.stage)
    measured = _max_diff(got, reference)
    tolerance = 4e-6 + (1 / 65535 if case.opcode_list == "OpcodeList1" else 0)
    matches = measured <= tolerance
    text = (
        f"matches (max {measured:.1e})" if matches else f"differs: max {measured:.3g}"
    )
    if matches and isinstance(case.opcodes[0], FixVignetteRadial):
        text = (
            f"matches its spacing-{VIGNETTE_MAP_SPACING} map (max {measured:.1e}); "
            f"per-pixel spec max {_max_diff(got, case.reference()):.1e}"
        )
    TABLE.record(case.section, _label(case), COLUMN_MU, text)
    assert matches, f"mu-image differs from the spec by {measured}"


@pytest.mark.skipif(not sdk_available(), reason="dng_validate not available")
@pytest.mark.parametrize("case", [_sdk_param(c) for c in CASES])
def test_dng_validate_matches_the_spec(case, out_dir):
    path = case.write(out_dir / "case.dng")
    measured = _max_diff(sdk_stage(path, case.stage), case.reference())
    matches = measured <= _sdk_tolerance(case)
    TABLE.record(
        case.section,
        _label(case),
        COLUMN_SDK,
        f"matches (max {measured:.1e})" if matches else f"differs: max {measured:.3g}",
        note=None if matches else case.sdk_issue,
    )
    assert matches, f"dng_validate differs from the spec by {measured}"


@pytest.mark.skipif(not sdk_available(), reason="dng_validate not available")
@pytest.mark.parametrize("case", [pytest.param(c, id=_label(c)) for c in TABLE_CASES])
def test_mu_image_matches_dng_validate_on_a_gain_table(case, out_dir):
    """mu-image's 16-bit render of a DNG with the table matches
    dng_validate's, and the table changes dng_validate's render clearly."""
    plain = case.write_plain(out_dir / "plain.dng")
    path = case.write(out_dir / "case.dng")
    sdk = sdk_render(path)
    assert _max_diff(sdk, sdk_render(plain)) > 0.05, "the table changes nothing"
    measured = _max_diff(mu_render(path, SDK_RENDERING_PARAMS), sdk)
    matches = measured <= PGTM_TOLERANCE
    TABLE.record(
        case.section,
        _label(case),
        COLUMN_MU_SDK,
        f"matches (max {measured:.1e})" if matches else f"differs: max {measured:.3g}",
    )
    assert matches, f"mu-image differs from dng_validate by {measured}"


@pytest.mark.skipif(not core_image_available, reason="Core Image not available")
@pytest.mark.parametrize("case", [_coreimage_param(c) for c in [*CASES, *TABLE_CASES]])
def test_coreimage_applies_the_opcode(case, out_dir):
    """Core Image reads only 16-bit DNGs, so every case is written as
    uint16. The opcode applies when the change it makes to Core Image's
    render follows the change it makes to mu-image's."""
    plain = case.write_plain(out_dir / "plain.dng", dtype="uint16")
    path = case.write(out_dir / "case.dng", dtype="uint16")
    mu_change = mu_render(path) - mu_render(plain)
    issue = _coreimage_issue(case)
    try:
        ci_change = coreimage_render(path) - coreimage_render(plain)
    except RendererError as error:
        TABLE.record(case.section, _label(case), COLUMN_CI, str(error), note=issue)
        raise
    ci_mean = float(np.mean(np.abs(ci_change)))
    mu_mean = float(np.mean(np.abs(mu_change)))
    if ci_mean < 1e-6:
        applies, text = False, "ignores the opcode"
    else:
        correlation = float(np.corrcoef(mu_change.ravel(), ci_change.ravel())[0, 1])
        applies = correlation >= 0.9 and 0.5 <= ci_mean / mu_mean <= 2
        text = (
            "applies it"
            if applies
            else f"applies it differently (correlation {correlation:.2f})"
        )
    TABLE.record(
        case.section,
        _label(case),
        COLUMN_CI,
        text,
        note=None if applies else issue,
    )
    assert applies, f"Core Image {text}"


def _file_param(case):
    marks = []
    if case.issue is not None:
        marks.append(pytest.mark.xfail(strict=True, reason=case.issue))
    return pytest.param(case, id=case.name, marks=marks)


@pytest.mark.skipif(not core_image_available, reason="Core Image not available")
@pytest.mark.parametrize("case", [_file_param(c) for c in FILE_CASES])
def test_coreimage_reads_the_file(case, out_dir):
    unit = rgb_ramp(*case.shape)
    if case.photometric == "CFA":
        unit = rggb(unit)
    path = write_synthetic_dng(
        out_dir / "file.dng", stored_samples(unit, case.dtype), case.photometric
    )
    row = f"Core Image: {case.name}"
    try:
        coreimage_render(path)
    except RendererError as error:
        TABLE.record(SECTION_FILES, row, "Result", str(error), note=case.issue)
        raise
    TABLE.record(SECTION_FILES, row, "Result", "reads it")


@pytest.mark.skipif(not sdk_available(), reason="dng_validate not available")
@pytest.mark.xfail(strict=True, reason=SDK_EMPTY_AREA, raises=RendererError)
def test_dng_validate_reads_an_area_of_all_zeros(out_dir):
    case = Case(
        "empty area",
        "LINEAR_RAW",
        "float32",
        "OpcodeList2",
        (_rgb(_DEGREE_3, area=(0, 0, 0, 0)),),
    )
    path = case.write(out_dir / "case.dng")
    try:
        sdk_stage(path, 2)
    except RendererError as error:
        TABLE.record(
            SECTION_FILES, EMPTY_AREA_CASE, "Result", str(error), note=SDK_EMPTY_AREA
        )
        raise
    TABLE.record(SECTION_FILES, EMPTY_AREA_CASE, "Result", "reads it")


def test_map_polynomial_encoder_round_trips():
    op = _rgb(
        [0.1, 0.9, -0.2],
        area=(3, 5, 70, 90),
        plane=1,
        planes=2,
        row_pitch=2,
        col_pitch=3,
    )
    (parsed,) = parse_opcode_list(opcode_list(op.encode()))
    assert parsed["type"] == "MapPolynomial"
    assert parsed["area"] == {"top": 3, "left": 5, "bottom": 70, "right": 90}
    assert (parsed["plane"], parsed["planes"]) == (1, 2)
    assert (parsed["row_pitch"], parsed["col_pitch"]) == (2, 3)
    np.testing.assert_array_equal(parsed["coefficients"], [0.1, 0.9, -0.2])


def test_fix_vignette_radial_encoder_round_trips():
    op = FixVignetteRadial((0.6, -0.3, 0.2), 0.45, 0.4)
    (parsed,) = parse_opcode_list(opcode_list(op.encode()))
    assert parsed["type"] == "FixVignetteRadial"
    np.testing.assert_array_equal(parsed["coefficients"], [0.6, -0.3, 0.2, 0.0, 0.0])
    assert (parsed["center_x"], parsed["center_y"]) == (0.45, 0.4)


def test_gain_map_encoder_round_trips():
    op = GainMap(
        (3, 4, 2),
        origin=(0.1, 0.2),
        spacing=(0.3, 0.25),
        area=(3, 5, 70, 90),
        plane=1,
        planes=2,
        row_pitch=2,
        col_pitch=3,
    )
    (parsed,) = parse_opcode_list(opcode_list(op.encode()))
    assert parsed["type"] == "GainMap"
    assert parsed["area"] == {"top": 3, "left": 5, "bottom": 70, "right": 90}
    assert (parsed["plane"], parsed["planes"]) == (1, 2)
    assert (parsed["row_pitch"], parsed["col_pitch"]) == (2, 3)
    assert (parsed["origin_v"], parsed["origin_h"]) == (0.1, 0.2)
    assert (parsed["spacing_v"], parsed["spacing_h"]) == (0.3, 0.25)
    np.testing.assert_array_equal(parsed["gain_values"], op.gains)


@pytest.mark.parametrize("data_type", ["float32", "float16"])
def test_profile_gain_table_map2_encoder_round_trips(data_type):
    table = ProfileGainTableMap2(0.6, data_type, points=(3, 4, 5))
    byteorder = "<" if sys.byteorder == "little" else ">"
    parsed = parse_profile_gain_table_map(
        table.encode(), is_version2=True, byteorder=byteorder
    )
    assert (parsed["points_v"], parsed["points_h"]) == (3, 4)
    assert parsed["num_table_points"] == 5
    assert (parsed["spacing_v"], parsed["spacing_h"]) == (0.5, 1 / 3)
    assert (parsed["origin_v"], parsed["origin_h"]) == (0.0, 0.0)
    np.testing.assert_array_equal(parsed["weights"], np.float32(table.weights))
    assert parsed["gamma"] == np.float32(0.6)
    np.testing.assert_array_equal(
        parsed["gains"], table.gains.astype(data_type).astype(np.float32)
    )


def test_fix_bad_pixels_constant_encoder_round_trips():
    (parsed,) = parse_opcode_list(opcode_list(FixBadPixelsConstant(4095, 2).encode()))
    assert parsed["type"] == "FixBadPixelsConstant"
    assert (parsed["constant"], parsed["bayer_phase"]) == (4095, 2)


def _opcode_list_tags(path):
    """{tag code: (TIFF type, bytes)} of the opcode lists in ``path``."""
    found = {}
    with tifffile.TiffFile(path) as tif:
        for page in [tif.pages[0], *(tif.pages[0].pages or [])]:
            for code in (51008, 51009, 51022):
                if code in page.tags:
                    tag = page.tags[code]
                    found[code] = (int(tag.dtype), bytes(tag.value))
    return found


def test_opcode_lists_are_written_as_undefined_and_survive_a_copy(out_dir):
    """The DNG spec types the opcode lists UNDEFINED (7); dng_validate
    rejects any other type. OpcodeList1 goes only in a CFA IFD. Copying
    and transcoding the page keeps every list byte for byte."""
    cfa = stored_samples(rggb(rgb_ramp(HEIGHT, WIDTH)), "uint16")
    lists = {
        name: opcode_list(_cfa([0.0, 0.5 + 0.1 * i]).encode())
        for i, name in enumerate(["OpcodeList1", "OpcodeList2", "OpcodeList3"])
    }
    src = write_synthetic_dng(out_dir / "src.dng", cfa, "CFA", lists)
    want = {
        code: (7, lists[name])
        for code, name in zip((51008, 51009, 51022), lists, strict=True)
    }
    assert _opcode_list_tags(src) == want
    transcode = PageEncoding(compression=COMPRESSION.NONE)
    for mode, encoding in [("copy", None), ("transcode", transcode)]:
        dst = out_dir / f"{mode}.dng"
        with DngFile(src) as dng:
            write_dng_from_page(
                dst, IfdPageSpec(page=dng.get_main_page(), transcode_encoding=encoding)
            )
        assert _opcode_list_tags(dst) == want, mode
