# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files

"""DNG opcodes on small synthetic DNGs, read by mu-image, dng_validate and
Core Image, each against a NumPy port of the DNG spec.

mu-image must match the spec. A known difference in dng_validate or Core
Image is a strict xfail that records the measured value. Every case
writes what it measured into a table of where the renderers differ from
the spec. The table goes to ``tests/test_outputs/`` on every run, and to
``docs/reference_renderer_differences.md`` when a run measured every
case, so a partial run or a machine without the reference renderers
leaves the document as it is.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import tifffile
from conftest import TEST_OUTPUTS_DIR
from synthetic_dng import (
    ReferenceTable,
    RendererError,
    core_image_available,
    coreimage_render,
    fix_vignette_radial,
    map_polynomial,
    mu_render,
    mu_stage,
    opcode_list,
    rgb_ramp,
    rggb,
    sdk_available,
    sdk_stage,
    stored_samples,
    write_synthetic_dng,
)
from test_fix_vignette import _reference as _fix_vignette_reference
from test_map_polynomial import _COEFFICIENTS, _PER_PLANE_SETS, _reference
from tifffile import COMPRESSION

from muraw.dngio import DngFile, IfdPageSpec, PageEncoding, write_dng_from_page
from muraw.raw_render import parse_opcode_list

# Core Image returns an empty image below 256 x 256 pixels.
HEIGHT = WIDTH = 256
FULL = (0, 0, HEIGHT, WIDTH)
DOC_PATH = Path(__file__).parent.parent / "docs" / "reference_renderer_differences.md"

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


# mi.fix_vignette's default map_spacing, which raw_render uses.
MU_VIGNETTE_MAP_SPACING = 8


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
        map at every ``MU_VIGNETTE_MAP_SPACING``-th column and row."""
        return self.reference(x, map_spacing=MU_VIGNETTE_MAP_SPACING)


@dataclass(frozen=True)
class Case:
    name: str
    photometric: str
    dtype: str
    opcode_list: str
    opcodes: tuple
    sdk_issue: str | None = None
    shape: tuple = (HEIGHT, WIDTH)

    @property
    def stage(self):
        return STAGE[self.opcode_list]

    @property
    def section(self):
        return self.opcodes[0].section

    def stored(self, dtype=None):
        """The samples the DNG stores, in the case's dtype or ``dtype``."""
        unit = rgb_ramp(*self.shape)
        if self.photometric == "CFA":
            unit = rggb(unit)
        return stored_samples(unit, dtype or self.dtype)

    def write(self, path, dtype=None):
        blob = opcode_list(*(op.encode() for op in self.opcodes))
        return write_synthetic_dng(
            path, self.stored(dtype), self.photometric, {self.opcode_list: blob}
        )

    def unit_input(self):
        stored = self.stored()
        if stored.dtype == np.uint16:
            return (stored / np.float32(65535)).astype(np.float32)
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


def _label(case):
    return f"{case.photometric} {case.dtype} {case.opcode_list}: {case.name}"


def _sdk_param(case):
    marks = []
    if case.sdk_issue is not None:
        marks.append(pytest.mark.xfail(strict=True, reason=case.sdk_issue))
    return pytest.param(case, id=_label(case), marks=marks)


def _coreimage_param(case):
    mark = pytest.mark.xfail(strict=True, reason=case.opcodes[0].coreimage_issue)
    return pytest.param(case, id=_label(case), marks=[mark])


SECTION_FILES = "Reading synthetic DNGs"
COLUMN_MU = "mu-image"
COLUMN_SDK = "dng_validate"
COLUMN_CI = "Core Image"


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
        "whether the opcode changes that render the way it changes mu-image's."
    ),
    sections={
        **{
            section: [COLUMN_MU, COLUMN_SDK, COLUMN_CI]
            for section in dict.fromkeys(case.section for case in CASES)
        },
        SECTION_FILES: ["Result"],
    },
)


@pytest.fixture(scope="module", autouse=True)
def reference_table():
    yield TABLE
    TABLE.write(TEST_OUTPUTS_DIR / "test_dng_opcodes_synthetic" / DOC_PATH.name)
    expected = {section: [] for section in TABLE.sections}
    for case in CASES:
        expected[case.section].append(_label(case))
    expected[SECTION_FILES] = [f"Core Image: {case.name}" for case in FILE_CASES] + [
        EMPTY_AREA_CASE
    ]
    if TABLE.is_complete(expected):
        TABLE.write(DOC_PATH)


@pytest.fixture
def out_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("synthetic_dng")


def _max_diff(got, want):
    return float(np.max(np.abs(np.asarray(got, np.float64) - want)))


def _sdk_tolerance(case):
    """dng_validate writes an integer CFA's stage 2 in 16-bit codes."""
    sample = SDK_UINT16_TOLERANCE if case.dtype == "uint16" else SDK_FLOAT_TOLERANCE
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
            f"matches its spacing-{MU_VIGNETTE_MAP_SPACING} map (max {measured:.1e}); "
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


@pytest.mark.skipif(not core_image_available, reason="Core Image not available")
@pytest.mark.parametrize("case", [_coreimage_param(c) for c in CASES])
def test_coreimage_applies_the_opcode(case, out_dir):
    """Core Image reads only 16-bit DNGs, so every case is written as
    uint16. The opcode applies when the change it makes to Core Image's
    render follows the change it makes to mu-image's."""
    plain = write_synthetic_dng(
        out_dir / "plain.dng", case.stored("uint16"), case.photometric
    )
    path = case.write(out_dir / "case.dng", dtype="uint16")
    mu_change = mu_render(path) - mu_render(plain)
    ci_change = coreimage_render(path) - coreimage_render(plain)
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
        note=None if applies else case.opcodes[0].coreimage_issue,
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
