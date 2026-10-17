# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files

"""Small DNGs written at test time, with the opcode under test, and the
three renderers that read them: Adobe's ``dng_validate``, Apple's Core
Image, and mu-image.

``dng_validate`` is compared at the stage right after the opcode list
(``-2`` for OpcodeList1 and OpcodeList2, ``-3`` for OpcodeList3), so
color processing and demosaic do not hide the opcode. Core Image renders
only a final image, so its check is whether the change the opcode makes
to that render follows the change it makes to mu-image's render.

``ReferenceTable`` collects what each test measured and writes it as a
markdown table of where each renderer differs from the DNG spec.
"""

import platform
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from conftest import DNG_VALIDATE_PATH

from muraw import raw_render
from muraw.dngio import DngFile, IfdDataSpec, write_dng_from_array
from muraw.tiff_metadata import MetadataTags

try:
    from muraw._dngio_coreimage import core_image_available, render_dng_coreimage
except ImportError:
    core_image_available = False

OPCODE_FIX_VIGNETTE_RADIAL = 3
OPCODE_FIX_BAD_PIXELS_CONSTANT = 4
OPCODE_MAP_POLYNOMIAL = 8
OPCODE_GAIN_MAP = 9
# Opcodes carry the oldest DNG version that reads them; FixVignetteRadial,
# FixBadPixelsConstant, MapPolynomial and GainMap arrived in DNG 1.3.
DNG_VERSION_1_3 = 0x01030000


def area_spec(area, plane, planes, row_pitch, col_pitch):
    """The 32-byte ``dng_area_spec``. ``area`` is (top, left, bottom,
    right). dng_validate 1.7.1 rejects an area of all zeros, which the
    spec reads as the whole image, so cases pass the image bounds."""
    top, left, bottom, right = area
    return struct.pack(
        ">iiiiIIII", top, left, bottom, right, plane, planes, row_pitch, col_pitch
    )


def opcode(opcode_id, body, min_version=DNG_VERSION_1_3, flags=0):
    """One opcode of an opcode list: id, version, flags, size, body."""
    return struct.pack(">IIII", opcode_id, min_version, flags, len(body)) + body


def map_polynomial(coefficients, area, plane=0, planes=1, row_pitch=1, col_pitch=1):
    """A MapPolynomial opcode with ``coefficients`` c0 .. c_degree."""
    degree = len(coefficients) - 1
    body = area_spec(area, plane, planes, row_pitch, col_pitch)
    body += struct.pack(f">I{degree + 1}d", degree, *coefficients)
    return opcode(OPCODE_MAP_POLYNOMIAL, body)


def fix_vignette_radial(params, center_x, center_y):
    """A FixVignetteRadial opcode: the five gain coefficients k0 .. k4
    (``params`` padded with zeros) and the optical center as a fraction of
    the image's width and height."""
    padded = [*params, *[0.0] * (5 - len(params))]
    body = struct.pack(">7d", *padded, center_x, center_y)
    return opcode(OPCODE_FIX_VIGNETTE_RADIAL, body)


def fix_bad_pixels_constant(constant, bayer_phase):
    """A FixBadPixelsConstant opcode: the code that marks a bad sample and
    the colour of the image's first sample (0 red, 1 green on a red row, 2
    green on a blue row, 3 blue)."""
    body = struct.pack(">II", constant, bayer_phase)
    return opcode(OPCODE_FIX_BAD_PIXELS_CONSTANT, body)


def gain_map(
    gains,
    area,
    origin,
    spacing,
    plane=0,
    planes=1,
    row_pitch=1,
    col_pitch=1,
):
    """A GainMap opcode. ``gains`` is (points_v, points_h, map_planes);
    ``origin`` and ``spacing`` are (vertical, horizontal) fractions of the
    image."""
    points_v, points_h, map_planes = gains.shape
    body = area_spec(area, plane, planes, row_pitch, col_pitch)
    body += struct.pack(">II", points_v, points_h)
    body += struct.pack(">dddd", spacing[0], spacing[1], origin[0], origin[1])
    body += struct.pack(">I", map_planes)
    body += struct.pack(f">{gains.size}f", *np.asarray(gains, np.float32).ravel())
    return opcode(OPCODE_GAIN_MAP, body)


def opcode_list(*opcodes):
    """The OpcodeList1/2/3 tag value: a count, then each opcode."""
    return struct.pack(">I", len(opcodes)) + b"".join(opcodes)


def rgb_ramp(height, width):
    """Linear RGB in [0.05, 0.95]: a ramp across, a ramp down, and a
    diagonal texture with a period of 17 pixels."""
    rows, cols = np.mgrid[0:height, 0:width]
    channels = [
        cols / (width - 1),
        rows / (height - 1),
        ((rows + cols) % 17) / 16,
    ]
    return np.stack([0.05 + 0.9 * channel for channel in channels], axis=-1)


def stored_samples(unit, dtype):
    """``unit`` samples in [0, 1] as the DNG stores them in ``dtype``."""
    if dtype == "uint16":
        return np.round(unit * 65535).astype(np.uint16)
    return unit.astype(dtype)


def bayer_mosaic(rgb, pattern):
    """The mosaic of an (H, W, 3) array for a Bayer ``pattern`` such as
    "GRBG": the colours of the top-left 2 x 2 block, row by row."""
    cfa = np.empty(rgb.shape[:2], rgb.dtype)
    for k, colour in enumerate(pattern):
        row, col = divmod(k, 2)
        cfa[row::2, col::2] = rgb[row::2, col::2, "RGB".index(colour)]
    return cfa


def rggb(rgb):
    """The RGGB mosaic of an (H, W, 3) array."""
    return bayer_mosaic(rgb, "RGGB")


def write_synthetic_dng(
    path, data, photometric, opcode_lists=None, black_level=0, cfa_pattern="RGGB"
):
    """Write ``data`` as a one-IFD DNG with an identity tone curve,
    ``cfa_pattern`` for a CFA, ``opcode_lists`` ({"OpcodeList2": bytes,
    …}), and a BlackLevel tag when ``black_level`` is not 0."""
    tags = MetadataTags()
    tags.add_tag("ProfileToneCurve", [0.0, 0.0, 1.0, 1.0])
    tags.add_tag("UniqueCameraModel", "Synthetic Test Camera")
    if black_level:
        tags.add_tag("BlackLevel", [black_level])
    for name, value in (opcode_lists or {}).items():
        tags.add_tag(name, value)
    write_dng_from_array(
        destination_file=path,
        data_spec=IfdDataSpec(
            data=data, photometric=photometric, cfa_pattern=cfa_pattern, extratags=tags
        ),
    )
    return path


class RendererError(RuntimeError):
    """A renderer could not read a file."""


def sdk_available():
    return DNG_VALIDATE_PATH.exists()


def sdk_version():
    """The dng_validate version string. Does not run the tool when it is
    not installed, so a CI machine can write a partial table."""
    if not sdk_available():
        return "dng_validate (not available)"
    result = subprocess.run([str(DNG_VALIDATE_PATH)], capture_output=True, text=True)
    for line in (result.stdout + result.stderr).splitlines():
        if "version" in line:
            return line.strip()
    return "dng_validate"


def _run_sdk(args):
    result = subprocess.run(
        [str(DNG_VALIDATE_PATH), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    errors = [
        line.strip().strip("*").strip()
        for line in (result.stdout + result.stderr).splitlines()
        if line.strip().startswith("*** Error")
    ]
    if result.returncode != 0 or errors:
        raise RendererError(errors[0] if errors else f"exit code {result.returncode}")


def _unit(samples):
    """Samples as float64 in [0, 1]; a uint16 dump holds 16-bit codes."""
    samples = np.asarray(samples)
    if samples.dtype == np.uint16:
        return samples.astype(np.float64) / 65535
    return samples.astype(np.float64)


def sdk_stage(path, stage):
    """``dng_validate``'s stage ``stage`` image of ``path``, in [0, 1].
    It writes an integer CFA's stage 2 in 16 bits even with ``-32``."""
    base = path.with_suffix(f".sdk_stage{stage}")
    _run_sdk(["-32", f"-{stage}", base, path])
    return np.squeeze(_unit(tifffile.imread(f"{base}.tif")))


def sdk_render(path):
    base = path.with_suffix(".sdk_render")
    _run_sdk(["-16", "-tif", base, path])
    return _unit(tifffile.imread(f"{base}.tif"))


def mu_stage(path, stage):
    """mu-image's image of ``path`` at the same stage as ``sdk_stage``."""
    with DngFile(path) as dng:
        page = dng.get_main_page()
        if stage == 3:
            return np.squeeze(_unit(page.get_camera_raw().realize()))
        if page.is_cfa:
            data, _ = page.get_cfa()
        else:
            data = page.get_linear_raw()
        stage2 = raw_render._linearize(page, data, page.photometric_name)
        return np.squeeze(_unit(stage2.realize()))


def mu_render(path):
    with DngFile(path) as dng:
        return _unit(dng.render_raw(output_dtype=np.uint16).realize())


def coreimage_render(path):
    """Core Image's render of ``path``, in [0, 1]. Core Image returns no
    image, or an empty one, for some files; that raises ``RendererError``."""
    try:
        rgb, _ = render_dng_coreimage(
            str(path), output_dtype=np.float32, use_system_camera_profiles=False
        )
    except RuntimeError as error:
        raise RendererError("Core Image returned no image") from error
    if rgb.size == 0:
        raise RendererError("Core Image returned an empty image")
    return rgb.astype(np.float64)


@dataclass
class ReferenceTable:
    """What each renderer did on each case, written as markdown.

    A table has sections; each section has columns and rows keyed by the
    case name. A cell may carry a note, written once under the table and
    referenced by number from every cell that uses it."""

    title: str
    intro: str
    sections: dict[str, list[str]]
    rows: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def record(self, section, case, column, text, note=None):
        if note is not None:
            if note not in self.notes:
                self.notes.append(note)
            text = f"{text} [{self.notes.index(note) + 1}]"
        self.rows.setdefault(section, {}).setdefault(case, {})[column] = text

    def is_complete(self, expected_cases):
        """True when every case in ``expected_cases`` ({section: [case]})
        has a cell in every column of its section."""
        for section, cases in expected_cases.items():
            for case in cases:
                cells = self.rows.get(section, {}).get(case, {})
                if any(column not in cells for column in self.sections[section]):
                    return False
        return True

    def markdown(self):
        lines = [f"# {self.title}", "", self.intro, ""]
        lines.append(
            f"Measured with {sdk_version()} and Core Image on macOS "
            f"{platform.mac_ver()[0]}."
        )
        for section, columns in self.sections.items():
            lines += ["", f"## {section}", ""]
            lines.append("| Case | " + " | ".join(columns) + " |")
            lines.append("|---" * (len(columns) + 1) + "|")
            for case, cells in self.rows.get(section, {}).items():
                values = [cells.get(column, "not run") for column in columns]
                lines.append(f"| {case} | " + " | ".join(values) + " |")
        if self.notes:
            lines += ["", "## Notes", ""]
            lines += [f"{i}. {note}" for i, note in enumerate(self.notes, start=1)]
        return "\n".join(lines) + "\n"

    def write(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.markdown())
