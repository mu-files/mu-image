"""Tests for writing a DNG with a SubIFD pyramid and validating roundtrip."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

import muimg
from muimg.dngio import IfdSpec
from conftest import DNG_VALIDATE_PATH, TEST_FILES_DIR, compute_diff_stats


OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_pyramid_subifd"


TEST_FILES = [
    ("asi676mc.cfa.jxl_lossy.2ifds.dng", True),
    ("canon_eos_r5.cfa.ljpeg.6ifds.dng", False),
    ("sony_dsc-rx100m7.cfa.ljpeg.2ifds.dng", True),
]


def _build_pyramid_rgb_u16(rgb_u16: np.ndarray) -> list[np.ndarray]:
    if rgb_u16.ndim != 3 or rgb_u16.shape[2] != 3:
        raise ValueError(f"rgb_u16 must be (H, W, 3), got {rgb_u16.shape}")

    h, w = rgb_u16.shape[:2]
    if min(h, w) < 256:
        return []

    levels: list[np.ndarray] = []
    cur = rgb_u16
    while True:
        next_h = cur.shape[0] // 2
        next_w = cur.shape[1] // 2
        if min(next_h, next_w) < 128:
            break
        down = cv2.resize(cur, (next_w, next_h), interpolation=cv2.INTER_AREA)
        if down.dtype != np.uint16:
            down = down.astype(np.uint16)
        levels.append(down)
        cur = down

    return levels


def _run_dng_validate_no_warnings(dng_path: Path, output_base: Path, timeout: int = 120) -> None:
    if not DNG_VALIDATE_PATH.exists():
        pytest.skip("dng_validate not available")

    result = subprocess.run(
        [str(DNG_VALIDATE_PATH), "-v", "-16", "-tif", str(output_base), str(dng_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(f"dng_validate failed: {result.stderr}")

    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    if "*** error:" in combined.lower():
        raise AssertionError(f"dng_validate produced errors:\n{combined}")


@pytest.fixture(scope="module")
def output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


@pytest.mark.parametrize("filename, generate_preview", TEST_FILES)
def test_write_subifd_pyramid_roundtrip(filename: str, generate_preview: bool, output_dir: Path):
    dng_path = TEST_FILES_DIR / filename
    if not dng_path.exists():
        pytest.skip(f"Test file not available: {filename}")

    with muimg.DngFile(dng_path) as dng:
        page = dng.get_main_page()
        assert page is not None
        if not page.is_cfa:
            pytest.skip("Main page is not CFA")

        cfa_result = page.get_cfa()
        assert cfa_result is not None
        _, cfa_pattern = cfa_result

        preview = None
        if generate_preview:
            decoded_u8 = dng.render(output_dtype=np.uint8, strict=False)
            assert decoded_u8 is not None
            preview = cv2.resize(
                decoded_u8,
                (min(decoded_u8.shape[1], 512), min(decoded_u8.shape[0], 512)),
                interpolation=cv2.INTER_AREA,
            )

        camera_rgb = dng.get_camera_rgb(demosaic_algorithm="RCD")
        assert camera_rgb is not None

        camera_rgb_u16 = np.clip((camera_rgb * 65535.0).round(), 0.0, 65535.0).astype(np.uint16)
        pyramid_levels = _build_pyramid_rgb_u16(camera_rgb_u16)
        if not pyramid_levels:
            pytest.skip("Image too small for pyramid")

        out_path = output_dir / f"{dng_path.stem}.pyramid.dng"

        subifds: list[IfdSpec] = [IfdSpec(data=page)]
        subifds += [
            IfdSpec(
                data=level,
                bits_per_pixel=16,
                photometric="linear_raw",
            )
            for level in pyramid_levels
        ]

        muimg.write_dng(
            destination_file=out_path,
            subifds=subifds,
            preview_image=preview,
        )

    with muimg.DngFile(out_path) as out_dng:
        out_pages = out_dng.get_flattened_pages()
        linear_pages = [p for p in out_pages if p.is_linear_raw]
        assert len(linear_pages) >= len(pyramid_levels)

        for i, expected in enumerate(pyramid_levels):
            got_arr = linear_pages[i].get_linear_raw()
            assert got_arr is not None
            assert got_arr.shape == expected.shape
            assert got_arr.dtype == expected.dtype
            assert np.array_equal(got_arr, expected)

    dv_base = output_dir / f"{dng_path.stem}.pyramid.dngvalidate"
    _run_dng_validate_no_warnings(out_path, dv_base)


def test_write_subifd_pyramid_roundtrip_cropped_activearea_asi(output_dir: Path):
    filename = "asi676mc.cfa.jxl_lossy.2ifds.dng"
    dng_path = TEST_FILES_DIR / filename
    if not dng_path.exists():
        pytest.skip(f"Test file not available: {filename}")

    with muimg.DngFile(dng_path) as dng:
        page = dng.get_main_page()
        assert page is not None
        if not page.is_cfa:
            pytest.skip("Main page is not CFA")

        ifd0_tags = page.get_ifd0_tags()

        decoded_orig_u8 = dng.render(output_dtype=np.uint8, strict=False)
        assert decoded_orig_u8 is not None

        cfa_result = page.get_cfa()
        assert cfa_result is not None
        cfa_u16_full, cfa_pattern = cfa_result
        assert cfa_u16_full.dtype == np.uint16

        crop_w, crop_h = 1100, 800
        if cfa_u16_full.shape[0] < crop_h or cfa_u16_full.shape[1] < crop_w:
            pytest.skip("Source CFA too small for requested crop")

        crop_top = (cfa_u16_full.shape[0] - crop_h) // 2
        crop_left = (cfa_u16_full.shape[1] - crop_w) // 2
        cfa_u16_crop = cfa_u16_full[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w].copy()

        aa_top, aa_left = 16, 38
        aa_bottom, aa_right = aa_top + 768, aa_left + 1024
        assert aa_bottom <= cfa_u16_crop.shape[0]
        assert aa_right <= cfa_u16_crop.shape[1]

        cfa_u16_active = cfa_u16_crop[aa_top:aa_bottom, aa_left:aa_right]

        from muimg import raw_render

        rgb_u16_active = raw_render.demosaic(cfa_u16_active, cfa_pattern, algorithm="RCD")
        assert rgb_u16_active.dtype == np.uint16
        pyramid_levels = _build_pyramid_rgb_u16(rgb_u16_active)
        if not pyramid_levels:
            pytest.skip("Image too small for pyramid")

        out_path = output_dir / f"{dng_path.stem}.cropped_activearea.pyramid.dng"

        pad = 4
        ifd0_tags.add_tag("ActiveArea", [pad, pad, crop_h - pad, crop_w - pad])
        ifd0_tags.add_tag("DefaultCropOrigin", [aa_left - pad, aa_top - pad])
        ifd0_tags.add_tag("DefaultCropSize", [aa_right - aa_left, aa_bottom - aa_top])

        assert aa_left - pad >= 2
        assert aa_top - pad >= 2
        assert (crop_w - pad) - aa_right >= 2
        assert (crop_h - pad) - aa_bottom >= 2

        subifds: list[IfdSpec] = [
            IfdSpec(
                data=cfa_u16_crop,
                bits_per_pixel=16,
                photometric="cfa",
                cfa_pattern=cfa_pattern,
            )
        ]
        subifds += [
            IfdSpec(
                data=level,
                bits_per_pixel=16,
                photometric="linear_raw",
                inherit_ifd0_tags_from_source=False,
            )
            for level in pyramid_levels
        ]

        muimg.write_dng(
            destination_file=out_path,
            ifd0_tags=ifd0_tags,
            subifds=subifds,
            preview_image=None,
        )

    with muimg.DngFile(out_path) as out_dng:
        out_main = out_dng.get_main_page()
        assert out_main is not None

        decoded_crop_u8 = out_dng.render(output_dtype=np.uint8, strict=False)
        assert decoded_crop_u8 is not None

        expected_crop_u8 = decoded_orig_u8[
            crop_top + aa_top : crop_top + aa_bottom,
            crop_left + aa_left : crop_left + aa_right,
        ]

        if decoded_crop_u8.shape != expected_crop_u8.shape:
            pytest.fail(
                f"Rendered crop shape mismatch: got={decoded_crop_u8.shape} expected={expected_crop_u8.shape}"
            )
        if decoded_crop_u8.dtype != expected_crop_u8.dtype:
            pytest.fail(
                f"Rendered crop dtype mismatch: got={decoded_crop_u8.dtype} expected={expected_crop_u8.dtype}"
            )

        stats = compute_diff_stats(decoded_crop_u8, expected_crop_u8)
        mean_ok = stats["mean"] < 0.01
        p99_ok = stats["p99"] < 0.5
        max_ok = stats["max"] <= 0.5
        if not (mean_ok and p99_ok and max_ok):
            base = output_dir / f"{dng_path.stem}.cropped_activearea.render"
            got_path = base.with_suffix(".got.tif")
            exp_path = base.with_suffix(".expected.tif")
            diff_path = base.with_suffix(".diff.tif")

            got_bgr = cv2.cvtColor(decoded_crop_u8, cv2.COLOR_RGB2BGR)
            exp_bgr = cv2.cvtColor(expected_crop_u8, cv2.COLOR_RGB2BGR)
            diff = cv2.absdiff(got_bgr, exp_bgr)
            diff_vis = np.clip(diff.astype(np.uint16) * 16, 0, 255).astype(np.uint8)

            cv2.imwrite(str(got_path), got_bgr)
            cv2.imwrite(str(exp_path), exp_bgr)
            cv2.imwrite(str(diff_path), diff_vis)

            raise AssertionError(
                f"Rendered crop diff too large: mean={stats['mean']:.4f}% p99={stats['p99']:.4f}% max={stats['max']:.4f}%. "
                f"Wrote {got_path.name}, {exp_path.name}, {diff_path.name} to {output_dir}"
            )

        out_pages = out_dng.get_flattened_pages()
        linear_pages = [p for p in out_pages if p.is_linear_raw]
        assert len(linear_pages) >= len(pyramid_levels)

        for i, expected in enumerate(pyramid_levels):
            got_arr = linear_pages[i].get_linear_raw()
            assert got_arr is not None
            assert got_arr.shape == expected.shape
            assert got_arr.dtype == expected.dtype
            assert np.array_equal(got_arr, expected)

    dv_base = output_dir / f"{dng_path.stem}.cropped_activearea.pyramid.dngvalidate"
    _run_dng_validate_no_warnings(out_path, dv_base)
