from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import tifffile

import muimg
from muimg.dngio import RawStageSelector
from muimg.dngio import IfdSpec
from conftest import DNG_VALIDATE_PATH, TEST_FILES_DIR, load_tiff, compute_diff_stats


OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "test_dng_validate_raw_stages"


TEST_FILES = [
    #"asi676mc.linearraw.jxl_lossy.1ifds.dng",
    "canon_eos_r5_mark_ii.linearraw.jxl_lossy.6ifds.dng",
    "canon_eos_r5.cfa.ljpeg.6ifds.dng",
]


DEBUG_FINAL_RAW_IFD_ONLY = True
DUMP_MUIMG_STAGE_TIFFS = True


def _run_dng_validate_dump_stages(dng_path: Path, stage1_base: Path, stage2_base: Path, timeout: int = 120) -> None:
    result = subprocess.run(
        [
            str(DNG_VALIDATE_PATH),
            "-v",
            "-16",
            "-1",
            str(stage1_base),
            "-2",
            str(stage2_base),
            str(dng_path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dng_validate failed: {result.stderr}")


@pytest.fixture(scope="module")
def output_dir() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


@pytest.mark.parametrize("filename", TEST_FILES)
def test_raw_stage_outputs_match_dng_validate(filename: str, output_dir: Path):
    if not DNG_VALIDATE_PATH.exists():
        pytest.skip("dng_validate not available")

    dng_path = TEST_FILES_DIR / filename
    if not dng_path.exists():
        pytest.skip(f"Test file not available: {filename}")

    stem = dng_path.stem

    with muimg.DngFile(dng_path) as dng:
        pages = dng.get_flattened_pages()

        for ifd_index, page in enumerate(pages):
            if not (page.is_cfa or page.is_linear_raw):
                continue

            # Roundtrip each IFD to isolate it, so dng_validate dumps correspond
            # unambiguously to this page.
            roundtrip_dng = output_dir / f"{stem}_ifd{ifd_index}.dng"

            muimg.write_dng(
                destination_file=roundtrip_dng,
                subifds=[IfdSpec(data=page)],
            )

            stage1_base = output_dir / f"{stem}_ifd{ifd_index}_stage1"
            stage2_base = output_dir / f"{stem}_ifd{ifd_index}_stage2"

            _run_dng_validate_dump_stages(roundtrip_dng, stage1_base, stage2_base)

            dv_stage1 = load_tiff(Path(str(stage1_base) + ".tif"))
            dv_stage2 = load_tiff(Path(str(stage2_base) + ".tif"))
            assert dv_stage1 is not None
            assert dv_stage2 is not None

            with muimg.DngFile(roundtrip_dng) as roundtrip_file:
                roundtrip_page = roundtrip_file.get_main_page()
                assert roundtrip_page is not None

                if roundtrip_page.is_cfa:
                    cfa_stage1 = roundtrip_page.get_cfa(stage=RawStageSelector.RAW)
                    cfa_stage2 = roundtrip_page.get_cfa(stage=RawStageSelector.LINEARIZED_PLUS_OPS)
                    assert cfa_stage1 is not None
                    assert cfa_stage2 is not None
                    mu_stage1, _ = cfa_stage1
                    mu_stage2, _ = cfa_stage2
                else:
                    mu_stage1 = roundtrip_page.get_linear_raw(stage=RawStageSelector.RAW)
                    mu_stage2 = roundtrip_page.get_linear_raw(stage=RawStageSelector.LINEARIZED_PLUS_OPS)

                assert mu_stage1 is not None
                assert mu_stage2 is not None

                assert np.issubdtype(mu_stage2.dtype, np.floating)

                if DUMP_MUIMG_STAGE_TIFFS:
                    mu_stage1_tif = output_dir / f"{stem}_ifd{ifd_index}_muimg_stage1.tif"
                    mu_stage2_u16_tif = output_dir / f"{stem}_ifd{ifd_index}_muimg_stage2_u16.tif"

                    try:
                        tifffile.imwrite(str(mu_stage1_tif), mu_stage1)
                    except Exception:
                        # Some viewers don't like single-channel; leave as-is and let tifffile decide.
                        tifffile.imwrite(str(mu_stage1_tif), mu_stage1)

                    # muimg stage2 is a normalized float image (expected range ~[0..1]).
                    # dng_validate -2 dumps stage2 as a uint16 TIFF using the full 16-bit range.
                    # For apples-to-apples comparisons and for easier human inspection, we
                    # scale muimg's float stage2 to uint16 [0..65535].
                    mu_stage2_u16 = np.clip(
                        (mu_stage2 * 65535.0).round(),
                        0.0,
                        65535.0,
                    ).astype(np.uint16)
                    tifffile.imwrite(str(mu_stage2_u16_tif), mu_stage2_u16)

            assert mu_stage1.shape == dv_stage1.shape
            assert mu_stage2.shape == dv_stage2.shape

            # Stage1 comparison:
            mu_stage1_compare = mu_stage1

            stats1 = compute_diff_stats(mu_stage1_compare, dv_stage1)

            mu_stage2_compare = mu_stage2
            # Stage2 is defined by our API to be normalized float output.
            assert np.issubdtype(mu_stage2.dtype, np.floating)

            if dv_stage2.dtype == np.uint16:
                # Stage2 comparison:
                # - muimg stage2 is normalized float (range-mapped) output.
                # - dng_validate -2 dumps stage2 as uint16 full-range.
                # Scale muimg to uint16 so compute_diff_stats compares like-for-like.
                mu_stage2_compare = np.clip(
                    (mu_stage2 * 65535.0).round(),
                    0.0,
                    65535.0,
                ).astype(np.uint16)

            stats2 = compute_diff_stats(mu_stage2_compare, dv_stage2)

            # Stage 1 should be essentially identical.
            if stats1["max"] >= 0.01:
                pytest.fail(f"Stage1 mismatch")

            # Stage 2 should be essentially identical (normalization + ActiveArea crop).
            if stats2["max"] >= 0.01:
                pytest.fail(f"Stage2 mismatch")
