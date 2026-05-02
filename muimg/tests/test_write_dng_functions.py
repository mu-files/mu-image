# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

"""Test write_dng(), write_dng_from_array(), and write_dng_from_page() functions.

Validates complete DNG file roundtrip by enumerating all pages from a source DNG,
creating IfdPageSpecs, writing to a new DNG, and comparing metadata and pixels.
"""

import logging
import numpy as np
import subprocess
import sys
from pathlib import Path
from tifffile import COMPRESSION

# Suppress tifffile warnings about "shaped series shape does not match page shape"
logging.getLogger('tifffile').setLevel(logging.CRITICAL)

from muimg.dngio import (
    write_dng,
    write_dng_from_page,
    DngFile,
    IfdPageSpec,
    PageOp,
    SubFileType,
    create_dng_from_page,
    PreviewParams,
    PyramidParams,
)
from conftest import run_dng_validate, compute_diff_stats
import cv2

# Test output configuration
USE_PERSISTENT_OUTPUT = False


def dump_comparison_images(img1: np.ndarray, img2: np.ndarray, output_dir: Path, prefix: str):
    """Dump comparison images to TIFF for visual inspection.
    
    Args:
        img1: First image (reference)
        img2: Second image (to compare)
        output_dir: Directory to save images
        prefix: Prefix for filenames
    """
    import tifffile
    
    dump_dir = output_dir / "pixel_comparison_dumps"
    dump_dir.mkdir(exist_ok=True, parents=True)
    
    img1_path = dump_dir / f"{prefix}_reference.tif"
    img2_path = dump_dir / f"{prefix}_output.tif"
    diff_path = dump_dir / f"{prefix}_diff.tif"
    
    # Save images and absolute difference
    tifffile.imwrite(img1_path, img1)
    tifffile.imwrite(img2_path, img2)
    abs_diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16)).astype(np.uint8)
    tifffile.imwrite(diff_path, abs_diff)
    
    print(f"      ⚠ Comparison images dumped to {dump_dir}/")
    print(f"        - {img1_path.name}")
    print(f"        - {img2_path.name}")
    print(f"        - {diff_path.name}")


def compare_page_metadata(src, out_page, spec=None, extra_skip_tags=None):
    """Compare metadata between source and output pages.
    
    Args:
        src: Source DngPage or MetadataTags
        out_page: Output DngPage
        spec: Optional IfdPageSpec to determine if transcode was required
        extra_skip_tags: Optional set of additional tag names to skip
    
    Returns:
        tuple: (compared_count, mismatches) where mismatches is a list of mismatch descriptions
    """
    from muimg.tiff_metadata import resolve_tag, MetadataTags
    
    # Get tag codes from both pages
    src_tags = src if isinstance(src, MetadataTags) else src.get_page_tags()
    out_tags = out_page.get_page_tags()
    
    src_tag_codes = {code for code, _, _, _, _ in src_tags}
    out_tag_codes = {code for code, _, _, _, _ in out_tags}
    
    missing_in_output = src_tag_codes - out_tag_codes
    extra_in_output = out_tag_codes - src_tag_codes
    
    if missing_in_output:
        missing_names = [resolve_tag(code)[1] or f"Tag_{code}" for code in missing_in_output]
        print(f"      WARNING: Tags missing in output: {missing_names}")
    if extra_in_output:
        extra_names = [resolve_tag(code)[1] or f"Tag_{code}" for code in extra_in_output]
        print(f"      WARNING: Extra tags in output: {extra_names}")
    
    # Tags that are always expected to differ (file layout, software, etc.)
    # These are in _TIFFWRITER_MANAGED_TAGS and managed automatically by tifffile
    SKIP_TAG_NAMES = {
        'StripOffsets',         # File layout dependent
        'StripByteCounts',      # File layout dependent
        'RowsPerStrip',         # File layout dependent, managed by tifffile
        'FreeOffsets',          # File layout dependent
        'FreeByteCounts',       # File layout dependent
        'Software',             # Changes from Adobe to tifffile.py
        'TileOffsets',          # File layout dependent
        'TileByteCounts',       # File layout dependent
        'SubIFDs',              # File layout dependent
        'DNGVersion',           # Our code uses 1.7.1.0, source may have 1.7.0.0
        'DNGBackwardVersion',   # Our code uses 1.7.1.0, source may have older
        'ExifTag',              # tifffile can't write ExifTag IFD pointer
    }
    
    # Add extra skip tags if provided
    if extra_skip_tags:
        SKIP_TAG_NAMES = SKIP_TAG_NAMES | extra_skip_tags
    
    # Conditionally skip PlanarConfiguration if it's the default (1/CONTIG)
    # tifffile only writes it when non-default
    src_planar = src_tags.get_tag('PlanarConfiguration')
    if src_planar == 1:
        SKIP_TAG_NAMES.add('PlanarConfiguration')
    
    # For pages with unsupported tiles, also skip compression-related tags
    if spec and spec.requires_transcode():
        SKIP_TAG_NAMES = SKIP_TAG_NAMES | {
            'Compression',      # Changes to NONE for unsupported tiles
            'TileWidth',        # May change when falling back to strips
            'TileLength',       # May change when falling back to strips
            'JXLDistance',      # Not present when falling back to uncompressed
            'JXLEffort',        # Not present when falling back to uncompressed
            'JXLDecodeSpeed',   # Not present when falling back to uncompressed
        }
    
    # Compare all tags from source page
    mismatches = []
    compared_count = 0
    
    # Enumerate all tags in source page
    for tag_code, _, _, _, _ in src_tags:
        # Resolve tag name
        _, tag_name, _ = resolve_tag(tag_code)
        if not tag_name:
            tag_name = f"Tag_{tag_code}"
        
        # Skip tags that are expected to differ
        if tag_name in SKIP_TAG_NAMES:
            continue
        
        # Get normalized tag values from both pages
        # For XMP, get as string to compare content directly
        return_type = str if tag_name == 'XMP' else None
        src_val = src_tags.get_tag(tag_name, return_type=return_type)
        out_val = out_page.get_tag(tag_name, return_type=return_type)
        
        # Check if tag exists in output (handle numpy arrays properly)
        if out_val is None or (isinstance(out_val, np.ndarray) and out_val.size == 0):
            mismatches.append(f"{tag_name}: missing in output")
            continue
        
        compared_count += 1
        
        # Compare tag values
        # Special handling for XMP - print diff if different
        if tag_name == 'XMP':
            if src_val != out_val:
                # Print first 500 chars of each to see the difference
                print(f"\n      XMP DIFF:")
                print(f"        Source (first 500): {src_val[:500]}")
                print(f"        Output (first 500): {out_val[:500]}")
                mismatches.append(f"{tag_name}: XMP content differs")
        # Handle numpy arrays
        elif isinstance(src_val, np.ndarray) or isinstance(out_val, np.ndarray):
            if not isinstance(src_val, np.ndarray) or not isinstance(out_val, np.ndarray):
                mismatches.append(f"{tag_name}: type mismatch (array vs non-array)")
            elif not np.array_equal(src_val, out_val):
                mismatches.append(f"{tag_name}: array mismatch")
        elif src_val != out_val:
            # For strings, show repr to see null terminators
            if isinstance(src_val, str) and isinstance(out_val, str):
                mismatches.append(f"{tag_name}: {repr(src_val)} != {repr(out_val)}")
            else:
                mismatches.append(f"{tag_name}: {src_val} != {out_val}")
    
    return compared_count, mismatches


def test_complete_file_roundtrip(tmp_path):
    """Test complete DNG file roundtrip with all pages.
    
    Enumerates all pages from canon_eos_r5_cfa_ljpeg_6ifds.dng, creates
    IfdPageSpecs for each, writes to a new DNG, and validates that the
    new file matches the original in both metadata and decoded pixels.
    """
    # Setup paths
    test_dir = Path(__file__).parent
    source_path = test_dir / "dngfiles" / "canon_eos_r5_cfa_ljpeg_6ifds.dng"
    
    if USE_PERSISTENT_OUTPUT:
        output_dir = test_dir / "test_outputs" / "test_write_dng_functions"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = tmp_path
    
    output_path = output_dir / "roundtrip_output.dng"
    
    # Verify source file exists
    assert source_path.exists(), f"Source file not found: {source_path}"
    
    print(f"\n{'='*80}")
    print(f"Testing complete file roundtrip")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")
    
    # Step 1: Load source DNG and enumerate pages
    print("Step 1: Loading source DNG and enumerating pages...")
    with DngFile(source_path) as source_dng:
        all_pages = source_dng.get_flattened_pages()
        print(f"  Found {len(all_pages)} total pages")
        
        # Print page structure for debugging
        for i, page in enumerate(all_pages):
            print(f"  Page {i}: "
                  f"is_ifd0={page.is_ifd0}, "
                  f"photometric={page.photometric_name}, "
                  f"size={page.imagewidth}x{page.imagelength}, "
                  f"compression={page.compression.name if page.compression else 'None'}, "
                  f"subfiletype={page.get_tag('NewSubfileType')}")
        
        # Step 3: Identify IFD structure
        # get_flattened_pages() returns IFD0 first, then SubIFDs
        print("\nStep 3: Identifying IFD structure...")
        assert len(all_pages) > 0, "No pages found in source DNG"
        
        ifd0_page = all_pages[0]
        assert ifd0_page.is_ifd0, "First page is not IFD0"
        subifd_pages = all_pages[1:]
        
        print(f"  IFD0: photometric={ifd0_page.photometric_name}, "
              f"size={ifd0_page.imagewidth}x{ifd0_page.imagelength}")
        for i, page in enumerate(subifd_pages):
            print(f"  SubIFD {i}: photometric={page.photometric_name}, "
                  f"size={page.imagewidth}x{page.imagelength}")
        
        print(f"  Total: 1 IFD0 + {len(subifd_pages)} SubIFDs")
        
        # Step 4: Create IfdPageSpecs
        print("\nStep 4: Creating IfdPageSpecs...")
        
        # Determine subfiletype from source page properties
        # IFD0: check if it's main or preview
        ifd0_subfiletype = SubFileType.MAIN_IMAGE if ifd0_page.is_main_image else SubFileType.PREVIEW_IMAGE
        
        ifd0_spec = IfdPageSpec(
            page=ifd0_page,
            subfiletype=ifd0_subfiletype,
            page_operation=PageOp.COPY,
        )
        print(f"  Created IFD0 spec: subfiletype={ifd0_subfiletype} "
              f"(is_main_image={ifd0_page.is_main_image})")
        
        subifd_specs = []
        for i, page in enumerate(subifd_pages):
            # Query each SubIFD page to determine if it's main or preview
            subfiletype = SubFileType.MAIN_IMAGE if page.is_main_image else SubFileType.PREVIEW_IMAGE
            
            spec = IfdPageSpec(
                page=page,
                subfiletype=subfiletype,
                page_operation=PageOp.COPY,
            )
            subifd_specs.append(spec)
            print(f"  Created SubIFD {i} spec: subfiletype={subfiletype} "
                  f"(is_main_image={page.is_main_image})")
        
        # Step 5: Write new DNG
        print("\nStep 5: Writing new DNG...")
        write_dng(
            destination_file=output_path,
            ifd0_spec=ifd0_spec,
            subifds=subifd_specs,
        )
        print(f"  Successfully wrote {output_path}")
        print(f"  File size: {output_path.stat().st_size:,} bytes")
    
    # Step 6: Validate new DNG
    print("\nStep 6: Validating new DNG with dng_validate...")
    validate_output = output_dir / "roundtrip_output_validate"
    
    try:
        run_dng_validate(
            output_path,
            validate_output,
            validate=True,
            indent="  "
        )
        print("  ✓ Validation passed (no errors/warnings)")
    except AssertionError as e:
        print(f"  ✗ Validation failed: {e}")
        raise
    
    # Step 7 & 8: Compare metadata and pixels
    print("\nStep 7 & 8: Comparing metadata and pixels...")
    
    # Get the specs we used to write the file (created earlier)
    all_specs = [ifd0_spec] + subifd_specs
    
    with DngFile(source_path) as source_dng, DngFile(output_path) as output_dng:
        source_pages = source_dng.get_flattened_pages()
        output_pages = output_dng.get_flattened_pages()
        
        assert len(source_pages) == len(output_pages), (
            f"Page count mismatch: source has {len(source_pages)}, "
            f"output has {len(output_pages)}"
        )
        
        for i, (src_page, out_page, spec) in enumerate(zip(source_pages, output_pages, all_specs)):
            print(f"\n  Comparing page {i}:")
            
            # Compare metadata
            print(f"    Metadata comparison:")
            compared_count, mismatches = compare_page_metadata(src_page, out_page, spec)
            
            if mismatches:
                print(f"      Tag mismatches:")
                for mismatch in mismatches:
                    print(f"        - {mismatch}")
                raise AssertionError(f"Page {i}: {len(mismatches)} tag mismatches found")
            
            print(f"      ✓ All {compared_count} compared tags match")
            
            # Compare decoded pixels
            print(f"    Pixel comparison:")
            
            # Decode both pages to RGB
            src_rgb = src_page.decode_to_rgb(output_dtype=np.uint16)
            out_rgb = out_page.decode_to_rgb(output_dtype=np.uint16)
            
            assert src_rgb is not None, f"Page {i}: Failed to decode source page"
            assert out_rgb is not None, f"Page {i}: Failed to decode output page"
            
            # Verify shapes match
            assert src_rgb.shape == out_rgb.shape, (
                f"Page {i}: RGB shape mismatch: {src_rgb.shape} != {out_rgb.shape}"
            )
            
            # Compute difference statistics
            diff_stats = compute_diff_stats(src_rgb, out_rgb)
            
            print(f"      Mean diff: {diff_stats['mean']:.4f}%")
            print(f"      P99 diff:  {diff_stats['p99']:.4f}%")
            print(f"      Max diff:  {diff_stats['max']:.4f}%")
            
            # All pages should be pixel-perfect:
            # - Copied pages: compressed data copied exactly
            # - Transcoded pages: decoded and stored uncompressed (no lossy compression)
            assert diff_stats['mean'] == 0.0, (
                f"Page {i}: Mean pixel difference {diff_stats['mean']:.4f}% "
                f"(expected 0.0% - exact match)"
            )
            assert diff_stats['max'] == 0.0, (
                f"Page {i}: Max pixel difference {diff_stats['max']:.4f}% "
                f"(expected 0.0% - exact match)"
            )
            
            print(f"      ✓ Pixels match exactly (0.0% difference)")
    
    # Step 9: Test CLI metadata command
    print("\nStep 9: Testing 'dng metadata' CLI command...")
    
    # Run metadata command on source file
    source_metadata_result = subprocess.run(
        [sys.executable, "-m", "muimg.cli", "dng", "metadata", str(source_path)],
        capture_output=True,
        text=True,
    )
    assert source_metadata_result.returncode == 0, f"Source metadata CLI failed: {source_metadata_result.stderr}"
    assert len(source_metadata_result.stdout) > 0, "Source metadata output is empty"
    
    # Run metadata command on output file
    output_metadata_result = subprocess.run(
        [sys.executable, "-m", "muimg.cli", "dng", "metadata", str(output_path)],
        capture_output=True,
        text=True,
    )
    assert output_metadata_result.returncode == 0, f"Output metadata CLI failed: {output_metadata_result.stderr}"
    assert len(output_metadata_result.stdout) > 0, "Output metadata output is empty"
    
    # Read EXIF tag names from source DNG file
    exif_tags = set()
    exif_dict = source_dng.get_tag('ExifTag')
    if exif_dict:
        for tag_name in exif_dict.keys():
            exif_tags.add(tag_name + ':')
    
    print(f"  EXIF tags to skip: {sorted(exif_tags)[:10]}... ({len(exif_tags)} total)")
    
    # Compare metadata outputs
    def normalize_metadata(text, exif_tags_to_skip):
        """Strip tags that differ, skip summary lines 4-5, stop at detailed SubIFD 3 section."""
        lines = []
        for line in text.split('\n'):
            # Stop at detailed SubIFD 3 section (which is IFD 4 in the CLI output)
            if '=== SubIFD 3 (--ifd 4) ===' in line:
                break
            # Skip summary lines for IFDs 4 and 5
            if line.startswith('4:') or line.startswith('5:'):
                continue
            # Skip EXIF tags
            if any(tag in line for tag in exif_tags_to_skip):
                continue
            # Skip tags that are added/changed by write process
            if any(x in line for x in ['ImageDescription:', 'XResolution:', 'YResolution:', 'ResolutionUnit:',
                                        'StripOffsets:', 'TileOffsets:', 'SubIFDs:', 'PlanarConfiguration:', 
                                        'Total file size:', 'DNGVersion:', 'DNGBackwardVersion:',
                                        'FocalPlaneXResolution:', 'FocalPlaneYResolution:',
                                        'ExifTag:']):
                continue
            lines.append(line)
        return '\n'.join(lines)
    
    source_normalized = normalize_metadata(source_metadata_result.stdout, exif_tags)
    output_normalized = normalize_metadata(output_metadata_result.stdout, exif_tags)
    
    if source_normalized == output_normalized:
        print(f"  ✓ CLI metadata matches (excluding ImageDescription and IFDs 4-5)")
    else:
        print(f"  ✗ CLI metadata mismatch - ALL DIFFERENCES:")
        src_lines = source_normalized.split('\n')
        out_lines = output_normalized.split('\n')
        
        # Show ALL differences
        diffs = []
        for i, (src, out) in enumerate(zip(src_lines, out_lines)):
            if src != out:
                diffs.append((i, src, out))
        
        for i, src, out in diffs:
            print(f"    Line {i}:")
            print(f"      Source: '{src}'")
            print(f"      Output: '{out}'")
        
        if len(src_lines) != len(out_lines):
            print(f"    Line count: source={len(src_lines)}, output={len(out_lines)}")
        
        assert False, f"CLI metadata comparison failed: {len(diffs)} differences found"
    
    print(f"\n{'='*80}")
    print(f"✓ Complete file roundtrip test PASSED")
    print(f"{'='*80}\n")


def test_write_dng_from_page_with_pyramid(tmp_path):
    """Test write_dng_from_page with pyramid generation.
    
    Takes the 4th sub-page (LINEAR_RAW) from canon_eos_r5_cfa_ljpeg_6ifds.dng,
    makes it the IFD0 of a new DNG file, and generates a 3-level JXL pyramid.
    Validates metadata and pixel scaling.
    """
    # Setup paths
    test_dir = Path(__file__).parent
    source_path = test_dir / "dngfiles" / "canon_eos_r5_cfa_ljpeg_6ifds.dng"
    
    if USE_PERSISTENT_OUTPUT:
        output_dir = test_dir / "test_outputs" / "test_write_dng_functions"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = tmp_path
    
    output_path = output_dir / "pyramid_from_page.dng"
    
    # Verify source file exists
    assert source_path.exists(), f"Source file not found: {source_path}"
    
    print(f"\n{'='*80}")
    print(f"Testing write_dng_from_page with pyramid generation")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")
    
    # Step 1: Load source and get SubIFD[2] (LINEAR_RAW page at index 3)
    print("Step 1: Loading source DNG and extracting SubIFD[2] (LINEAR_RAW)...")
    with DngFile(source_path) as source_dng:
        all_pages = source_dng.get_flattened_pages()
        
        # Page 0 is IFD0, pages 1-5 are SubIFDs
        # Index 3 = SubIFD[2] which is LINEAR_RAW 2048x1366
        source_page = all_pages[3]
        source_ifd0 = all_pages[0]
        
        print(f"  Source page 3 (SubIFD[2]): photometric={source_page.photometric_name}, "
              f"size={source_page.imagewidth}x{source_page.imagelength}, "
              f"compression={source_page.compression}")
        
        assert source_page.photometric_name == "LINEAR_RAW", (
            f"Expected LINEAR_RAW photometric, got {source_page.photometric_name}"
        )
        
        # Decode source page for later pixel comparison
        source_decoded = source_page.decode_to_rgb()
        print(f"  Decoded source: shape={source_decoded.shape}, dtype={source_decoded.dtype}")
        
        # Step 2: Write new DNG with pyramid
        # Note: Must happen inside the with block while source_dng is still open
        # because write_dng_from_page needs to read data from source_page
        print("\nStep 2: Writing new DNG with 3-level JXL pyramid...")
        write_dng_from_page(
            output_path,
            IfdPageSpec(
                page=source_page,
                extratags=source_ifd0.get_page_tags()
            ),
            pyramid=PyramidParams(
                levels=3,
                compression=COMPRESSION.JPEGXL_DNG,
                compression_args={'distance': 0.0, 'effort': 5}
            )
        )
    
    print(f"  Successfully wrote {output_path}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")
    
    # Step 3: Validate with dng_validate
    print("\nStep 3: Validating new DNG with dng_validate...")
    try:
        run_dng_validate(
            output_path,
            output_dir / "validate_output",
            validate=True,
            indent="  "
        )
        print("  ✓ Validation passed (no errors/warnings)")
    except AssertionError as e:
        print(f"  ✗ Validation failed: {e}")
        raise
    
    # Step 4: Compare metadata and pixels
    print("\nStep 4: Comparing metadata and pixels...")
    
    with DngFile(output_path) as output_dng:
        output_pages = output_dng.get_flattened_pages()
        
        # Should have 1 IFD0 + 3 SubIFDs (pyramid levels)
        assert len(output_pages) == 4, (
            f"Expected 4 pages (1 IFD0 + 3 pyramid levels), got {len(output_pages)}"
        )
        
        output_ifd0 = output_pages[0]
        
        print(f"\n  Output structure:")
        for i, page in enumerate(output_pages):
            print(f"    Page {i}: is_ifd0={page.is_ifd0}, photometric={page.photometric_name}, "
                  f"size={page.imagewidth}x{page.imagelength}, compression={page.compression}")
        
        # Compare IFD0 metadata
        # IFD0 should have merged tags from source IFD0 and source page 4
        print(f"\n  Comparing IFD0 metadata:")
        print(f"    Note: IFD0 merges tags from source IFD0 and source SubIFD 3")
        
        # Additional tags to skip when upleveling a preview SubIFD to main IFD0
        extra_skip = {
            'NewSubfileType',  # Changes from PreviewImage (1) to MainImage (0)
            'CacheVersion',  # Preview-only tag, correctly stripped when upleveling to main
            'PreviewApplicationName', 'PreviewApplicationVersion',  # Preview metadata
            'PreviewSettingsDigest', 'PreviewColorSpace', 'PreviewDateTime',
            'RawDataUniqueID', 'NewRawImageDigest',  # Digest tags
            'ImageDescription',  # Different convention between source and tifffile
        }
        
        # Merge source tags: IFD0 | page (page tags override)
        merged_tags = source_ifd0.get_page_tags() | source_page.get_page_tags()
        compared_count, mismatches = compare_page_metadata(merged_tags, output_ifd0, extra_skip_tags=extra_skip)
        
        if mismatches:
            print(f"      Tag mismatches:")
            for mismatch in mismatches:
                print(f"        - {mismatch}")
            raise AssertionError(f"IFD0: {len(mismatches)} tag mismatches found")
        
        print(f"      ✓ All {compared_count} compared tags match")
        
        # Compare pixels for each level
        print(f"\n  Comparing decoded pixels:")
        
        for i, out_page in enumerate(output_pages):
            print(f"\n    Level {i} (size={out_page.imagewidth}x{out_page.imagelength}):")
            
            # Decode output page
            out_decoded = out_page.decode_to_rgb()
            
            # Resize source to match output size
            # cv2.resize expects (width, height) not (height, width)
            src_resized = cv2.resize(
                source_decoded,
                (out_page.imagewidth, out_page.imagelength),
                interpolation=cv2.INTER_AREA
            )
            
            # Compare pixels
            diff_stats = compute_diff_stats(src_resized, out_decoded)
            
            print(f"      Mean diff: {diff_stats['mean']:.4f}%")
            print(f"      P99 diff:  {diff_stats['p99']:.4f}%")
            print(f"      Max diff:  {diff_stats['max']:.4f}%")
            
            # For level 0 (full resolution), expect very close match
            # For pyramid levels, allow some difference due to JXL lossy compression and scaling
            if i == 0:
                # Level 0: should be very close (JXL distance=1.0 is near-lossless)
                mean_threshold = 1.0
                max_threshold = 5.0
            else:
                # Pyramid levels: allow more difference due to Lanczos vs INTER_AREA + compression
                # Lanczos can have ringing artifacts that cause higher max differences
                # Differences accumulate at deeper levels due to repeated filtering
                mean_threshold = 2.0
                max_threshold = 50.0
            
            # Check thresholds and dump TIFFs if they fail
            mean_failed = diff_stats['mean'] >= mean_threshold
            max_failed = diff_stats['max'] >= max_threshold
            
            if mean_failed or max_failed:
                dump_comparison_images(src_resized, out_decoded, output_dir, f"level{i}")
            
            if mean_failed:
                assert False, (
                    f"Level {i}: Mean pixel difference {diff_stats['mean']:.4f}% too high "
                    f"(expected < {mean_threshold}%)"
                )
            if max_failed:
                assert False, (
                    f"Level {i}: Max pixel difference {diff_stats['max']:.4f}% too high "
                    f"(expected < {max_threshold}%)"
                )
            
            print(f"      ✓ Pixels match within tolerance")
    
    print(f"\n{'='*80}")
    print(f"✓ write_dng_from_page with pyramid test PASSED")
    print(f"{'='*80}\n")


def test_render_raw_scaling_consistency(tmp_path):
    """Test render_raw consistency across different scaling and processing paths.
    
    Validates that:
    1. render_raw with scale=0.25 returns SubIFD[2] exactly
    2. Rendering SubIFD[2] directly matches scaled render
    3. create_dng_from_page preserves rendering
    4. create_dng_from_page with preview generates consistent renders
    5. create_dng_from_page with demosaic+pyramid generates consistent renders
    """
    test_file = Path(__file__).parent / "dngfiles" / "canon_eos_r5_cfa_ljpeg_6ifds.dng"
    assert test_file.exists(), f"Test file not found: {test_file}"
    
    # Setup output directory
    if USE_PERSISTENT_OUTPUT:
        output_dir = Path(__file__).parent / "test_outputs" / "test_write_dng_functions"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = tmp_path
    
    print(f"\n{'='*80}")
    print(f"Testing render_raw scaling consistency")
    print(f"Source: {test_file}")
    print(f"{'='*80}\n")
    
    # Open source DNG
    with DngFile(test_file) as dng:
        # Get flattened pages
        pages = dng.get_flattened_pages()
        main_page = dng.get_main_page()
        subifd2 = pages[3]  # SubIFD[2]
        
        print(f"Main page: {main_page.photometric_name}, {main_page.shape}")
        print(f"SubIFD[2]: {subifd2.photometric_name}, {subifd2.shape}\n")
        
        # Test 1: render_raw with scale=0.25 should use SubIFD[2]
        print("Test 1: render_raw(scale=0.25) should use SubIFD[2]...")
        render_scaled = dng.render_raw(output_dtype=np.uint8, scale=0.25)
        assert render_scaled is not None
        print(f"  Scaled render shape: {render_scaled.shape}")
        
        # Test 2: Render SubIFD[2] directly
        print("\nTest 2: Render SubIFD[2] directly...")
        render_subifd2 = subifd2.render_raw(output_dtype=np.uint8)
        assert render_subifd2 is not None
        print(f"  SubIFD[2] render shape: {render_subifd2.shape}")
        
        # Compare Test 1 and Test 2 - should match exactly
        print("\n  Comparing scaled render vs SubIFD[2] render...")
        assert render_scaled.shape == render_subifd2.shape, \
            f"Shape mismatch: {render_scaled.shape} != {render_subifd2.shape}"
        
        diff_stats = compute_diff_stats(render_scaled, render_subifd2)
        print(f"    Mean diff: {diff_stats['mean']:.4f}%")
        print(f"    Max diff:  {diff_stats['max']:.4f}%")
        assert diff_stats['mean'] == 0.0 and diff_stats['max'] == 0.0, \
            "Renders should match exactly"
        print("  ✓ Renders match exactly")
        
        # Test 3: create_dng_from_page on SubIFD[2]
        print("\nTest 3: create_dng_from_page(SubIFD[2])...")
        dng_from_page = create_dng_from_page(subifd2)
        
        # Validate and save DNG
        test3_path = output_dir / "test3_from_page.dng"
        dng_from_page.write_to(test3_path)
        print(f"  Saved to: {test3_path}")
        run_dng_validate(test3_path, output_dir / "test3_from_page")
        
        render_from_page = dng_from_page.get_main_page().render_raw(output_dtype=np.uint8)
        assert render_from_page is not None
        print(f"  Render from page shape: {render_from_page.shape}")
        
        print("  Comparing with SubIFD[2] render...")
        diff_stats = compute_diff_stats(render_subifd2, render_from_page)
        print(f"    Mean diff: {diff_stats['mean']:.4f}%")
        print(f"    Max diff:  {diff_stats['max']:.4f}%")
        assert diff_stats['mean'] == 0.0 and diff_stats['max'] == 0.0, \
            "Renders should match exactly"
        print("  ✓ Renders match exactly")
        
        # Test 4: create_dng_from_page with scale=0.25 and preview
        print("\nTest 4: create_dng_from_page(main, scale=0.25, preview)...")
        preview_params = PreviewParams(max_dimension=512)
        dng_with_preview = create_dng_from_page(
            main_page,
            scale=0.25,
            preview=preview_params
        )
        
        # Validate and save DNG
        test4_path = output_dir / "test4_with_preview.dng"
        dng_with_preview.write_to(test4_path)
        print(f"  Saved to: {test4_path}")
        run_dng_validate(test4_path, output_dir / "test4_with_preview")
        
        render_with_preview = dng_with_preview.render_raw(output_dtype=np.uint8)
        assert render_with_preview is not None
        print(f"  Render with preview shape: {render_with_preview.shape}")
        
        print("  Comparing with SubIFD[2] render...")
        diff_stats = compute_diff_stats(render_subifd2, render_with_preview)
        print(f"    Mean diff: {diff_stats['mean']:.4f}%")
        print(f"    Max diff:  {diff_stats['max']:.4f}%")
        # Allow differences due to compression (LJPEG vs JPEGXL_DNG)
        mean_failed = diff_stats['mean'] >= 1.0
        max_failed = diff_stats['max'] >= 40.0
        if mean_failed or max_failed:
            dump_comparison_images(render_subifd2, render_with_preview, output_dir, "test4_preview")
            # Copy DNG to comparison folder
            import shutil
            dump_dir = output_dir / "pixel_comparison_dumps"
            shutil.copy(test4_path, dump_dir / "test4_preview.dng")
            print(f"        - test4_preview.dng")
            assert False, f"Renders differ too much: mean={diff_stats['mean']:.2f}%, max={diff_stats['max']:.2f}%"
        print(f"  ✓ Renders match within threshold")
        
        # Test 5: create_dng_from_page with JXL compression and pyramid
        print("\nTest 5: create_dng_from_page(IfdPageSpec with JXL, pyramid)...")
        from muimg.dngio import IfdPageSpec, PageOp
        main_page_spec = IfdPageSpec(
            page=main_page,
            page_operation=(PageOp.TRANSCODE, COMPRESSION.JPEGXL_DNG),
            compression_args={'distance': 0.1, 'effort': 5}
        )
        pyramid_params = PyramidParams(levels=3, compression=COMPRESSION.JPEGXL_DNG)
        dng_with_pyramid = create_dng_from_page(
            main_page_spec,
            pyramid=pyramid_params
        )
        
        # Validate and save DNG
        test5_path = output_dir / "test5_with_pyramid.dng"
        dng_with_pyramid.write_to(test5_path)
        print(f"  Saved to: {test5_path}")
        run_dng_validate(
             test5_path, 
             output_dir / "test5_with_pyramid"
        )
        
        # Render main page and resize to 0.25
        render_pyramid_main = dng_with_pyramid.render_raw(output_dtype=np.uint8)
        assert render_pyramid_main is not None
        print(f"  Pyramid main render shape: {render_pyramid_main.shape}")
        
        # Resize to match SubIFD[2] size
        target_h, target_w = render_subifd2.shape[:2]
        render_pyramid_resized = cv2.resize(
            render_pyramid_main,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA
        )
        print(f"  Resized to: {render_pyramid_resized.shape}")
        
        print("  Comparing resized render with SubIFD[2] render...")
        diff_stats = compute_diff_stats(render_subifd2, render_pyramid_resized)
        print(f"    Mean diff: {diff_stats['mean']:.4f}%")
        print(f"    Max diff:  {diff_stats['max']:.4f}%")
        # Allow differences due to compression (LJPEG vs JPEGXL_DNG)
        mean_failed = diff_stats['mean'] >= 0.75
        max_failed = diff_stats['max'] >= 40.0
        if mean_failed or max_failed:
            dump_comparison_images(render_subifd2, render_pyramid_resized, output_dir, "test5_pyramid_resized")
            # Copy DNG to comparison folder
            import shutil
            dump_dir = output_dir / "pixel_comparison_dumps"
            shutil.copy(test5_path, dump_dir / "test5_pyramid_resized.dng")
            print(f"        - test5_pyramid_resized.dng")
            assert False, f"Renders differ too much: mean={diff_stats['mean']:.2f}%, max={diff_stats['max']:.2f}%"
        print(f"  ✓ Resized render matches within threshold")
        
        # Also check pyramid level 2 (second pyramid level after main)
        print("\n  Rendering pyramid level 2...")
        pyramid_pages = dng_with_pyramid.get_flattened_pages()
        pyramid_level2 = pyramid_pages[2]  # SubIFD[1]
        render_pyramid_level2 = pyramid_level2.render_raw(output_dtype=np.uint8)
        assert render_pyramid_level2 is not None
        print(f"  Pyramid level 2 render shape: {render_pyramid_level2.shape}")
        
        print("  Comparing pyramid level 2 with SubIFD[2] render...")
        diff_stats = compute_diff_stats(render_subifd2, render_pyramid_level2)
        print(f"    Mean diff: {diff_stats['mean']:.4f}%")
        print(f"    Max diff:  {diff_stats['max']:.4f}%")
        mean_failed = diff_stats['mean'] >= 1.0
        max_failed = diff_stats['max'] >= 40.0
        if mean_failed or max_failed:
            dump_comparison_images(render_subifd2, render_pyramid_level2, output_dir, "test5_pyramid_level2")
            # Copy DNG to comparison folder
            import shutil
            dump_dir = output_dir / "pixel_comparison_dumps"
            shutil.copy(test5_path, dump_dir / "test5_pyramid_level2.dng")
            print(f"        - test5_pyramid_level2.dng")
            assert False, f"Renders differ too much: mean={diff_stats['mean']:.2f}%, max={diff_stats['max']:.2f}%"
        print(f"  ✓ Pyramid level 2 matches within threshold")
    
    print(f"\n{'='*80}")
    print(f"✓ All render_raw scaling consistency tests PASSED")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Allow running test directly for debugging
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_complete_file_roundtrip(Path(tmpdir))
