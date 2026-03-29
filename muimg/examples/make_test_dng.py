#!/usr/bin/env python3
"""Create a size-constrained test DNG file.

This example demonstrates how to use muimg to create test DNG files that fit
within a target size by iteratively scaling down the image resolution.

The script:
- Copies a DNG file with optional scaling and JXL compression
- If source is already small enough, copies without JXL 
- Otherwise, iteratively scales down by powers of 2 until output fits target size
- Always demosaics to LINEAR_RAW for consistency
- Sets UniqueCameraModel to "muimg: test"

Usage:
    python make_test_dng.py input.dng output.dng --target-size 1048576
    python make_test_dng.py input.dng output.dng --target-size 1048576 --min-scale 0.25
    python make_test_dng.py input.dng output.dng --target-size 1048576 --generate-preview
"""

import argparse
import io
import sys
from pathlib import Path


def make_test_dng(
    input_file: Path,
    output_file: Path,
    target_size: int,
    min_scale: float = 0.25,
    generate_preview: bool = False,
    jxl_distance: float = 0.5,
    jxl_effort: int = 8,
    dry_run: bool = False,
    preserve_unique_camera_model: bool = False,
) -> tuple[int, float, bool]:
    """Create a test DNG file that fits within target size.
    
    Args:
        input_file: Source DNG file path
        output_file: Destination DNG file path
        target_size: Target size in bytes
        min_scale: Minimum allowed scale factor (must be <= 1.0, power of 2)
        generate_preview: If True, generate preview/thumbnail
        jxl_distance: JXL compression distance (0=lossless, >0=lossy)
        jxl_effort: JXL compression effort (1-9)
        dry_run: If True, process to memory only, don't write output
        preserve_unique_camera_model: If True, preserve original UniqueCameraModel instead of replacing with "muimg: test"
        
    Returns:
        Tuple of (final_size_bytes, scale_used, used_jxl)
        
    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If processing fails
    """
    from muimg import dngio
    from muimg.tiff_metadata import MetadataTags
    
    if not input_file.exists():
        raise ValueError(f"Input file does not exist: {input_file}")
    
    if min_scale > 1.0:
        raise ValueError(f"min_scale must be <= 1.0, got {min_scale}")
    
    # Verify min_scale is a power of 2
    scale_check = min_scale
    while scale_check < 1.0:
        scale_check *= 2.0
    if abs(scale_check - 1.0) > 1e-6:
        raise ValueError(f"min_scale must be a power of 2 (e.g., 1.0, 0.5, 0.25), got {min_scale}")
    
    source_size = input_file.stat().st_size
    
    # Create metadata override
    ifd0_tags = MetadataTags()
    if not preserve_unique_camera_model:
        ifd0_tags.add_tag("UniqueCameraModel", "muimg: test")
    
    # If source is already small enough - still need to copy to inject new metadata and generate preview
    if source_size <= target_size:
        if not dry_run:
            dngio.copy_dng(
                source_file=input_file,
                destination_file=output_file,
                generate_preview=generate_preview,
                ifd0_tags=ifd0_tags,
            )
        return source_size, 1.0, False
    
    # Source is too large - use JXL compression with iterative power-of-2 scaling
    scale = 1.0
    
    while scale >= min_scale:
        stream = io.BytesIO()
        
        dngio.copy_dng(
            source_file=input_file,
            destination_file=stream,
            scale=scale,
            demosaic=True,
            demosaic_algorithm="DNGSDK_BILINEAR",
            jxl_distance=jxl_distance,
            jxl_effort=jxl_effort,
            generate_preview=generate_preview,
            ifd0_tags=ifd0_tags,
        )
        
        size = stream.tell()
        
        if size <= target_size:
            if not dry_run:
                with open(output_file, "wb") as f:
                    f.write(stream.getvalue())
            return size, scale, True
        
        # Scale down by power of 2
        scale *= 0.5
    
    # Reached min_scale, write final result even if over target
    if not dry_run:
        with open(output_file, "wb") as f:
            f.write(stream.getvalue())
    
    return size, scale * 2.0, True  # Return the scale we actually used


def main():
    parser = argparse.ArgumentParser(
        description="Create a size-constrained test DNG file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create 1MB test file
  %(prog)s input.dng output.dng --target-size 1048576
  
  # Create test file with minimum scale of 0.125 (1/8)
  %(prog)s input.dng output.dng --target-size 1048576 --min-scale 0.125
  
  # Create test file with preview
  %(prog)s input.dng output.dng --target-size 1048576 --generate-preview
  
  # Dry run to see what would happen
  %(prog)s input.dng output.dng --target-size 1048576 --dry-run
        """
    )
    
    parser.add_argument("input_file", type=Path, help="Source DNG file")
    parser.add_argument("output_file", type=Path, help="Destination DNG file")
    parser.add_argument(
        "--target-size",
        type=int,
        default=1048576,
        help="Target size in bytes (default: 1048576 = 1MB)"
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.25,
        help="Minimum allowed scale factor, must be power of 2 (default: 0.25)"
    )
    parser.add_argument(
        "--generate-preview",
        action="store_true",
        help="Generate preview/thumbnail in output DNG"
    )
    parser.add_argument(
        "--jxl-distance",
        type=float,
        default=0.5,
        help="JXL compression distance, 0=lossless (default: 0.5)"
    )
    parser.add_argument(
        "--jxl-effort",
        type=int,
        default=8,
        help="JXL compression effort, 1-9 (default: 8)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process to memory only, don't write output file"
    )
    parser.add_argument(
        "--preserve-unique-camera-model",
        action="store_true",
        help="Preserve original UniqueCameraModel instead of replacing with 'muimg: test'"
    )
    
    args = parser.parse_args()
    
    try:
        final_size, scale, used_jxl = make_test_dng(
            input_file=args.input_file,
            output_file=args.output_file,
            target_size=args.target_size,
            min_scale=args.min_scale,
            generate_preview=args.generate_preview,
            jxl_distance=args.jxl_distance,
            jxl_effort=args.jxl_effort,
            dry_run=args.dry_run,
            preserve_unique_camera_model=args.preserve_unique_camera_model,
        )
        
        size_mb = final_size / (1024 * 1024)
        target_mb = args.target_size / (1024 * 1024)
        compression = "JXL" if used_jxl else "uncompressed"
        
        mode = "[DRY RUN] " if args.dry_run else ""
        
        if scale < 1.0:
            print(f"{mode}Created: {args.output_file.name}")
            print(f"  Size: {size_mb:.2f}MB (target: {target_mb:.2f}MB)")
            print(f"  Compression: {compression}")
            print(f"  Scale: {scale:.3f}")
        else:
            print(f"{mode}Created: {args.output_file.name}")
            print(f"  Size: {size_mb:.2f}MB (target: {target_mb:.2f}MB)")
            print(f"  Compression: {compression}")
        
        if final_size > args.target_size:
            print(f"  Warning: Output exceeds target (min_scale={args.min_scale} reached)")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
