"""Command-line interface for muimg DNG processing."""

import argparse
import logging
from .dng import convert_raw


def main():
    """CLI entry point for DNG conversion."""
    parser = argparse.ArgumentParser(description="Convert DNG files to JPG using Core Image")
    parser.add_argument("input_dng", help="Input DNG file path")
    parser.add_argument("output_jpg", help="Output JPG file path")
    parser.add_argument("--temperature", type=float, help="White balance temperature")
    parser.add_argument("--tint", type=float, help="White balance tint")
    parser.add_argument("--exposure", type=float, help="Exposure adjustment in stops")
    parser.add_argument("--noise-reduction", type=float, help="Noise reduction amount")
    parser.add_argument("--orientation", type=int, help="Image orientation")
    parser.add_argument("--no-xmp", action="store_true", help="Don't use XMP metadata")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    # Convert DNG
    success = convert_raw(
        file=args.input_dng,
        output_path=args.output_jpg,
        temperature=args.temperature,
        tint=args.tint,
        exposure=args.exposure,
        noise_reduction=args.noise_reduction,
        orientation=args.orientation,
        use_xmp=not args.no_xmp,
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
