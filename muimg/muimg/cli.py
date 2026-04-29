# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

"""Command-line interface for muimg."""

import logging
import sys
from pathlib import Path

import click

from .raw_render import DemosaicAlgorithm

logger = logging.getLogger(__name__)



@click.group()
@click.option("-v", "--verbose", count=True, help="Increase verbosity")
def cli(verbose):
    """muimg - Image processing utilities."""
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


@cli.command(name="convert-image")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--bit-depth", type=click.Choice(["8", "16"]), default="8", help="Output bit depth (8 or 16)")
def convert_image(input_file, output_file, bit_depth):
    """Convert image file to another format."""
    import numpy as np
    from .imgio import convert_imgformat
    
    success = convert_imgformat(
        file=input_file,
        output=output_file,
        output_dtype=np.uint16 if bit_depth == "16" else np.uint8
    )

    sys.exit(0 if success else 1)


def _format_bytes(size):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _parse_ifd_spec(ifd_spec):
    """Parse IFD specification string to flattened page index.
    
    Args:
        ifd_spec: String like 'ifd0', 'subifd0', 'subifd1', etc., or None for main page
        
    Returns:
        Tuple of (ifd_index, ifd_name) where ifd_index is the flattened page index,
        or (None, "main raw page") if ifd_spec is None
        
    Raises:
        click.ClickException: If IFD specification is invalid
    """
    import re
    
    if ifd_spec is None:
        return None, "main raw page"
    elif ifd_spec == "ifd0":
        return 0, "ifd0"
    elif match := re.match(r"subifd(\d+)$", ifd_spec):
        subifd_num = int(match.group(1))
        return subifd_num + 1, ifd_spec  # SubIFD[0] is flattened index 1
    else:
        raise click.ClickException(
            f"Invalid IFD specification '{ifd_spec}'. Use 'ifd0' or 'subifd0', 'subifd1', etc."
        )


def _get_page_from_ifd_spec(dng_file, ifd_spec):
    """Get page from DNG file based on IFD specification.
    
    Args:
        dng_file: DngFile instance
        ifd_spec: String like 'ifd0', 'subifd0', 'subifd1', etc., or None for main page
        
    Returns:
        Tuple of (page, index, name) where:
            - page: Page object
            - index: Flattened page index (or None for main page)
            - name: IFD name string (e.g., "ifd0", "subifd0", "main raw page")
        
    Raises:
        click.ClickException: If IFD specification is invalid or page not found
    """
    ifd_index, ifd_name = _parse_ifd_spec(ifd_spec)
    
    if ifd_index is None:
        # Default: get main raw page
        page = dng_file.get_main_page()
        if page is None:
            raise click.ClickException("No main raw page found")
    else:
        # Get specific IFD by index
        pages = dng_file.get_flattened_pages()
        if ifd_index >= len(pages):
            raise click.ClickException(
                f"IFD index {ifd_index} out of range. File has {len(pages)} IFD(s)."
            )
        page = pages[ifd_index]
    
    return page, ifd_index, ifd_name

@cli.group(name="dng")
def dng():
    """DNG file operations."""
    pass

@dng.command(name="metadata")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--ifd", type=int, help="Show specific IFD (0=IFD0, 1+=SubIFDs)")
@click.option("--tag", multiple=True, help="Show only tag(s) matching regex pattern(s) (case-insensitive)")
@click.option("--exclude-tag", multiple=True, help="Exclude tag(s) matching regex pattern(s) (case-insensitive)")
@click.option("--summary", is_flag=True, help="Show only IFD summary and exit")
def dng_metadata(input_file, ifd, tag, exclude_tag, summary):
    """Display DNG file metadata."""
    import re
    import os
    from tifffile import PHOTOMETRIC, COMPRESSION
    from .dngio import DngFile
    from .tiff_metadata import LOCAL_TIFF_TAGS, TIFF_TAG_TYPE_REGISTRY, SubFileType
    from .common import enum_display_name
    
    try:
        f = DngFile(input_file)
        pages = f.get_flattened_pages()
    except Exception as e:
        click.echo(f"Error opening DNG file: {e}", err=True)
        sys.exit(1)
    
    if not pages:
        click.echo("No IFDs found in file", err=True)
        sys.exit(1)
    
    # Compile regex patterns for tag filtering (inclusion)
    tag_patterns = []
    if tag:
        for pattern in tag:
            try:
                tag_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                click.echo(f"Error: Invalid --tag regex pattern '{pattern}': {e}", err=True)
                sys.exit(1)
    
    # Compile regex patterns for tag exclusion
    exclude_patterns = []
    if exclude_tag:
        for pattern in exclude_tag:
            try:
                exclude_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                click.echo(f"Error: Invalid --exclude-tag regex pattern '{pattern}': {e}", err=True)
                sys.exit(1)
    
    # Show summary of available IFDs
    click.echo(f"File contains {len(pages)} IFD(s):")
    for i, p in enumerate(pages):
        ifd_type = "IFD0" if i == 0 else f"SubIFD[{i-1}]"
        photometric = p.photometric_name or "Unknown"
        width = p.imagewidth or "?"
        length = p.imagelength or "?"
        
        # Get rendered size and format dimensions
        rendered_size = p.get_rendered_size()
        if rendered_size != (width, length) and width != "?" and length != "?":
            size_str = f"{rendered_size[0]}x{rendered_size[1]} (from {width}x{length})"
        else:
            size_str = f"{width}x{length}"
        
        # Get bit depth
        bits_per_sample = p.bitspersample if hasattr(p, 'bitspersample') else 16
        photometric_with_bits = f"{photometric}{bits_per_sample}"
        
        # Calculate compressed size
        compressed_size = sum(p.databytecounts) if hasattr(p, 'databytecounts') else 0
        
        # Calculate uncompressed size (in bits to handle bit-packed data correctly)
        if width != "?" and length != "?":
            samples_per_pixel = p.samplesperpixel if hasattr(p, 'samplesperpixel') else 1
            uncompressed_size_bits = width * length * samples_per_pixel * bits_per_sample
            compressed_size_bits = compressed_size * 8
            compression_ratio = compressed_size_bits / uncompressed_size_bits if uncompressed_size_bits > 0 else 0
        else:
            compression_ratio = 0
        
        # Get compression type name
        if hasattr(p, 'compression') and p.compression != COMPRESSION.NONE:
            compression_name = enum_display_name(COMPRESSION, p.compression, " compression")
        else:
            compression_name = "uncompressed"
        
        # Add NewSubfileType info if present
        subfile_info = ""
        if (subfile_type := p.get_tag("NewSubfileType")) is not None:
            subfile_info = f" ({enum_display_name(SubFileType, subfile_type)})"
        
        summary_indent = "" if i == 0 else "  "  # Extra indent for SubIFDs
        file_size_str = _format_bytes(compressed_size)
        click.echo(f"{i}:{summary_indent} {ifd_type}{subfile_info} - {photometric_with_bits}, {size_str}, {file_size_str} ({compression_ratio:.2f}x {compression_name})")
    
    # Show total file size
    total_size = os.path.getsize(input_file)
    click.echo(f"Total file size: {_format_bytes(total_size)}")
    
    # If --summary flag is set, exit early
    if summary:
        sys.exit(0)
    
    click.echo()
    
    # Determine which IFDs to show
    if ifd is not None:
        if ifd < 0 or ifd >= len(pages):
            click.echo(f"Error: IFD {ifd} not found (file has {len(pages)} IFDs)", err=True)
            sys.exit(1)
        pages_to_show = [pages[ifd]]
        ifd_indices = [ifd]
    else:
        # Default: show all IFDs
        pages_to_show = pages
        ifd_indices = list(range(len(pages)))
    
    # Track XMP content for duplicate detection
    xmp_tracker = {}
    
    # Track validation issues
    validation_issues = []
    
    # Display metadata for each IFD
    for idx, (page, ifd_num) in enumerate(zip(pages_to_show, ifd_indices)):
        if idx > 0:
            click.echo()  # Blank line between IFDs
        
        # Use TIFF terminology for header
        if ifd_num == 0:
            ifd_label = "IFD0"
            indent = ""
            header_indent = ""
            actual_ifd_type = "dng_ifd0"
        else:
            ifd_label = f"SubIFD {ifd_num - 1}"
            indent = "  "  # Indent SubIFD tags
            header_indent = "  "  # Indent SubIFD header
            actual_ifd_type = ("dng_raw:cfa" if page.photometric == PHOTOMETRIC.CFA 
                              else "dng_raw" if page.photometric == PHOTOMETRIC.LINEAR_RAW else "other")
        click.echo(f"{header_indent}=== {ifd_label} (--ifd {ifd_num}) ===")
        
        # Get NewSubfileType once for reuse in validation and display
        newsubfiletype = page.get_tag("NewSubfileType") or 0
        
        # Check if IFD0 contains main raw image (for validation)
        ifd0_is_main_raw = (ifd_num == 0 and 
                           newsubfiletype == SubFileType.MAIN_IMAGE and 
                           (page.photometric == PHOTOMETRIC.CFA or page.photometric == PHOTOMETRIC.LINEAR_RAW))
        
        # Iterate through actual file tags (tifffile API)
        for tag_code, tag in page.tags.items():
            dtype = tag.dtype
            count = tag.count
            # Get tag name from registry
            tag_name = LOCAL_TIFF_TAGS.get(tag_code)
            if tag_name is None:
                tag_name = f"Tag{tag_code}"
            
            # Filter if requested (regex matching, case-insensitive)
            # First apply inclusion filter (--tag)
            if tag_patterns:
                if not any(pattern.search(tag_name) for pattern in tag_patterns):
                    continue
            
            # Then apply exclusion filter (--exclude-tag)
            if exclude_patterns:
                if any(pattern.search(tag_name) for pattern in exclude_patterns):
                    continue
            
            # Validate DNG IFD for known tags
            issue_number = None
            if tag_name in TIFF_TAG_TYPE_REGISTRY:
                tag_spec = TIFF_TAG_TYPE_REGISTRY[tag_name]
                if tag_spec.dng_ifd != "any":
                    # Normalize expected DNG IFD
                    expected_dng_ifd = tag_spec.dng_ifd
                    # ifd0/exif/dng_profile -> dng_ifd0
                    if expected_dng_ifd in ("ifd0", "exif", "dng_profile"):
                        expected_dng_ifd = "dng_ifd0"
                    # When IFD0 is main raw, also normalize dng_raw -> dng_ifd0
                    elif ifd0_is_main_raw and expected_dng_ifd == "dng_raw":
                        expected_dng_ifd = "dng_ifd0"
                    
                    # Check if tag is in wrong IFD
                    if actual_ifd_type == expected_dng_ifd:
                        pass  # Valid
                    elif expected_dng_ifd == "dng_raw" and actual_ifd_type == "dng_raw:cfa":
                        pass  # Allow 'dng_raw' tags in 'dng_raw:cfa' IFDs (CFA is a type of raw)
                    elif expected_dng_ifd == "dng_raw:cfa" and page.photometric == PHOTOMETRIC.CFA:
                        pass  # Allow 'dng_raw:cfa' tags in any CFA IFD (including IFD0 if it's main CFA)
                    elif expected_dng_ifd == "dng_preview" and newsubfiletype in (SubFileType.PREVIEW_IMAGE, SubFileType.ALT_PREVIEW_IMAGE):
                        pass  # Preview tags are valid in preview IFDs
                    else:
                        # Tag is in wrong IFD
                        issue_number = len(validation_issues) + 1
                        validation_issues.append((
                            issue_number,
                            tag_name,
                            actual_ifd_type,
                            tag_spec.dng_ifd  # Store original spec for display
                        ))
            
            # Get value from page for proper formatting, fallback to raw tag value
            value = page.get_tag(tag_code)
            if value is None:
                value = tag.value
            
            # Convert enum values to display names if applicable
            tag_spec = TIFF_TAG_TYPE_REGISTRY.get(tag_name)
            if tag_spec and tag_spec.enum_class and value is not None:
                # Special case: PhotometricInterpretation has a custom name property
                if tag_name == "PhotometricInterpretation":
                    value = page.photometric_name
                else:
                    value = enum_display_name(tag_spec.enum_class, value)
            
            # Display the tag with appropriate indentation
            _display_tag(tag_name, value, indent, tag_code, dtype, count, xmp_tracker, ifd_num, issue_number)
    
    # Display validation issues summary if any were found
    if validation_issues:
        click.echo()
        click.echo("=== DNG Validation Issues ===")
        for issue_num, tag_name, actual_type, expected_spec in validation_issues:
            click.echo(f"*-{issue_num}: {tag_name} found in {actual_type} IFD, should be in {expected_spec} IFD")
    
    sys.exit(0)

def _display_tag(tag_name, value, indent="", tag_code=None, dtype=None, count=None, xmp_tracker=None, ifd_num=None, issue_number=None):
    """Format and display a single tag value.
    
    Args:
        tag_name: Name of the tag
        value: Already-converted value from page.get_tag()
        indent: String to prepend to each line (for SubIFD indentation)
        tag_code: Numeric tag code (for unknown tags)
        dtype: TIFF dtype code (for unknown tags)
        count: Element count (for unknown tags)
        xmp_tracker: Dict for tracking XMP content across IFDs (None to disable duplicate detection)
        ifd_num: Current IFD number (for duplicate messages)
        issue_number: Validation issue number (if tag is in wrong IFD)
    """
    import numpy as np
    from .tiff_metadata import XmpMetadata, TiffType
    
    def echo(text):
        """Helper to echo with indentation."""
        click.echo(f"{indent}{text}")
    
    # For unknown tags, show tag code and type info
    is_unknown = tag_name.startswith("Tag") and tag_code is not None
    if is_unknown and dtype is not None:
        # Try to get TiffType name from dtype code
        try:
            tiff_type = TiffType(dtype)
            dtype_name = f"{tiff_type.name}({dtype})"
        except ValueError:
            dtype_name = f'Type{dtype}'
        tag_display = f"{tag_name} ({tag_code}, {dtype_name})"
    else:
        tag_display = tag_name
    
    # Prepend issue marker if this tag has a validation issue
    if issue_number is not None:
        tag_display = f"[*-{issue_number}] {tag_display}"
    
    # Handle None first
    if value is None:
        echo(f"{tag_display}: None")
        return
    
    # Special handling for EXIF/GPS dictionaries - enumerate fields
    if tag_name in ("ExifTag", "GPSTag") and isinstance(value, dict):
        echo(f"{tag_display}: {len(value)} fields")
        # Display each EXIF field with extra indentation
        for key, val in value.items():
            # Recursively display each field
            _display_tag(str(key), val, indent + "  ", None, None, None)
        return

    # Special formatting for DNG version tags (4-tuple from get_tag)
    if tag_name in ("DNGVersion", "DNGBackwardVersion") and isinstance(value, tuple) and len(value) == 4:
        version_str = f"{value[0]}.{value[1]}.{value[2]}.{value[3]}"
        echo(f"{tag_display}: {version_str}")
        return
    
    # Special handling for OpcodeList tags - show opcode names
    if tag_name in ('OpcodeList1', 'OpcodeList2', 'OpcodeList3'):
        from . import raw_render
        summary = raw_render.get_opcode_summary(bytes(value), detailed=True)
        num_bytes = len(bytes(value))
        
        # Display multi-line summary
        lines = summary.split('\n')
        echo(f"{tag_display}: {lines[0]} ({num_bytes} bytes)")
        for line in lines[1:]:
            echo(f"  {line}")
        return

    # Special handling for XMP - show raw XML with duplicate detection
    if isinstance(value, XmpMetadata):
        # Get the formatted XMP string for comparison and display
        xmp_str = value.get_formatted_string(strip_whitespace=True, filter_blank_lines=True)
        
        # Check for duplicates
        if xmp_str and xmp_str in xmp_tracker:
            # Duplicate found - show reference to first occurrence
            first_ifd = xmp_tracker[xmp_str]
            first_ifd_label = "IFD0" if first_ifd == 0 else f"SubIFD {first_ifd - 1}"
            echo(f"{tag_display}: duplicate of {first_ifd_label}")
            return
        
        # First occurrence - track it
        if xmp_str:
            xmp_tracker[xmp_str] = ifd_num
        
            # Display the full XMP
            echo(f"{tag_display}:")
            for line in xmp_str.splitlines():
                echo(f"  {line}")
        else:
            size = len(str(value))
            echo(f"{tag_display}: XML metadata, {size} bytes")
        return

    # Strings
    if isinstance(value, str):
        display_str = f"{value[:200]}... ({len(value)} chars)" if len(value) > 200 else value
        echo(f"{tag_display}: {display_str}")
        return

    # Bytes (binary data) - handle both bytes and numpy byte arrays
    byte_value = None
    if isinstance(value, bytes):
        byte_value = value
    elif isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind in ('S', 'V'):
        byte_value = value.item()
    
    if byte_value is not None and isinstance(byte_value, bytes):
        if len(byte_value) > 100:
            echo(f"{tag_display}: Binary data, {len(byte_value)} bytes")
        else:
            echo(f"{tag_display}: {byte_value}")
        return
    
    # Convert to numpy array for easier handling
    if not isinstance(value, np.ndarray):
        value = np.atleast_1d(value)
    
    # Simple scalars
    if value.size == 1:
        scalar = value.item()
        if isinstance(scalar, float):
            scalar = round(scalar, 4)
        echo(f"{tag_display}: {scalar}")
        return
    
    # 2D arrays (matrices) - display in matrix format
    if value.ndim == 2:
        echo(f"{tag_display}:")
        for i in range(min(value.shape[0], 3)):  # Max 3 lines
            row_str = "  " + "  ".join(f"{round(v, 4):8.4f}" for v in value[i])
            echo(row_str)
        if value.shape[0] > 3:
            echo(f"  ... ({value.shape[0]} rows total)")
        return
    
    # Helper to format values with appropriate precision
    def format_value(v):
        """Format a value with appropriate precision."""
        val = v.item() if isinstance(v, (np.integer, np.floating)) else v
        if isinstance(val, float):
            return round(val, 4)
        return val
    
    # 1D arrays
    if value.ndim == 1:
        # Small arrays (≤9 elements): show full array
        if value.size <= 9:
            clean_list = [format_value(v) for v in value.flat]
            echo(f"{tag_display}: {clean_list}")
            return
        
        # Larger 1D arrays: show first 9 with truncation
        clean_list = [format_value(v) for v in value.flat[:9]]
        # Format as list with ... inside brackets
        list_str = str(clean_list)[:-1]  # Remove closing bracket
        echo(f"{tag_display}: {list_str}, ... ({value.size} elements)]")
        return
    
    # Very large arrays: show type, element count, and byte size
    dtype_name = value.dtype.name
    num_elements = value.size
    num_bytes = value.nbytes
    echo(f"{tag_display}: {dtype_name} array, {num_elements} elements, {num_bytes} bytes")

@dng.command(name="raw-stage")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.argument("stage", type=click.Choice(["raw", "linearized", "linearized-plus-ops", "camera-rgb"]))
@click.option("--ifd", type=str, help="IFD to extract (ifd0, subifd0, subifd1, etc.). Default: main raw page")
@click.option("--demosaic", type=str, help="Demosaic CFA data using specified algorithm (OPENCV_EA, VNG, RCD, AHD)")
def dng_raw_stage(input_file, output_file, stage, ifd, demosaic):
    """Extract raw image data at a specific pipeline stage.
    
    \b
    Stages:
      raw                   - Raw sensor data (decoded, no processing)
      linearized            - After linearization and normalization
      linearized-plus-ops   - After OpcodeList2 (includes MapPolynomial)
      camera-rgb            - Demosaiced camera RGB
    
    \b
    Examples:
      muimg dng raw-stage input.dng output.tif linearized
      muimg dng raw-stage input.dng output.tif camera-rgb --ifd subifd2
      muimg dng raw-stage input.dng output.tif linearized --demosaic OPENCV_EA
      muimg dng raw-stage input.dng output.tif linearized-plus-ops --demosaic RCD
    """
    import numpy as np
    import tifffile
    from .dngio import DngFile, RawStageSelector
    from . import raw_render
    
    # Parse IFD specification before opening file
    ifd_index, ifd_name = _parse_ifd_spec(ifd)
    
    try:
        # Open DNG file and select page
        with DngFile(input_file) as dng:
            if ifd_index is None:
                # Default: get main raw page
                page = dng.get_main_page()
                if page is None:
                    click.echo("Error: No main raw page found", err=True)
                    sys.exit(1)
            else:
                # Get specific IFD by index
                pages = dng.get_flattened_pages()
                if ifd_index >= len(pages):
                    click.echo(f"Error: IFD index {ifd_index} out of range. File has {len(pages)} IFD(s).", err=True)
                    sys.exit(1)
                
                page = pages[ifd_index]
            
            # Validate page is raw
            if not (page.is_cfa or page.is_linear_raw):
                click.echo(f"Error: Selected IFD is not a raw page ({page.photometric_name})", err=True)
                sys.exit(1)
            
            # Extract data based on stage
            data = None
            cfa_pattern = None
            
            if stage == "camera-rgb":
                # Special case: camera-rgb uses get_camera_rgb_raw()
                # Use specified demosaic algorithm or default to OPENCV_EA
                if demosaic is not None:
                    algorithm = DemosaicAlgorithm.lookup(demosaic)
                else:
                    algorithm = DemosaicAlgorithm.OPENCV_EA
                data = page.get_camera_rgb_raw(demosaic_algorithm=algorithm)
                if data is None:
                    click.echo("Error: Failed to extract camera RGB", err=True)
                    sys.exit(1)
            else:
                # Map stage names to RawStageSelector
                stage_map = {
                    "raw": RawStageSelector.RAW,
                    "linearized": RawStageSelector.LINEARIZED,
                    "linearized-plus-ops": RawStageSelector.LINEARIZED_PLUS_OPS,
                }
                
                stage_selector = stage_map[stage]
                
                if page.is_cfa:
                    # CFA: use get_cfa() with stage selector
                    cfa_data, cfa_pattern = page.get_cfa(stage_selector)
                    data = cfa_data
                    
                    # Apply demosaic if requested
                    if demosaic is not None:
                        algorithm = DemosaicAlgorithm.lookup(demosaic)
                        data = raw_render.demosaic(
                            data, cfa_pattern, algorithm=algorithm
                        )
                else:
                    # LINEAR_RAW: use get_linear_raw() with stage selector
                    if demosaic is not None:
                        click.echo("Warning: --demosaic ignored for LINEAR_RAW page (already RGB)", err=True)
                    data = page.get_linear_raw(stage_selector)
                    if data is None:
                        click.echo(f"Error: Failed to extract {stage} stage", err=True)
                        sys.exit(1)
            
            if data is None:
                click.echo(f"Error: Failed to extract {stage} stage", err=True)
                sys.exit(1)
            
            # Convert to uint16
            data_uint16 = raw_render.convert_dtype(data, np.uint16)
            
            # Save as TIFF
            tifffile.imwrite(output_file, data_uint16)
            
            # Report actual extracted data dimensions
            if data.ndim == 3:
                h, w, c = data.shape
                dims_str = f"{w}x{h}x{c}"
            else:
                h, w = data.shape
                dims_str = f"{w}x{h}"
            
            click.echo(f"Extracted {stage} stage from {ifd_name} ({page.photometric_name}): {dims_str} -> {output_file}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@dng.command(name="copy")
@click.argument("input_dng", type=click.Path(exists=True))
@click.argument("output_dng", type=click.Path())
@click.option("--ifd", type=str, help="IFD to copy (ifd0, subifd0, subifd1, etc.). Default: main raw page")
@click.option(
    "--scale",
    type=float,
    default=1.0,
    help="Scale factor for image (default: 1.0). If != 1.0, forces conversion to LINEAR_RAW",
)
@click.option(
    "--demosaic", is_flag=True, help="Convert CFA to LINEAR_RAW (demosaic)"
)
@click.option(
    "--demosaic-algorithm",
    type=str,
    default="OPENCV_EA",
    help="Demosaic algorithm (OPENCV_EA, VNG, RCD)",
)
@click.option("--preview", is_flag=True, help="Generate preview/thumbnail")
@click.option(
    "--preview-max-dim",
    type=int,
    default=1024,
    help="Maximum preview dimension (default: 1024)",
)
@click.option(
    "--jxl-distance",
    type=float,
    default=None,
    help="JXL compression distance (0=lossless mode, >0=lossy with specified quality). None=uncompressed.",
)
@click.option(
    "--jxl-effort",
    type=int,
    default=None,
    help="JXL compression effort (1-9, default 5 if jxl-distance is set)",
)
@click.option(
    "--tag",
    multiple=True,
    help="Set/override tag as NAME=VALUE (can be specified multiple times)",
)
@click.option(
    "--strip-tag",
    multiple=True,
    help="Tag name to strip (can be specified multiple times)",
)
@click.option(
    "--pyramid-levels",
    type=int,
    default=0,
    help="Number of pyramid levels to generate (default: 0 = none)",
)
def dng_copy(
    input_dng,
    output_dng,
    ifd,
    scale,
    demosaic,
    demosaic_algorithm,
    strip_tag,
    preview,
    preview_max_dim,
    jxl_distance,
    jxl_effort,
    tag,
    pyramid_levels,
):
    """Create a new DNG file with optional transformations.
    
    Apply scale, demosaic (CFA to LINEAR_RAW), strip tags, and/or generate preview.
    """
    from . import dngio
    from .tiff_metadata import MetadataTags
    
    # Map algorithm string to enum
    demosaic_algorithm_enum = DemosaicAlgorithm.lookup(demosaic_algorithm)
    
    # Parse --strip-tag options, supporting comma-separated lists
    strip_tags_set = None
    if strip_tag:
        strip_tags_set = set()
        for tag_spec in strip_tag:
            # Split by comma and strip whitespace from each tag name
            for tag_name in tag_spec.split(','):
                tag_name = tag_name.strip()
                if tag_name:  # Skip empty strings
                    strip_tags_set.add(tag_name)
    
    # Parse --tag NAME=VALUE options into MetadataTags
    extra_tags = None
    if tag:
        extra_tags = MetadataTags()
        for tag_spec in tag:
            if "=" not in tag_spec:
                click.echo(f"Error: Invalid tag format '{tag_spec}'. Use NAME=VALUE", err=True)
                sys.exit(1)
            name, value = tag_spec.split("=", 1)
            extra_tags.add_tag(name.strip(), value.strip())
    
    try:
        # Build compression args and determine compression type
        from tifffile import COMPRESSION
        
        compression = None
        compression_args = None
        
        if jxl_distance is not None or jxl_effort is not None:
            compression = COMPRESSION.JPEGXL_DNG
            compression_args = {}
            if jxl_distance is not None:
                compression_args['distance'] = jxl_distance
            if jxl_effort is not None:
                compression_args['effort'] = jxl_effort
        
        # Open source DNG and get page based on IFD spec
        with dngio.DngFile(input_dng) as dng_file:
            page, _, _ = _get_page_from_ifd_spec(dng_file, ifd)
            
            # Determine page_operation based on explicit compression request
            if compression is not None:
                # TRANSCODE mode - decompress and recompress with specified compression
                page_operation = (dngio.PageOp.TRANSCODE, compression)
            else:
                # COPY mode - preserve source compression (even when demosaicing/scaling)
                page_operation = dngio.PageOp.COPY
            
            # Create IfdPageSpec
            page_spec = dngio.IfdPageSpec(
                page=page,
                page_operation=page_operation,
                compression_args=compression_args,
                strip_tags=strip_tags_set,
            )
            
            # Write using write_dng_from_page
            preview_params = None
            if preview:
                preview_params = dngio.PreviewParams(
                    max_dimension=preview_max_dim,
                    compression=COMPRESSION.JPEG,
                    compression_args={'level': 90}
                )
            
            pyramid_params = None
            if pyramid_levels > 0:
                pyramid_params = dngio.PyramidParams(
                    levels=pyramid_levels,
                    compression=COMPRESSION.JPEGXL_DNG,
                    compression_args={'distance': 1.0}
                )
            
            dngio.write_dng_from_page(
                destination_file=output_dng,
                page=page_spec,
                scale=scale,
                demosaic=demosaic,
                demosaic_algorithm=demosaic_algorithm_enum,
                preview=preview_params,
                pyramid=pyramid_params,
                ifd0_extratags=extra_tags,
                ifd0_strip_tags=strip_tags_set,
            )
        click.echo(f"Successfully copied DNG to {output_dng}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Failed to copy DNG")
        sys.exit(1)


@dng.command(name="convert")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--ifd", type=str, help="IFD to extract (ifd0, subifd0, subifd1, etc.). Default: main raw page")
@click.option("--temperature", type=float, help="White balance temperature")
@click.option("--tint", type=float, help="White balance tint")
@click.option("--exposure", type=float, help="Exposure adjustment in stops")
@click.option("--orientation", type=int, help="Image orientation")
@click.option("--bit-depth", type=click.Choice(["8", "16"]), default="8", help="Output bit depth (8 or 16)")
@click.option("--no-xmp", is_flag=True, help="Don't use XMP metadata")
@click.option("--use-coreimage", is_flag=True, help="Use Core Image pipeline on macOS if available")
def dng_convert(
    input_file, output_file, ifd, temperature, tint, exposure, orientation, bit_depth, no_xmp, use_coreimage
):
    """Convert DNG file to image file (.tif, .jpg, .png, .jxl) with processing options."""
    import numpy as np
    import time
    from .dngio import DngFile
    from .imgio import convert_dng
    
    # Map bit depth to numpy dtype
    output_dtype = np.uint16 if bit_depth == "16" else np.uint8
    
    # Map CLI parameter names to XMP-style rendering_params names
    params = {}
    if temperature is not None:
        params['Temperature'] = temperature
    if tint is not None:
        params['Tint'] = tint
    if exposure is not None:
        params['Exposure2012'] = exposure
    if orientation is not None:
        params['orientation'] = orientation
    
    # Convert DNG
    t_start = time.perf_counter()
    try:
        # Open DNG and select page
        with DngFile(input_file) as dng:
            # Pass dng if no specific IFD, otherwise pass page
            if ifd is None:
                file_arg = dng
                ifd_name = "main raw page"
            else:
                file_arg, _, ifd_name = _get_page_from_ifd_spec(dng, ifd)
            
            # Handle preview pages differently from raw pages
            if ifd is not None and not (file_arg.is_cfa or file_arg.is_linear_raw):
                # Validate parameters before rendering
                if params:
                    click.echo("Error: Rendering parameters (--temperature, --tint, --exposure, --orientation) not allowed for preview pages", err=True)
                    sys.exit(1)
                
                # Preview page: decode directly to RGB
                from .imgio import write_image
                rgb = file_arg.decode_to_rgb(output_dtype=output_dtype)
                if rgb is None:
                    click.echo(f"Error: Failed to decode preview page", err=True)
                    sys.exit(1)

                # Write with metadata (set Orientation=HORIZONTAL since preview is already rotated)
                metadata = dng.get_ifd0_tags()
                metadata.add_tag("Orientation", Orientation.HORIZONTAL)
                success = write_image(rgb, output_file, metadata=metadata)
                if not success:
                    click.echo(f"Error: Failed to write output file", err=True)
                    sys.exit(1)
            else:
                # Raw page: use convert_dng for full rendering pipeline
                success = convert_dng(
                    file=file_arg,
                    output=output_file,
                    output_dtype=output_dtype,
                    demosaic_algorithm=DemosaicAlgorithm.OPENCV_EA,
                    strict=False,
                    use_xmp=not no_xmp,
                    rendering_params=params,
                    use_coreimage_if_available=use_coreimage,
                )
                
                if not success:
                    click.echo(f"Error: Failed to convert {input_file}", err=True)
                    sys.exit(1)
            
            # Shared output message
            # Get photometric from page (file_arg is either DngPage or DngFile)
            if isinstance(file_arg, DngFile):
                page = file_arg.get_main_page()
                photometric = page.photometric_name if page else "unknown"
            else:
                photometric = file_arg.photometric_name
            w, h = file_arg.get_rendered_size(rendering_params=params)
            
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            click.echo(f"Converted {ifd_name} ({photometric}, {w}x{h}) to {output_file} in {elapsed_ms:.0f}ms")
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _load_dng_settings(settings_file: Path) -> tuple[list[Path], list[dict]]:
    """Load DNG rendering settings from CSV file.
    
    Returns:
        Tuple of (file_list, settings_list) where:
        - file_list: List of absolute paths to DNG files from CSV
        - settings_list: List of rendering_params dicts, one per file in same order
    
    If filename in CSV is relative, it's resolved relative to the CSV file's directory.
    Only includes non-None values in the rendering params.
    """
    from dataclasses import dataclass
    from .csv import CsvReader
    
    @dataclass
    class DngRenderSettings:
        """Per-file DNG rendering settings loaded from CSV."""
        filename: str
        Temperature: float | None = None
        Tint: float | None = None
        Exposure2012: float | None = None
        orientation: int | None = None
    
    file_list = []
    settings_list = []
    csv_dir = settings_file.parent
    
    with CsvReader(settings_file, header_schema=[], data_type=DngRenderSettings) as reader:        
        for settings in reader:  
            # Resolve file path: if absolute use as-is, else relative to CSV dir
            file_path = Path(settings.filename)
            if not file_path.is_absolute():
                file_path = csv_dir / file_path
            file_list.append(file_path)
            
            # Build rendering params dict with only non-None values
            rendering_params = {}
            if settings.Temperature is not None:
                rendering_params['Temperature'] = settings.Temperature
            if settings.Tint is not None:
                rendering_params['Tint'] = settings.Tint
            if settings.Exposure2012 is not None:
                rendering_params['Exposure2012'] = settings.Exposure2012
            if settings.orientation is not None:
                rendering_params['orientation'] = settings.orientation
            
            settings_list.append(rendering_params)
    
    return file_list, settings_list


@dng.command(name="batch-convert")
@click.argument("input", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path())
@click.option("--format", "output_format", type=click.Choice(["tif", "jxl", "jpg"]), default="tif", help="Output format")
@click.option("--bit-depth", type=click.Choice(["8", "16"]), default="16", help="Output bit depth (8 or 16)")
@click.option("--temperature", type=float, help="White balance temperature")
@click.option("--tint", type=float, help="White balance tint")
@click.option("--exposure", type=float, help="Exposure adjustment in stops")
@click.option("--orientation", type=int, help="Image orientation")
@click.option("--no-xmp", is_flag=True, help="Don't use XMP metadata")
@click.option("--use-coreimage", is_flag=True, help="Use Core Image pipeline on macOS if available")
@click.option("--scale", type=float, default=1.0, help="Resolution scale factor (e.g., 0.5 for half size)")
@click.option("--num-workers", type=int, default=4, help="Number of parallel processing threads")
def batch_convert_dngs(
    input, output_folder, output_format, bit_depth, temperature, tint, 
    exposure, orientation, no_xmp, use_coreimage, scale, num_workers
):
    """Convert DNG files to TIFF/JXL/JPG images.
    
    INPUT can be either:
    - A folder containing DNG files (will scan for *.dng)
    - A CSV file with per-file settings (filename,Temperature,Tint,Exposure2012,orientation)
    """
    import io
    import numpy as np
    from .dngio import decode_dng, DngFile, DemosaicAlgorithm
    from .imgio import ImageSequencePipeline
    
    # Set process title for easier identification in task managers
    try:
        import setproctitle
        setproctitle.setproctitle("muimg: batch-convert")
    except ImportError:
        pass  # setproctitle is optional
    
    # Build base rendering_params dict from CLI options
    base_rendering_params = {}
    if temperature is not None:
        base_rendering_params['Temperature'] = temperature
    if tint is not None:
        base_rendering_params['Tint'] = tint
    if exposure is not None:
        base_rendering_params['Exposure2012'] = exposure
    if orientation is not None:
        base_rendering_params['Orientation'] = orientation
    
    # Determine file list and settings based on input type
    input_path = Path(input)
    settings_list = []
    
    if input_path.is_file() and input_path.suffix.lower() == '.csv':
        # CSV file drives the file list
        dng_files, settings_list = _load_dng_settings(input_path)
        click.echo(f"Loaded {len(dng_files)} files from {input}")
    elif input_path.is_dir():
        # Scan input folder for DNG files
        dng_files = sorted(input_path.glob("*.dng"))
        
        if not dng_files:
            click.echo(f"Error: No DNG files found in {input}", err=True)
            sys.exit(1)
        
        click.echo(f"Found {len(dng_files)} DNG files in {input}")
    else:
        click.echo(f"Error: INPUT must be either a folder or a CSV file", err=True)
        sys.exit(1)
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine output dtype
    output_dtype = np.uint16 if bit_depth == "16" else np.uint8
    
    # Custom DNG consumer - decodes DNG and encodes to output format
    def dng_consumer(task: tuple[int, str, bytes]) -> tuple[int, str, bytes | None]:
        from .imgio import write_image
        
        index, file_path, blob = task
        try:
            # Create DngFile from blob
            dng_file = DngFile(io.BytesIO(blob))
            
            # Build rendering params: use settings from CSV or base params from CLI
            if settings_list:
                # CSV mode: use settings from list, merge with base
                rendering_params = base_rendering_params.copy()
                rendering_params.update(settings_list[index])
            else:
                # Folder mode: use base params from CLI
                rendering_params = base_rendering_params.copy()
            
            # Decode with rendering params and scale
            img, metadata = decode_dng(
                file=dng_file,
                output_dtype=output_dtype,
                demosaic_algorithm=DemosaicAlgorithm.OPENCV_EA,
                use_coreimage_if_available=use_coreimage,
                use_xmp=not no_xmp,
                rendering_params=rendering_params,
                strict=False,
                scale=scale,
            )
            
            if img is None:
                logger.warning(f"Frame {index}: Failed to decode {Path(file_path).name}")
                return (index, file_path, None)
            
            # Encode to output format with metadata
            output_stream = io.BytesIO()
            write_image(img, output_stream, output_format_stream=output_format, metadata=metadata)
            encoded_blob = output_stream.getvalue()
            
            return (index, file_path, encoded_blob)
        except Exception as e:
            logger.error(f"Frame {index}: Error decoding {Path(file_path).name} ({type(e).__name__}): {e}")
            return (index, file_path, None)
    
    # Create and run the pipeline
    try:
        pipeline = ImageSequencePipeline(
            source_files=dng_files,
            output_folder=output_path,
            output_format=output_format,
            output_dtype=output_dtype,
            consumer=dng_consumer,
            num_workers=num_workers,
            task_name="DNG Batch Convert",
        )
        
        click.echo(f"Converting {len(dng_files)} DNGs to {output_format.upper()}...")
        
        import time
        start_time = time.perf_counter()
        pipeline.run()
        elapsed = time.perf_counter() - start_time
        
        click.echo(f"Successfully converted {len(dng_files)} files to {output_folder}")
        click.echo(f"Processed {len(dng_files)} files in {elapsed:.2f}s ({len(dng_files)/elapsed:.2f} files/s)")
        
        # Print queue stats
        stats = pipeline.get_queue_stats()
        if "task_queue" in stats:
            q = stats["task_queue"]
            click.echo(f"  Queue stats - Task queue: avg_depth={q['avg_depth']:.1f}, empty_time={q['empty_time']:.1f}s")
        if "writer_queue" in stats:
            q = stats["writer_queue"]
            click.echo(f"  Queue stats - Writer queue: avg_depth={q['avg_depth']:.1f}, empty_time={q['empty_time']:.1f}s")
        
    except Exception as e:
        logger.exception("Failed to convert DNGs")
        sys.exit(1)


@dng.command(name="batch-to-video")
@click.argument("input", type=click.Path(exists=True))
@click.argument("output_mp4", type=click.Path())
@click.option("--temperature", type=float, help="White balance temperature")
@click.option("--tint", type=float, help="White balance tint")
@click.option("--exposure", type=float, help="Exposure adjustment in stops")
@click.option("--orientation", type=int, help="Image orientation")
@click.option("--no-xmp", is_flag=True, help="Don't use XMP metadata")
@click.option("--use-coreimage", is_flag=True, help="Use Core Image pipeline on macOS if available")
@click.option("--resolution", type=str, default="1920x1080", help="Output video resolution (e.g., '1920x1080')")
@click.option("--codec", type=str, default="h264", help="Video codec (h264, hevc, vp9)")
@click.option("--crf", type=int, default=20, help="Constant Rate Factor for quality (lower=better)")
@click.option("--bit-depth", type=click.Choice(["8", "10"]), default="8", help="Video bit depth (8 or 10)")
@click.option("--frame-rate", type=float, default=30, help="Output frame rate in fps (supports fractional rates, e.g., 0.5 for 1 frame every 2 seconds)")
@click.option("--num-workers", type=int, default=4, help="Number of parallel processing threads")
@click.option("--overlay-txt", is_flag=True, help="Add filename overlay to each frame")
def convert_dngs_to_video(
    input, output_mp4, temperature, tint, exposure, orientation, 
    no_xmp, use_coreimage, resolution, codec, crf, bit_depth, frame_rate, num_workers, overlay_txt
):
    """Convert DNG files to MP4 video.
    
    INPUT can be either:
    - A folder containing DNG files (will scan for *.dng)
    - A CSV file with per-file settings (filename,Temperature,Tint,Exposure2012,orientation)
    """
    import io
    import numpy as np
    from .dngio import decode_dng
    from .videoio import VideoEncodePipeline
    
    # Set process title for easier identification in task managers
    try:
        import setproctitle
        setproctitle.setproctitle(f"muimg: encoding {Path(output_mp4).name}")
    except ImportError:
        pass  # setproctitle is optional
    
    # Parse resolution if provided
    resolution_tuple = None
    if resolution:
        try:
            width, height = resolution.split('x')
            resolution_tuple = (int(width), int(height))
        except ValueError:
            click.echo(f"Error: Invalid resolution format '{resolution}'. Use format like '1920x1080'", err=True)
            sys.exit(1)
    
    # Build base rendering_params dict from CLI options
    base_rendering_params = {}
    if temperature is not None:
        base_rendering_params['Temperature'] = temperature
    if tint is not None:
        base_rendering_params['Tint'] = tint
    if exposure is not None:
        base_rendering_params['Exposure2012'] = exposure
    if orientation is not None:
        base_rendering_params['orientation'] = orientation
    
    # Determine file list and settings based on input type
    input_path = Path(input)
    settings_list = []
    
    if input_path.is_file() and input_path.suffix.lower() == '.csv':
        # CSV file drives the file list
        dng_files, settings_list = _load_dng_settings(input_path)
        click.echo(f"Loaded {len(dng_files)} files from {input}")
    elif input_path.is_dir():
        # Scan input folder for DNG files
        dng_files = sorted(input_path.glob("*.dng"))
        
        if not dng_files:
            click.echo(f"Error: No DNG files found in {input}", err=True)
            sys.exit(1)
        
        click.echo(f"Found {len(dng_files)} DNG files in {input}")
    else:
        click.echo(f"Error: INPUT must be either a folder or a CSV file", err=True)
        sys.exit(1)
    
    # Build video encoding config
    config = {
        "codec": codec,
        "crf": crf,
        "bit_depth": int(bit_depth),
        "frame_rate": frame_rate,
    }
    
    # Determine output dtype based on bit depth
    output_dtype = np.uint16 if int(bit_depth) == 10 else np.uint8
    
    # Create custom consumer function for DNG decoding
    def dng_consumer(task):
        """Custom consumer: decodes DNG blob using decode_dng with rendering params."""
        index, file_path, blob = task
        try:
            # Create DngFile from blob
            from .dngio import DngFile
            dng_file = DngFile(io.BytesIO(blob))
            
            # Build rendering params: use settings from CSV or base params from CLI
            if settings_list:
                # CSV mode: use settings from list, merge with base
                rendering_params = base_rendering_params.copy()
                rendering_params.update(settings_list[index])
            else:
                # Folder mode: use base params from CLI
                rendering_params = base_rendering_params.copy()
            
            # Calculate scale before decoding for efficiency
            # Scaling during decode_dng is much more efficient as it avoids the full render path
            # at full resolution
            # Get render size from DNG to compute scale
            # Pass rendering_params so it uses orientation from params if specified
            render_width, render_height = dng_file.get_rendered_size(
                rendering_params=rendering_params
            )
            
            target_width, target_height = resolution_tuple
            scale_w = target_width / render_width
            scale_h = target_height / render_height
            scale = min(scale_w, scale_h)
            
            # Debug output
            orientation = rendering_params.get('orientation', 'None')
            logger.info(f"Frame {index} ({Path(file_path).name}): orientation={orientation}, "
                       f"rendered_size={render_width}x{render_height}, scale={scale:.3f}")

            # Decode with scaling applied during rendering
            img, _ = decode_dng(
                file=dng_file,
                output_dtype=output_dtype,
                demosaic_algorithm=DemosaicAlgorithm.OPENCV_EA,
                use_coreimage_if_available=use_coreimage,
                use_xmp=not no_xmp,
                rendering_params=rendering_params,
                strict=False,
                scale=scale,
            )
            
            if img is None:
                logger.warning(f"Frame {index}: Failed to decode {Path(file_path).name}")
                return (index, None)
            
            # Apply letterboxing
            current_height, current_width = img.shape[:2]
            target_width, target_height = resolution_tuple
            
            logger.info(f"Frame {index}: decoded image size={current_width}x{current_height}, "
                       f"target={target_width}x{target_height}")
            
            # Create black canvas at target resolution
            canvas = np.zeros((target_height, target_width, img.shape[2]), dtype=img.dtype)
            
            # Calculate centered position for scaled image
            y_offset = (target_height - current_height) // 2
            x_offset = (target_width - current_width) // 2
            
            # Place scaled image on canvas
            canvas[y_offset:y_offset+current_height, x_offset:x_offset+current_width] = img
            img = canvas
            
            # Add filename overlay if requested
            if overlay_txt:
                from .videoio import add_text_overlay
                filename = Path(file_path).name
                img = add_text_overlay(img, filename, position="bottom-left")
            
            return (index, img)
            
        except Exception as e:
            logger.error(f"Frame {index}: Error decoding {Path(file_path).name} ({type(e).__name__}): {e}")
            return (index, None)
    
    # Create and run the pipeline
    try:
        pipeline = VideoEncodePipeline(
            source_files=dng_files,
            output_path=output_mp4,
            resolution=resolution_tuple,
            config=config,
            consumer=dng_consumer,
            num_workers=num_workers,
        )
        
        click.echo(f"Encoding {len(dng_files)} DNGs to {output_mp4}...")
        
        import time
        start_time = time.perf_counter()
        pipeline.run()
        elapsed = time.perf_counter() - start_time
        
        # Count successful frames (exclude None results)
        successful_frames = len(dng_files)  # Assume all succeeded unless we track failures
        fps = successful_frames / elapsed if elapsed > 0 else 0
        
        click.echo(f"Successfully created video: {output_mp4}")
        click.echo(f"Encoded {successful_frames} frames in {elapsed:.2f}s ({fps:.2f} fps)")
        
        # Print queue statistics (indented)
        stats = pipeline.get_queue_stats()
        click.echo(
            f"  Queue stats - Task queue: avg_depth={stats['task_queue']['avg_depth']:.1f}, "
            f"empty_time={stats['task_queue']['empty_time']:.1f}s"
        )
        click.echo(
            f"  Queue stats - Writer queue: avg_depth={stats['writer_queue']['avg_depth']:.1f}, "
            f"empty_time={stats['writer_queue']['empty_time']:.1f}s"
        )
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Failed to encode video")
        sys.exit(1)


@cli.group(name="google-photos")
def google_photos():
    """Google Photos integration commands."""
    pass


@google_photos.command()
@click.option(
    "--credentials",
    type=click.Path(exists=True),
    help="Path to OAuth2 credentials JSON file",
)
@click.option(
    "--token-path",
    type=click.Path(),
    help="Path to store/load refresh token",
)
@click.option("--force", is_flag=True, help="Force re-authentication")
def auth(credentials, token_path, force):
    """Authenticate with Google Photos API.

    This will open a browser window for OAuth2 authentication.
    The refresh token will be saved for future automated access.
    """
    from .google_photos import GooglePhotosClient
    
    creds_path = Path(credentials) if credentials else None
    token_path = Path(token_path) if token_path else None

    client = GooglePhotosClient(
        credentials_path=creds_path, token_path=token_path
    )

    if client.authenticate(force_reauth=force):
        click.echo(f"✓ Successfully authenticated")
        click.echo(f"✓ Token saved to: {client.token_path}")
        click.echo("\nYou can now use 'muimg google-photos upload' for automated uploads")
        sys.exit(0)
    else:
        click.echo("✗ Authentication failed", err=True)
        sys.exit(1)


@google_photos.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--album", help="Album name (will be created if doesn't exist)")
@click.option(
    "--token-path",
    type=click.Path(),
    help="Path to stored refresh token",
)
def upload(image_path, album, token_path):
    """Upload an image to Google Photos."""
    from .google_photos import GooglePhotosClient
    
    image_path = Path(image_path)
    token_path = Path(token_path) if token_path else None

    client = GooglePhotosClient(token_path=token_path)

    if not client.authenticate():
        click.echo(
            "✗ Authentication failed. Run 'muimg google-photos auth' first.",
            err=True,
        )
        sys.exit(1)

    try:
        result = client.upload_image(image_path, album_title=album)
        click.echo(f"✓ Uploaded: {image_path.name}")
        if album:
            click.echo(f"  Album: {album}")
        click.echo(f"  URL: {result['productUrl']}")
        sys.exit(0)
    except Exception as e:
        click.echo(f"✗ Upload failed: {e}", err=True)
        logger.exception("Upload error")
        sys.exit(1)


@google_photos.command(name="list-albums")
@click.option(
    "--token-path",
    type=click.Path(),
    help="Path to stored refresh token",
)
def list_albums(token_path):
    """List all albums in Google Photos."""
    from .google_photos import GooglePhotosClient
    
    token_path = Path(token_path) if token_path else None

    client = GooglePhotosClient(token_path=token_path)

    if not client.authenticate():
        click.echo(
            "✗ Authentication failed. Run 'muimg google-photos auth' first.",
            err=True,
        )
        sys.exit(1)

    try:
        albums = client.list_albums()
        if not albums:
            click.echo("No albums found")
        else:
            click.echo(f"Found {len(albums)} albums:\n")
            for album in albums:
                click.echo(f"  • {album['title']}")
                click.echo(f"    URL: {album['productUrl']}")
                click.echo()
        sys.exit(0)
    except Exception as e:
        click.echo(f"✗ Failed to list albums: {e}", err=True)
        logger.exception("List albums error")
        sys.exit(1)


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
