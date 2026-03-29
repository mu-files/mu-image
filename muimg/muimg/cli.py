"""Command-line interface for muimg."""

import logging
import sys
from pathlib import Path

import click

from .imgio import convert_imgformat
from .google_photos import GooglePhotosClient

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


@cli.group(name="dng")
def dng():
    """DNG file operations."""
    pass


@dng.command(name="metadata")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--ifd", type=int, help="Show specific IFD (0=IFD0, 1+=SubIFDs)")
@click.option("--tag", "filter_tags", multiple=True, help="Show only specific tag(s) (case-insensitive)")
def dng_metadata(input_file, ifd, filter_tags):
    """Display DNG file metadata."""
    from . import dngio
    from .tiff_metadata import TIFF_TAG_TYPE_REGISTRY, TIFF_DTYPES, LOCAL_TIFF_TAGS
    import numpy as np
    
    # Normalize filter tags to lowercase for case-insensitive matching
    filter_tags_lower = set(tag.lower() for tag in filter_tags) if filter_tags else None
    
    try:
        dng_file = dngio.DngFile(input_file)
    except Exception as e:
        click.echo(f"Error opening DNG file: {e}", err=True)
        sys.exit(1)
    
    # Get flattened pages for flat numbering (IFD0, then SubIFDs)
    pages = dng_file.get_flattened_pages()
    
    if not pages:
        click.echo("No IFDs found in file", err=True)
        sys.exit(1)
    
    # Show summary of available IFDs
    click.echo(f"File contains {len(pages)} IFD(s):")
    for i, p in enumerate(pages):
        ifd_type = "IFD0" if i == 0 else f"SubIFD[{i-1}]"
        photometric = p.photometric_name or "Unknown"
        width = p.imagewidth or "?"
        length = p.imagelength or "?"
        summary_indent = "" if i == 0 else "  "  # Extra indent for SubIFDs
        click.echo(f"{i}:{summary_indent} {ifd_type} - {photometric}, {width}x{length}")
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
    
    # Display metadata for each IFD
    for idx, (page, ifd_num) in enumerate(zip(pages_to_show, ifd_indices)):
        if idx > 0:
            click.echo()  # Blank line between IFDs
        
        # Use TIFF terminology for header
        if ifd_num == 0:
            ifd_label = "IFD0"
            indent = ""
            header_indent = ""
        else:
            ifd_label = f"SubIFD {ifd_num - 1}"
            indent = "  "  # Indent SubIFD tags
            header_indent = "  "  # Indent SubIFD header
        click.echo(f"{header_indent}=== {ifd_label} (--ifd {ifd_num}) ===")
        
        # Get all tags using DngPage API
        page_tags = page.get_page_tags()
        
        # Iterate through tags
        for tag_code, dtype, count, _, _ in page_tags:
            # Get tag name from registry
            tag_name = LOCAL_TIFF_TAGS.get(tag_code)
            if tag_name is None:
                tag_name = f"Tag{tag_code}"
            
            # Filter if requested (case-insensitive)
            if filter_tags_lower and tag_name.lower() not in filter_tags_lower:
                continue
            
            # Get the converted value using DngPage API
            value = page.get_tag(tag_code)
            
            # Display the tag with appropriate indentation
            _display_tag(tag_name, value, indent, tag_code, dtype, count)
    
    sys.exit(0)


def _display_tag(tag_name, value, indent="", tag_code=None, dtype=None, count=None):
    """Format and display a single tag value.
    
    Args:
        tag_name: Name of the tag
        value: Already-converted value from page.get_tag()
        indent: String to prepend to each line (for SubIFD indentation)
        tag_code: Numeric tag code (for unknown tags)
        dtype: TIFF dtype code (for unknown tags)
        count: Element count (for unknown tags)
    """
    import numpy as np
    from .tiff_metadata import XmpMetadata, TIFF_DTYPES
    
    def echo(text):
        """Helper to echo with indentation."""
        click.echo(f"{indent}{text}")
    
    # For unknown tags, show tag code and type info
    is_unknown = tag_name.startswith("Tag") and tag_code is not None
    if is_unknown and dtype is not None:
        dtype_info = TIFF_DTYPES.get(dtype, {})
        dtype_name = dtype_info.get('name', f'Type{dtype}')
        tag_display = f"{tag_name} ({tag_code}, {dtype_name})"
    else:
        tag_display = tag_name
    
    # Special handling for XMP
    if isinstance(value, XmpMetadata):
        size = len(str(value))
        echo(f"{tag_display}: XML metadata, {size} bytes")
        return
    
    # Special handling for EXIF/GPS dictionaries - enumerate fields
    if isinstance(value, dict) and tag_name in ("ExifTag", "GPSTag"):
        echo(f"{tag_display}: {len(value)} fields")
        # Display each EXIF field with extra indentation
        for key, val in value.items():
            # Recursively display each field
            _display_tag(str(key), val, indent + "  ", None, None, None)
        return
    
    # Handle None
    if value is None:
        echo(f"{tag_display}: None")
        return
    
    # Strings
    if isinstance(value, str):
        if len(value) > 200:
            echo(f"{tag_display}: {value[:200]}... ({len(value)} chars)")
        else:
            echo(f"{tag_display}: {value}")
        return
    
    # Special formatting for DNG version tags (4-tuple from get_tag)
    if tag_name in ("DNGVersion", "DNGBackwardVersion") and isinstance(value, tuple) and len(value) == 4:
        version_str = f"{value[0]}.{value[1]}.{value[2]}.{value[3]}"
        echo(f"{tag_display}: {version_str}")
        return
    
    # Bytes (binary data) - handle both bytes and numpy byte arrays
    if isinstance(value, bytes):
        if len(value) > 100:
            echo(f"{tag_display}: Binary data, {len(value)} bytes")
        else:
            echo(f"{tag_display}: {value}")
        return
    
    # Check for numpy scalar bytes (0-d array with bytes dtype)
    if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind in ('S', 'V'):
        byte_value = value.item()
        if isinstance(byte_value, bytes):
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
        echo(f"{tag_display}: {value.item()}")
        return
    
    # 2D arrays (matrices) - display in matrix format
    if value.ndim == 2:
        echo(f"{tag_display}:")
        for i in range(min(value.shape[0], 3)):  # Max 3 lines
            row_str = "  " + "  ".join(f"{v:8.4f}" for v in value[i])
            echo(row_str)
        if value.shape[0] > 3:
            echo(f"  ... ({value.shape[0]} rows total)")
        return
    
    # Small 1D arrays (≤9 elements): show full array
    if value.size <= 9:
        clean_list = [v.item() if isinstance(v, (np.integer, np.floating)) else v for v in value.flat]
        echo(f"{tag_display}: {clean_list}")
        return
    
    # Medium 1D arrays (9 < size ≤ 16): show first 9 with truncation
    if value.size <= 16:
        clean_list = [v.item() if isinstance(v, (np.integer, np.floating)) else v for v in value.flat[:9]]
        echo(f"{tag_display}: {clean_list}... ({value.size} elements)")
        return
    
    # Medium arrays (16 < count ≤ 100): format as multi-line
    if value.size <= 100:
        # Try to reshape into matrix format
        if value.size == 9:
            value = value.reshape(3, 3)
        elif value.size == 6:
            value = value.reshape(2, 3)
        
        if value.ndim == 2:
            # Multi-line matrix display (like dng_validate)
            echo(f"{tag_display}:")
            for i in range(min(value.shape[0], 3)):
                row_str = "  " + "  ".join(f"{v:8.4f}" for v in value[i])
                echo(row_str)
            if value.shape[0] > 3:
                echo(f"  ... ({value.shape[0]} rows total)")
        else:
            # 1D array - show first 9 elements
            clean_values = [v.item() if isinstance(v, (np.integer, np.floating)) else v for v in value[:9]]
            echo(f"{tag_display}: {clean_values}... ({value.size} elements)")
        return
    
    # Very large arrays: show type, element count, and byte size
    dtype_name = value.dtype.name
    num_elements = value.size
    num_bytes = value.nbytes
    echo(f"{tag_display}: {dtype_name} array, {num_elements} elements, {num_bytes} bytes")


@dng.command(name="convert")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--temperature", type=float, help="White balance temperature")
@click.option("--tint", type=float, help="White balance tint")
@click.option("--exposure", type=float, help="Exposure adjustment in stops")
@click.option("--orientation", type=int, help="Image orientation")
@click.option("--bit-depth", type=click.Choice(["8", "16"]), default="8", help="Output bit depth (8 or 16)")
@click.option("--no-xmp", is_flag=True, help="Don't use XMP metadata")
@click.option("--use-coreimage", is_flag=True, help="Use Core Image pipeline on macOS if available")
def dng_convert(
    input_file, output_file, temperature, tint, exposure, orientation, bit_depth, no_xmp, use_coreimage
):
    """Convert DNG file to image file with processing options."""
    import numpy as np
    
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
    # Note: noise_reduction not currently supported in rendering_params
    
    success = convert_imgformat(
        file=input_file,
        output_path=output_file,
        output_dtype=output_dtype,
        use_xmp=not no_xmp,
        use_coreimage_if_available=use_coreimage,
        **params,
    )

    sys.exit(0 if success else 1)


@cli.command(name="convert-image")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--bit-depth", type=click.Choice(["8", "16"]), default="8", help="Output bit depth (8 or 16)")
def convert_image(input_file, output_file, bit_depth):
    """Convert image file to another format."""
    import numpy as np
    
    # Map bit depth to numpy dtype
    output_dtype = np.uint16 if bit_depth == "16" else np.uint8
    
    success = convert_imgformat(
        file=input_file,
        output_path=output_file,
        output_dtype=output_dtype,
    )

    sys.exit(0 if success else 1)


@dng.command(name="copy")
@click.argument("input_dng", type=click.Path(exists=True))
@click.argument("output_dng", type=click.Path())
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
    default="RCD",
    help="Demosaic algorithm (RCD, VNG, OPENCV_EA)",
)
@click.option(
    "--strip-tag",
    multiple=True,
    help="Tag name to strip (can be specified multiple times)",
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
    help="JXL compression distance (0=lossless, >0=lossy). None=uncompressed.",
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
def dng_copy(
    input_dng,
    output_dng,
    scale,
    demosaic,
    demosaic_algorithm,
    strip_tag,
    preview,
    preview_max_dim,
    jxl_distance,
    jxl_effort,
    tag,
):
    """Copy DNG file with optional transformations.
    
    Apply scale, demosaic (CFA to LINEAR_RAW), strip tags, and/or generate preview.
    """
    from . import dngio
    from .tiff_metadata import MetadataTags
    
    strip_tags_set = set(strip_tag) if strip_tag else None
    
    # Parse --tag NAME=VALUE options into MetadataTags
    ifd0_tags = None
    if tag:
        ifd0_tags = MetadataTags()
        for tag_spec in tag:
            if "=" not in tag_spec:
                click.echo(f"Error: Invalid tag format '{tag_spec}'. Use NAME=VALUE", err=True)
                sys.exit(1)
            name, value = tag_spec.split("=", 1)
            ifd0_tags.add_tag(name.strip(), value.strip())
    
    try:
        dngio.copy_dng(
            source_file=input_dng,
            destination_file=output_dng,
            scale=scale,
            demosaic=demosaic,
            demosaic_algorithm=demosaic_algorithm,
            strip_tags=strip_tags_set,
            generate_preview=preview,
            preview_max_dimension=preview_max_dim,
            jxl_distance=jxl_distance,
            jxl_effort=jxl_effort,
            ifd0_tags=ifd0_tags,
        )
        click.echo(f"Successfully copied DNG to {output_dng}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Failed to copy DNG")
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
