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


@cli.command(name="convert-dng")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--temperature", type=float, help="White balance temperature")
@click.option("--tint", type=float, help="White balance tint")
@click.option("--exposure", type=float, help="Exposure adjustment in stops")
@click.option("--noise-reduction", type=float, help="Noise reduction amount")
@click.option("--orientation", type=int, help="Image orientation")
@click.option("--bit-depth", type=click.Choice(["8", "16"]), default="8", help="Output bit depth (8 or 16)")
@click.option("--no-xmp", is_flag=True, help="Don't use XMP metadata")
@click.option("--use-coreimage", is_flag=True, help="Use Core Image pipeline on macOS if available")
def convert_dng(
    input_file, output_file, temperature, tint, exposure, noise_reduction, orientation, bit_depth, no_xmp, use_coreimage
):
    """Convert DNG file to image file with processing options."""
    import numpy as np
    
    # Map bit depth to numpy dtype
    output_dtype = np.uint16 if bit_depth == "16" else np.uint8
    
    success = convert_imgformat(
        file=input_file,
        output_path=output_file,
        temperature=temperature,
        tint=tint,
        exposure=exposure,
        noise_reduction=noise_reduction,
        orientation=orientation,
        output_dtype=output_dtype,
        use_xmp=not no_xmp,
        use_coreimage_if_available=use_coreimage,
    )

    sys.exit(0 if success else 1)


@cli.command(name="convert-image")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--bit-depth", type=click.Choice(["8", "16"]), default="8", help="Output bit depth (8 or 16)")
def convert_image_cmd(input_file, output_file, bit_depth):
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
