"""Command-line interface for muimg tools."""
import click
import setproctitle

from pathlib import Path
from typing import Optional

# Package imports
from .calib import calib_prep, calib_compare
from .common import setup_logging


@click.group()
@click.option("-v", "--verbose", "verbosity", count=True, help="Enable verbose logging.")
@click.pass_context
def cli(ctx, verbosity):
    """muimg - DNG and raw image processing utilities."""
    setup_logging(verbosity)
    ctx.ensure_object(dict)
    ctx.obj['verbosity'] = verbosity


@cli.group()
def calib():
    """Camera calibration commands."""
    pass


@calib.command()
@click.argument('input_dng', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dng', type=click.Path(path_type=Path))
@click.option('--dcp-file', type=click.Path(exists=True, path_type=Path), help='DCP file to extract color matrix from (optional)')
@click.option('--analog-balance', type=str, help='Analog balance as 3 comma-separated floats (e.g., "1.0,1.2,0.8")')
@click.option('--analog-balance-compose', type=str, help='Analog balance to compose into color matrix as diag(AB) * matrix (e.g., "1.0,1.2,0.8")')
@click.option('--neutral-coordinate', type=str, help='Image coordinate for neutral sampling as "x,y" (e.g., "1664,1080")')
@click.option('--neutral-color', type=str, help='Neutral color as 3 comma-separated floats (e.g., "0.64,1.0,0.96")')
@click.option('--force', is_flag=True, help='Overwrite output file if it exists')
def prep(input_dng: Path, output_dng: Path, dcp_file: Optional[Path], analog_balance: Optional[str], analog_balance_compose: Optional[str], neutral_coordinate: Optional[str], neutral_color: Optional[str], force: bool):
    """Prepare calibrated DNG by embedding color matrix from DCP file.
    
    INPUT_DNG: Input DNG file path
    OUTPUT_DNG: Output DNG file path
    
    If --dcp-file is not provided, uses identity matrix and unknown illuminant.
    """
    setproctitle.setproctitle(f"muimg-calib-prep")
    
    # Validate that only one analog balance option is provided
    if analog_balance and analog_balance_compose:
        click.echo("Error: Cannot specify both --analog-balance and --analog-balance-compose")
        return 1
    
    # Parse analog balance if provided
    analog_balance_array = None
    if analog_balance:
        try:
            values = [float(x.strip()) for x in analog_balance.split(',')]
            if len(values) != 3:
                click.echo(f"Error: Analog balance must have exactly 3 values, got {len(values)}")
                return 1
            analog_balance_array = values
        except ValueError as e:
            click.echo(f"Error: Invalid analog balance format: {e}")
            return 1
    
    # Parse analog balance compose if provided
    analog_balance_compose_array = None
    if analog_balance_compose:
        try:
            values = [float(x.strip()) for x in analog_balance_compose.split(',')]
            if len(values) != 3:
                click.echo(f"Error: Analog balance compose must have exactly 3 values, got {len(values)}")
                return 1
            analog_balance_compose_array = values
        except ValueError as e:
            click.echo(f"Error: Invalid analog balance compose format: {e}")
            return 1
    
    # Validate that only one neutral option is provided
    if neutral_coordinate and neutral_color:
        click.echo("Error: Cannot specify both --neutral-coordinate and --neutral-color")
        return 1
    
    # Parse neutral coordinate if provided
    neutral_coordinate_tuple = None
    if neutral_coordinate:
        try:
            values = [int(x.strip()) for x in neutral_coordinate.split(',')]
            if len(values) != 2:
                click.echo(f"Error: Neutral coordinate must have exactly 2 values, got {len(values)}")
                return 1
            neutral_coordinate_tuple = tuple(values)
        except ValueError as e:
            click.echo(f"Error: Invalid neutral coordinate format: {e}")
            return 1
    
    # Parse neutral color if provided
    neutral_color_array = None
    if neutral_color:
        try:
            values = [float(x.strip()) for x in neutral_color.split(',')]
            if len(values) != 3:
                click.echo(f"Error: Neutral color must have exactly 3 values, got {len(values)}")
                return 1
            # Validate that values are in reasonable range (0-1)
            for i, val in enumerate(values):
                if val < 0 or val > 1:
                    click.echo(f"Warning: Neutral color value {val} at index {i} is outside [0,1] range")
            neutral_color_array = values
        except ValueError as e:
            click.echo(f"Error: Invalid neutral color format: {e}")
            return 1
    
    return calib_prep(input_dng, dcp_file, output_dng, force, analog_balance_array, analog_balance_compose_array, neutral_coordinate_tuple, neutral_color_array)


@calib.command()
@click.argument('dcp_file1', type=click.Path(exists=True, path_type=Path))
@click.argument('dcp_file2', type=click.Path(exists=True, path_type=Path))
def compare(dcp_file1, dcp_file2):
    """Compare two DCP files and analyze their color matrices.
    
    DCP_FILE1: First DCP file path
    DCP_FILE2: Second DCP file path
    
    Prints illuminants for each file, matrix1 for each, and inv(matrix2) * matrix1.
    """
    setproctitle.setproctitle(f"muimg-calib-compare")
    return calib_compare(dcp_file1, dcp_file2)


def main():
    """Entry point for console scripts."""
    try:
        cli()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1
    return 0


if __name__ == '__main__':
    main()
