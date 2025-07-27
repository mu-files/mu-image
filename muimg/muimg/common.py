"""Common utilities for muimg."""

import logging
import sys


def setup_logging(verbosity: int = 0) -> None:
    """
    Set up logging configuration based on verbosity level.
    
    Args:
        verbosity: Verbosity level (0=ERROR, 1=WARNING, 2=INFO, 3+=DEBUG)
    """
    if verbosity == 0:
        level = logging.ERROR
    elif verbosity == 1:
        level = logging.WARNING
    elif verbosity == 2:
        level = logging.INFO
    else:
        level = logging.DEBUG
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    
    # Set specific logger levels
    logger = logging.getLogger('muimg')
    logger.setLevel(level)
