"""Common utilities for muimg."""

import logging
import sys
from enum import Enum
from typing import Type, Optional


def enum_display_name(enum_class: Type[Enum], value: int, suffix: str = "") -> str:
    """
    Get display name for an enum value.
    
    Converts enum member name to display format (e.g., MAIN_IMAGE -> MainImage).
    Returns formatted value string if enum member not found.
    
    Args:
        enum_class: The enum class to look up
        value: The numeric value to find
        suffix: Optional suffix to append (e.g., " compression")
        
    Returns:
        Display name string (e.g., "MainImage") or "Type{value}" if not found
    """
    try:
        member = enum_class(value)
        display = ''.join(word.capitalize() for word in member.name.split('_'))
        return f"{display}{suffix}" if suffix else display
    except ValueError:
        return f"Type{value}{suffix}" if suffix else f"Type{value}"


def enum_from_value(enum_class: Type[Enum], value: int) -> Optional[Enum]:
    """
    Get enum member from numeric value.
    
    Args:
        enum_class: The enum class to look up
        value: The numeric value to find
        
    Returns:
        Enum member or None if not found
    """
    try:
        return enum_class(value)
    except ValueError:
        return None


def enum_from_string(enum_class: Type[Enum], value: str) -> Enum:
    """
    Get enum member from string value.
    
    For string enums (inheriting from str and Enum), this looks up the member
    by its string value. Raises KeyError if not found.
    
    Args:
        enum_class: The enum class to look up
        value: The string value to find
        
    Returns:
        Enum member
        
    Raises:
        KeyError: If value not found in enum
    """
    # For str enums, we can iterate and compare values
    for member in enum_class:
        if member.value == value:
            return member
    raise KeyError(f"'{value}' is not a valid {enum_class.__name__}")


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
