"""
Utility functions for PeopleOS.

Contains shared helper functions used across multiple modules.
"""

import os
from typing import Any

import yaml


def load_config() -> dict:
    """Load configuration and bind mutable desktop storage to PEOPLEOS_HOME.

    The source config remains immutable application configuration. When the
    desktop/local product sets PEOPLEOS_HOME, mutable persistence is redirected
    to the user's OS-native PeopleOS data directory so upgrades never write
    into or depend on the installed application bundle.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    peopleos_home = os.getenv('PEOPLEOS_HOME')
    if peopleos_home:
        root = os.path.abspath(os.path.expanduser(peopleos_home))
        persistence = config.setdefault('persistence', {})
        persistence['database_path'] = os.path.join(root, 'data', 'peopleos.db')

    return config


def check_config_version(loaded_config: dict, expected_version: str) -> bool:
    """
    Returns True if versions match, False if migration needed.
    
    Args:
        loaded_config: The loaded configuration dictionary.
        expected_version: The expected version string.
        
    Returns:
        True if versions match, False otherwise.
    """
    loaded_version = loaded_config.get('version', '0.0.0')
    return loaded_version == expected_version


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if denominator is zero.
    
    Args:
        numerator: The numerator.
        denominator: The denominator.
        default: Value to return if division by zero.
        
    Returns:
        Result of division or default value.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between a minimum and maximum.
    
    Args:
        value: The value to clamp.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        
    Returns:
        Clamped value.
    """
    return max(min_val, min(max_val, value))


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value (e.g., 0.25 for 25%).
        decimals: Number of decimal places.
    """
    return f"{value * 100:.{decimals}f}%"
