"""
Utility functions for PeopleOS.

Contains shared helper functions used across multiple modules.
"""

import os
from typing import Any

import yaml


def load_config() -> dict:
    """
    Load configuration from config.yaml.
    
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config.yaml is not found.
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
        
    Returns:
        Formatted percentage string (e.g., "25.0%").
    """
    return f"{value * 100:.{decimals}f}%"


def is_valid_file_extension(file_path: str, allowed_extensions: list[str]) -> bool:
    """
    Check if a file has an allowed extension.
    
    Args:
        file_path: Path to the file.
        allowed_extensions: List of allowed extensions (without dots).
        
    Returns:
        True if extension is allowed, False otherwise.
    """
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    return ext in allowed_extensions


def get_file_extension(file_path: str) -> str:
    """
    Get the file extension without the dot.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        File extension in lowercase.
    """
    return os.path.splitext(file_path)[1].lower().lstrip('.')


# Error message templates - written in friendly, non-technical language for HR users
ERROR_MESSAGES = {
    "file_load_failed": "We couldn't read your file. Please make sure it's a CSV, Excel-exported CSV, JSON, or SQLite file and try again.",
    "insufficient_data": "Your file has {count} employees, but we need at least 50 to spot meaningful patterns. Please upload a larger dataset.",
    "missing_columns": "Your file is missing some important information: {columns}. You can rename your columns to match, or check the Data Format Guide below.",
    "model_training_failed": "We couldn't analyze risk patterns right now, but you can still view workforce metrics in the Overview tab.",
    "llm_unavailable": "The AI assistant isn't available right now. All analytics and predictions are still working.",
    "duplicate_ids": "Found {count} employees with the same ID. Each employee needs a unique identifier - please check your data.",
    "negative_values": "Found {count} rows with impossible values (like negative salary or age). We've automatically removed these.",
    "invalid_data_types": "Some data was in an unexpected format. We've automatically fixed what we could - everything should work fine."
}


def get_error_message(error_key: str, **kwargs: Any) -> str:
    """
    Get a user-friendly error message.
    
    Args:
        error_key: Key for the error message template.
        **kwargs: Values to format into the message.
        
    Returns:
        Formatted error message.
    """
    template = ERROR_MESSAGES.get(error_key, "An unexpected error occurred. Please try again.")
    try:
        return template.format(**kwargs)
    except KeyError:
        return template
