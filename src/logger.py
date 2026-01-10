"""
Centralized logging module for PeopleOS.

Provides a configured logger with RotatingFileHandler and StreamHandler.
Includes PII sanitization to prevent logging of sensitive employee data.
"""

import logging
import hashlib
import os
from logging.handlers import RotatingFileHandler
from typing import Any

import yaml


# Load configuration
def _load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Return defaults if config not found
        return {
            'app': {'log_level': 'INFO'},
            'logging': {
                'file_path': 'logs/peopleos.log',
                'max_file_size_mb': 10,
                'backup_count': 5,
                'format': '%(asctime)s | %(levelname)s | %(module)s | %(message)s'
            }
        }


_config = _load_config()

# PII-sensitive field patterns to redact
PII_FIELDS = frozenset({
    'employeeid', 'employee_id', 'name', 'first_name', 'last_name',
    'email', 'phone', 'address', 'ssn', 'social_security',
    'date_of_birth', 'dob', 'national_id'
})


def _hash_value(value: Any) -> str:
    """Hash a value for safe logging."""
    return hashlib.sha256(str(value).encode()).hexdigest()[:12]


def sanitize_for_logging(data: dict) -> dict:
    """
    Removes PII fields before logging.
    
    Args:
        data: Dictionary potentially containing PII fields.
        
    Returns:
        Sanitized dictionary with PII fields hashed or redacted.
    """
    if not isinstance(data, dict):
        return data
    
    sanitized = {}
    for key, value in data.items():
        key_lower = key.lower().replace(' ', '_')
        
        if key_lower in PII_FIELDS:
            if key_lower == 'employeeid' or key_lower == 'employee_id':
                # Hash employee IDs instead of fully redacting
                sanitized[key] = f"[HASHED:{_hash_value(value)}]"
            else:
                # Fully redact other PII
                sanitized[key] = "[REDACTED]"
        elif isinstance(value, dict):
            # Recursively sanitize nested dictionaries
            sanitized[key] = sanitize_for_logging(value)
        elif isinstance(value, list):
            # Sanitize lists of dictionaries
            sanitized[key] = [
                sanitize_for_logging(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized


def get_logger(module_name: str) -> logging.Logger:
    """
    Returns a configured logger instance for the given module.
    
    Args:
        module_name: Name of the module requesting the logger.
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(f"peopleos.{module_name}")
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Get config values
    log_config = _config.get('logging', {})
    app_config = _config.get('app', {})
    
    log_level = getattr(logging, app_config.get('log_level', 'INFO').upper(), logging.INFO)
    log_format = log_config.get('format', '%(asctime)s | %(levelname)s | %(module)s | %(message)s')
    log_file = log_config.get('file_path', 'logs/peopleos.log')
    max_bytes = log_config.get('max_file_size_mb', 10) * 1024 * 1024
    backup_count = log_config.get('backup_count', 5)
    
    logger.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    try:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        # Log to console if file handler fails
        logger.warning(f"Could not create file handler: {e}. Logging to console only.")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger
