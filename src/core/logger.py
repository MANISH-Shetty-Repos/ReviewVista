# src/core/logger.py
"""
Structured logging for the ReviewVista platform.
Replaces all print() debug statements with proper logging.
"""

import logging
import sys
from config.settings import BASE_DIR


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger instance."""
    logger = logging.getLogger(f"reviewvista.{name}")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger
