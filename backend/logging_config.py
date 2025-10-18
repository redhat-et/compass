"""Logging configuration for Compass backend."""

import logging
import sys
from pathlib import Path

# Log levels
LOG_LEVEL = logging.INFO
DEBUG_MODE = False  # Set to True to enable DEBUG level logs with full prompts/responses

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_file: str = None, debug: bool = False):
    """
    Configure logging for the application.

    Args:
        log_file: Optional path to log file. If None, logs to stdout only.
        debug: If True, enable DEBUG level logging (includes full prompts/responses)
    """
    global DEBUG_MODE
    DEBUG_MODE = debug

    # Set root logger level
    level = logging.DEBUG if debug else LOG_LEVEL
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[]  # Clear default handlers
    )

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(console_formatter)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"Logging to file: {log_file}")

    # Log startup
    logging.info("=" * 80)
    logging.info("Compass Backend Starting")
    logging.info(f"Log Level: {logging.getLevelName(level)}")
    logging.info(f"Debug Mode: {debug}")
    logging.info("=" * 80)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger
    """
    return logging.getLogger(name)
