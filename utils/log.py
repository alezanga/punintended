import logging
from pathlib import Path

from transformers import logging as hf_logging


def setup_logging(log_folder: Path, logger_name: str = "my_logger") -> logging.Logger:
    # Create a custom logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Set the logger to the lowest level

    # Create handlers
    info_handler = logging.FileHandler(log_folder / "info.log", mode="a")
    warn_handler = logging.FileHandler(log_folder / "warning.log", mode="a")
    error_handler = logging.FileHandler(log_folder / "error.log", mode="a")

    # Set levels for handlers
    info_handler.setLevel(logging.DEBUG)  # Capture DEBUG and INFO messages
    warn_handler.setLevel(logging.WARNING)  # Capture WARNING and above
    error_handler.setLevel(logging.ERROR)  # Capture ERROR and above

    # Create formatters and add them to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    info_handler.setFormatter(formatter)
    warn_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(info_handler)
    logger.addHandler(warn_handler)
    logger.addHandler(error_handler)

    # Configure Hugging Face logger
    hf_logger = hf_logging.get_logger()
    hf_logger.setLevel(logging.DEBUG)  # Set the level for Hugging Face logger

    # Add the same handlers to the Hugging Face logger
    hf_logger.addHandler(info_handler)
    hf_logger.addHandler(warn_handler)
    hf_logger.addHandler(error_handler)

    return logger
