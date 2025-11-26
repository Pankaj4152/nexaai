import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import LOGS_DIR, config


def get_logger(name: str):
    """
    Returns a configured logger.
    Ensures each logger is created only once.
    """
    logger = logging.getLogger(name)

    if logger.handlers:  # Already configured
        return logger

    logger.setLevel(config.log_level.upper())

    # --- Log Format ---
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- Console Handler ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File Handler ---
    log_file = Path(LOGS_DIR) / "app.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=1_000_000,  # 1 MB
        backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
