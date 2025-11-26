from pathlib import Path
from typing import List

from config import config
from logger import get_logger

logger = get_logger(__name__)


def is_valid_image(path: Path) -> bool:
    """
    Validate that path exists and is an allowed image type.
    """
    if not path.exists():
        logger.warning(f"File does not exist: {path}")
        return False

    if path.suffix.lower() not in config.allowed_extensions:
        logger.warning(f"Unsupported file format: {path}")
        return False

    return True


def scan_image_folder(folder_path: str) -> List[str]:
    """
    Recursively scans a folder for valid images.
    Returns a list of full file paths.
    """
    folder = Path(folder_path)

    if not folder.exists():
        logger.error(f"Folder not found: {folder}")
        return []

    if not folder.is_dir():
        logger.error(f"Expected directory, got file: {folder}")
        return []

    images = []

    for file in folder.rglob("*"):  # recursive
        if file.suffix.lower() in config.allowed_extensions:
            images.append(str(file))

    logger.info(f"Found {len(images)} images in {folder_path}")
    return images


def ensure_directory(path: str):
    """
    Ensures a directory exists, creates it if needed.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory: {p}")
