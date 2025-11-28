from pathlib import Path
from typing import List

from config import config
from logger import get_logger
from utils.json_db import load_database, valid_db_record

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


def filter_existing_images(
        image_paths: List[str],
        existing_db: List[dict]
) -> List[str]:
    """
    Filters out images that are already present in the existing database.
    And check if processed and saved in db correctly.
    """
    db_paths = [record["path"] for record in existing_db]
    existing_paths = []
    for record in  list(set(db_paths) & set(image_paths)):
        if valid_db_record(record):
            existing_paths.append(record["path"])
        else:
            logger.debug(f"Invalid DB record, skipping path check: {record}")


    filtered = [p for p in image_paths if p not in existing_paths]

    logger.info(f"Filtered images: {len(image_paths)} -> {len(filtered)} (existing: {len(existing_paths)})")
    return filtered

def fetch_processed_images_paths(
        existing_db: List[dict]
) -> List[str]:
    """
    Fetches the list of image paths that have already been processed
    and are present in the existing database.
    """
    processed_paths = [record["path"] for record in existing_db]

    logger.info(f"Fetched {len(processed_paths)} processed image paths from database.")
    return processed_paths