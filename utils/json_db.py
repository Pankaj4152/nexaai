import json
from pathlib import Path
from typing import List, Dict, Any

from config import config
from logger import get_logger

logger = get_logger(__name__)


def load_database(db_path: str = None) -> List[Dict[str, Any]]:
    """
    Loads image metadata database from JSON.
    Returns empty list if file doesn't exist.
    """
    db_path = Path(db_path or config.db_path)

    if not db_path.exists():
        logger.warning(f"Database file not found: {db_path}")
        return []

    try:
        with db_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error(f"Invalid DB format: expected list, got {type(data)}")
            return []

        logger.info(f"Loaded database: {db_path} | {len(data)} records")
        return data

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON DB: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading DB: {e}")
        return []


def save_database(
    records: List[Dict[str, Any]], db_path: str = None
) -> bool:
    """
    Saves list of records to JSON database file.
    Returns True if success, False otherwise.
    """
    db_path = Path(db_path or config.db_path)

    try:
        with db_path.open("w", encoding="utf-8") as f:
            json.dump(
                records,
                f,
                indent=2,
                ensure_ascii=False
            )

        logger.info(f"Saved {len(records)} records to DB: {db_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save DB: {e}")
        return False
