from typing import Optional, Dict, List
from pathlib import Path
import time

from services.vlm_service import VLMService
from services.embedder_service import EmbedderService

from config import config
from logger import get_logger


logger = get_logger(__name__)


class ImageProcessorService:
    def __init__(self, vlm: VLMService, embedder: EmbedderService):
        self.vlm = vlm
        self.embedder = embedder



    @staticmethod
    def _validate_image(image_path: str) -> bool:
        path = Path(image_path)
        if not path.exists():
            logger.error(f"Image not found: {image_path}")
            return False
        if path.suffix.lower() not in config.allowed_extensions:
            logger.error(f"Unsupported file format: {image_path}")
            return False
        return True


    def process_image(self, image_path: str) -> Optional[Dict]:
        if not self._validate_image(image_path):
            return None

        image_path = str(image_path)
        image_name = Path(image_path).name

        logger.info(f"Processing image: {image_name}")

        start_time = time.time()

        # Step 1: Generate description
        description = self.vlm.generate_description(image_path)
        if not description:
            logger.error(f"Failed to generate description for {image_name}")
            return None

        # Step 2: Generate embedding
        embedding = self.embedder.encode(description)
        if embedding is None:
            logger.error(f"Failed to generate embedding for {image_name}")
            return None

        elapsed = time.time() - start_time

        logger.info(f"Completed: {image_name} | {elapsed:.2f}s")

        return {
            "path": image_path,
            "filename": image_name,
            "description": description,
            "embedding": embedding.tolist()  # serialized for JSON writing
        }


    def process_images(self, image_paths: List[str]) -> List[Dict]:
        results = []

        for idx, path in enumerate(image_paths, start=1):
            logger.info(f"[{idx}/{len(image_paths)}] Starting: {path}")

            result = self.process_image(path)
            if result:
                results.append(result)

        logger.info(f"Batch processing completed: {len(results)}/{len(image_paths)} successful")

        return results
