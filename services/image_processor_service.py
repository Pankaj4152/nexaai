from typing import Optional, Dict, List
from pathlib import Path
import time

from tqdm import tqdm

from services.vlm_service import VLMService
from services.embedder_service import EmbedderService

from config import config
from logger import get_logger


logger = get_logger(__name__)


class ImageProcessorService:
    def __init__(self, vlm: VLMService, embedder: EmbedderService):
        logger.debug("Initializing ImageProcessorService...")
        self.vlm = vlm
        self.embedder = embedder
        logger.debug("ImageProcessorService initialized.")



    @staticmethod
    def _validate_image(image_path: str) -> bool:
        path = Path(image_path)
        logger.debug(f"Validating image: {image_path}")
        if not path.exists():
            logger.warning(f"Image missing: {image_path}")
            return False
        if path.suffix.lower() not in config.allowed_extensions:
            logger.warning(f"Unsupported file format: {image_path}")
            return False
        return True


    def process_image(self, image_path: str) -> Optional[Dict]:
        logger.info(f"Starting processing: {image_path}")
        if not self._validate_image(image_path):
            logger.warning(f"Image validation failed: {image_path}")
            return None

        image_path = str(image_path)
        image_name = Path(image_path).name

        logger.info(f"Processing image: {image_name}")

        start_time = time.time()
        logger.debug(f"Processing started at {start_time}")

        # Step 1: Generate description
        try:
            logger.debug(f"Generating description for {image_name}")
            description = self.vlm.generate_description(image_path)
        except Exception as e:
            logger.exception(f"Exception during description generation for {image_name}: {e}")
            return None
        
        if not description:
            logger.error(f"VLM returned empty description for {image_name}")
            return None 
        logger.debug(f"Description generated for {image_name}: {description}")

        # Step 2: Generate embedding
        try:
            logger.debug(f"Encoding embedding for {image_name}")
            embedding = self.embedder.encode(description)
        except Exception as e:
            logger.exception(f"Exception during embedding generation for {image_name}: {e}")
            return None
        
        if embedding is None:
            logger.error(f"Embedding generation failed (None returned) for {image_name}")
            return None

        elapsed = time.time() - start_time
        logger.info(f"Completed: {image_name} in {elapsed:.2f}s")
        logger.debug(f"Embedding shape for {image_name}: {getattr(embedding, 'shape', 'unknown')}")


        return {
            "path": image_path,
            "filename": image_name,
            "description": description,
            "embedding": embedding.tolist()  # serialized for JSON writing
        }


    def process_images(self, image_paths: List[str]) -> List[Dict]:
        logger.info(f"Batch processing started. Total images: {len(image_paths)}")
        results = []

        with tqdm(total=len(image_paths), desc="Processing images", unit="img") as pbar:
            for idx, path in enumerate(image_paths, start=1):
                pbar.set_description(f"Processing {Path(path).name}")
                logger.debug(f"Batch step {idx}/{len(image_paths)} -> {path}")

                result = self.process_image(path)
                if result:
                    results.append(result)
                    logger.debug(f"Successfully processed: {path}")
                else:
                    logger.warning(f"Failed to process: {path}")
                pbar.update(1)

        logger.info(f"Batch processing completed -> {len(results)}/{len(image_paths)} successful")
        return results
