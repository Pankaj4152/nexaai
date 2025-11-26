from typing import List, Optional, Union
import numpy as np

from sentence_transformers import SentenceTransformer

from config import config
from logger import get_logger


logger = get_logger(__name__)



class EmbedderService:
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading embedder model: {config.embedder_model_path}")

            self.model = SentenceTransformer(
                config.embedder_model_path,
                # device=config.embedder_device
            )

            logger.info("Embedder loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load embedder model: {e}")
            raise RuntimeError(f"Embedder initialization failed: {e}")
        

    def encode(self, text: str, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Encode a single text into a vector.
        Returns None on failure.
        """

        if not self.model:
            logger.error("Embedder model not initialized.")
            return None

        try:
            logger.debug(f"Encoding text -> {text[:60]}...")
            emb = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            ).astype(np.float32)

            if normalize:
                emb = self.normalize(emb)

            return emb

        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            return None
        
    def encode_batch(self, texts: List[str], normalize: bool = True) -> List[np.ndarray]:
        """
        Encode multiple texts at once (efficient).
        Returns list of numpy vectors.
        """

        if not self.model:
            logger.error("Embedder model not initialized.")
            return []

        try:
            logger.debug(f"Batch encoding {len(texts)} texts...")

            embeddings = self.model.encode(
                texts,
                batch_size=config.embedder_batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            ).astype(np.float32)

            if normalize:
                embeddings = np.vstack([self.normalize(v) for v in embeddings])

            return list(embeddings)

        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            return []

    @staticmethod
    def normalize(vec: np.ndarray) -> np.ndarray:
        """
        Normalizes vector to unit length.
        Helps cosine similarity behave correctly.
        """

        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec

        return vec / norm