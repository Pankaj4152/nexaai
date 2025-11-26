import numpy as np
from typing import List, Tuple
from logger import get_logger

logger = get_logger(__name__)


class SimpleIndexer:
    """
    Minimal, clean indexer using cosine similarity.
    Works well for small to medium datasets (< 10k).
    """

    def __init__(self, embeddings: np.ndarray):
        """
        embeddings: 2D numpy array (N, D)
        """
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        self.embeddings = embeddings.astype(np.float32)
        logger.info(f"Indexer initialized with {len(self.embeddings)} vectors.")


    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)


    def query(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """
        Returns:
            List of (index, similarity)
        """
        sims = []

        for idx, emb in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_vec, emb)
            sims.append((idx, sim))


        sims.sort(key=lambda x: x[1], reverse=True)

        return sims[:top_k]
