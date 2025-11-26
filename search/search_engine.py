import numpy as np
from typing import List, Dict, Tuple, Any

from utils.json_db import load_database
from search.indexer import SimpleIndexer
from services.embedder_service import EmbedderService

from config import config
from logger import get_logger

logger = get_logger(__name__)


class SearchEngine:
    def __init__(self, embedder: EmbedderService):
        self.embedder = embedder
        self.db = load_database()

        if not self.db:
            logger.warning("Empty or missing database. Search will return no results.")

        self._build_embedding_matrix()


    def _build_embedding_matrix(self):
        vectors = []

        for entry in self.db:
            emb = entry.get("embedding")
            if emb:
                vectors.append(np.array(emb, dtype=np.float32))
            else:
                vectors.append(None)

        # Filter out invalid entries
        self.valid_entries = [
            (i, vec)
            for i, vec in enumerate(vectors)
            if isinstance(vec, np.ndarray)
        ]

        if not self.valid_entries:
            logger.error("No valid embeddings found in DB.")
            self.indexer = None
            return

        # Stack embeddings
        matrix = np.vstack([vec for _, vec in self.valid_entries])
        self.indexer = SimpleIndexer(matrix)

        logger.info(f"Search engine ready with {len(self.valid_entries)} vectors.")


    def search(self, query: str, top_k: int = None, min_similarity: float = None):
        if self.indexer is None:
            logger.error("Search index not available.")
            return []

        top_k = top_k or config.top_k
        min_similarity = min_similarity or config.min_similarity

        # Encode query
        q_emb = self.embedder.encode(query)
        if q_emb is None:
            logger.error("Query embedding failed.")
            return []

        # Query the index
        raw_results = self.indexer.query(q_emb, top_k=top_k)

        # Filter and map results
        results = []
        for rank, (idx, score) in enumerate(raw_results):
            if score < min_similarity:
                continue

            db_index = self.valid_entries[idx][0]
            record = self.db[db_index]

            results.append({
                "score": float(score),
                "path": record["path"],
                "filename": record["filename"],
                "description": record["description"]
            })

        return results
