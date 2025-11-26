import json
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer


class SearchEngine:
    """
    Semantic search engine for finding images by text queries
    Uses cosine similarity between query and image embeddings.
    """

    def __init__(self, database_file="image_database.json"):
        
        # Load image database
        try:
            with open(database_file, 'r', encoding='utf-8') as f:
                self.image_database = json.load(f)
        except FileNotFoundError:
            print(f"Database not found: {database_file}")
            raise
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            raise


        # convert embeddings to numpy arrays
        self.has_embeddings = False
        for item in self.image_database:
            if 'embedding' in item:
                item['embedding'] = np.array(item['embedding'])
                self.has_embeddings = True
            else:
                print("Warning: Some images lack embeddings.")
                self.has_embeddings = False
                break

        if not self.has_embeddings:
            print("No Embeddings found in database.")
        else:
            print("Embeddings loaded successfully.")

        
        # Load text embedding model
        if self.has_embeddings:
            try:
                embedder_model = "all-MiniLM-L6-v2"
                self.embedder = SentenceTransformer(embedder_model)
                print(f"Loaded text embedding model: {embedder_model}")
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
                self.has_embeddings = False
                self.embedder = None
                raise
        else:
            self.embedder = None

    def consine_similarity(self, ve1, vec2):
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(ve1, vec2)
        norm1 = np.linalg.norm(ve1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.5) -> List[Tuple[float, Dict]]:
        """
        Search for images similar to the query.

        Args:
            query (str): Text query.
            top_k (int): Number of top results to return.
            min_similarity (float): Minimum similarity threshold for results.
        Returns:
            List of tuples (similarity, image_data) for top_k results.
        """
        if not self.has_embeddings or self.embedder is None:
            print("Search not available: No embeddings in database.")
            return []
        

        # Embed the query
        try:
            query_embedding = self.embedder.encode(
                query,
                convert_to_numpy=True,
                show_progress_bar=False
            )   
        except Exception as e:
            print(f"Failed to embed query: {e}")
            return []
        
        # Compute similarities
        similarities = []
        for item in self.image_database:
            # print(item["path"])
            if item['embedding'] is None:
                continue

            similarity = self.consine_similarity(query_embedding, item['embedding'])
            # print(f"Similarity with {item['filename']}: {similarity:.4f}")
            if similarity >= min_similarity:
                similarities.append((similarity, item))


        # print(similarities)
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        # print(f"Found {len(similarities)} results with similarity >= {min_similarity}")
        # Return top K results
        results = similarities[:top_k]

        return results
    

if __name__ == "__main__":
    engine = SearchEngine(database_file="image_database.json")
    # query = "images of aeroplane within them flying in the sky"
    while True:
        query = input("Enter search query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        results = engine.search(query, top_k=3, min_similarity=0.5)
        if( len(results) == 0 ):
            print("No results found.")
            continue

        for sim, item in results:
            print(f"{item['path']}, {sim:.4f}")
            # print(f"Similarity: {sim:.4f}, Image: {item['filename']}, Description: {item['description']}")
            print("=" * 100)
            print("=" * 100)