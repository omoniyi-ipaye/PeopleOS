"""
Vector Engine module for PeopleOS.

Provides local semantic search using FAISS and sentence-transformers.
"""

from typing import Any, Optional

import faiss
from sentence_transformers import SentenceTransformer

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('vector_engine')


class VectorEngine:
    """
    Engine for semantic search and discovery.
    
    Uses sentence-transformers for embeddings and FAISS for efficient similarity search.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the Vector Engine with a transformer model."""
        self.config = load_config()
        self.vector_config = self.config.get('vector_db', {})
        
        try:
            # Note: In a local-first production environment, we'd cache these models
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
            
        self.index: Optional[faiss.Index] = None
        self.metadata: list[dict[str, Any]] = []
        self.dimension: int = 384  # Default for all-MiniLM-L6-v2
    
    def build_index(self, texts: list[str], metadata: list[dict[str, Any]]) -> None:
        """
        Build a FAISS index from a list of texts.
        
        Args:
            texts: List of review/performance texts.
            metadata: List of corresponding metadata (e.g., EmployeeID, Dept).
        """
        if not texts:
            logger.warning("Empty text list provided for indexing")
            return
            
        try:
            logger.info(f"Generating embeddings for {len(texts)} records...")
            embeddings = self.model.encode(texts, show_progress_bar=False)
            self.dimension = embeddings.shape[1]
            
            # FAISS IndexFlatL2 for simple Euclidean similarity
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings.astype('float32'))
            self.metadata = metadata
            
            logger.info(f"Successfully built FAISS index with dimension {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to build vector index: {str(e)}")
            self.index = None
    
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Search for records semantically similar to the query.
        
        Args:
            query: The search query string.
            top_k: Number of results to return.
            
        Returns:
            List of matching records with their metadata and scores.
        """
        if self.index is None or not self.metadata:
            logger.warning("Search called but index is not built")
            return []
            
        try:
            # Embed the query
            query_vector = self.model.encode([query])
            
            # Search the index
            distances, indices = self.index.search(query_vector.astype('float32'), top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.metadata):
                    # Lower distance = higher similarity for L2
                    result = self.metadata[idx].copy()
                    result['similarity_score'] = float(1 / (1 + distances[0][i])) # Convert to similarity 0-1
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []

    def is_initialized(self) -> bool:
        """Check if the index is ready for searching."""
        return self.index is not None
