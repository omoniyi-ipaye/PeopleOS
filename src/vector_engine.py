"""
Vector Engine module for PeopleOS.

Provides optional local semantic search using FAISS and sentence-transformers.
The heavy vector/transformer dependencies are imported only when this capability
is explicitly initialized so the core PeopleOS runtime can boot without them.
"""

from typing import Any, Optional

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('vector_engine')


class VectorEngine:
    """Optional semantic-search engine backed by local embeddings and FAISS."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.config = load_config()
        self.vector_config = self.config.get('vector_db', {})
        self.index: Optional[Any] = None
        self.metadata: list[dict[str, Any]] = []
        self.dimension: int = 384

        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Vector search dependencies are not installed. "
                "Install requirements-advanced.txt to enable semantic search."
            ) from exc

        self._faiss = faiss
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as exc:
            logger.error(f"Failed to load embedding model: {exc}")
            raise

    def build_index(self, texts: list[str], metadata: list[dict[str, Any]]) -> None:
        if not texts:
            logger.warning("Empty text list provided for indexing")
            return

        try:
            logger.info(f"Generating embeddings for {len(texts)} records...")
            embeddings = self.model.encode(texts, show_progress_bar=False)
            self.dimension = embeddings.shape[1]
            self.index = self._faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings.astype('float32'))
            self.metadata = metadata
            logger.info(f"Successfully built FAISS index with dimension {self.dimension}")
        except Exception as exc:
            logger.error(f"Failed to build vector index: {exc}")
            self.index = None

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.index is None or not self.metadata:
            logger.warning("Search called but index is not built")
            return []

        try:
            query_vector = self.model.encode([query])
            distances, indices = self.index.search(query_vector.astype('float32'), top_k)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['similarity_score'] = float(1 / (1 + distances[0][i]))
                    results.append(result)
            return results
        except Exception as exc:
            logger.error(f"Semantic search failed: {exc}")
            return []

    def is_initialized(self) -> bool:
        return self.index is not None
