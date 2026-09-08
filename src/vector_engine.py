"""
Vector Engine module for PeopleOS.

Provides optional local semantic search using FAISS and sentence-transformers.
The heavy vector/transformer dependencies are imported only when this capability
is explicitly initialized so the core PeopleOS runtime can boot without them.
"""

from typing import Any, Optional
from copy import deepcopy
import numpy as np

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('vector_engine')

DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
DEFAULT_EMBEDDING_REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'


class VectorEngine:
    """Optional semantic-search engine backed by local embeddings and FAISS."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, *, model=None,
                 faiss_backend=None, model_revision: Optional[str] = None):
        self.config = load_config()
        self.vector_config = self.config.get('vector_db', {})
        self.index: Optional[Any] = None
        self.metadata: list[dict[str, Any]] = []
        self.dimension: int = 384
        self.model_name = model_name
        # Pin the evaluated default; custom models require their own validation.
        self.model_revision = model_revision or (
            DEFAULT_EMBEDDING_REVISION if model_name == DEFAULT_EMBEDDING_MODEL else None
        )

        if model is not None and faiss_backend is not None:
            self.model = model
            self._faiss = faiss_backend
            return

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
            self.model = SentenceTransformer(model_name, revision=self.model_revision)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as exc:
            logger.error(f"Failed to load embedding model: {exc}")
            raise

    def build_index(self, texts: list[str], metadata: list[dict[str, Any]]) -> None:
        # A rebuild replaces the previous dataset even if the new one is empty
        # or invalid. Never leave old employee results queryable after a refresh.
        self.index, self.metadata = None, []
        if len(texts) != len(metadata):
            raise ValueError('Texts and metadata must align one-to-one')
        if not texts:
            return
        if any(not isinstance(t, str) or not t.strip() for t in texts) or any(not isinstance(m, dict) for m in metadata):
            raise ValueError('Index records require nonempty text and dictionary metadata')
        embeddings = np.asarray(self.model.encode(texts, show_progress_bar=False), dtype='float32')
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts) or embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
            raise ValueError('Embedding rows must align with records and contain finite vectors')
        index = self._faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        self.dimension = embeddings.shape[1]
        self.metadata = deepcopy(metadata)
        self.index = index

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        if not isinstance(query, str) or not query.strip():
            return []
        if self.index is None or not self.metadata:
            logger.warning("Search called but index is not built")
            return []

        try:
            query_vector = np.asarray(self.model.encode([query]), dtype='float32')
            if query_vector.shape != (1, self.dimension) or not np.isfinite(query_vector).all():
                raise ValueError('Query embedding does not match the finite index dimension')
            distances, indices = self.index.search(query_vector, min(top_k, len(self.metadata)))

            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.metadata) and np.isfinite(distances[0][i]) and distances[0][i] >= 0:
                    result = deepcopy(self.metadata[idx])
                    result['squared_l2_distance'] = float(distances[0][i])
                    result['similarity_score'] = float(1 / (1 + distances[0][i]))
                    result['score_semantics'] = 'inverse_squared_l2_distance_not_probability_or_validated_relevance'
                    results.append(result)
            return results
        except Exception as exc:
            logger.error(f"Semantic search failed: {exc}")
            raise RuntimeError("Semantic search failed; results are unavailable") from exc

    def is_initialized(self) -> bool:
        return self.index is not None
