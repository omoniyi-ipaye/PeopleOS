"""
Vector Engine module for PeopleOS.

Provides optional local semantic search using FAISS and sentence-transformers.
The heavy vector/transformer dependencies are imported only when this capability
is explicitly initialized so the core PeopleOS runtime can boot without them.
"""

from typing import Any, Optional
from copy import deepcopy
from threading import RLock
import numpy as np

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('vector_engine')

DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
DEFAULT_EMBEDDING_REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
PROVENANCE_KEYS = ('workspace_id', 'dataset_id', 'generation', 'current_fingerprint')


class VectorEngine:
    """Optional semantic-search engine backed by local embeddings and FAISS."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, *, model=None,
                 faiss_backend=None, model_revision: Optional[str] = None):
        self.config = load_config()
        self.vector_config = self.config.get('vector_db', {})
        self._lock = RLock()
        self.index: Optional[Any] = None
        self.metadata: list[dict[str, Any]] = []
        self.dimension: int = 384
        self.index_provenance: Optional[dict[str, Any]] = None
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

    def _ensure_lock(self) -> RLock:
        # A few legacy tests and consumers construct the class with __new__.
        # Preserve that compatibility while giving normal instances a shared
        # lock across refresh and search.
        lock = getattr(self, '_lock', None)
        if lock is None:
            lock = RLock()
            self._lock = lock
        return lock

    def _clear_index_locked(self) -> None:
        self.index = None
        self.metadata = []
        self.index_provenance = None

    def clear_index(self) -> None:
        """Drop all indexed text and its provenance from process memory."""
        with self._ensure_lock():
            self._clear_index_locked()

    def build_index(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]],
        *,
        provenance: Optional[dict[str, Any]] = None,
    ) -> None:
        """Build a complete replacement index under one refresh/search lock.

        A refresh clears the live index before validation or embedding. If any
        step fails, the old dataset cannot remain queryable. The optional
        provenance is required by the API consumer, which rejects unbound
        indexes, while the legacy in-process UI may still use an unbound index.
        """
        with self._ensure_lock():
            self._clear_index_locked()
            if len(texts) != len(metadata):
                raise ValueError('Texts and metadata must align one-to-one')
            if not texts:
                return
            if (any(not isinstance(t, str) or not t.strip() for t in texts)
                    or any(not isinstance(m, dict) for m in metadata)):
                raise ValueError('Index records require nonempty text and dictionary metadata')
            if provenance is not None and not isinstance(provenance, dict):
                raise ValueError('Index provenance must be a dictionary when provided')
            try:
                embeddings = np.asarray(
                    self.model.encode(texts, show_progress_bar=False), dtype='float32'
                )
            except Exception as exc:
                raise RuntimeError('Embedding backend failed; index is unavailable') from exc
            if (embeddings.ndim != 2 or embeddings.shape[0] != len(texts)
                    or embeddings.shape[1] == 0 or not np.isfinite(embeddings).all()):
                raise ValueError('Embedding rows must align with records and contain finite vectors')
            try:
                index = self._faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
            except Exception as exc:
                raise RuntimeError('Vector index backend failed; index is unavailable') from exc
            indexed_count = getattr(index, 'ntotal', len(texts))
            if indexed_count != len(texts):
                raise RuntimeError('Vector index record count does not match metadata')
            self.dimension = embeddings.shape[1]
            self.metadata = deepcopy(metadata)
            self.index_provenance = deepcopy(provenance) if provenance is not None else None
            self.index = index

    def matches_provenance(self, provenance: Optional[dict[str, Any]]) -> bool:
        """Return whether the live index belongs to the exact active snapshot."""
        with self._ensure_lock():
            indexed = getattr(self, 'index_provenance', None)
            if not isinstance(indexed, dict) or not isinstance(provenance, dict):
                return False
            return all(
                key in indexed and key in provenance
                and indexed[key] is not None and provenance[key] is not None
                and indexed[key] == provenance[key]
                for key in PROVENANCE_KEYS
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        provenance: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        with self._ensure_lock():
            if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
                raise ValueError('top_k must be a positive integer')
            if not isinstance(query, str) or not query.strip():
                return []
            if self.index is None or not self.metadata:
                logger.warning("Search called but index is not built")
                return []
            if provenance is not None:
                indexed = getattr(self, 'index_provenance', None)
                if not isinstance(indexed, dict) or not all(
                    key in indexed and key in provenance
                    and indexed[key] is not None and provenance[key] is not None
                    and indexed[key] == provenance[key]
                    for key in PROVENANCE_KEYS
                ):
                    raise RuntimeError('Semantic search index does not match the active dataset snapshot')

            try:
                query_vector = np.asarray(self.model.encode([query]), dtype='float32')
                if query_vector.shape != (1, self.dimension) or not np.isfinite(query_vector).all():
                    raise ValueError('Query embedding does not match the finite index dimension')
                distances, indices = self.index.search(query_vector, min(top_k, len(self.metadata)))
                distances = np.asarray(distances)
                indices = np.asarray(indices)
                if (distances.ndim != 2 or indices.ndim != 2
                        or distances.shape != indices.shape or distances.shape[0] != 1):
                    raise ValueError('Vector backend returned malformed search results')

                results = []
                for i, idx in enumerate(indices[0]):
                    if (not np.issubdtype(indices.dtype, np.integer)
                            or not 0 <= idx < len(self.metadata)
                            or not np.isfinite(distances[0][i])
                            or distances[0][i] < 0):
                        continue
                    result = deepcopy(self.metadata[idx])
                    distance = float(distances[0][i])
                    result['squared_l2_distance'] = distance
                    result['similarity_score'] = float(1 / (1 + distance))
                    result['score_semantics'] = 'inverse_squared_l2_distance_not_probability_or_validated_relevance'
                    results.append(result)
                return results
            except Exception as exc:
                logger.error(f"Semantic search failed: {exc}")
                raise RuntimeError("Semantic search failed; results are unavailable") from exc

    def is_initialized(self) -> bool:
        with self._ensure_lock():
            return self.index is not None and bool(getattr(self, 'metadata', []))
