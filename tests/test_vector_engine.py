"""
Tests for the VectorEngine module.
"""

import pandas as pd
import pytest
import numpy as np

from src.vector_engine import VectorEngine


class _FakeIndex:
    def __init__(self, dimension):
        self.vectors = np.empty((0, dimension), dtype='float32')

    def add(self, vectors):
        self.vectors = vectors.copy()

    def search(self, query, top_k):
        distances = ((self.vectors - query[0]) ** 2).sum(axis=1)
        order = np.argsort(distances)[:top_k]
        return distances[order][None, :], order[None, :]


class _FakeFaiss:
    IndexFlatL2 = _FakeIndex


class _FakeModel:
    def encode(self, texts, **_kwargs):
        return np.asarray([
            [float('python' in text.lower()), float('sales' in text.lower()), float('team' in text.lower())]
            for text in texts
        ], dtype='float32')


@pytest.fixture
def vector_engine():
    """Exercise index/search behavior without downloading an external model."""
    return VectorEngine(model=_FakeModel(), faiss_backend=_FakeFaiss())


class TestVectorEngine:
    """Test cases for VectorEngine class."""

    def test_is_initialized_initially_false(self, vector_engine):
        """Test that engine is not initialized before building index."""
        engine = vector_engine
        assert engine.is_initialized() is False

    def test_build_index_success(self, vector_engine):
        """Test that FAISS index builds correctly."""
        texts = [
            "Exceptional leader with Python skills.",
            "Underperforming in sales targets.",
            "Great teamwork and communication."
        ]
        metadata = [
            {'EmployeeID': 'E001', 'Dept': 'Engineering'},
            {'EmployeeID': 'E002', 'Dept': 'Sales'},
            {'EmployeeID': 'E003', 'Dept': 'HR'}
        ]
        
        engine = vector_engine
        engine.build_index(texts, metadata)
        
        assert engine.is_initialized() is True

    def test_search_returns_results(self, vector_engine):
        """Test that semantic search returns relevant records."""
        texts = [
            "Exceptional leader with Python skills.",
            "Underperforming in sales targets.",
            "Great teamwork and communication."
        ]
        metadata = [
            {'EmployeeID': 'E001', 'Dept': 'Engineering', 'PerformanceText': texts[0]},
            {'EmployeeID': 'E002', 'Dept': 'Sales', 'PerformanceText': texts[1]},
            {'EmployeeID': 'E003', 'Dept': 'HR', 'PerformanceText': texts[2]}
        ]
        
        engine = vector_engine
        engine.build_index(texts, metadata)
        
        results = engine.search("Python programming leader", top_k=2)
        
        assert len(results) > 0
        assert 'similarity_score' in results[0]
        # Best match should be E001 (Python leader)
        assert results[0]['EmployeeID'] == 'E001'

    def test_search_on_uninitialized_engine_returns_empty(self, vector_engine):
        """Test that searching on an uninitialized engine returns empty list."""
        engine = vector_engine
        results = engine.search("Some query", top_k=5)
        
        assert results == []
