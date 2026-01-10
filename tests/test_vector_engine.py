"""
Tests for the VectorEngine module.
"""

import pandas as pd
import pytest

from src.vector_engine import VectorEngine


class TestVectorEngine:
    """Test cases for VectorEngine class."""

    def test_is_initialized_initially_false(self):
        """Test that engine is not initialized before building index."""
        engine = VectorEngine()
        assert engine.is_initialized() is False

    def test_build_index_success(self):
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
        
        engine = VectorEngine()
        engine.build_index(texts, metadata)
        
        assert engine.is_initialized() is True

    def test_search_returns_results(self):
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
        
        engine = VectorEngine()
        engine.build_index(texts, metadata)
        
        results = engine.search("Python programming leader", top_k=2)
        
        assert len(results) > 0
        assert 'similarity_score' in results[0]
        # Best match should be E001 (Python leader)
        assert results[0]['EmployeeID'] == 'E001'

    def test_search_on_uninitialized_engine_returns_empty(self):
        """Test that searching on an uninitialized engine returns empty list."""
        engine = VectorEngine()
        results = engine.search("Some query", top_k=5)
        
        assert results == []
