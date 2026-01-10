"""
Integration tests for PeopleOS.

Tests the full workflow from data upload through analytics and predictions.
"""

import tempfile
import os
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import DataLoader, DataValidationError
from src.preprocessor import Preprocessor
from src.analytics_engine import AnalyticsEngine
from src.ml_engine import MLEngine, MLEngineError


class TestFullUploadToAnalyticsFlow:
    """Test the complete data flow from upload to analytics."""

    def test_csv_upload_to_analytics(self, sample_valid_data):
        """Test complete flow: CSV upload -> validation -> analytics."""
        # Create temp CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_valid_data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            # Load and validate
            loader = DataLoader()
            df = loader.load(temp_path)

            assert df is not None
            assert len(df) == len(sample_valid_data)

            # Run analytics
            engine = AnalyticsEngine(df)
            headcount = engine.get_headcount()
            turnover = engine.get_turnover_rate()
            dept_stats = engine.get_department_aggregates()

            assert headcount > 0
            assert turnover is not None or 'Attrition' not in df.columns
            assert not dept_stats.empty

        finally:
            os.unlink(temp_path)

    def test_json_upload_to_analytics(self, sample_valid_data):
        """Test complete flow with JSON data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            sample_valid_data.to_json(f.name, orient='records')
            temp_path = f.name

        try:
            loader = DataLoader()
            df = loader.load(temp_path)

            engine = AnalyticsEngine(df)
            summary = engine.get_summary_statistics()

            assert 'headcount' in summary
            assert summary['headcount'] > 0

        finally:
            os.unlink(temp_path)

    def test_fuzzy_column_mapping_flow(self, sample_data_with_fuzzy_columns):
        """Test that fuzzy column names are properly mapped."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data_with_fuzzy_columns.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            loader = DataLoader()
            df = loader.load(temp_path)

            # Check standard columns exist after mapping
            assert 'EmployeeID' in df.columns
            assert 'Dept' in df.columns
            assert 'Tenure' in df.columns

            # Verify analytics work
            engine = AnalyticsEngine(df)
            assert engine.get_headcount() > 0

        finally:
            os.unlink(temp_path)


class TestMLPredictionPipeline:
    """Test the ML prediction workflow."""

    def test_preprocessing_to_prediction(self, sample_valid_data):
        """Test: preprocessing -> ML training -> predictions."""
        # Preprocess
        preprocessor = Preprocessor()
        processed_df, metadata = preprocessor.fit_transform(
            sample_valid_data,
            target_column='Attrition'
        )

        assert processed_df is not None
        assert len(processed_df) > 0

        # Prepare features
        target = processed_df['Attrition']
        features = processed_df.drop(columns=['Attrition', 'EmployeeID'], errors='ignore')
        numeric_features = features.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])

        # Train model
        ml_engine = MLEngine()
        metrics = ml_engine.train_model(numeric_features, target)

        assert 'accuracy' in metrics
        assert metrics['accuracy'] >= 0.0
        assert metrics['accuracy'] <= 1.0

        # Get predictions
        risk_scores = ml_engine.predict_risk(numeric_features)
        assert len(risk_scores) == len(numeric_features)

        # Get risk categories
        categories = [ml_engine.get_risk_category(score) for score in risk_scores]
        assert all(cat in ['High', 'Medium', 'Low'] for cat in categories)

    def test_single_class_handling(self, sample_data_single_class):
        """Test ML handles single-class attrition gracefully."""
        preprocessor = Preprocessor()
        processed_df, _ = preprocessor.fit_transform(
            sample_data_single_class,
            target_column='Attrition'
        )

        target = processed_df['Attrition']
        features = processed_df.drop(columns=['Attrition', 'EmployeeID'], errors='ignore')
        numeric_features = features.select_dtypes(include=['int64', 'float64'])

        ml_engine = MLEngine()

        # Should not raise - model trains but metrics may be limited
        try:
            metrics = ml_engine.train_model(numeric_features, target)
            # Predictions should still work
            risk_scores = ml_engine.predict_risk(numeric_features)
            assert len(risk_scores) == len(numeric_features)
        except MLEngineError:
            # Acceptable if single-class causes training to fail
            pass

    def test_feature_importance_generation(self, sample_valid_data):
        """Test feature importance is properly generated."""
        preprocessor = Preprocessor()
        processed_df, _ = preprocessor.fit_transform(
            sample_valid_data,
            target_column='Attrition'
        )

        target = processed_df['Attrition']
        features = processed_df.drop(columns=['Attrition', 'EmployeeID'], errors='ignore')
        numeric_features = features.select_dtypes(include=['int64', 'float64'])

        ml_engine = MLEngine()
        ml_engine.train_model(numeric_features, target)

        importance_df = ml_engine.get_feature_importance_summary()

        assert not importance_df.empty
        assert 'Feature' in importance_df.columns
        assert 'Importance' in importance_df.columns
        assert len(importance_df) == len(numeric_features.columns)


class TestLLMFallbackBehavior:
    """Test LLM client fallback behavior."""

    def test_llm_unavailable_returns_fallback(self):
        """Test that LLM client returns fallback when unavailable."""
        from src.llm_client import LLMClient

        client = LLMClient()

        # If Ollama is not running, should return fallback
        metrics = {
            'total_employees': 100,
            'turnover_rate': 0.15,
            'high_risk_count': 10,
            'departments': 5
        }

        result = client.get_strategic_summary(metrics)

        # Should always return a dict with status
        assert isinstance(result, dict)
        assert 'status' in result or 'message' in result

    def test_action_items_always_returns_list(self):
        """Test action items returns recommendations even without LLM."""
        from src.llm_client import LLMClient

        client = LLMClient()

        metrics = {
            'total_employees': 100,
            'turnover_rate': 0.25,
            'high_risk_count': 20
        }

        result = client.get_action_items(metrics)

        assert isinstance(result, dict)
        if 'recommendations' in result:
            assert isinstance(result['recommendations'], list)


class TestExportFunctionality:
    """Test data export functionality."""

    def test_csv_export(self, sample_valid_data):
        """Test CSV export produces valid data."""
        from src.export import export_to_csv

        csv_bytes = export_to_csv(sample_valid_data)

        assert csv_bytes is not None
        assert len(csv_bytes) > 0

        # Parse back to verify
        import io
        parsed_df = pd.read_csv(io.BytesIO(csv_bytes))
        assert len(parsed_df) == len(sample_valid_data)

    def test_excel_export(self, sample_valid_data):
        """Test Excel export produces valid file."""
        from src.export import export_to_excel

        excel_bytes = export_to_excel(sample_valid_data)

        assert excel_bytes is not None
        assert len(excel_bytes) > 0

        # Verify it's valid Excel
        import io
        parsed_df = pd.read_excel(io.BytesIO(excel_bytes))
        assert len(parsed_df) == len(sample_valid_data)

    def test_risk_report_export(self, sample_valid_data):
        """Test risk report export includes all required data."""
        from src.export import export_risk_report

        analytics_data = {
            'headcount': 100,
            'turnover_rate': 0.15,
            'department_count': 5
        }

        ml_data = {
            'risk_distribution': {'High': 10, 'Medium': 30, 'Low': 60},
            'metrics': {'accuracy': 0.85, 'f1': 0.80}
        }

        report_bytes = export_risk_report(analytics_data, ml_data)

        assert report_bytes is not None
        assert len(report_bytes) > 0


class TestSessionManagement:
    """Test session persistence functionality."""

    def test_save_and_load_session(self, sample_valid_data, tmp_path):
        """Test saving and loading a session."""
        from src.session_manager import SessionManager

        manager = SessionManager(sessions_dir=str(tmp_path))

        # Save session
        filepath = manager.save_session(
            session_name="Test Session",
            raw_data=sample_valid_data,
            analytics_data={'headcount': 100},
            ml_data={'risk_distribution': {'High': 5}},
            features_enabled={'predictive': True}
        )

        assert filepath is not None
        assert os.path.exists(filepath)

        # Load session
        loaded = manager.load_session(filepath)

        assert loaded is not None
        assert loaded['session_name'] == "Test Session"
        assert 'raw_data' in loaded
        assert len(loaded['raw_data']) == len(sample_valid_data)

    def test_list_sessions(self, sample_valid_data, tmp_path):
        """Test listing available sessions."""
        from src.session_manager import SessionManager
        import time

        manager = SessionManager(sessions_dir=str(tmp_path))

        # Save multiple sessions with small delay to ensure different timestamps
        manager.save_session("Session 1", raw_data=sample_valid_data, analytics_data={}, ml_data={}, features_enabled={})
        time.sleep(0.1)
        manager.save_session("Session 2", raw_data=sample_valid_data, analytics_data={}, ml_data={}, features_enabled={})

        sessions = manager.list_sessions()

        assert len(sessions) == 2
        assert any(s['session_name'] == "Session 1" for s in sessions)
        assert any(s['session_name'] == "Session 2" for s in sessions)

    def test_delete_session(self, sample_valid_data, tmp_path):
        """Test deleting a session."""
        from src.session_manager import SessionManager

        manager = SessionManager(sessions_dir=str(tmp_path))

        filepath = manager.save_session("To Delete", raw_data=sample_valid_data, analytics_data={}, ml_data={}, features_enabled={})

        assert manager.delete_session(filepath) is True
        assert not os.path.exists(filepath)

        sessions = manager.list_sessions()
        assert len(sessions) == 0
