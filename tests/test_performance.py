"""
Performance benchmark tests for PeopleOS.

Validates that operations complete within acceptable time limits.
"""

import time
import tempfile
import os

import pandas as pd
import numpy as np
import pytest


class TestAppLoadPerformance:
    """Test application startup performance."""

    @pytest.mark.timeout(5)
    def test_config_load_under_1_second(self):
        """Config should load in under 1 second."""
        from src.utils import load_config

        start = time.time()
        config = load_config()
        elapsed = time.time() - start

        assert config is not None
        assert elapsed < 1.0, f"Config load took {elapsed:.2f}s, expected < 1s"

    @pytest.mark.timeout(5)
    def test_data_loader_init_fast(self):
        """DataLoader initialization should be fast."""
        from src.data_loader import DataLoader

        start = time.time()
        loader = DataLoader()
        elapsed = time.time() - start

        assert elapsed < 0.5, f"DataLoader init took {elapsed:.2f}s, expected < 0.5s"

    @pytest.mark.timeout(5)
    def test_ml_engine_init_fast(self):
        """MLEngine initialization should be fast."""
        from src.ml_engine import MLEngine

        start = time.time()
        engine = MLEngine()
        elapsed = time.time() - start

        assert elapsed < 1.0, f"MLEngine init took {elapsed:.2f}s, expected < 1s"


class TestModelTrainingPerformance:
    """Test model training performance."""

    @pytest.mark.timeout(30)
    def test_model_training_under_30_seconds(self, sample_data_large):
        """Model training on 1000 rows should complete in under 30 seconds."""
        from src.preprocessor import Preprocessor
        from src.ml_engine import MLEngine

        # Preprocess
        preprocessor = Preprocessor()
        processed_df, _ = preprocessor.fit_transform(
            sample_data_large,
            target_column='Attrition'
        )

        target = processed_df['Attrition']
        features = processed_df.drop(columns=['Attrition', 'EmployeeID'], errors='ignore')
        numeric_features = features.select_dtypes(include=['int64', 'float64'])

        # Time training
        ml_engine = MLEngine()
        start = time.time()
        metrics = ml_engine.train_model(numeric_features, target)
        elapsed = time.time() - start

        assert metrics is not None
        assert elapsed < 30.0, f"Training took {elapsed:.2f}s, expected < 30s"

    @pytest.mark.timeout(15)
    def test_prediction_under_5_seconds(self, sample_data_large):
        """Predictions on 1000 rows should complete in under 5 seconds."""
        from src.preprocessor import Preprocessor
        from src.ml_engine import MLEngine

        preprocessor = Preprocessor()
        processed_df, _ = preprocessor.fit_transform(
            sample_data_large,
            target_column='Attrition'
        )

        target = processed_df['Attrition']
        features = processed_df.drop(columns=['Attrition', 'EmployeeID'], errors='ignore')
        numeric_features = features.select_dtypes(include=['int64', 'float64'])

        ml_engine = MLEngine()
        ml_engine.train_model(numeric_features, target)

        # Time predictions
        start = time.time()
        risk_scores = ml_engine.predict_risk(numeric_features)
        elapsed = time.time() - start

        assert len(risk_scores) == len(numeric_features)
        assert elapsed < 5.0, f"Predictions took {elapsed:.2f}s, expected < 5s"


class TestSHAPPerformance:
    """Test SHAP computation performance."""

    @pytest.mark.timeout(20)
    def test_shap_computation_under_10_seconds(self, sample_valid_data):
        """SHAP values computation should complete in under 10 seconds."""
        from src.preprocessor import Preprocessor
        from src.ml_engine import MLEngine

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

        # Time SHAP driver computation for single employee
        start = time.time()
        drivers = ml_engine.get_risk_drivers(0, numeric_features)
        elapsed = time.time() - start

        assert len(drivers) > 0
        assert elapsed < 10.0, f"SHAP took {elapsed:.2f}s, expected < 10s"


class TestAnalyticsPerformance:
    """Test analytics computation performance."""

    @pytest.mark.timeout(10)
    def test_analytics_under_5_seconds(self, sample_data_large):
        """Analytics on 1000 rows should complete in under 5 seconds."""
        from src.analytics_engine import AnalyticsEngine

        start = time.time()

        engine = AnalyticsEngine(sample_data_large)
        headcount = engine.get_headcount()
        turnover = engine.get_turnover_rate()
        dept_stats = engine.get_department_aggregates()
        summary = engine.get_summary_statistics()

        elapsed = time.time() - start

        assert headcount > 0
        assert not dept_stats.empty
        assert elapsed < 5.0, f"Analytics took {elapsed:.2f}s, expected < 5s"

    @pytest.mark.timeout(10)
    def test_correlation_under_3_seconds(self, sample_data_large):
        """Correlation computation should complete in under 3 seconds."""
        from src.analytics_engine import AnalyticsEngine

        engine = AnalyticsEngine(sample_data_large)

        start = time.time()
        correlations = engine.get_correlations()
        elapsed = time.time() - start

        assert elapsed < 3.0, f"Correlations took {elapsed:.2f}s, expected < 3s"


class TestDataLoadingPerformance:
    """Test data loading performance."""

    @pytest.mark.timeout(10)
    def test_csv_load_under_5_seconds(self, sample_data_large):
        """Loading 1000-row CSV should complete in under 5 seconds."""
        from src.data_loader import DataLoader

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data_large.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            loader = DataLoader()

            start = time.time()
            df = loader.load(temp_path)
            elapsed = time.time() - start

            assert df is not None
            assert len(df) == len(sample_data_large)
            assert elapsed < 5.0, f"CSV load took {elapsed:.2f}s, expected < 5s"

        finally:
            os.unlink(temp_path)

    @pytest.mark.timeout(10)
    def test_preprocessing_under_5_seconds(self, sample_data_large):
        """Preprocessing 1000 rows should complete in under 5 seconds."""
        from src.preprocessor import Preprocessor

        preprocessor = Preprocessor()

        start = time.time()
        processed_df, metadata = preprocessor.fit_transform(
            sample_data_large,
            target_column='Attrition'
        )
        elapsed = time.time() - start

        assert processed_df is not None
        assert elapsed < 5.0, f"Preprocessing took {elapsed:.2f}s, expected < 5s"


class TestExportPerformance:
    """Test export performance."""

    @pytest.mark.timeout(10)
    def test_csv_export_under_3_seconds(self, sample_data_large):
        """CSV export should complete in under 3 seconds."""
        from src.export import export_to_csv

        start = time.time()
        csv_bytes = export_to_csv(sample_data_large)
        elapsed = time.time() - start

        assert csv_bytes is not None
        assert elapsed < 3.0, f"CSV export took {elapsed:.2f}s, expected < 3s"

    @pytest.mark.timeout(10)
    def test_excel_export_under_5_seconds(self, sample_data_large):
        """Excel export should complete in under 5 seconds."""
        from src.export import export_to_excel

        start = time.time()
        excel_bytes = export_to_excel(sample_data_large)
        elapsed = time.time() - start

        assert excel_bytes is not None
        assert elapsed < 5.0, f"Excel export took {elapsed:.2f}s, expected < 5s"


class TestMemoryUsage:
    """Test memory efficiency."""

    def test_large_dataset_memory_efficient(self):
        """Large dataset processing should not exceed reasonable memory."""
        import tracemalloc

        from src.preprocessor import Preprocessor
        from src.analytics_engine import AnalyticsEngine

        # Create larger dataset
        np.random.seed(42)
        n = 5000

        large_df = pd.DataFrame({
            'EmployeeID': [f'EMP{i:06d}' for i in range(1, n + 1)],
            'Dept': np.random.choice(['Eng', 'Sales', 'HR', 'Mkt', 'Fin'], n),
            'Tenure': np.random.uniform(0.1, 20, n).round(1),
            'Salary': np.random.uniform(30000, 200000, n).round(0),
            'LastRating': np.random.uniform(1, 5, n).round(1),
            'Age': np.random.randint(20, 65, n),
            'Attrition': np.random.choice([0, 1], n, p=[0.85, 0.15])
        })

        tracemalloc.start()

        # Process data
        preprocessor = Preprocessor()
        processed_df, _ = preprocessor.fit_transform(large_df, target_column='Attrition')

        engine = AnalyticsEngine(processed_df)
        engine.get_summary_statistics()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak memory should be under 500MB
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 500, f"Peak memory {peak_mb:.1f}MB exceeds 500MB limit"
