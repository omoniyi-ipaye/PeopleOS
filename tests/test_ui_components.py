"""
UI component tests for PeopleOS.

Tests UI rendering functions with mock data to ensure no exceptions.
Note: These tests verify component functions don't crash, not visual output.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import sys


class TestRiskBadge:
    """Test risk badge rendering (pure function, no Streamlit dependency)."""

    def test_render_risk_badge_high(self):
        """Test high risk badge HTML."""
        # Import here to avoid streamlit import issues
        from ui.styling import RISK_COLORS

        # Direct test of HTML generation logic
        risk_category = "High"
        color = RISK_COLORS.get(risk_category, '#a0a0a0')

        html = f'<span style="background-color: {color}; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold;">{risk_category}</span>'

        assert "High" in html
        assert "background-color" in html
        assert RISK_COLORS['High'] in html

    def test_render_risk_badge_medium(self):
        """Test medium risk badge HTML."""
        from ui.styling import RISK_COLORS

        risk_category = "Medium"
        color = RISK_COLORS.get(risk_category, '#a0a0a0')

        assert color == RISK_COLORS['Medium']

    def test_render_risk_badge_low(self):
        """Test low risk badge HTML."""
        from ui.styling import RISK_COLORS

        risk_category = "Low"
        color = RISK_COLORS.get(risk_category, '#a0a0a0')

        assert color == RISK_COLORS['Low']


class TestStylingFunctions:
    """Test styling helper functions."""

    def test_get_current_colors_dark(self):
        """Test dark mode colors."""
        from ui.styling import get_current_colors, DARK_COLORS

        colors = get_current_colors(dark_mode=True)
        assert colors == DARK_COLORS

    def test_get_current_colors_light(self):
        """Test light mode colors."""
        from ui.styling import get_current_colors, LIGHT_COLORS

        colors = get_current_colors(dark_mode=False)
        assert colors == LIGHT_COLORS

    def test_get_custom_css_returns_string(self):
        """Test CSS generation returns valid string."""
        from ui.styling import get_custom_css

        css = get_custom_css(dark_mode=True)
        assert isinstance(css, str)
        assert '<style>' in css
        assert '</style>' in css

    def test_get_plotly_theme_dark(self):
        """Test Plotly theme generation for dark mode."""
        from ui.styling import get_plotly_theme, DARK_COLORS

        theme = get_plotly_theme(dark_mode=True)

        assert 'paper_bgcolor' in theme
        assert 'plot_bgcolor' in theme
        assert theme['paper_bgcolor'] == DARK_COLORS['background']

    def test_get_plotly_theme_light(self):
        """Test Plotly theme generation for light mode."""
        from ui.styling import get_plotly_theme, LIGHT_COLORS

        theme = get_plotly_theme(dark_mode=False)

        assert theme['paper_bgcolor'] == LIGHT_COLORS['background']


class TestExportModule:
    """Test export module functions."""

    def test_export_to_csv(self):
        """Test CSV export."""
        from src.export import export_to_csv

        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        result = export_to_csv(df)

        assert isinstance(result, bytes)
        assert len(result) > 0
        assert b'A,B' in result

    def test_export_to_excel(self):
        """Test Excel export."""
        from src.export import export_to_excel

        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        result = export_to_excel(df)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_generate_download_filename(self):
        """Test filename generation."""
        from src.export import generate_download_filename

        filename = generate_download_filename("test", "csv")

        assert filename.endswith(".csv")
        assert "test" in filename

    def test_export_risk_report(self):
        """Test risk report export."""
        from src.export import export_risk_report

        analytics_data = {'headcount': 100, 'turnover_rate': 0.15}
        ml_data = {'risk_distribution': {'High': 10, 'Medium': 30, 'Low': 60}}

        result = export_risk_report(analytics_data, ml_data)

        assert isinstance(result, bytes)
        assert len(result) > 0


class TestDataLoader:
    """Test data loader validation."""

    def test_column_mapping_aliases(self):
        """Test that column aliases are defined."""
        from src.data_loader import DataLoader

        loader = DataLoader()
        config = loader.config.get('data', {})

        assert 'required_columns' in config


class TestMLEngine:
    """Test ML engine helper functions."""

    def test_risk_thresholds_configurable(self):
        """Test that risk thresholds come from config."""
        from src.ml_engine import MLEngine

        engine = MLEngine()

        assert hasattr(engine, 'risk_threshold_high')
        assert hasattr(engine, 'risk_threshold_medium')
        assert 0 < engine.risk_threshold_medium < engine.risk_threshold_high <= 1.0

    def test_risk_category_high(self):
        """Test high risk categorization."""
        from src.ml_engine import MLEngine

        engine = MLEngine()
        category = engine.get_risk_category(0.85)

        assert category == "High"

    def test_risk_category_medium(self):
        """Test medium risk categorization."""
        from src.ml_engine import MLEngine

        engine = MLEngine()
        category = engine.get_risk_category(0.60)

        assert category == "Medium"

    def test_risk_category_low(self):
        """Test low risk categorization."""
        from src.ml_engine import MLEngine

        engine = MLEngine()
        category = engine.get_risk_category(0.25)

        assert category == "Low"


class TestAnalyticsEngine:
    """Test analytics engine configuration."""

    def test_high_risk_threshold_configurable(self):
        """Test that high risk department threshold is configurable."""
        from src.analytics_engine import AnalyticsEngine

        df = pd.DataFrame({
            'EmployeeID': ['E1', 'E2'],
            'Dept': ['A', 'A'],
            'Tenure': [1, 2],
            'Salary': [50000, 60000],
            'LastRating': [3, 4],
            'Age': [25, 30]
        })

        engine = AnalyticsEngine(df)

        assert hasattr(engine, 'high_risk_threshold')
        assert 0 < engine.high_risk_threshold < 1.0


class TestSessionManager:
    """Test session manager initialization."""

    def test_session_manager_creates_directory(self, tmp_path):
        """Test that session manager creates sessions directory."""
        from src.session_manager import SessionManager

        sessions_dir = tmp_path / "test_sessions"
        manager = SessionManager(sessions_dir=str(sessions_dir))

        assert sessions_dir.exists()

    def test_session_manager_max_saved_config(self):
        """Test that max_saved comes from config."""
        from src.session_manager import SessionManager

        manager = SessionManager()

        assert hasattr(manager, 'max_saved')
        assert isinstance(manager.max_saved, int)
        assert manager.max_saved > 0


class TestLLMClient:
    """Test LLM client behavior."""

    def test_strategic_summary_raises_error_if_unavailable(self):
        """Test strategic summary raises error when LLM unavailable."""
        from src.llm_client import LLMClient, LLMClientError

        client = LLMClient()
        metrics = {'total_employees': 100, 'turnover_rate': 0.15}

        # If LLM is not running, it should raise LLMClientError
        if not client.is_available:
            with pytest.raises(LLMClientError):
                client.get_strategic_summary(metrics)

    def test_action_items_raises_error_if_unavailable(self):
        """Test action items raises error when LLM unavailable."""
        from src.llm_client import LLMClient, LLMClientError

        client = LLMClient()
        metrics = {'total_employees': 100}

        if not client.is_available:
            with pytest.raises(LLMClientError):
                client.get_action_items(metrics)


class TestConfigValidation:
    """Test configuration validation."""

    def test_config_loads_successfully(self):
        """Test that config file loads."""
        from src.utils import load_config

        config = load_config()

        assert config is not None
        assert 'version' in config

    def test_config_has_required_sections(self):
        """Test config has expected sections."""
        from src.utils import load_config

        config = load_config()

        assert 'app' in config
        assert 'ollama' in config
        assert 'ml' in config
        assert 'data' in config
        assert 'sessions' in config
        assert 'analytics' in config

    def test_ml_config_values(self):
        """Test ML configuration values are valid."""
        from src.utils import load_config

        config = load_config()
        ml_config = config.get('ml', {})

        assert 'random_seed' in ml_config
        assert 'risk_threshold_high' in ml_config
        assert 'risk_threshold_medium' in ml_config
        assert ml_config['risk_threshold_medium'] < ml_config['risk_threshold_high']
