"""
PeopleOS - Main Application Entry Point

A local-first People Analytics platform for HR data analysis,
predictive modeling, and strategic insights.

This is the entry point that wires together all modules.
No business logic should be in this file.
"""

import sys

# Python version check
if sys.version_info < (3, 10):
    raise RuntimeError("PeopleOS requires Python 3.10 or higher")

from typing import Any

import pandas as pd
import streamlit as st

from src.data_loader import DataLoader, DataValidationError
from src.database import get_database
from src.merge_engine import get_merge_engine, MergeResult
from src.preprocessor import Preprocessor, PreprocessingError
from src.analytics_engine import AnalyticsEngine
from src.ml_engine import MLEngine, MLEngineError, get_cached_model
from src.llm_client import LLMClient
from src.session_manager import SessionManager, get_session_manager
from src.logger import get_logger
from src.utils import load_config, check_config_version, get_error_message
from src.nlp_engine import NLPEngine
from src.compensation_engine import CompensationEngine
from src.succession_engine import SuccessionEngine
from src.team_dynamics_engine import TeamDynamicsEngine
from src.vector_engine import VectorEngine
from src.insight_interpreter import InsightInterpreter
from src.fairness_engine import FairnessEngine

from ui.dashboard_layout import (
    setup_page_config, inject_custom_css, render_sidebar,
    render_overview_tab, render_diagnostics_tab,
    render_future_radar_tab, render_strategic_advisor_tab,
    render_semantic_search_tab, render_empty_state, render_error_state
)

logger = get_logger('main')

# Expected config version
EXPECTED_VERSION = "1.1.0"


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'features_enabled' not in st.session_state:
        st.session_state.features_enabled = {'predictive': False, 'nlp': False}
    if 'analytics_data' not in st.session_state:
        st.session_state.analytics_data = {}
    if 'ml_data' not in st.session_state:
        st.session_state.ml_data = {}
    if 'llm_data' not in st.session_state:
        st.session_state.llm_data = {}
    if 'llm_available' not in st.session_state:
        st.session_state.llm_available = False
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = 'Unknown'
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    if 'ml_engine' not in st.session_state:
        st.session_state.ml_engine = None
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True
    # New feature data
    if 'nlp_data' not in st.session_state:
        st.session_state.nlp_data = {}
    if 'comp_data' not in st.session_state:
        st.session_state.comp_data = {}
    if 'succ_data' not in st.session_state:
        st.session_state.succ_data = {}
    if 'team_data' not in st.session_state:
        st.session_state.team_data = {}
    if 'vector_engine' not in st.session_state:
        st.session_state.vector_engine = None
    if 'fairness_data' not in st.session_state:
        st.session_state.fairness_data = {}
    if 'merge_result' not in st.session_state:
        st.session_state.merge_result = None
    if 'persistence_enabled' not in st.session_state:
        st.session_state.persistence_enabled = True
    if 'database_loaded' not in st.session_state:
        st.session_state.database_loaded = False


def check_config() -> bool:
    """
    Check configuration version.
    
    Returns:
        True if config is valid.
    """
    try:
        config = load_config()
        if not check_config_version(config, EXPECTED_VERSION):
            logger.warning(f"Config version mismatch. Expected {EXPECTED_VERSION}")
            st.warning(f"Configuration version mismatch. Please update config.yaml to version {EXPECTED_VERSION}")
        return True
    except FileNotFoundError:
        st.error("Configuration file not found. Please ensure config.yaml exists.")
        return False


def initialize_llm_client() -> LLMClient:
    """
    Initialize and check LLM client.
    
    Returns:
        LLMClient instance.
    """
    client = LLMClient()
    st.session_state.llm_available = client.is_connected()
    if client.is_connected():
        model_info = client.get_model_info()
        st.session_state.llm_model = model_info.get('model', 'Unknown')
    return client


def load_and_validate_data(file_content: bytes, file_name: str) -> tuple:
    """
    Load, validate, and merge uploaded data with existing database.

    Args:
        file_content: Raw file content.
        file_name: Original file name.

    Returns:
        Tuple of (DataFrame, validation_report, merge_result, error_message).
    """
    import tempfile
    import os

    loader = DataLoader()
    file_ext = file_name.split('.')[-1].lower()

    try:
        # Save to temp file for full validation
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            # Use load_and_merge to persist and merge with existing data
            result = loader.load_and_merge(tmp_path, file_name)
            df = result['df']
            merge_result = result.get('merge_result')
            report = result['report']
            return df, report, merge_result, None
        finally:
            os.unlink(tmp_path)

    except DataValidationError as e:
        return None, None, None, str(e)
    except Exception as e:
        logger.error(f"Data loading error: {type(e).__name__}: {str(e)}")
        return None, None, None, get_error_message('file_load_failed')


def load_from_database() -> tuple:
    """
    Load existing data from the persistent database.

    Returns:
        Tuple of (DataFrame, features_enabled, error_message).
    """
    try:
        loader = DataLoader()
        df = loader.load_from_database()

        if df is not None and not df.empty:
            return df, loader.features_enabled, None
        return None, {}, None

    except Exception as e:
        logger.error(f"Database load error: {type(e).__name__}: {str(e)}")
        return None, {}, str(e)


def process_data(df: pd.DataFrame, features_enabled: dict) -> tuple:
    """
    Process data through the preprocessing pipeline.

    Args:
        df: Raw DataFrame.
        features_enabled: Dictionary of enabled features.

    Returns:
        Tuple of (processed_df, feature_metadata, error_message).
    """
    try:
        preprocessor = Preprocessor()

        if features_enabled.get('predictive', False) and 'Attrition' in df.columns:
            processed_df, metadata = preprocessor.fit_transform(df, target_column='Attrition')
        else:
            processed_df, metadata = preprocessor.fit_transform(df, target_column=None)

        return processed_df, metadata, None
    except PreprocessingError as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return None, None, str(e)
    except Exception as e:
        logger.error(f"Unexpected preprocessing error: {type(e).__name__}: {str(e)}")
        return None, None, "Data preprocessing failed. Please check your data format."


def run_analytics(df: pd.DataFrame) -> dict:
    """
    Run analytics on the data.
    
    Args:
        df: DataFrame to analyze.
        
    Returns:
        Dictionary with analytics results.
    """
    engine = AnalyticsEngine(df)
    
    return {
        'headcount': engine.get_headcount(),
        'turnover_rate': engine.get_turnover_rate(),
        'department_count': df['Dept'].nunique() if 'Dept' in df.columns else 0,
        'tenure_mean': df['Tenure'].mean() if 'Tenure' in df.columns else None,
        'dept_stats': engine.get_department_aggregates(),
        'tenure_distribution': engine.get_tenure_distribution(),
        'age_distribution': engine.get_age_distribution(),
        'salary_bands': engine.get_salary_bands(),
        'high_risk_depts': engine.get_high_risk_departments(),
        'correlations': engine.get_correlations() if 'Attrition' in df.columns else pd.DataFrame(),
        'summary': engine.get_summary_statistics(),
        'temporal_stats': engine.get_temporal_stats()
    }


def run_ml_predictions(df: pd.DataFrame, features_enabled: dict) -> tuple[dict, Any]:
    """
    Run ML predictions on the data.

    Args:
        df: Processed DataFrame.
        features_enabled: Dictionary of enabled features.

    Returns:
        Tuple of (Dictionary with ML results, MLEngine instance).
    """
    if not features_enabled.get('predictive', False):
        return {}, None

    if 'Attrition' not in df.columns:
        return {}, None

    try:
        ml_engine = get_cached_model()

        # Prepare features and target
        target = df['Attrition']
        features = df.drop(columns=['Attrition', 'EmployeeID'], errors='ignore')

        # Select only numeric columns for training
        numeric_features = features.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])

        if numeric_features.empty or len(numeric_features.columns) < 2:
            logger.warning("Not enough numeric features for ML training")
            return {}, None

        # Train model
        metrics = ml_engine.train_model(numeric_features, target)

        # Get predictions
        risk_scores = ml_engine.predict_risk(numeric_features)

        # Categorize risks
        risk_categories = [ml_engine.get_risk_category(score) for score in risk_scores]

        # Count by category
        risk_distribution = {
            'High': risk_categories.count('High'),
            'Medium': risk_categories.count('Medium'),
            'Low': risk_categories.count('Low')
        }

        # Create high-risk employee list
        high_risk_mask = [cat == 'High' for cat in risk_categories]
        high_risk_df = df[high_risk_mask].copy()
        high_risk_df['Risk_Score'] = [score for score, is_high in zip(risk_scores, high_risk_mask) if is_high]
        high_risk_df['Risk_Category'] = 'High'

        # Feature importances as DataFrame
        importance_df = ml_engine.get_feature_importance_summary()

        return {
            'metrics': metrics,
            'risk_distribution': risk_distribution,
            'high_risk_employees': high_risk_df.head(20),
            'feature_importances': importance_df,
            'model_trained': True
        }, ml_engine

    except MLEngineError as e:
        logger.error(f"ML Error: {str(e)}")
        st.session_state.error_message = get_error_message('model_training_failed')
        return {}, None
    except Exception as e:
        logger.error(f"Unexpected ML error: {type(e).__name__}: {str(e)}")
        return {}, None


def get_llm_insights(llm_client: LLMClient, analytics_data: dict) -> dict:
    """
    Get LLM-powered insights.

    Args:
        llm_client: LLM client instance.
        analytics_data: Analytics data for context.

    Returns:
        Dictionary with LLM responses.
    """
    metrics = {
        'total_employees': analytics_data.get('headcount', 0),
        'turnover_rate': analytics_data.get('turnover_rate'),
        'high_risk_count': analytics_data.get('summary', {}).get('attrition_count', 0),
        'departments': analytics_data.get('department_count', 0)
    }

    return {
        'strategic_summary': llm_client.get_strategic_summary(metrics),
        'action_items': llm_client.get_action_items(metrics)
    }


def run_nlp_analysis(df: pd.DataFrame, llm_client: LLMClient, features_enabled: dict) -> dict:
    """
    Run NLP analysis on performance text.

    Args:
        df: DataFrame with PerformanceText column.
        llm_client: LLM client for AI-powered analysis.
        features_enabled: Dictionary of enabled features.

    Returns:
        Dictionary with NLP analysis results.
    """
    if not features_enabled.get('nlp', False):
        return {}

    if 'PerformanceText' not in df.columns:
        return {}

    try:
        nlp_engine = NLPEngine(llm_client)
        return nlp_engine.process_all(df)
    except Exception as e:
        logger.error(f"NLP analysis error: {type(e).__name__}: {str(e)}")
        return {}


def run_compensation_analysis(df: pd.DataFrame) -> dict:
    """
    Run compensation analysis.

    Args:
        df: DataFrame with employee data.

    Returns:
        Dictionary with compensation analysis results.
    """
    if 'Salary' not in df.columns:
        return {}

    try:
        comp_engine = CompensationEngine(df)
        return comp_engine.analyze_all()
    except Exception as e:
        logger.error(f"Compensation analysis error: {type(e).__name__}: {str(e)}")
        return {}


def run_succession_analysis(df: pd.DataFrame, ml_data: dict) -> dict:
    """
    Run succession planning analysis.

    Args:
        df: DataFrame with employee data.
        ml_data: ML predictions for risk scores.

    Returns:
        Dictionary with succession analysis results.
    """
    try:
        # Get risk scores from ML predictions if available
        risk_scores = None
        if ml_data and 'high_risk_employees' in ml_data:
            high_risk = ml_data['high_risk_employees']
            if not high_risk.empty and 'Risk_Score' in high_risk.columns:
                risk_scores = high_risk[['EmployeeID', 'Risk_Score']].copy()
                risk_scores.columns = ['EmployeeID', 'risk_score']

        succ_engine = SuccessionEngine(df, risk_scores)
        return succ_engine.analyze_all()
    except Exception as e:
        logger.error(f"Succession analysis error: {type(e).__name__}: {str(e)}")
        return {}


def run_team_dynamics_analysis(df: pd.DataFrame, nlp_data: dict) -> dict:
    """
    Run team dynamics analysis.

    Args:
        df: DataFrame with employee data.
        nlp_data: NLP analysis results for sentiment integration.

    Returns:
        Dictionary with team dynamics results.
    """
    try:
        team_engine = TeamDynamicsEngine(df, nlp_data if nlp_data else None)
        return team_engine.analyze_all()
    except Exception as e:
        logger.error(f"Team dynamics analysis error: {type(e).__name__}: {str(e)}")
        return {}


def run_vector_indexing(df: pd.DataFrame) -> VectorEngine:
    """Initialize and index vector engine."""
    if 'PerformanceText' not in df.columns:
        return None

    try:
        engine = VectorEngine()
        texts = df['PerformanceText'].fillna('').tolist()
        metadata = df[['EmployeeID', 'Dept', 'PerformanceText']].to_dict('records')
        engine.build_index(texts, metadata)
        return engine
    except Exception as e:
        logger.error(f"Vector indexing failed: {str(e)}")
        return None


def run_fairness_analysis(df: pd.DataFrame, ml_data: dict) -> dict:
    """
    Run fairness and bias detection analysis.

    Args:
        df: DataFrame with employee data.
        ml_data: ML predictions for bias analysis.

    Returns:
        Dictionary with fairness analysis results.
    """
    config = load_config()
    fairness_config = config.get('fairness', {})

    if not fairness_config.get('enabled', True):
        return {}

    try:
        # Get risk predictions if available
        predictions = None
        if ml_data and ml_data.get('model_trained', False):
            high_risk = ml_data.get('high_risk_employees')
            if high_risk is not None and not high_risk.empty:
                predictions = high_risk[['EmployeeID', 'Risk_Score']].copy()
                predictions.columns = ['EmployeeID', 'risk_score']

        fairness_engine = FairnessEngine(df, predictions)
        results = fairness_engine.analyze_all()

        # Log any critical findings
        if results.get('summary', {}).get('overall_status') == 'Critical':
            logger.warning("Critical fairness issues detected in the data")

        return results
    except Exception as e:
        logger.error(f"Fairness analysis error: {type(e).__name__}: {str(e)}")
        return {}


def main():
    """Main application entry point."""
    # Setup page
    setup_page_config()

    # Initialize state (needed before inject_custom_css for dark_mode)
    initialize_session_state()

    # Inject CSS with theme preference
    inject_custom_css(st.session_state.dark_mode)

    # Check config
    if not check_config():
        return

    # Initialize LLM client and session manager
    llm_client = initialize_llm_client()
    session_manager = get_session_manager()

    # Load from persistent database on startup (if not already loaded)
    if not st.session_state.database_loaded and not st.session_state.data_loaded:
        db_df, db_features, db_error = load_from_database()
        if db_df is not None and not db_df.empty:
            st.session_state.raw_data = db_df
            st.session_state.data_loaded = True
            st.session_state.features_enabled = db_features
            st.session_state.database_loaded = True
            logger.info(f"Loaded {len(db_df)} employees from persistent storage")

    # Handle session loading
    if 'load_session' in st.session_state and st.session_state.load_session:
        session_data = session_manager.load_session(st.session_state.load_session)
        if session_data and 'raw_data' in session_data and session_data['raw_data'] is not None:
            st.session_state.raw_data = session_data['raw_data']
            st.session_state.data_loaded = True
            st.session_state.features_enabled = session_data.get('features_enabled', {})
            # Rerun analytics
            st.session_state.analytics_data = run_analytics(st.session_state.raw_data)
            ml_data, ml_engine = run_ml_predictions(
                st.session_state.raw_data,
                st.session_state.features_enabled
            )
            st.session_state.ml_data = ml_data
            st.session_state.ml_engine = ml_engine
            st.success(f"Session loaded: {session_data.get('session_name', 'Unknown')}")
        st.session_state.load_session = None

    # Render sidebar
    sidebar_result = render_sidebar(
        data_loaded=st.session_state.data_loaded,
        features_enabled=st.session_state.features_enabled,
        session_manager=session_manager
    )

    # Handle session saving
    if 'save_session' in st.session_state and st.session_state.save_session:
        session_manager.save_session(
            st.session_state.save_session,
            st.session_state.raw_data,
            st.session_state.analytics_data,
            st.session_state.ml_data,
            st.session_state.features_enabled
        )
        st.success(f"Session saved: {st.session_state.save_session}")
        st.session_state.save_session = None
    
    # Process data and run analysis if file uploaded
    uploaded_file = sidebar_result.get('uploaded_file')
    if uploaded_file is not None:
        file_content = uploaded_file.read()
        df, report, merge_result, error = load_and_validate_data(file_content, uploaded_file.name)

        if error:
            render_error_state(error)
            return

        if df is not None:
            st.session_state.raw_data = df
            st.session_state.data_loaded = True
            st.session_state.features_enabled = report.get('features_enabled', {})
            st.session_state.refresh_llm = True
            st.session_state.merge_result = merge_result

            # Show merge summary (what was added/updated)
            if merge_result:
                summary = merge_result.get_summary_text()
                if merge_result.added > 0 or merge_result.updated > 0:
                    st.success(f"Data imported: {summary}")
                elif merge_result.unchanged > 0:
                    st.info(f"Data imported: {summary}")

            # Show validation warnings
            for warning in report.get('warnings', []):
                st.warning(warning)

            # Clear cached analytics to force refresh
            st.session_state.analytics_data = {}
            st.session_state.ml_data = {}
            st.session_state.nlp_data = {}
            st.session_state.comp_data = {}
            st.session_state.succ_data = {}
            st.session_state.team_data = {}
            st.session_state.fairness_data = {}
            st.session_state.vector_engine = None

            # Analytics + ML
            st.session_state.analytics_data = run_analytics(df)
            ml_data, ml_engine = run_ml_predictions(df, st.session_state.features_enabled)
            st.session_state.ml_data = ml_data
            st.session_state.ml_engine = ml_engine

    # Run remaining analysis if data is loaded but state is empty
    if st.session_state.data_loaded and st.session_state.raw_data is not None:
        df = st.session_state.raw_data
        
        # Ensure analytics data is present
        if not st.session_state.analytics_data:
            st.session_state.analytics_data = run_analytics(df)

        # Ensure ML data is present
        if not st.session_state.ml_data:
            ml_data, ml_engine = run_ml_predictions(df, st.session_state.features_enabled)
            st.session_state.ml_data = ml_data
            st.session_state.ml_engine = ml_engine

        # NLP analysis
        if not st.session_state.nlp_data:
            st.session_state.nlp_data = run_nlp_analysis(df, llm_client, st.session_state.features_enabled)

        # Compensation
        if not st.session_state.comp_data:
            st.session_state.comp_data = run_compensation_analysis(df)

        # Succession
        if not st.session_state.succ_data:
            st.session_state.succ_data = run_succession_analysis(df, st.session_state.ml_data)

        # Vector indexing
        if st.session_state.vector_engine is None:
            st.session_state.vector_engine = run_vector_indexing(df)

        # Team Dynamics
        if not st.session_state.team_data:
            st.session_state.team_data = run_team_dynamics_analysis(df, st.session_state.nlp_data)

        # Fairness
        if not st.session_state.fairness_data:
            st.session_state.fairness_data = run_fairness_analysis(df, st.session_state.ml_data)

        # LLM Insights
        if st.session_state.get('refresh_llm', True) or st.session_state.get('refresh_executive_briefing', False):
            st.session_state.llm_data = get_llm_insights(llm_client, st.session_state.analytics_data)
            
            if llm_client.is_available:
                with st.spinner("Generating Executive Briefing..."):
                    summary_stats = st.session_state.analytics_data.get('summary', {})
                    exec_briefing = llm_client.get_executive_briefing(summary_stats, st.session_state.analytics_data)
                    st.session_state.llm_data['executive_briefing'] = exec_briefing
            
            st.session_state.refresh_llm = False
            st.session_state.refresh_executive_briefing = False

    # Render main content
    if not st.session_state.data_loaded:
        render_empty_state()
        return
    
    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🔍 Diagnostics",
        "🔮 Future Radar",
        "🔎 Semantic Search",
        "🧠 Strategic Advisor"
    ])

    with tab1:
        # Initialize interpreter if LLM is available
        insight_interpreter = None
        if llm_client and llm_client.is_available:
            insight_interpreter = InsightInterpreter(llm_client)
        
        render_overview_tab(
            st.session_state.analytics_data,
            insight_interpreter=insight_interpreter
        )

    with tab2:
        render_diagnostics_tab(
            st.session_state.analytics_data,
            st.session_state.comp_data,
            st.session_state.succ_data,
            st.session_state.team_data,
            st.session_state.nlp_data,
            st.session_state.raw_data
        )

    with tab3:
        render_future_radar_tab(
            st.session_state.ml_data,
            features_enabled=st.session_state.features_enabled.get('predictive', False),
            full_df=st.session_state.raw_data,
            ml_engine=st.session_state.ml_engine
        )

    with tab4:
        render_semantic_search_tab(
            st.session_state.vector_engine,
            df=st.session_state.raw_data
        )

    with tab5:
        render_strategic_advisor_tab(
            st.session_state.llm_data,
            st.session_state.analytics_data.get('summary', {}),
            llm_available=st.session_state.llm_available
        )


if __name__ == "__main__":
    main()
