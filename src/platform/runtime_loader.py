"""Fast dataset activation for the PeopleOS transition runtime.

The legacy AppState.load_data() path initializes every engine, trains the predictive
model, and may build transformer embeddings synchronously. That makes a simple
upload perform model lifecycle work. This module keeps upload deterministic and
fast: load + preprocess + initialize read-only analytical engines. Predictive
training and vector indexing remain explicit downstream operations.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.logger import get_logger

logger = get_logger('runtime_loader')


_EXCLUDED_MODEL_COLUMNS = {
    'EmployeeID', 'Attrition', 'PerformanceText', 'RatingHistory',
    'HireDate', 'PromotionDate',
}


def _prepare_predictive_inputs(state) -> None:
    """Prepare features/target without fitting a model."""
    state.features_df = None
    state.target_series = None
    state.model_metrics = None
    state.risk_scores = None
    state.ml_engine = None
    state.fairness_engine = None

    if not state.features_enabled.get('predictive', False) or state.processed_df is None:
        return

    from src.data_loader import GOLDEN_SCHEMA

    potential_features = GOLDEN_SCHEMA['required'] + GOLDEN_SCHEMA['optional']
    feature_cols = [
        column for column in state.processed_df.columns
        if column in potential_features and column not in _EXCLUDED_MODEL_COLUMNS
    ]
    state.features_df = state.processed_df[feature_cols].select_dtypes(
        include=['int64', 'float64', 'int32', 'float32']
    )

    interview_cols = [column for column in state.processed_df.columns if column.startswith('InterviewScore_')]
    if interview_cols:
        interview_df = state.processed_df[interview_cols].select_dtypes(include=['number'])
        state.features_df = pd.concat([state.features_df, interview_df], axis=1)

    state.target_series = state.processed_df['Attrition']


def _initialize_read_only_engines(state) -> None:
    """Initialize engines that do not require fitting a predictive model or embeddings."""
    from src.analytics_engine import AnalyticsEngine
    from src.compensation_engine import CompensationEngine
    from src.experience_engine import ExperienceEngine
    from src.insight_interpreter import InsightInterpreter
    from src.nlp_engine import NLPEngine
    from src.quality_of_hire_engine import QualityOfHireEngine
    from src.safe_llm_client import SafeLLMClient as LLMClient
    from src.scenario_engine import ScenarioEngine
    from src.sentiment_engine import SentimentEngine
    from src.structural_engine import StructuralEngine
    from src.succession_engine import SuccessionEngine
    from src.survival_engine import SurvivalEngine
    from src.team_dynamics_engine import TeamDynamicsEngine

    raw = state.raw_df
    if raw is None:
        return

    state.analytics_engine = AnalyticsEngine(raw)

    def safe(factory, name: str):
        try:
            return factory()
        except Exception as exc:
            logger.warning('%s initialization skipped: %s', name, exc)
            return None

    state.compensation_engine = safe(lambda: CompensationEngine(raw), 'CompensationEngine')
    state.succession_engine = safe(lambda: SuccessionEngine(raw, None), 'SuccessionEngine')
    state.team_dynamics_engine = safe(lambda: TeamDynamicsEngine(raw), 'TeamDynamicsEngine')

    # Vector search is an optional advanced capability and must never block data activation.
    state.vector_engine = None

    try:
        state.llm_client = LLMClient()
        if state.llm_client.is_available:
            state.features_enabled['llm'] = True
        state.nlp_engine = NLPEngine(state.llm_client)
        state.insight_interpreter = InsightInterpreter(state.llm_client)
    except Exception as exc:
        logger.warning('Local LLM initialization degraded: %s', exc)
        state.llm_client = None
        state.nlp_engine = None
        state.insight_interpreter = InsightInterpreter()

    state.survival_engine = (
        safe(lambda: SurvivalEngine(raw), 'SurvivalEngine')
        if 'Tenure' in raw.columns and 'Attrition' in raw.columns else None
    )

    cols_lower = [column.lower() for column in raw.columns]
    has_qoh = (
        'hiresource' in cols_lower or
        'interviewscore' in cols_lower or
        any(column.startswith('interviewscore_') for column in cols_lower)
    )
    state.quality_of_hire_engine = safe(lambda: QualityOfHireEngine(raw), 'QualityOfHireEngine') if has_qoh else None

    has_structural = 'Tenure' in raw.columns and (
        'YearsInCurrentRole' in raw.columns or 'ManagerID' in raw.columns
    )
    state.structural_engine = safe(lambda: StructuralEngine(raw), 'StructuralEngine') if has_structural else None

    state.sentiment_engine = safe(
        lambda: SentimentEngine(employee_df=raw, enps_df=state.enps_df, onboarding_df=state.onboarding_df),
        'SentimentEngine',
    )
    state.experience_engine = safe(lambda: ExperienceEngine(raw), 'ExperienceEngine')
    state.scenario_engine = safe(
        lambda: ScenarioEngine(
            employee_df=raw,
            ml_engine=None,
            survival_engine=state.survival_engine,
            compensation_engine=state.compensation_engine,
        ),
        'ScenarioEngine',
    )


def load_dataset(state, file_path: str, file_name: str = 'upload') -> Dict[str, Any]:
    """Load and activate a dataset without training models or building embeddings."""
    result = state.data_loader.load_and_merge(file_path, file_name)
    state.raw_df = result['df']
    state.features_enabled = state.data_loader.features_enabled.copy()
    state.features_enabled['llm'] = False

    if 'Attrition' in state.raw_df.columns:
        state.processed_df, state.preprocessing_metadata = state.preprocessor.fit_transform(
            state.raw_df, target_column='Attrition'
        )
        state.features_enabled['predictive'] = True
    else:
        state.processed_df, state.preprocessing_metadata = state.preprocessor.fit_transform(state.raw_df)

    _prepare_predictive_inputs(state)
    _initialize_read_only_engines(state)

    return {
        'rows_loaded': len(state.raw_df),
        'columns': list(state.raw_df.columns),
        'features_enabled': state.features_enabled,
        'merge_result': result.get('merge_result'),
        'report': result.get('report'),
        'deferred': {
            'predictive_training': bool(state.features_enabled.get('predictive', False)),
            'vector_indexing': bool(state.features_enabled.get('nlp', False)),
        },
    }
