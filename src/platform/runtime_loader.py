"""Fast, deterministic dataset activation for PeopleOS.

Activation establishes an explicit population contract before any analysis:
- historical_df preserves all validated rows/snapshots;
- raw_df is the latest observation per employee (current-state population);
- active_df is the current population with observed Attrition == 0.

Read-only analytics initialize from current-state data. Predictive training and
vector indexing remain explicit downstream operations and never reuse an
activation-fitted transform for model evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from types import SimpleNamespace
from copy import deepcopy
from uuid import uuid4
from src.platform.provenance import frame_fingerprint
from src.platform.runtime_lock import runtime_mutation

import pandas as pd

from src.logger import get_logger
from src.population import active_population, resolve_current_population

logger = get_logger('runtime_loader')


def pay_basis_is_confirmed(frame: pd.DataFrame) -> bool:
    """Require explicit comparable units, including when restoring old artifacts."""
    periods = [column for column in ('PayPeriod', 'PayFrequency') if column in frame]
    annual = {'annual', 'annually', 'year', 'yearly', 'per year', 'per annum', 'annualized', 'annualised'}
    if not periods or 'Currency' not in frame or frame.empty:
        return False
    if any(not frame[column].astype('string').str.strip().str.lower().isin(annual).all() for column in periods):
        return False
    currencies = frame['Currency'].astype('string').str.strip().str.upper()
    return bool(currencies.notna().all() and currencies.str.fullmatch('[A-Z]{3}').all() and currencies.nunique() == 1)


def _prepare_predictive_inputs(state) -> None:
    """Record predictive readiness without fitting/evaluating a model."""
    state.features_df = None
    state.target_series = None
    state.model_metrics = None
    state.risk_scores = None
    state.ml_engine = None
    state.fairness_engine = None

    raw = state.raw_df
    if raw is None or 'Attrition' not in raw.columns:
        state.features_enabled['predictive'] = False
        return
    known = raw['Attrition'].dropna()
    state.features_enabled['predictive'] = bool(
        not known.empty
        and set(known.astype(int).unique()).issubset({0, 1})
        and known.nunique() == 2
    )


def _initialize_read_only_engines(state) -> None:
    from src.analytics_engine import AnalyticsEngine
    from src.compensation_engine import CompensationEngine
    from src.experience_engine import ExperienceEngine
    from src.fairness_engine import FairnessEngine
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

    def safe(factory, name: str):
        try:
            return factory()
        except Exception as exc:
            logger.warning('%s initialization skipped: %s', name, exc)
            return None

    state.analytics_engine = AnalyticsEngine(raw)
    state.fairness_engine = safe(lambda: FairnessEngine(raw), 'FairnessEngine') if 'Attrition' in raw else None
    state.compensation_engine = safe(lambda: CompensationEngine(raw), 'CompensationEngine')
    state.succession_engine = safe(lambda: SuccessionEngine(raw, None), 'SuccessionEngine')
    state.team_dynamics_engine = safe(lambda: TeamDynamicsEngine(raw), 'TeamDynamicsEngine')
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

    state.survival_engine = safe(lambda: SurvivalEngine(raw), 'SurvivalEngine') if {'Tenure', 'Attrition'}.issubset(raw.columns) else None

    cols_lower = [column.lower() for column in raw.columns]
    has_qoh = 'hiresource' in cols_lower or 'interviewscore' in cols_lower or any(c.startswith('interviewscore_') for c in cols_lower)
    state.quality_of_hire_engine = safe(lambda: QualityOfHireEngine(raw), 'QualityOfHireEngine') if has_qoh else None

    has_structural = 'Tenure' in raw.columns and ('YearsInCurrentRole' in raw.columns or 'ManagerID' in raw.columns)
    state.structural_engine = safe(lambda: StructuralEngine(raw), 'StructuralEngine') if has_structural else None
    state.sentiment_engine = safe(lambda: SentimentEngine(employee_df=raw, enps_df=state.enps_df, onboarding_df=state.onboarding_df), 'SentimentEngine')
    state.experience_engine = safe(lambda: ExperienceEngine(raw), 'ExperienceEngine')
    state.scenario_engine = safe(
        lambda: ScenarioEngine(employee_df=raw, ml_engine=None, survival_engine=state.survival_engine, compensation_engine=state.compensation_engine),
        'ScenarioEngine',
    )


def _populate_dataframe(
    state,
    historical: pd.DataFrame,
    *,
    feature_flags: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Activate validated rows from any source without fitting a predictive model."""
    current, population = resolve_current_population(historical.copy())
    # Keep original measurements in the versioned source, but never let unknown
    # units become apparent facts in engines, agent evidence or model inputs.
    pay_confirmed = pay_basis_is_confirmed(historical)
    if not pay_confirmed:
        for column in ('Salary', 'StartingSalary', 'SalaryGrowth', 'MarketSalary', 'SalaryMidpoint', 'BandMidpoint', 'CompaRatio'):
            if column in current:
                current[column] = float('nan')

    if 'EmployeeID' not in current or current['EmployeeID'].isna().any() or current['EmployeeID'].astype(str).str.strip().eq('').any():
        raise ValueError('Every workforce row requires a nonempty EmployeeID')
    state.enps_df = None
    state.onboarding_df = None
    state.nlp_results = None
    state.model_provenance = None
    state.scenario_cache = {}
    state.historical_df = historical.copy()
    state.raw_df = current
    state.active_df = active_population(current)
    state.population_resolution = population
    state.features_enabled = dict(feature_flags or getattr(state.data_loader, 'features_enabled', {}) or {})
    state.features_enabled.setdefault('predictive', False)
    state.features_enabled.setdefault('nlp', False)
    state.features_enabled['llm'] = False
    state.features_enabled['compensation'] = pay_confirmed

    state.processed_df, state.preprocessing_metadata = state.preprocessor.fit_transform(
        current,
        target_column='Attrition' if 'Attrition' in current.columns else '__no_target__',
    )
    if state.preprocessing_metadata is not None:
        state.preprocessing_metadata['fit_scope'] = 'current_dataset_compatibility_only'
        state.preprocessing_metadata['not_valid_for_model_evaluation'] = True

    _prepare_predictive_inputs(state)
    _initialize_read_only_engines(state)

    return {
        'rows_loaded': len(current),
        'source_rows': population.source_rows,
        'unique_employees': population.unique_employees,
        'snapshot_history': population.snapshot_history,
        'as_of_date': population.as_of_date,
        'active_rows': len(state.active_df),
        'columns': list(current.columns),
        'features_enabled': state.features_enabled,
        'deferred': {
            'predictive_training': bool(state.features_enabled.get('predictive', False)),
            'vector_indexing': bool(state.features_enabled.get('nlp', False)),
        },
    }


def prepare_dataframe(state, historical, *, feature_flags=None, workspace_id='local', dataset_id=None):
    """Build an isolated candidate; failures cannot partially replace live engines."""
    candidate = SimpleNamespace(**state.__dict__)
    candidate.preprocessor = deepcopy(state.preprocessor)
    if feature_flags is None:
        feature_flags = {'predictive': 'Attrition' in historical, 'nlp': 'PerformanceText' in historical and bool(historical['PerformanceText'].notna().any())}
    result = _populate_dataframe(candidate, historical, feature_flags=feature_flags)
    candidate.runtime_provenance = {
        'workspace_id': workspace_id, 'dataset_id': dataset_id,
        'generation': uuid4().hex, 'current_fingerprint': frame_fingerprint(candidate.raw_df),
        'source_rows': len(historical), 'current_rows': len(candidate.raw_df),
        'active_rows': len(candidate.active_df),
        'unknown_status_rows': int(candidate.raw_df.Attrition.isna().sum()) if 'Attrition' in candidate.raw_df else 0,
        'pay_basis_confirmed': candidate.features_enabled['compensation'],
        'reporting_currency': (historical['Currency'].astype('string').str.strip().str.upper().iloc[0]
                               if candidate.features_enabled['compensation'] else None),
        'pay_basis_message': ('Annual amounts and a shared currency are declared in the source.'
                              if candidate.features_enabled['compensation'] else
                              'Pay outputs are unavailable. Reimport with annual pay and one shared currency declared; workforce counts remain available.'),
        'population_contract': 'observed_status' if 'Attrition' in candidate.raw_df else 'active_only_input',
    }
    result['provenance'] = candidate.runtime_provenance
    result['auxiliary_inputs_cleared'] = True
    return candidate, result


@runtime_mutation
def activate_dataframe(state, historical, *, feature_flags=None, workspace_id='local', dataset_id=None):
    candidate, result = prepare_dataframe(state, historical, feature_flags=feature_flags, workspace_id=workspace_id, dataset_id=dataset_id)
    state.__dict__.update(candidate.__dict__)
    return result


@runtime_mutation
def load_dataset(state, file_path: str, file_name: str = 'upload', *, salary_basis=None, salary_currency=None) -> Dict[str, Any]:
    """Activate a complete upload as its own snapshot; never blend workspaces via SQLite."""
    loader = deepcopy(state.data_loader)
    frame = loader.load(file_path)
    if salary_basis is not None:
        if salary_basis != 'annual':
            raise ValueError('Salary basis confirmation must be annual; PeopleOS does not guess conversion factors.')
        if not any(column in frame for column in ('PayPeriod', 'PayFrequency')):
            frame['PayPeriod'] = 'annual'
    if salary_currency is not None:
        import re
        currency = salary_currency.strip().upper()
        if not re.fullmatch('[A-Z]{3}', currency):
            raise ValueError('Enter a three-letter reporting currency, such as EUR.')
        if 'Currency' in frame and not frame['Currency'].astype('string').str.upper().eq(currency).all():
            raise ValueError('Confirmed currency conflicts with the source Currency column. No amounts were converted.')
        frame['Currency'] = currency
    activation = activate_dataframe(state, frame, feature_flags=loader.features_enabled.copy())
    state.data_loader = loader
    activation['merge_result'] = None
    activation['report'] = loader.get_column_mapping_report()
    return activation
