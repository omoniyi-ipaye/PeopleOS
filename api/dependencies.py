"""Shared FastAPI application state.

AppState is now a compatibility container only. All dataset activation flows through
`src.platform.runtime_loader`, which establishes current/active population contracts
and initializes read-only analytics. Predictive training/activation is available only
through the explicit model lifecycle API.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data_loader import DataLoader
from src.logger import get_logger
from src.preprocessor import Preprocessor
from src.utils import load_config

logger = get_logger('api_dependencies')


class AppState:
    """Compatibility state container for the local PeopleOS runtime."""

    _instance: Optional['AppState'] = None

    def __new__(cls) -> 'AppState':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return

        self.config = load_config()
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()

        self.historical_df: Optional[pd.DataFrame] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.active_df: Optional[pd.DataFrame] = None
        self.population_resolution: Any = None

        self.processed_df: Optional[pd.DataFrame] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.target_series: Optional[pd.Series] = None
        self.preprocessing_metadata: Optional[Dict[str, Any]] = None

        self.analytics_engine: Any = None
        self.ml_engine: Any = None
        self.compensation_engine: Any = None
        self.succession_engine: Any = None
        self.team_dynamics_engine: Any = None
        self.fairness_engine: Any = None
        self.vector_engine: Any = None
        self.nlp_engine: Any = None
        self.llm_client: Any = None
        self.insight_interpreter: Any = None
        self.survival_engine: Any = None
        self.quality_of_hire_engine: Any = None
        self.structural_engine: Any = None
        self.sentiment_engine: Any = None
        self.experience_engine: Any = None
        self.scenario_engine: Any = None

        self.enps_df: Optional[pd.DataFrame] = None
        self.onboarding_df: Optional[pd.DataFrame] = None

        self.model_metrics: Optional[Dict[str, Any]] = None
        self.risk_scores: Optional[pd.DataFrame] = None
        self.nlp_results: Optional[Dict[str, Any]] = None

        self.features_enabled = {'predictive': False, 'nlp': False, 'llm': False}
        self._initialized = True

    def load_data(self, file_path: str, file_name: str = 'upload') -> Dict[str, Any]:
        """Load a file through the same governed activation path used by upload."""
        from src.platform.runtime_loader import load_dataset
        return load_dataset(self, file_path, file_name)

    def load_from_database(self) -> bool:
        """Restore the durable active local dataset, then fall back to legacy SQLite.

        Canonical dataset artifacts preserve full snapshot histories and therefore
        take precedence over the one-row-per-employee compatibility database.
        """
        from src.platform.runtime_loader import activate_dataframe

        try:
            from src.platform.local_dataset_store import load_dataset_artifact
            from src.platform.workspace import WorkspaceStore

            workspace = WorkspaceStore().get_workspace('local')
            if workspace.active_dataset_id:
                persisted = load_dataset_artifact(workspace.active_dataset_id)
                if persisted is not None and not persisted.empty:
                    # Reconstruct feature availability from the persisted source.
                    feature_flags = {
                        'predictive': 'Attrition' in persisted.columns,
                        'nlp': 'PerformanceText' in persisted.columns and persisted['PerformanceText'].notna().any(),
                    }
                    activate_dataframe(self, persisted, feature_flags=feature_flags)
                    return True
        except Exception as exc:
            logger.warning('Canonical dataset restore unavailable; trying SQLite fallback: %s', exc)

        df = self.data_loader.load_from_database()
        if df is None or df.empty:
            return False
        activate_dataframe(self, df, feature_flags=self.data_loader.features_enabled.copy())
        return True

    def _initialize_engines(self) -> None:
        """Deprecated compatibility hook; initializes read-only runtime only."""
        if self.raw_df is None or self.raw_df.empty:
            return
        from src.platform.runtime_loader import activate_dataframe

        source = self.historical_df if self.historical_df is not None else self.raw_df
        activate_dataframe(self, source, feature_flags=self.features_enabled.copy())

    def has_data(self) -> bool:
        return self.raw_df is not None and not self.raw_df.empty

    def get_employee_by_id(self, employee_id: str) -> Optional[pd.Series]:
        """Legacy lookup retained for non-governed compatibility code."""
        if self.raw_df is None or 'EmployeeID' not in self.raw_df.columns:
            return None
        matches = self.raw_df[self.raw_df['EmployeeID'].astype(str) == str(employee_id)]
        return None if matches.empty else matches.iloc[0]

    def get_employee_risk(self, employee_id: str) -> Optional[Dict[str, Any]]:
        """Legacy lookup; enterprise API does not expose individual risk outputs."""
        if self.risk_scores is None or 'EmployeeID' not in self.risk_scores.columns:
            return None
        matches = self.risk_scores[self.risk_scores['EmployeeID'].astype(str) == str(employee_id)]
        if matches.empty:
            return None
        row = matches.iloc[0]
        result: Dict[str, Any] = {
            'risk_score': float(row['risk_score']),
            'risk_category': row['risk_category'],
        }
        for column in ('uncertainty_lower', 'uncertainty_upper', 'ci_lower', 'ci_upper'):
            if column in row.index and pd.notna(row[column]):
                result[column] = float(row[column])
        return result

    def get_employee_index(self, employee_id: str) -> Optional[int]:
        if self.processed_df is None or 'EmployeeID' not in self.processed_df.columns:
            return None
        matches = self.processed_df[self.processed_df['EmployeeID'].astype(str) == str(employee_id)]
        return None if matches.empty else int(matches.index[0])

    def reset(self) -> None:
        """Reset runtime state and clear the local persistent employee store."""
        try:
            from src.database import get_database
            get_database().clear_all_data()
        except Exception as exc:
            logger.error('Failed to clear database during reset: %s', exc)

        self.historical_df = None
        self.raw_df = None
        self.active_df = None
        self.population_resolution = None
        self.processed_df = None
        self.features_df = None
        self.target_series = None
        self.preprocessing_metadata = None

        for name in (
            'analytics_engine', 'ml_engine', 'compensation_engine', 'succession_engine',
            'team_dynamics_engine', 'fairness_engine', 'vector_engine', 'nlp_engine',
            'llm_client', 'insight_interpreter', 'survival_engine',
            'quality_of_hire_engine', 'structural_engine', 'sentiment_engine',
            'experience_engine', 'scenario_engine',
        ):
            setattr(self, name, None)

        self.enps_df = None
        self.onboarding_df = None
        self.model_metrics = None
        self.risk_scores = None
        self.nlp_results = None
        self.features_enabled = {'predictive': False, 'nlp': False, 'llm': False}


def get_app_state() -> AppState:
    return AppState()
