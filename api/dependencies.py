"""
Shared dependencies for FastAPI routes.

Manages application state including loaded data and initialized engines.
"""

import os
import sys
from typing import Optional, Dict, Any
from functools import lru_cache

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.logger import get_logger

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.analytics_engine import AnalyticsEngine
from src.ml_engine import MLEngine
from src.compensation_engine import CompensationEngine
from src.succession_engine import SuccessionEngine
from src.team_dynamics_engine import TeamDynamicsEngine
from src.fairness_engine import FairnessEngine
from src.vector_engine import VectorEngine
from src.nlp_engine import NLPEngine
from src.llm_client import LLMClient
from src.insight_interpreter import InsightInterpreter
from src.survival_engine import SurvivalEngine
from src.quality_of_hire_engine import QualityOfHireEngine
from src.structural_engine import StructuralEngine
from src.sentiment_engine import SentimentEngine
from src.experience_engine import ExperienceEngine
from src.scenario_engine import ScenarioEngine
from src.utils import load_config
logger = get_logger('api_dependencies')


class AppState:
    """
    Singleton class to manage application state.

    Holds loaded data and initialized engine instances.
    """
    _instance: Optional["AppState"] = None

    def __new__(cls) -> "AppState":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.config = load_config()
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()

        # Data state
        self.raw_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.target_series: Optional[pd.Series] = None
        self.preprocessing_metadata: Optional[Dict[str, Any]] = None

        # Engine instances
        self.analytics_engine: Optional[AnalyticsEngine] = None
        self.ml_engine: Optional[MLEngine] = None
        self.compensation_engine: Optional[CompensationEngine] = None
        self.succession_engine: Optional[SuccessionEngine] = None
        self.team_dynamics_engine: Optional[TeamDynamicsEngine] = None
        self.fairness_engine: Optional[FairnessEngine] = None
        self.vector_engine: Optional[VectorEngine] = None
        self.nlp_engine: Optional[NLPEngine] = None
        self.llm_client: Optional[LLMClient] = None
        self.insight_interpreter: Optional[InsightInterpreter] = None
        self.survival_engine: Optional[SurvivalEngine] = None
        self.quality_of_hire_engine: Optional[QualityOfHireEngine] = None
        self.structural_engine: Optional[StructuralEngine] = None
        self.sentiment_engine: Optional[SentimentEngine] = None
        self.experience_engine: Optional[ExperienceEngine] = None
        self.scenario_engine: Optional[ScenarioEngine] = None

        # Survey data for sentiment analysis
        self.enps_df: Optional[pd.DataFrame] = None
        self.onboarding_df: Optional[pd.DataFrame] = None

        # ML results
        self.model_metrics: Optional[Dict[str, Any]] = None
        self.risk_scores: Optional[pd.DataFrame] = None
        self.nlp_results: Optional[Dict[str, Any]] = None

        # Feature flags
        self.features_enabled = {
            'predictive': False,
            'nlp': False,
            'llm': False
        }

        self._initialized = True

    def load_data(self, file_path: str, file_name: str = "upload") -> Dict[str, Any]:
        """
        Load data from file and initialize engines.

        Args:
            file_path: Path to the data file.
            file_name: Original file name.

        Returns:
            Dictionary with load results.
        """
        # Load and merge data
        result = self.data_loader.load_and_merge(file_path, file_name)
        self.raw_df = result['df']
        self.features_enabled = self.data_loader.features_enabled.copy()

        # Preprocess data
        if 'Attrition' in self.raw_df.columns:
            self.processed_df, self.preprocessing_metadata = self.preprocessor.fit_transform(
                self.raw_df,
                target_column='Attrition'
            )
            self.features_enabled['predictive'] = True
        else:
            self.processed_df, self.preprocessing_metadata = self.preprocessor.fit_transform(
                self.raw_df
            )

        # Initialize engines
        self._initialize_engines()

        return {
            'rows_loaded': len(self.raw_df),
            'columns': list(self.raw_df.columns),
            'features_enabled': self.features_enabled,
            'merge_result': result.get('merge_result'),
            'report': result.get('report')
        }

    def load_from_database(self) -> bool:
        """
        Load data from persistent database.

        Returns:
            True if data was loaded successfully.
        """
        df = self.data_loader.load_from_database()

        if df is None or df.empty:
            return False

        self.raw_df = df
        self.features_enabled = self.data_loader.features_enabled.copy()

        # Preprocess data
        if 'Attrition' in self.raw_df.columns:
            self.processed_df, self.preprocessing_metadata = self.preprocessor.fit_transform(
                self.raw_df,
                target_column='Attrition'
            )
            self.features_enabled['predictive'] = True
        else:
            self.processed_df, self.preprocessing_metadata = self.preprocessor.fit_transform(
                self.raw_df
            )

        # Initialize engines
        self._initialize_engines()

        return True

    def _initialize_engines(self) -> None:
        """Initialize all analysis engines with current data."""
        if self.raw_df is None:
            return

        # Analytics engine (always available)
        self.analytics_engine = AnalyticsEngine(self.raw_df)

        # Compensation engine
        try:
            self.compensation_engine = CompensationEngine(self.raw_df)
        except Exception:
            self.compensation_engine = None

        # ML engine (requires Attrition column)
        if self.features_enabled.get('predictive', False) and self.processed_df is not None:
            try:
                self.ml_engine = MLEngine()

                # Dynamically identify features from Golden Schema
                from src.data_loader import GOLDEN_SCHEMA
                potential_features = GOLDEN_SCHEMA['required'] + GOLDEN_SCHEMA['optional']

                # Prepare features and target
                feature_cols = [
                    c for c in self.processed_df.columns
                    if c in potential_features and c not in ['EmployeeID', 'Attrition', 'PerformanceText', 'RatingHistory', 'HireDate', 'PromotionDate']
                ]

                self.features_df = self.processed_df[feature_cols].select_dtypes(
                    include=['int64', 'float64', 'int32', 'float32']
                )

                # Add any interview dimensions dynamically
                interview_cols = [c for c in self.processed_df.columns if c.startswith('InterviewScore_')]
                if interview_cols:
                    interview_df = self.processed_df[interview_cols].select_dtypes(include=['number'])
                    self.features_df = pd.concat([self.features_df, interview_df], axis=1)

                self.target_series = self.processed_df['Attrition']
                
                # Train model
                self.model_metrics = self.ml_engine.train_model(
                    self.features_df,
                    self.target_series
                )

                # Generate risk scores
                risk_scores = self.ml_engine.predict_risk(self.features_df)
                risk_with_confidence = self.ml_engine.predict_risk_with_confidence(self.features_df)

                self.risk_scores = pd.DataFrame({
                    'EmployeeID': self.processed_df['EmployeeID'].values,
                    'risk_score': risk_scores,
                    'risk_category': [self.ml_engine.get_risk_category(s) for s in risk_scores],
                    'ci_lower': risk_with_confidence['ci_lower'].values,
                    'ci_upper': risk_with_confidence['ci_upper'].values,
                    'confidence_level': risk_with_confidence['confidence_level'].values
                })

            except Exception:
                self.ml_engine = None
                self.model_metrics = None
                self.risk_scores = None

        # Succession engine
        try:
            self.succession_engine = SuccessionEngine(
                self.raw_df,
                self.risk_scores
            )
        except Exception:
            self.succession_engine = None

        # Team dynamics engine
        try:
            self.team_dynamics_engine = TeamDynamicsEngine(self.raw_df)
        except Exception:
            self.team_dynamics_engine = None

        # Fairness engine (requires predictions)
        if self.risk_scores is not None:
            try:
                predictions = self.risk_scores['risk_category'].map(
                    {'High': 1, 'Medium': 1, 'Low': 0}
                ).values
                self.fairness_engine = FairnessEngine(self.raw_df, predictions)
            except Exception:
                self.fairness_engine = None

        # Vector engine (for semantic search)
        if self.features_enabled.get('nlp', False) and 'PerformanceText' in self.raw_df.columns:
            try:
                self.vector_engine = VectorEngine()
                texts = self.raw_df['PerformanceText'].dropna().tolist()
                metadata = []
                for _, row in self.raw_df[self.raw_df['PerformanceText'].notna()].iterrows():
                    metadata.append({
                        'EmployeeID': row['EmployeeID'],
                        'Dept': row.get('Dept', 'Unknown'),
                        'text': row['PerformanceText']
                    })
                self.vector_engine.build_index(texts, metadata)
            except Exception:
                self.vector_engine = None

        # LLM client & NLP Engine
        try:
            self.llm_client = LLMClient()
            if self.llm_client.is_available:
                self.features_enabled['llm'] = True

            # NLP engine (always available with fallback)
            self.nlp_engine = NLPEngine(self.llm_client)

            # Initialize interpreter (works with or without LLM)
            self.insight_interpreter = InsightInterpreter(self.llm_client)
        except Exception:
            self.llm_client = None
            self.nlp_engine = None
            self.insight_interpreter = InsightInterpreter()

        # Survival Analysis engine (requires Tenure and Attrition)
        if 'Tenure' in self.raw_df.columns and 'Attrition' in self.raw_df.columns:
            try:
                self.survival_engine = SurvivalEngine(self.raw_df)
            except Exception as e:
                self.survival_engine = None

        # Quality of Hire engine (requires HireSource or interview scores)
        cols_lower = [c.lower() for c in self.raw_df.columns]
        has_qoh_data = (
            'hiresource' in cols_lower or
            'interviewscore' in cols_lower or
            any(c.startswith('interviewscore_') for c in cols_lower)
        )
        if has_qoh_data:
            try:
                self.quality_of_hire_engine = QualityOfHireEngine(self.raw_df)
                logger.info("QualityOfHireEngine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize QualityOfHireEngine: {e}")
                self.quality_of_hire_engine = None
        else:
            logger.info("QualityOfHireEngine skipped: missing HireSource or interview columns")
            self.quality_of_hire_engine = None

        # Structural engine (requires Tenure and YearsInCurrentRole or ManagerID)
        has_structural_data = (
            'Tenure' in self.raw_df.columns and
            ('YearsInCurrentRole' in self.raw_df.columns or 'ManagerID' in self.raw_df.columns)
        )
        if has_structural_data:
            try:
                self.structural_engine = StructuralEngine(self.raw_df)
            except Exception as e:
                self.structural_engine = None

        # Sentiment engine (initialized with any available survey data)
        # Will be re-initialized when survey data is uploaded
        try:
            self.sentiment_engine = SentimentEngine(
                employee_df=self.raw_df,
                enps_df=self.enps_df,
                onboarding_df=self.onboarding_df
            )
        except Exception as e:
            self.sentiment_engine = None

        # Experience engine (always available - uses unified data)
        try:
            self.experience_engine = ExperienceEngine(self.raw_df)
            logger.info("ExperienceEngine initialized successfully")
        except Exception as e:
            logger.warning(f"ExperienceEngine initialization failed: {e}")
            self.experience_engine = None

        # Scenario engine (uses ML/Survival engines when available)
        try:
            self.scenario_engine = ScenarioEngine(
                employee_df=self.raw_df,
                ml_engine=self.ml_engine,
                survival_engine=self.survival_engine,
                compensation_engine=self.compensation_engine
            )
            logger.info("ScenarioEngine initialized successfully")
        except Exception as e:
            logger.warning(f"ScenarioEngine initialization failed: {e}")
            self.scenario_engine = None

    def has_data(self) -> bool:
        """Check if data is loaded."""
        return self.raw_df is not None and not self.raw_df.empty

    def get_employee_by_id(self, employee_id: str) -> Optional[pd.Series]:
        """Get employee data by ID."""
        if self.raw_df is None:
            return None

        matches = self.raw_df[self.raw_df['EmployeeID'] == employee_id]
        if matches.empty:
            return None

        return matches.iloc[0]

    def get_employee_risk(self, employee_id: str) -> Optional[Dict[str, Any]]:
        """Get risk data for an employee."""
        if self.risk_scores is None:
            return None

        matches = self.risk_scores[self.risk_scores['EmployeeID'] == employee_id]
        if matches.empty:
            return None

        row = matches.iloc[0]
        return {
            'risk_score': float(row['risk_score']),
            'risk_category': row['risk_category'],
            'ci_lower': float(row['ci_lower']) if pd.notna(row['ci_lower']) else None,
            'ci_upper': float(row['ci_upper']) if pd.notna(row['ci_upper']) else None,
            'confidence_level': float(row['confidence_level']) if pd.notna(row['confidence_level']) else None
        }

    def get_employee_index(self, employee_id: str) -> Optional[int]:
        """Get the index of an employee in the processed dataframe."""
        if self.processed_df is None:
            return None

        matches = self.processed_df[self.processed_df['EmployeeID'] == employee_id]
        if matches.empty:
            return None

        return matches.index[0]

    def reset(self) -> None:
        """Reset all state and clear persistent database."""
        # Clear database if available
        try:
            from src.database import get_database
            db = get_database()
            db.clear_all_data()
        except Exception as e:
            from src.logger import get_logger
            get_logger('app_state').error(f"Failed to clear database during reset: {e}")

        self.raw_df = None
        self.processed_df = None
        self.features_df = None
        self.target_series = None
        self.preprocessing_metadata = None
        self.analytics_engine = None
        self.ml_engine = None
        self.compensation_engine = None
        self.succession_engine = None
        self.team_dynamics_engine = None
        self.fairness_engine = None
        self.vector_engine = None
        self.nlp_engine = None
        self.survival_engine = None
        self.quality_of_hire_engine = None
        self.structural_engine = None
        self.sentiment_engine = None
        self.experience_engine = None
        self.scenario_engine = None
        self.enps_df = None
        self.onboarding_df = None
        self.model_metrics = None
        self.risk_scores = None
        self.nlp_results = None
        self.features_enabled = {
            'predictive': False,
            'nlp': False,
            'llm': False
        }


def get_app_state() -> AppState:
    """Get the singleton AppState instance."""
    return AppState()
