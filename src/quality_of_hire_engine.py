"""
Quality of Hire Analysis Engine for PeopleOS.

Correlates pre-hire signals (interview scores, source, assessments) with
post-hire outcomes (performance, retention, promotion) to optimize hiring.

Key Insights Generated:
- "Candidates scoring high on 'Curiosity' have 20% higher performance after 1 year"
- "Employee referrals have 85% retention at 12 months vs 65% for job boards"
- "Technical interview score has weak correlation with actual performance"

Strategic Value:
- Identify which hiring signals actually predict success
- Optimize interview rubrics based on data
- Improve source channel allocation
- Reduce cost-per-quality-hire
"""

import pandas as pd
import numpy as np
from src.population import resolve_current_population
from typing import Dict, List, Any
from datetime import datetime, timedelta
from scipy import stats

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('quality_of_hire_engine')

# Minimum sample sizes
MIN_SAMPLE_FOR_CORRELATION = 20
MIN_SAMPLE_FOR_SOURCE = 10
MIN_COHORT_SIZE = 10
ROLE_MIX_COLUMNS = ('Dept', 'JobLevel', 'JobTitle')


class QualityOfHireEngineError(Exception):
    """Custom exception for quality of hire engine errors."""
    pass


class QualityOfHireEngine:
    """
    Quality of Hire analysis engine.

    Correlates pre-hire indicators with post-hire success metrics
    to identify which hiring signals predict employee performance.

    HR Value:
    - Identify most predictive interview dimensions
    - Rank hiring sources by quality
    - Build data-driven hiring rubrics
    - Calculate ROI by source channel
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize Quality of Hire Engine.

        Args:
            df: DataFrame with employee data including:
                Required:
                - EmployeeID: Unique identifier
                - HireDate: Date of hire
                - LastRating: Performance metric (outcome)

                Recommended pre-hire signals:
                - HireSource: Source of hire (Referral, LinkedIn, etc.)
                - InterviewScore: Overall interview score (1-5)
                - InterviewScore_Technical, _Cultural, _Curiosity
                - AssessmentScore: Pre-employment assessment (0-100)
                - PriorExperienceYears: Years of experience

                Recommended post-hire outcomes:
                - Attrition: 0/1 (for retention analysis)
                - PromotionCount: Number of promotions
                - Tenure: Years at company
        """
        self.df, self.population_resolution = resolve_current_population(df)
        self.config = load_config()
        self.qoh_config = self.config.get('quality_of_hire', {})

        # Load configuration
        self.performance_window = self.qoh_config.get('performance_window_months', 12)
        self.retention_window = self.qoh_config.get('retention_window_months', 12)
        self.min_cohort_size = self.qoh_config.get('min_cohort_size', MIN_COHORT_SIZE)
        self.expected_sources = self.qoh_config.get('sources', [
            'Referral', 'LinkedIn', 'Agency', 'JobBoard', 'Internal', 'Website'
        ])
        self.role_mix_columns = [column for column in ROLE_MIX_COLUMNS if column in self.df.columns]

        # Quality score weights
        self.quality_weights = self.qoh_config.get('quality_score_weights', {
            'performance': 0.4,
            'retention': 0.3,
            'promotion': 0.2,
            'ramp_time': 0.1
        })

        # Check available columns
        self._identify_available_columns()
        for col in ['LastRating', 'Tenure', 'PromotionCount', 'Attrition', *self.prehire_columns]:
            if col in self.df:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if self.has_performance:
            self.df['LastRating'] = self.df['LastRating'].where(self.df['LastRating'].between(1, 5))
        if self.has_promotion:
            self.df['PromotionCount'] = self.df['PromotionCount'].where(self.df['PromotionCount'] >= 0)

        self.warnings: List[str] = []
        self._validate_data()

        logger.info(f"QualityOfHireEngine initialized with {len(df)} employees")

    def _identify_available_columns(self) -> None:
        """Identify which pre-hire and post-hire columns are available."""
        # Pre-hire signals
        self.has_hire_source = 'HireSource' in self.df.columns
        self.has_hire_date = 'HireDate' in self.df.columns
        self.has_interview_score = 'InterviewScore' in self.df.columns
        self.has_assessment = 'AssessmentScore' in self.df.columns
        self.has_prior_experience = 'PriorExperienceYears' in self.df.columns

        # Interview dimensions
        self.interview_dimensions = [
            col for col in self.df.columns
            if col.startswith('InterviewScore_')
        ]

        # Post-hire outcomes
        self.has_performance = 'LastRating' in self.df.columns
        self.has_attrition = 'Attrition' in self.df.columns
        self.has_promotion = 'PromotionCount' in self.df.columns
        self.has_tenure = 'Tenure' in self.df.columns

        # All pre-hire signals
        self.prehire_columns = []
        if self.has_interview_score:
            self.prehire_columns.append('InterviewScore')
        if self.has_assessment:
            self.prehire_columns.append('AssessmentScore')
        if self.has_prior_experience:
            self.prehire_columns.append('PriorExperienceYears')
        self.prehire_columns.extend(self.interview_dimensions)

    def _validate_data(self) -> None:
        """Validate required columns exist and data quality."""
        required = ['EmployeeID']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise QualityOfHireEngineError(f"Missing required columns: {missing}")

        if not self.has_hire_source:
            self._add_warning(
                "HireSource column not found. Source effectiveness analysis unavailable."
            )

        if not self.has_performance:
            self._add_warning(
                "LastRating column not found. Performance correlation analysis unavailable."
            )

        if not self.prehire_columns:
            self._add_warning(
                "No pre-hire signal columns found (InterviewScore, AssessmentScore, etc.). "
                "Pre-hire to post-hire correlation analysis unavailable."
            )

    def _add_warning(self, warning: str) -> None:
        """Add a warning message for HR review."""
        self.warnings.append(warning)
        logger.warning(warning)

    def _check_sample_size(self, n: int, context: str) -> bool:
        """Check if sample size is sufficient."""
        if n < MIN_SAMPLE_FOR_CORRELATION:
            self._add_warning(
                f"{context}: Sample size ({n}) too small for reliable analysis. "
                f"Minimum recommended: {MIN_SAMPLE_FOR_CORRELATION}"
            )
            return False
        return True

    @staticmethod
    def _valid_numeric_mask(df: pd.DataFrame, column: str, low: float, high: float) -> pd.Series:
        """Return a finite, domain-valid mask without changing the source frame."""
        if column not in df.columns:
            return pd.Series(False, index=df.index)
        values = pd.to_numeric(df[column], errors='coerce')
        return values.between(low, high) & np.isfinite(values)

    def _window_eligible_mask(self, df: pd.DataFrame, window_months: int) -> pd.Series:
        """Identify rows with the configured duration exposure for an outcome."""
        if 'Tenure' not in df.columns:
            return pd.Series(False, index=df.index)
        tenure = pd.to_numeric(df['Tenure'], errors='coerce')
        return tenure.ge(float(window_months) / 12) & np.isfinite(tenure)

    def _performance_masks(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Return recorded and performance-window-qualified rating masks."""
        recorded = self._valid_numeric_mask(df, 'LastRating', 1, 5)
        return recorded, recorded & self._window_eligible_mask(df, self.performance_window)

    def _retention_masks(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Return recorded attrition and duration-qualified retention masks."""
        recorded = self._valid_numeric_mask(df, 'Attrition', 0, 1)
        return recorded, recorded & self._window_eligible_mask(df, self.retention_window)

    def _outcome_maturity(self, recorded: int, eligible: int, has_exposure: bool, window_months: int) -> str:
        """Describe whether an outcome is measured at the configured exposure."""
        if not has_exposure:
            return 'unavailable_no_duration_exposure'
        if eligible < MIN_SAMPLE_FOR_SOURCE:
            return f'insufficient_{window_months}mo_exposure'
        if recorded < MIN_SAMPLE_FOR_SOURCE:
            return 'insufficient_observed_outcomes'
        return 'duration_qualified_observed'

    def _role_mix(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Return aggregate role composition percentages, never employee rows."""
        mix: Dict[str, Dict[str, float]] = {}
        for column in self.role_mix_columns:
            values = df[column].where(df[column].notna()).astype('string').str.strip()
            values = values[values.ne('') & values.ne('<NA>')]
            if values.empty:
                continue
            counts = values.value_counts().sort_index()
            mix[column] = {str(key): round(float(value / len(df)), 3) for key, value in counts.items()}
        return mix

    def _configured_quality_weights(self) -> Dict[str, float]:
        """Return configured weights for evidence accounting, including unsupported components."""
        return {key: float(value) for key, value in self.quality_weights.items()}

    def calculate_source_effectiveness(self) -> pd.DataFrame:
        """
        Analyze effectiveness of each hiring source.

        Compares sources on multiple dimensions:
        - Average performance rating
        - Retention rate (if attrition data available)
        - Promotion rate (if promotion data available)
        - Time to productivity (if available)

        Returns:
            DataFrame with source effectiveness metrics:
            - HireSource
            - hire_count
            - avg_performance
            - retention_rate_12mo
            - quality_score (composite)

        HR Action:
        - Increase investment in high-quality sources
        - Review or terminate low-performing agency relationships
        - Expand referral bonus programs if referrals perform well
        """
        if not self.has_hire_source:
            return pd.DataFrame()

        df = self.df.copy()
        results = []
        quality_weights = self._quality_comparison_weights()
        configured_weights = self._configured_quality_weights()
        sources = (
            df['HireSource'].astype('string').str.strip().replace('', pd.NA).fillna('Unknown')
        )

        for source in sources.unique():
            # Missing source provenance is a real cohort, not an empty
            # category. Keeping it visible prevents sources_analyzed and
            # percentages from silently dropping rows from the denominator.
            source_df = df.loc[sources == source].copy()
            n = len(source_df)

            if n < MIN_SAMPLE_FOR_SOURCE:
                continue

            result = {
                'HireSource': source,
                'hire_count': n,
                'total_hires': n,
                'pct_of_total': round(n / len(df) * 100, 1)
            }

            # Performance metrics
            if self.has_performance:
                performance_recorded, performance_eligible = self._performance_masks(source_df)
                performance_values = source_df.loc[performance_eligible, 'LastRating']
                result['avg_performance'] = round(performance_values.mean(), 2) if len(performance_values) else None
                result['high_performers'] = int((performance_values >= 4.0).sum())
                result['high_performer_rate'] = round(
                    result['high_performers'] / len(performance_values) * 100, 1
                ) if len(performance_values) else None
                result['performance_recorded_observations'] = int(performance_recorded.sum())
                result['performance_observations'] = int(performance_eligible.sum())
                result['performance_coverage'] = round(result['performance_recorded_observations'] / n, 3)
                result['performance_window_observations'] = int(performance_eligible.sum())
                result['performance_window_coverage'] = round(result['performance_observations'] / n, 3)
                result['performance_maturity'] = self._outcome_maturity(
                    result['performance_recorded_observations'],
                    result['performance_observations'],
                    self.has_tenure,
                    self.performance_window,
                )

            # Retention metrics
            if self.has_attrition:
                retention_recorded, retention_eligible = self._retention_masks(source_df)
                retention_values = source_df.loc[retention_eligible, 'Attrition']
                result['retention_recorded_observations'] = int(retention_recorded.sum())
                result['retention_eligible_hires'] = int(self._window_eligible_mask(source_df, self.retention_window).sum()) if self.has_tenure else None
                result['retention_observations'] = int(retention_eligible.sum())
                result['outcome_observations'] = result['retention_observations']
                result['retention_recorded_coverage'] = round(result['retention_recorded_observations'] / n, 3)
                result['retention_coverage'] = round(result['retention_observations'] / n, 3)
                result['retention_maturity'] = self._outcome_maturity(
                    result['retention_recorded_observations'],
                    result['retention_observations'],
                    self.has_tenure,
                    self.retention_window,
                )
                result['attrition_count'] = int(retention_values.sum()) if len(retention_values) else None
                result['retention_rate'] = round(
                    1 - retention_values.mean(), 3
                ) if len(retention_values) else None
                result['retention_rate_pct'] = round(result['retention_rate'] * 100, 1) if result['retention_rate'] is not None else None

            # Promotion metrics
            if self.has_promotion:
                result['avg_promotions'] = round(source_df['PromotionCount'].mean(), 2)
                result['promoted_count'] = int((source_df['PromotionCount'] > 0).sum())
                result['promotion_rate'] = round(
                    result['promoted_count'] / source_df['PromotionCount'].count() * 100, 1
                ) if source_df['PromotionCount'].count() else None
                result['promotion_observations'] = int(source_df['PromotionCount'].count())

            # Tenure metrics
            if self.has_tenure:
                result['avg_tenure'] = round(source_df['Tenure'].mean(), 2)

            # Pre-hire scores (for reference)
            if self.has_interview_score:
                result['avg_interview_score'] = round(
                    source_df['InterviewScore'].mean(), 2
                )

            # Calculate composite quality score
            quality_score = self._calculate_quality_score(source_df)
            result['quality_score'] = round(quality_score, 1)
            _, observations = self._quality_measurements(source_df)
            result['quality_components'] = list(quality_weights)
            result['quality_weights'] = quality_weights
            result['effective_quality_weights'] = quality_weights
            result['configured_quality_weights'] = configured_weights
            result['excluded_quality_components'] = [
                key for key, weight in configured_weights.items()
                if weight > 0 and key not in quality_weights
            ]
            result['component_observations'] = observations
            result['component_coverage'] = {key: count / n for key, count in observations.items()}
            result['minimum_component_observations'] = MIN_SAMPLE_FOR_SOURCE
            result['quality_unavailable_reason'] = None
            result['quality_semantics'] = 'retrospective_heuristic_fixed_dataset_components_not_validated_hire_quality'
            result['quality_claim'] = 'descriptive_observed_composite_not_hiring_effectiveness'
            result['quality_comparison_status'] = 'comparable_within_effective_construct' if np.isfinite(quality_score) else 'insufficient_component_support'
            result['quality_comparison_basis'] = 'same dataset-wide effective components and weights; every scored source must meet the component floor'
            result['role_mix'] = self._role_mix(source_df)
            result['role_mix_columns'] = list(result['role_mix'])

            # Missing outcome measurements do not earn a failing grade.
            if not np.isfinite(quality_score):
                result['quality_score'] = None
                result['grade'] = 'Unavailable'
                unsupported = [key for key in quality_weights if observations[key] < MIN_SAMPLE_FOR_SOURCE]
                result['quality_unavailable_reason'] = (
                    'Insufficient measured observations for: ' + ', '.join(unsupported)
                    if quality_weights else 'No outcome component has sufficient measured support in this dataset'
                )
                result['recommendation'] = 'Collect valid outcome measurements before comparing source composites'
            elif quality_score >= 80:
                result['grade'] = 'A'
                result['recommendation'] = 'Increase investment'
            elif quality_score >= 65:
                result['grade'] = 'B'
                result['recommendation'] = 'Maintain current level'
            elif quality_score >= 50:
                result['grade'] = 'C'
                result['recommendation'] = 'Review and optimize'
            else:
                result['grade'] = 'D'
                result['recommendation'] = 'Consider reducing or eliminating'
            if np.isfinite(quality_score):
                result['recommendation'] = 'Compare measured coverage, role mix and exposure; validate prospectively before changing sourcing decisions'

            results.append(result)

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df = result_df.sort_values('quality_score', ascending=False)

        return result_df

    def _quality_measurements(self, df: pd.DataFrame):
        """Return duration-qualified component scores and their measured counts."""
        components = {}
        observations = {'performance': 0, 'retention': 0, 'promotion': 0}
        masks = {
            'performance': self._performance_masks(df)[1],
            'retention': self._retention_masks(df)[1],
            'promotion': self._valid_numeric_mask(df, 'PromotionCount', 0, float('inf')),
        }
        columns = {'performance': 'LastRating', 'retention': 'Attrition', 'promotion': 'PromotionCount'}
        for key, column in columns.items():
            if column not in df:
                continue
            values = pd.to_numeric(df.loc[masks[key], column], errors='coerce')
            observations[key] = int(len(values))
            if values.empty:
                continue
            mean = float(values.mean())
            components[key] = ((mean - 1) / 4 * 100 if key == 'performance'
                               else (1 - mean) * 100 if key == 'retention'
                               else min(mean / 2, 1) * 100)
        return components, observations

    def _quality_comparison_weights(self) -> Dict[str, float]:
        """Define one construct for this dataset, never reweight per source.

        Entirely unmeasured components are omitted once for the dataset. A
        partially measured component remains required so weak support cannot
        silently turn into a different composite construct for one source.
        The minimum is a reporting guard, not evidence of statistical validity.
        """
        _, observations = self._quality_measurements(self.df)
        weights = {key: float(self.quality_weights.get(key, 0)) for key in observations}
        if any(not np.isfinite(w) or w < 0 for w in weights.values()):
            raise QualityOfHireEngineError('Quality weights must be finite and non-negative')
        weights = {key: weight for key, weight in weights.items()
                   if weight > 0 and observations[key] > 0}
        total = sum(weights.values())
        return {key: weight / total for key, weight in weights.items()} if total else {}

    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Fixed dataset component weights; insufficient cohort evidence is missing."""
        components, observations = self._quality_measurements(df)
        weights = self._quality_comparison_weights()
        if not weights or any(observations[key] < MIN_SAMPLE_FOR_SOURCE for key in weights):
            return np.nan
        return sum(components[key] * weight for key, weight in weights.items())

    def correlate_prehire_posthire(
        self,
        outcome_column: str = 'LastRating'
    ) -> Dict[str, Any]:
        """
        Calculate correlations between pre-hire signals and post-hire outcomes.

        This is the core analysis for optimizing interview processes.

        Args:
            outcome_column: Post-hire metric to correlate against

        Returns:
            Dictionary with correlation analysis:
            {
                'correlations': [
                    {'predictor': 'InterviewScore_Curiosity',
                     'correlation': 0.42, 'p_value': 0.001,
                     'interpretation': 'Strong positive predictor'}
                ],
                'best_predictors': [...],
                'non_predictors': [...]
            }

        HR Action:
        - Increase weight of highly predictive interview dimensions
        - Consider removing non-predictive assessments
        - Retrain interviewers on predictive traits
        """
        if outcome_column not in self.df.columns:
            return {
                'available': False,
                'reason': f'Outcome column {outcome_column} not found'
            }

        if not self.prehire_columns:
            return {
                'available': False,
                'reason': 'No pre-hire signal columns available'
            }

        df = self.df.copy()
        if outcome_column == 'LastRating':
            outcome_valid = self._performance_masks(df)[1]
            outcome_recorded = int(self._performance_masks(df)[0].sum())
            outcome_maturity = self._outcome_maturity(
                outcome_recorded,
                int(outcome_valid.sum()),
                self.has_tenure,
                self.performance_window,
            )
        elif outcome_column == 'Attrition':
            outcome_valid = self._retention_masks(df)[1]
            outcome_recorded = int(self._retention_masks(df)[0].sum())
            outcome_maturity = self._outcome_maturity(
                outcome_recorded,
                int(outcome_valid.sum()),
                self.has_tenure,
                self.retention_window,
            )
        else:
            outcome_valid = self._valid_numeric_mask(df, outcome_column, -float('inf'), float('inf'))
            outcome_recorded = int(outcome_valid.sum())
            outcome_maturity = 'observed_outcome_window_not_defined'
        results = {
            'available': True,
            'outcome_column': outcome_column,
            'outcome_observations': int(outcome_valid.sum()),
            'outcome_recorded_observations': outcome_recorded,
            'outcome_maturity': outcome_maturity,
            'correlations': [],
            'best_predictors': [],
            'non_predictors': [],
            'recommendations': [],
            'measurement_gaps': [],
        }

        for predictor in self.prehire_columns:
            predictor_values = pd.to_numeric(df[predictor], errors='coerce')
            predictor_valid = predictor_values.notna() & np.isfinite(predictor_values)
            valid_mask = predictor_valid & outcome_valid
            valid_df = pd.DataFrame({predictor: predictor_values, outcome_column: pd.to_numeric(df[outcome_column], errors='coerce')}).loc[valid_mask]
            gap_reason = None
            if len(valid_df) < MIN_SAMPLE_FOR_CORRELATION:
                gap_reason = 'insufficient_paired_observations'
            elif valid_df[predictor].nunique() < 2:
                gap_reason = 'predictor_has_no_variation'
            elif valid_df[outcome_column].nunique() < 2:
                gap_reason = 'outcome_has_no_variation'
            if gap_reason:
                results['measurement_gaps'].append({
                    'predictor': predictor,
                    'display_name': predictor.replace('InterviewScore_', '').replace('_', ' '),
                    'paired_observations': int(len(valid_df)),
                    'predictor_observations': int(predictor_valid.sum()),
                    'outcome_observations': int(outcome_valid.sum()),
                    'minimum_paired_observations': MIN_SAMPLE_FOR_CORRELATION,
                    'reason': gap_reason,
                })
                continue

            # Calculate Pearson correlation
            correlation, p_value = stats.pearsonr(
                valid_df[predictor],
                valid_df[outcome_column]
            )

            # Interpret correlation strength
            abs_corr = abs(correlation)
            if abs_corr >= 0.5:
                strength = 'Strong'
            elif abs_corr >= 0.3:
                strength = 'Moderate'
            elif abs_corr >= 0.1:
                strength = 'Weak'
            else:
                strength = 'Negligible'

            direction = 'positive' if correlation > 0 else 'negative'
            is_significant = p_value < 0.05

            # Calculate effect size (how much outcome differs per unit of predictor)
            if valid_df[predictor].std() > 0:
                # Standardized regression coefficient approximation
                effect_size = correlation * (valid_df[outcome_column].std() / valid_df[predictor].std())
            else:
                effect_size = 0

            # Clean predictor name for display
            display_name = predictor.replace('InterviewScore_', '').replace('_', ' ')

            corr_result = {
                'predictor': predictor,
                'display_name': display_name,
                'correlation': round(correlation, 3),
                'abs_correlation': round(abs_corr, 3),
                'p_value': round(p_value, 4),
                'is_significant': is_significant,
                'strength': strength,
                'direction': direction,
                'sample_size': len(valid_df),
                'interpretation': f"{strength} {direction} predictor" + (" (significant)" if is_significant else " (not significant)")
            }

            corr_result['insight'] = None
            corr_result['interpretation'] = f"{strength} {direction} observed association; no causal or percentage uplift is estimated."
            corr_result['multiple_testing_adjusted'] = False

            results['correlations'].append(corr_result)

        # Sort by absolute correlation
        results['correlations'].sort(key=lambda x: x['abs_correlation'], reverse=True)

        # Identify best predictors
        results['best_predictors'] = [
            c for c in results['correlations']
            if c['is_significant'] and c['abs_correlation'] >= 0.2
        ]

        # Identify non-predictors
        results['non_predictors'] = [
            c for c in results['correlations']
            if not c['is_significant'] or c['abs_correlation'] < 0.1
        ]

        # Generate recommendations
        if results['best_predictors']:
            top_predictor = results['best_predictors'][0]
            results['recommendations'].append(
                f"REVIEW: '{top_predictor['display_name']}' has the strongest observed association "
                f"of {outcome_column} (r={top_predictor['correlation']:.2f}). "
                "Validate it on future independent cohorts before changing selection weights."
            )

        if results['non_predictors']:
            weak_predictors = [c['display_name'] for c in results['non_predictors'][:3]]
            results['recommendations'].append(
                f"REVIEW: These signals show weak/no correlation with performance: "
                f"{', '.join(weak_predictors)}. Absence of sample significance does not establish absence of usefulness."
            )

        if results['measurement_gaps']:
            results['recommendations'].append(
                f"COLLECT: {len(results['measurement_gaps'])} pre-hire signal(s) lack the minimum "
                f"{MIN_SAMPLE_FOR_CORRELATION} paired observations or outcome variation; no relationship is reported for them."
            )

        return results

    def get_hiring_insights(self) -> Dict[str, Any]:
        """
        Generate comprehensive hiring insights.

        Combines source effectiveness and correlation analysis into
        actionable recommendations.

        Returns:
            Dictionary with strategic hiring insights.
        """
        results = {
            'summary': {},
            'top_sources': [],
            'top_predictors': [],
            'red_flags': [],
            'recommendations': [],
            'roi_analysis': {}
        }

        df = self.df.copy()
        performance_recorded, performance_eligible = self._performance_masks(df)
        retention_recorded, retention_eligible = self._retention_masks(df)

        # Summary statistics
        results['summary'] = {
            'total_employees': len(df),
            'total_hires': len(df),
            'sources_analyzed': df['HireSource'].nunique() if self.has_hire_source else 0,
            'prehire_signals_available': len(self.prehire_columns),
            'avg_performance': round(df.loc[performance_eligible, 'LastRating'].mean(), 2) if performance_eligible.any() else None,
            'overall_retention': round(1 - df.loc[retention_eligible, 'Attrition'].mean(), 3) if retention_eligible.any() else None,
            'performance_recorded_observations': int(performance_recorded.sum()),
            'performance_observations': int(performance_eligible.sum()),
            'performance_coverage': round(float(performance_recorded.sum()) / len(df), 3) if len(df) else 0,
            'performance_window_coverage': round(float(performance_eligible.sum()) / len(df), 3) if len(df) else 0,
            'retention_recorded_observations': int(retention_recorded.sum()),
            'retention_observations': int(retention_eligible.sum()),
            'retention_recorded_coverage': round(float(retention_recorded.sum()) / len(df), 3) if len(df) else 0,
            'retention_coverage': round(float(retention_eligible.sum()) / len(df), 3) if len(df) else 0,
            'performance_maturity': self._outcome_maturity(int(performance_recorded.sum()), int(performance_eligible.sum()), self.has_tenure, self.performance_window),
            'retention_maturity': self._outcome_maturity(int(retention_recorded.sum()), int(retention_eligible.sum()), self.has_tenure, self.retention_window),
            'performance_window_months': self.performance_window,
            'retention_window_months': self.retention_window,
            'role_mix_columns': list(self.role_mix_columns),
        }

        # Source analysis
        source_df = self.calculate_source_effectiveness()
        if not source_df.empty:
            # Top sources
            top_sources = source_df.dropna(subset=['quality_score']).head(3).to_dict('records')
            results['top_sources'] = top_sources

            # Red flags - sources with low quality or retention
            if 'retention_rate' in source_df.columns:
                low_retention = source_df[(source_df['retention_rate'] < 0.7)
                                          & (source_df['outcome_observations'] >= MIN_SAMPLE_FOR_SOURCE)]
                for _, row in low_retention.iterrows():
                    results['red_flags'].append({
                        'type': 'Low Retention Source',
                        'source': row['HireSource'],
                        'retention_rate': row['retention_rate'],
                        'message': f"{row['HireSource']} has only {row['retention_rate_pct']}% retention"
                    })

            if 'quality_score' in source_df.columns:
                low_quality = source_df[source_df['quality_score'] < 50]
                for _, row in low_quality.iterrows():
                    if row['HireSource'] not in [rf['source'] for rf in results['red_flags']]:
                        results['red_flags'].append({
                            'type': 'Low Quality Source',
                            'source': row['HireSource'],
                            'quality_score': row['quality_score'],
                            'message': f"{row['HireSource']} has low quality score ({row['quality_score']})"
                        })

        # Correlation analysis
        if self.has_performance and self.prehire_columns:
            corr_results = self.correlate_prehire_posthire()
            if corr_results.get('available'):
                results['top_predictors'] = corr_results.get('best_predictors', [])[:5]
                results['recommendations'].extend(corr_results.get('recommendations', []))

        # ROI analysis (simplified)
        if not source_df.empty and self.has_performance:
            # Calculate relative quality by source
            overall_quality = df.loc[performance_eligible, 'LastRating'].mean() if performance_eligible.any() else None

            for _, row in source_df.iterrows():
                if (row.get('performance_observations', 0) >= MIN_SAMPLE_FOR_SOURCE
                        and 'avg_performance' in row and pd.notna(row['avg_performance']) and pd.notna(overall_quality)):
                    quality_diff = row['avg_performance'] - overall_quality
                    roi_indicator = 'Above Average' if quality_diff > 0.2 else (
                        'Below Average' if quality_diff < -0.2 else 'Average'
                    )

                    results['roi_analysis'][row['HireSource']] = {
                        'source': row['HireSource'],
                        'quality_vs_average': round(quality_diff, 2),
                        'roi_indicator': roi_indicator,
                        'metric_semantics': 'relative_recorded_rating_not_return_on_investment',
                        'claim_boundary': 'descriptive_difference_not_hiring_effectiveness_or_financial_roi',
                        'recommendation': row.get('recommendation', '')
                    }

        # Generate final recommendations
        if results['top_sources']:
            top_source = results['top_sources'][0]
            results['recommendations'].append(
                f"Review source composition and follow-up: {top_source['HireSource']} has the highest measured heuristic composite "
                f"({top_source['quality_score']}) in this sample. Validate independently before changing sourcing decisions."
            )

        if results['red_flags']:
            results['recommendations'].append(
                f"INVESTIGATE: {len(results['red_flags'])} source(s) flagged for review. "
                "See red_flags for details."
            )

        return results

    def analyze_cohort_performance(
        self,
        cohort_column: str = 'HireSource',
        min_tenure_months: int = 6
    ) -> pd.DataFrame:
        """
        Analyze performance trajectory by cohort.

        Useful for comparing how different cohorts perform over time.

        Args:
            cohort_column: Column to group by (default: HireSource)
            min_tenure_months: Minimum tenure to include in analysis

        Returns:
            DataFrame with cohort performance metrics.
        """
        if cohort_column not in self.df.columns:
            return pd.DataFrame()

        if not self.has_tenure:
            return pd.DataFrame()  # Cannot attest the requested exposure threshold.
        df = self.df.copy()

        # Filter by minimum tenure
        if self.has_tenure:
            min_tenure_years = min_tenure_months / 12
            df = df[df['Tenure'] >= min_tenure_years]

        if len(df) < MIN_SAMPLE_FOR_CORRELATION:
            return pd.DataFrame()

        results = []

        for cohort in df[cohort_column].unique():
            cohort_df = df[df[cohort_column] == cohort]
            total_cohort_df = self.df[self.df[cohort_column] == cohort]

            if len(cohort_df) < MIN_SAMPLE_FOR_SOURCE:
                continue

            result = {
                cohort_column: cohort,
                'count': len(cohort_df),
                'total_hires': len(total_cohort_df),
                'mature_hires': len(cohort_df),
            }

            # Performance metrics
            if self.has_performance:
                performance_recorded, performance_eligible = self._performance_masks(cohort_df)
                performance_values = cohort_df.loc[performance_eligible, 'LastRating']
                result['avg_performance'] = round(performance_values.mean(), 2) if len(performance_values) else None
                result['performance_std'] = round(performance_values.std(), 2) if len(performance_values) else None
                result['high_performer_pct'] = round(
                    (performance_values >= 4.0).mean() * 100, 1
                ) if len(performance_values) else None
                result['low_performer_pct'] = round(
                    (performance_values <= 2.5).mean() * 100, 1
                ) if len(performance_values) else None
                result['performance_recorded_observations'] = int(performance_recorded.sum())
                result['performance_observations'] = int(performance_eligible.sum())
                result['performance_coverage'] = round(float(performance_recorded.sum()) / len(total_cohort_df), 3) if len(total_cohort_df) else 0
                result['performance_maturity'] = self._outcome_maturity(
                    result['performance_recorded_observations'],
                    result['performance_observations'],
                    self.has_tenure,
                    self.performance_window,
                )
            # Observed retained share; not a survival-adjusted retention rate.
            if self.has_attrition:
                recorded, eligible = self._retention_masks(cohort_df)
                known = cohort_df.loc[eligible, 'Attrition']
                result['retention_recorded_observations'] = int(recorded.sum())
                result['retention_eligible_hires'] = int(self._window_eligible_mask(cohort_df, self.retention_window).sum()) if self.has_tenure else None
                result['retention_observations'] = int(eligible.sum())
                result['retention_recorded_coverage'] = round(float(recorded.sum()) / len(total_cohort_df), 3) if len(total_cohort_df) else 0
                result['retention_coverage'] = round(float(eligible.sum()) / len(total_cohort_df), 3) if len(total_cohort_df) else 0
                result['retention_rate'] = round(1 - float(known.mean()), 3) if len(known) else None
                result['outcome_observations'] = result['retention_observations']
                result['retention_maturity'] = self._outcome_maturity(
                    result['retention_recorded_observations'],
                    result['retention_observations'],
                    self.has_tenure,
                    self.retention_window,
                )

            # Tenure
            if self.has_tenure:
                result['avg_tenure'] = round(cohort_df['Tenure'].mean(), 2)

            results.append(result)

        result_df = pd.DataFrame(results)
        if not result_df.empty and 'avg_performance' in result_df.columns:
            result_df = result_df.sort_values('avg_performance', ascending=False)

        return result_df

    def get_new_hire_risk_assessment(self, months_since_hire: int = 6) -> pd.DataFrame:
        """Retired: pre-hire heuristics are not validated individual risk models."""
        self._add_warning(
            'Individual new-hire risk assessment is unavailable; pre-hire signals may only be used for aggregate observational analysis.'
        )
        return pd.DataFrame()

    def analyze_all(self) -> Dict[str, Any]:
        """
        Run complete quality of hire analysis.

        Returns:
            Dictionary with all quality of hire results:
            - source_effectiveness: Source-by-source metrics
            - correlations: Pre-hire to post-hire correlations
            - insights: Strategic hiring insights
            - cohort_analysis: Performance by cohort
            - new_hire_risks: Risk assessment for recent hires
            - recommendations: Actionable insights
            - warnings: Data quality warnings
        """
        logger.info("Running full quality of hire analysis")

        # Clear warnings
        self.warnings = []

        results = {
            'source_effectiveness': [],
            'correlations': {},
            'retention_correlations': {},
            'insights': {},
            'cohort_analysis': [],
            'new_hire_risks': [],
            'summary': {},
            'recommendations': [],
            'warnings': []
        }

        # Source effectiveness
        source_df = self.calculate_source_effectiveness()
        if not source_df.empty:
            results['source_effectiveness'] = source_df.to_dict('records')

        # Correlations
        if self.has_performance:
            corr_results = self.correlate_prehire_posthire('LastRating')
            results['correlations'] = corr_results

            # Also check correlation with retention
            if self.has_attrition:
                retention_corr = self.correlate_prehire_posthire('Attrition')
                results['retention_correlations'] = retention_corr

        # Insights
        results['insights'] = self.get_hiring_insights()

        # Cohort analysis
        cohort_df = self.analyze_cohort_performance()
        if not cohort_df.empty:
            results['cohort_analysis'] = cohort_df.to_dict('records')

        # New hire risks
        new_hire_df = self.get_new_hire_risk_assessment()
        if not new_hire_df.empty:
            results['new_hire_risks'] = new_hire_df.to_dict('records')

        # Summary
        results['summary'] = {
            'total_employees': len(self.df),
            'total_hires': len(self.df),
            'has_hire_source': self.has_hire_source,
            'has_interview_scores': self.has_interview_score,
            'has_assessment': self.has_assessment,
            'prehire_signals_count': len(self.prehire_columns),
            'sources_analyzed': len(results['source_effectiveness']),
            'best_source': next((row['HireSource'] for row in results['source_effectiveness'] if row.get('quality_score') is not None and pd.notna(row['quality_score'])), None),
            'best_source_semantics': 'descriptive_composite_only_not_hiring_effectiveness',
            'top_predictor': results['correlations'].get('best_predictors', [{}])[0].get('predictor') if results.get('correlations', {}).get('best_predictors') else None,
            'new_hires_at_risk': len([r for r in results['new_hire_risks'] if r.get('risk_category') in ['High', 'Medium']]),
        }
        performance_recorded, performance_eligible = self._performance_masks(self.df)
        retention_recorded, retention_eligible = self._retention_masks(self.df)
        results['summary'].update({
            'performance_recorded_observations': int(performance_recorded.sum()),
            'performance_observations': int(performance_eligible.sum()),
            'performance_coverage': round(float(performance_recorded.sum()) / len(self.df), 3) if len(self.df) else 0,
            'performance_window_coverage': round(float(performance_eligible.sum()) / len(self.df), 3) if len(self.df) else 0,
            'retention_recorded_observations': int(retention_recorded.sum()),
            'retention_observations': int(retention_eligible.sum()),
            'retention_recorded_coverage': round(float(retention_recorded.sum()) / len(self.df), 3) if len(self.df) else 0,
            'retention_coverage': round(float(retention_eligible.sum()) / len(self.df), 3) if len(self.df) else 0,
            'performance_maturity': self._outcome_maturity(int(performance_recorded.sum()), int(performance_eligible.sum()), self.has_tenure, self.performance_window),
            'retention_maturity': self._outcome_maturity(int(retention_recorded.sum()), int(retention_eligible.sum()), self.has_tenure, self.retention_window),
            'performance_window_months': self.performance_window,
            'retention_window_months': self.retention_window,
            'role_mix_columns': list(self.role_mix_columns),
        })

        # Compile recommendations
        if results['insights'].get('recommendations'):
            results['recommendations'].extend(results['insights']['recommendations'])

        if results['correlations'].get('recommendations'):
            results['recommendations'].extend(results['correlations']['recommendations'])

        # Add warnings
        results['warnings'] = self.warnings.copy()
        if results['summary']['performance_maturity'] != 'duration_qualified_observed':
            results['warnings'].append(
                'Performance evidence is not fully duration-qualified for the configured '
                f"{self.performance_window}-month window; recorded ratings and qualified observations are reported separately."
            )
        if results['summary']['retention_maturity'] != 'duration_qualified_observed':
            results['warnings'].append(
                'Retention is unavailable or immature for the configured '
                f"{self.retention_window}-month window; current attrition status is not treated as duration-qualified retention."
            )
        results['warnings'].append(
            'Source scores are descriptive composites of observed outcomes. They do not measure hiring effectiveness, source ROI, causation or expected future performance.'
        )
        results['warnings'].extend([
            'Performance and retention denominators are duration-qualified observations; total hires and recorded-but-unqualified outcomes remain separate.',
            'Prospective independent-cohort validation is required before changing hiring criteria, source allocation or selection weights.',
        ])

        logger.info(f"Quality of hire analysis complete. Warnings: {len(self.warnings)}")
        return results
