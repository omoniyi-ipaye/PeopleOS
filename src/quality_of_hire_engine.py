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

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from scipy import stats

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('quality_of_hire_engine')

# Minimum sample sizes
MIN_SAMPLE_FOR_CORRELATION = 20
MIN_SAMPLE_FOR_SOURCE = 10
MIN_COHORT_SIZE = 10


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
        self.df = df.copy()
        self.config = load_config()
        self.qoh_config = self.config.get('quality_of_hire', {})

        # Load configuration
        self.performance_window = self.qoh_config.get('performance_window_months', 12)
        self.retention_window = self.qoh_config.get('retention_window_months', 12)
        self.min_cohort_size = self.qoh_config.get('min_cohort_size', MIN_COHORT_SIZE)
        self.expected_sources = self.qoh_config.get('sources', [
            'Referral', 'LinkedIn', 'Agency', 'JobBoard', 'Internal', 'Website'
        ])

        # Quality score weights
        self.quality_weights = self.qoh_config.get('quality_score_weights', {
            'performance': 0.4,
            'retention': 0.3,
            'promotion': 0.2,
            'ramp_time': 0.1
        })

        # Check available columns
        self._identify_available_columns()

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

        for source in df['HireSource'].unique():
            source_df = df[df['HireSource'] == source]
            n = len(source_df)

            if n < MIN_SAMPLE_FOR_SOURCE:
                continue

            result = {
                'HireSource': source,
                'hire_count': n,
                'pct_of_total': round(n / len(df) * 100, 1)
            }

            # Performance metrics
            if self.has_performance:
                result['avg_performance'] = round(source_df['LastRating'].mean(), 2)
                result['high_performers'] = int((source_df['LastRating'] >= 4.0).sum())
                result['high_performer_rate'] = round(
                    result['high_performers'] / n * 100, 1
                )

            # Retention metrics
            if self.has_attrition:
                result['attrition_count'] = int(source_df['Attrition'].sum())
                result['retention_rate'] = round(
                    1 - source_df['Attrition'].mean(), 3
                )
                result['retention_rate_pct'] = round(result['retention_rate'] * 100, 1)

            # Promotion metrics
            if self.has_promotion:
                result['avg_promotions'] = round(source_df['PromotionCount'].mean(), 2)
                result['promoted_count'] = int((source_df['PromotionCount'] > 0).sum())
                result['promotion_rate'] = round(
                    result['promoted_count'] / n * 100, 1
                )

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

            # Determine grade
            if quality_score >= 80:
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

            results.append(result)

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df = result_df.sort_values('quality_score', ascending=False)

        return result_df

    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate composite quality score for a cohort (0-100 scale).

        Weights:
        - Performance: 40%
        - Retention: 30%
        - Promotion: 20%
        - Ramp time: 10%
        """
        score = 0
        weights = self.quality_weights
        total_weight = 0

        # Performance component (0-100)
        if self.has_performance:
            # Convert 1-5 rating to 0-100 scale
            avg_rating = df['LastRating'].mean()
            performance_score = ((avg_rating - 1) / 4) * 100  # 1=0, 5=100
            score += performance_score * weights.get('performance', 0.4)
            total_weight += weights.get('performance', 0.4)

        # Retention component (0-100)
        if self.has_attrition:
            retention_rate = 1 - df['Attrition'].mean()
            retention_score = retention_rate * 100
            score += retention_score * weights.get('retention', 0.3)
            total_weight += weights.get('retention', 0.3)

        # Promotion component (0-100)
        if self.has_promotion:
            # Assume 0-3 promotions range, normalize to 0-100
            avg_promotions = df['PromotionCount'].mean()
            promotion_score = min(avg_promotions / 2, 1) * 100  # Cap at 2 promotions = 100
            score += promotion_score * weights.get('promotion', 0.2)
            total_weight += weights.get('promotion', 0.2)

        # Normalize by actual weights used
        if total_weight > 0:
            score = score / total_weight

        return score

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
        results = {
            'available': True,
            'outcome_column': outcome_column,
            'correlations': [],
            'best_predictors': [],
            'non_predictors': [],
            'recommendations': []
        }

        for predictor in self.prehire_columns:
            # Get valid pairs (non-null for both)
            valid_df = df[[predictor, outcome_column]].dropna()

            if len(valid_df) < MIN_SAMPLE_FOR_CORRELATION:
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

            # Generate insight
            if is_significant and abs_corr >= 0.2:
                pct_effect = round(abs_corr * 100, 0)
                if correlation > 0:
                    corr_result['insight'] = (
                        f"Candidates scoring high on '{display_name}' tend to have "
                        f"~{pct_effect}% higher {outcome_column}"
                    )
                else:
                    corr_result['insight'] = (
                        f"Candidates scoring high on '{display_name}' tend to have "
                        f"~{pct_effect}% lower {outcome_column}"
                    )

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
                f"PRIORITIZE: '{top_predictor['display_name']}' is the strongest predictor "
                f"of {outcome_column} (r={top_predictor['correlation']:.2f}). "
                "Increase its weight in hiring decisions."
            )

        if results['non_predictors']:
            weak_predictors = [c['display_name'] for c in results['non_predictors'][:3]]
            results['recommendations'].append(
                f"REVIEW: These signals show weak/no correlation with performance: "
                f"{', '.join(weak_predictors)}. Consider revising or removing from rubric."
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

        # Summary statistics
        results['summary'] = {
            'total_employees': len(df),
            'sources_analyzed': df['HireSource'].nunique() if self.has_hire_source else 0,
            'prehire_signals_available': len(self.prehire_columns),
            'avg_performance': round(df['LastRating'].mean(), 2) if self.has_performance else None,
            'overall_retention': round(1 - df['Attrition'].mean(), 3) if self.has_attrition else None
        }

        # Source analysis
        source_df = self.calculate_source_effectiveness()
        if not source_df.empty:
            # Top sources
            top_sources = source_df.head(3).to_dict('records')
            results['top_sources'] = top_sources

            # Red flags - sources with low quality or retention
            if 'retention_rate' in source_df.columns:
                low_retention = source_df[source_df['retention_rate'] < 0.7]
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
            overall_quality = df['LastRating'].mean() if self.has_performance else 3.0

            for _, row in source_df.iterrows():
                if 'avg_performance' in row:
                    quality_diff = row['avg_performance'] - overall_quality
                    roi_indicator = 'Above Average' if quality_diff > 0.2 else (
                        'Below Average' if quality_diff < -0.2 else 'Average'
                    )

                    results['roi_analysis'][row['HireSource']] = {
                        'source': row['HireSource'],
                        'quality_vs_average': round(quality_diff, 2),
                        'roi_indicator': roi_indicator,
                        'recommendation': row.get('recommendation', '')
                    }

        # Generate final recommendations
        if results['top_sources']:
            top_source = results['top_sources'][0]
            results['recommendations'].append(
                f"EXPAND: {top_source['HireSource']} is your highest-quality source "
                f"(score: {top_source['quality_score']}). Consider increasing investment."
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

            if len(cohort_df) < MIN_SAMPLE_FOR_SOURCE:
                continue

            result = {
                cohort_column: cohort,
                'count': len(cohort_df)
            }

            # Performance metrics
            if self.has_performance:
                result['avg_performance'] = round(cohort_df['LastRating'].mean(), 2)
                result['performance_std'] = round(cohort_df['LastRating'].std(), 2)
                result['high_performer_pct'] = round(
                    (cohort_df['LastRating'] >= 4.0).mean() * 100, 1
                )
                result['low_performer_pct'] = round(
                    (cohort_df['LastRating'] <= 2.5).mean() * 100, 1
                )

            # Retention
            if self.has_attrition:
                result['retention_rate'] = round(1 - cohort_df['Attrition'].mean(), 3)

            # Tenure
            if self.has_tenure:
                result['avg_tenure'] = round(cohort_df['Tenure'].mean(), 2)

            results.append(result)

        result_df = pd.DataFrame(results)
        if not result_df.empty and 'avg_performance' in result_df.columns:
            result_df = result_df.sort_values('avg_performance', ascending=False)

        return result_df

    def get_new_hire_risk_assessment(self, months_since_hire: int = 6) -> pd.DataFrame:
        """
        Assess risk for recent hires based on pre-hire signals.

        Identifies new hires who may need additional support based on
        historically weak pre-hire signals.

        Args:
            months_since_hire: Look at hires within this timeframe

        Returns:
            DataFrame with new hire risk assessment.
        """
        if not self.has_hire_date or not self.prehire_columns:
            return pd.DataFrame()

        df = self.df.copy()

        # Parse hire dates
        df['HireDate_parsed'] = pd.to_datetime(df['HireDate'], errors='coerce')
        cutoff_date = datetime.now() - timedelta(days=months_since_hire * 30)

        # Filter to recent hires
        recent_hires = df[df['HireDate_parsed'] >= cutoff_date].copy()

        if len(recent_hires) == 0:
            return pd.DataFrame()

        # Calculate risk score based on pre-hire signals
        results = []

        # Get baseline (company average) for each signal
        baselines = {}
        for col in self.prehire_columns:
            baselines[col] = df[col].mean()

        for _, row in recent_hires.iterrows():
            risk_factors = []
            risk_score = 0

            for col in self.prehire_columns:
                if pd.notna(row[col]) and baselines[col] > 0:
                    # Calculate how far below average
                    deviation = (row[col] - baselines[col]) / baselines[col]
                    if deviation < -0.2:  # More than 20% below average
                        display_name = col.replace('InterviewScore_', '').replace('_', ' ')
                        impact = 'High' if deviation < -0.4 else 'Medium'
                        
                        risk_factors.append({
                            'factor': display_name,
                            'impact': impact,
                            'direction': 'Increase Risk',
                            'score': round(abs(deviation) * 10, 1),
                            'description': f"{display_name} score ({row[col]}) is {abs(deviation)*100:.0f}% below the company average ({baselines[col]:.1f})."
                        })
                        risk_score += abs(deviation) * 10

            risk_category = 'High' if risk_score > 30 else ('Medium' if risk_score > 15 else 'Low')

            results.append({
                'EmployeeID': row['EmployeeID'],
                'HireDate': row['HireDate'],
                'HireSource': row.get('HireSource', ''),
                'Dept': row.get('Dept', ''),
                'risk_score': round(risk_score, 1),
                'risk_category': risk_category,
                'risk_factors': risk_factors,
                'risk_factors_text': '; '.join([f"{f['factor']} ({f['impact']})" for f in risk_factors]) if risk_factors else 'None identified',
                'recommendation': 'Additional onboarding support' if risk_category == 'High' else ''
            })

        result_df = pd.DataFrame(results)
        return result_df.sort_values('risk_score', ascending=False)

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
            'has_hire_source': self.has_hire_source,
            'has_interview_scores': self.has_interview_score,
            'has_assessment': self.has_assessment,
            'prehire_signals_count': len(self.prehire_columns),
            'sources_analyzed': len(results['source_effectiveness']),
            'best_source': results['source_effectiveness'][0]['HireSource'] if results['source_effectiveness'] else None,
            'top_predictor': results['correlations'].get('best_predictors', [{}])[0].get('predictor') if results.get('correlations', {}).get('best_predictors') else None,
            'new_hires_at_risk': len([r for r in results['new_hire_risks'] if r.get('risk_category') == 'High'])
        }

        # Compile recommendations
        if results['insights'].get('recommendations'):
            results['recommendations'].extend(results['insights']['recommendations'])

        if results['correlations'].get('recommendations'):
            results['recommendations'].extend(results['correlations']['recommendations'])

        # Add warnings
        results['warnings'] = self.warnings.copy()

        logger.info(f"Quality of hire analysis complete. Warnings: {len(self.warnings)}")
        return results
