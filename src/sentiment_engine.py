"""
Sentiment Analysis Engine for PeopleOS.

Processes employee sentiment data including:
- eNPS (Employee Net Promoter Score) analysis
- Onboarding survey trajectory (30/60/90 day)
- Early warning detection for declining engagement
"""

import pandas as pd
from typing import Dict, Any, List, Optional

from src.utils import load_config
from src.population import resolve_current_population
from src.logger import get_logger


class SentimentEngine:
    """
    Engine for employee sentiment and survey analysis.

    Provides:
    - eNPS calculation and analysis by cohort
    - Onboarding trajectory monitoring (30/60/90 day surveys)
    - Early warning detection for at-risk employees
    """

    def __init__(
        self,
        employee_df: pd.DataFrame,
        enps_df: Optional[pd.DataFrame] = None,
        onboarding_df: Optional[pd.DataFrame] = None
    ):
        """
        Initialize SentimentEngine with data.

        Args:
            employee_df: Core employee DataFrame
            enps_df: Optional eNPS survey data
            onboarding_df: Optional onboarding survey data
        """
        self.employee_df, _ = resolve_current_population(employee_df)
        self.enps_df = enps_df.copy() if enps_df is not None else None
        self.onboarding_df = onboarding_df.copy() if onboarding_df is not None else None
        self.survey_coverage: Dict[str, Dict[str, int]] = {}

        self.config = load_config()
        self.sentiment_config = self.config.get('sentiment', {})
        self.enps_config = self.sentiment_config.get('enps', {})
        self.onboarding_config = self.sentiment_config.get('onboarding', {})
        self.early_warning_config = self.sentiment_config.get('early_warning', {})

        self.logger = get_logger('sentiment_engine')

        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare and validate survey data."""
        self.enps_df = self._match_survey_population(self.enps_df, 'enps')
        self.onboarding_df = self._match_survey_population(self.onboarding_df, 'onboarding')
        if self.enps_df is not None and 'eNPSScore' not in self.enps_df:
            self.survey_coverage['enps']['invalid_score_rows'] = len(self.enps_df)
            self.survey_coverage['enps']['valid_score_rows'] = 0
            self.enps_df = None
        if self.enps_df is not None:
            # Ensure date columns are datetime
            if 'SurveyDate' in self.enps_df.columns:
                self.enps_df['SurveyDate'] = pd.to_datetime(
                    self.enps_df['SurveyDate'],
                    errors='coerce', utc=True, format='mixed'
                )

            # Categorize eNPS responses
            promoter_threshold = self.enps_config.get('promoter_threshold', 9)
            detractor_threshold = self.enps_config.get('detractor_threshold', 6)

            if 'eNPSScore' in self.enps_df.columns:
                values = pd.to_numeric(self.enps_df['eNPSScore'], errors='coerce')
                self.survey_coverage['enps']['invalid_score_rows'] = int((~values.between(0, 10)).sum())
                self.survey_coverage['enps']['valid_score_rows'] = int(values.between(0, 10).sum())
                self.enps_df = self.enps_df.loc[values.between(0, 10)].copy()
                self.enps_df['eNPSScore'] = values.loc[self.enps_df.index]
                self.enps_df['eNPSCategory'] = self.enps_df['eNPSScore'].apply(
                    lambda x: 'Promoter' if x >= promoter_threshold
                    else ('Detractor' if x <= detractor_threshold else 'Passive')
                )

        if self.onboarding_df is not None:
            for col in ['OverallScore', 'ClarityOfRole', 'ManagerSupport', 'TeamIntegration', 'ToolsAccess', 'TrainingQuality']:
                if col in self.onboarding_df:
                    values = pd.to_numeric(self.onboarding_df[col], errors='coerce')
                    self.onboarding_df[col] = values.where(values.between(1, 5))
            if 'SurveyDate' in self.onboarding_df.columns:
                self.onboarding_df['SurveyDate'] = pd.to_datetime(
                    self.onboarding_df['SurveyDate'],
                    errors='coerce', utc=True, format='mixed'
                )
            # A snapshot measure uses one latest response per employee/type.
            # With no dates, stable input order is the documented fallback.
            frame = self.onboarding_df
            if 'SurveyType' not in frame:
                self.survey_coverage['onboarding']['unsupported_type_rows'] = len(frame)
                frame = frame.iloc[:0].assign(SurveyType=pd.Series(dtype=str))
            else:
                supported = frame['SurveyType'].isin(['30-day', '60-day', '90-day'])
                self.survey_coverage['onboarding']['unsupported_type_rows'] = int((~supported).sum())
                frame = frame.loc[supported].copy()
            before = len(frame)
            if 'SurveyDate' in frame:
                frame = frame.sort_values('SurveyDate', kind='stable', na_position='first')
            self.onboarding_df = frame.drop_duplicates(['EmployeeID', 'SurveyType'], keep='last')
            self.survey_coverage['onboarding']['superseded_rows'] = before - len(self.onboarding_df)
            self.survey_coverage['onboarding']['latest_response_rows'] = len(self.onboarding_df)
            self.survey_coverage['onboarding']['valid_overall_score_rows'] = (
                int(self.onboarding_df['OverallScore'].count()) if 'OverallScore' in self.onboarding_df else 0
            )

    def _match_survey_population(self, frame: Optional[pd.DataFrame], name: str) -> Optional[pd.DataFrame]:
        """Only identified members of the resolved employee population contribute.

        IDs are compared exactly: guessing a numeric/string conversion here could
        silently join different source identities. Ingestion owns normalization.
        """
        coverage = {'input_rows': 0, 'matched_rows': 0, 'unmatched_rows': 0,
                    'missing_employee_id_rows': 0}
        self.survey_coverage[name] = coverage
        if frame is None:
            return None
        frame = frame.copy()
        coverage['input_rows'] = len(frame)
        if 'EmployeeID' not in frame:
            coverage['missing_employee_id_rows'] = len(frame)
            return frame.iloc[:0].assign(EmployeeID=pd.Series(dtype=object))
        missing = frame['EmployeeID'].isna() | frame['EmployeeID'].astype(str).str.strip().eq('')
        ids = self.employee_df['EmployeeID'].dropna() if 'EmployeeID' in self.employee_df else []
        matched = ~missing & frame['EmployeeID'].isin(ids)
        coverage['missing_employee_id_rows'] = int(missing.sum())
        coverage['unmatched_rows'] = int((~missing & ~matched).sum())
        coverage['matched_rows'] = int(matched.sum())
        return frame.loc[matched].copy()

    # =========================================================================
    # eNPS ANALYSIS
    # =========================================================================

    def calculate_enps(
        self,
        group_by: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate eNPS (Employee Net Promoter Score).

        eNPS = %Promoters - %Detractors

        Args:
            group_by: Optional column to segment by (Dept, Location, etc.)
            date_from: Optional start date for filtering
            date_to: Optional end date for filtering

        Returns:
            Dictionary with eNPS metrics.
        """
        if self.enps_df is None or self.enps_df.empty:
            return {
                'available': False,
                'reason': 'No valid eNPS responses matched to the employee population',
                'survey_coverage': self.survey_coverage
            }

        df = self.enps_df.copy()
        allowed_group_fields = {'Dept', 'Location', 'BusinessUnit', 'Country', 'Region'}
        minimum_group_size = int(self.enps_config.get('min_group_size', 10))
        if group_by and group_by not in allowed_group_fields:
            return {
                'available': False,
                'reason': 'Unsupported eNPS grouping field',
                'allowed_group_fields': sorted(allowed_group_fields),
                'survey_coverage': self.survey_coverage,
            }
        if group_by and group_by not in self.employee_df.columns:
            return {
                'available': False,
                'reason': f'{group_by} is unavailable in the governed employee population',
                'survey_coverage': self.survey_coverage,
            }

        # Date-only endpoints include the entire UTC day. A missing date
        # column cannot satisfy a requested date filter.
        if (date_from or date_to) and 'SurveyDate' not in df:
            return {'available': False, 'reason': 'SurveyDate is required for date filtering'}
        try:
            start = pd.to_datetime(date_from, utc=True) if date_from else None
            end = pd.to_datetime(date_to, utc=True) if date_to else None
            if (start is not None and pd.isna(start)) or (end is not None and pd.isna(end)):
                raise ValueError('Invalid date')
            if start is not None and end is not None and start > end:
                raise ValueError('Reversed date range')
        except (ValueError, TypeError):
            return {'available': False, 'reason': 'Invalid date range'}
        if start is not None:
            df = df[df['SurveyDate'] >= start]
        if end is not None:
            if len(date_to.strip()) == 10:
                df = df[df['SurveyDate'] < end + pd.Timedelta(days=1)]
            else:
                df = df[df['SurveyDate'] <= end]

        if df.empty:
            return {
                'available': False,
                'reason': 'No data in selected date range'
            }

        # Merge with employee data for grouping
        if group_by:
            # Organization grouping is sourced from the governed employee
            # population. A survey upload cannot redefine an employee's cohort.
            df = df.drop(columns=[group_by], errors='ignore')
            df = df.merge(
                self.employee_df[['EmployeeID', group_by]],
                on='EmployeeID',
                how='left', validate='many_to_one'
            )

        def calc_enps_score(data):
            total = len(data)
            if total == 0:
                return None
            promoters = (data['eNPSCategory'] == 'Promoter').sum()
            detractors = (data['eNPSCategory'] == 'Detractor').sum()
            return round((promoters - detractors) / total * 100, 1)

        # Overall eNPS
        overall_enps = calc_enps_score(df)

        result = {
            'available': True,
            'survey_coverage': self.survey_coverage,
            'measurement_semantics': 'valid_matched_survey_responses_not_unique_employee_prevalence',
            'overall_enps': overall_enps,
            'total_responses': len(df),
            'promoters': int((df['eNPSCategory'] == 'Promoter').sum()),
            'passives': int((df['eNPSCategory'] == 'Passive').sum()),
            'detractors': int((df['eNPSCategory'] == 'Detractor').sum()),
            'promoter_pct': round((df['eNPSCategory'] == 'Promoter').sum() / len(df) * 100, 1),
            'passive_pct': round((df['eNPSCategory'] == 'Passive').sum() / len(df) * 100, 1),
            'detractor_pct': round((df['eNPSCategory'] == 'Detractor').sum() / len(df) * 100, 1)
        }

        # Calculate by group if specified
        if group_by and group_by in df.columns:
            by_group = []
            group_labels = df[group_by].astype('string').str.strip().replace('', pd.NA).fillna('Unknown')
            suppressed_groups = 0
            suppressed_responses = 0
            for group_name in group_labels.unique():
                group_data = df[group_labels == group_name]
                if len(group_data) < minimum_group_size:
                    suppressed_groups += 1
                    suppressed_responses += len(group_data)
                    continue
                enps = calc_enps_score(group_data)
                by_group.append({
                    'group': group_name,
                    'enps': enps,
                    'responses': len(group_data),
                    'promoters': int((group_data['eNPSCategory'] == 'Promoter').sum()),
                    'detractors': int((group_data['eNPSCategory'] == 'Detractor').sum())
                })

            by_group.sort(key=lambda x: x['enps'] if x['enps'] is not None else -100, reverse=True)
            result['by_group'] = by_group
            result['minimum_group_size'] = minimum_group_size
            result['suppressed_group_count'] = suppressed_groups
            result['suppressed_response_count'] = suppressed_responses

        # Add benchmark interpretation
        if overall_enps is not None:
            if overall_enps >= 50:
                result['interpretation'] = 'Excellent - Strong employee advocacy'
            elif overall_enps >= 20:
                result['interpretation'] = 'Good - Positive employee sentiment'
            elif overall_enps >= 0:
                result['interpretation'] = 'Neutral - Room for improvement'
            else:
                result['interpretation'] = 'Poor - Significant engagement issues'

        return result

    def get_enps_trends(
        self,
        period: str = 'month',
        group_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get eNPS trends over time.

        Args:
            period: 'week', 'month', or 'quarter'
            group_by: Optional column to segment by

        Returns:
            Dictionary with trend data.
        """
        if self.enps_df is None or 'SurveyDate' not in self.enps_df.columns:
            return {
                'available': False,
                'reason': 'No eNPS survey data with dates available'
            }

        df = self.enps_df.dropna(subset=['SurveyDate']).copy()
        if df.empty:
            return {'available': False, 'reason': 'No valid dated eNPS responses'}
        if period not in {'week', 'month', 'quarter'}:
            return {'available': False, 'reason': 'Unsupported trend period'}
        # All source timestamps were normalized to UTC before period grouping.
        df['SurveyDate'] = df['SurveyDate'].dt.tz_localize(None)

        # Set period
        if period == 'week':
            df['Period'] = df['SurveyDate'].dt.to_period('W').astype(str)
        elif period == 'quarter':
            df['Period'] = df['SurveyDate'].dt.to_period('Q').astype(str)
        else:  # month
            df['Period'] = df['SurveyDate'].dt.to_period('M').astype(str)

        # Calculate eNPS per period
        trends = []
        for period_val in sorted(df['Period'].unique()):
            period_data = df[df['Period'] == period_val]
            total = len(period_data)
            promoters = (period_data['eNPSCategory'] == 'Promoter').sum()
            detractors = (period_data['eNPSCategory'] == 'Detractor').sum()
            enps = round((promoters - detractors) / total * 100, 1) if total > 0 else None

            trends.append({
                'period': period_val,
                'enps': enps,
                'responses': total,
                'promoters': int(promoters),
                'detractors': int(detractors)
            })

        # Calculate trend direction
        if len(trends) >= 2:
            recent = trends[-1]['enps']
            previous = trends[-2]['enps']
            if recent is not None and previous is not None:
                change = recent - previous
                trend_direction = 'improving' if change > 0 else ('declining' if change < 0 else 'stable')
            else:
                trend_direction = 'unknown'
                change = None
        else:
            trend_direction = 'insufficient_data'
            change = None

        return {
            'available': True,
            'period_type': period,
            'trends': trends,
            'trend_direction': trend_direction,
            'recent_change': change
        }

    def get_enps_drivers(self) -> Dict[str, Any]:
        """
        Analyze what's driving eNPS scores.

        Returns correlation with sub-scores and identifies key drivers.
        """
        if self.enps_df is None:
            return {'available': False, 'reason': 'No eNPS data available'}

        # Expected sub-score columns
        sub_scores = [
            'EngagementScore', 'WellbeingScore', 'GrowthScore',
            'ManagerScore', 'CultureScore', 'RecommendScore'
        ]

        available_scores = [s for s in sub_scores if s in self.enps_df.columns]

        if not available_scores:
            return {
                'available': False,
                'reason': 'No sub-score columns available for driver analysis'
            }

        drivers = []
        for score in available_scores:
            pairs = self.enps_df[['eNPSScore', score]].apply(pd.to_numeric, errors='coerce')
            pairs = pairs.replace([float('inf'), -float('inf')], float('nan')).dropna()
            if len(pairs) < 10 or (pairs.nunique() < 2).any():
                continue
            corr = pairs['eNPSScore'].corr(pairs[score])
            drivers.append({'dimension': score.replace('Score', ''), 'correlation': round(corr, 3),
                            'avg_score': round(pairs[score].mean(), 2), 'sample_size': len(pairs),
                            'impact': 'High' if abs(corr) >= .5 else 'Medium' if abs(corr) >= .3 else 'Low',
                            'metric_semantics': 'paired_observational_correlation_not_causal_driver'})
        drivers.sort(key=lambda item: abs(item['correlation']), reverse=True)
        # Different survey instruments can use different scales; no universal low-score threshold.
        improvement_areas = []

        return {
            'available': bool(drivers),
            'drivers': drivers,
            'top_driver': drivers[0]['dimension'] if drivers else None,
            'improvement_areas': [d['dimension'] for d in improvement_areas],
            'recommendations': self._generate_enps_recommendations(drivers, improvement_areas)
        }

    def _generate_enps_recommendations(
        self,
        drivers: List[Dict],
        improvement_areas: List[Dict]
    ) -> List[str]:
        """Generate recommendations based on eNPS analysis."""
        recommendations = []

        for area in improvement_areas:
            dim = area['dimension']
            if dim == 'Manager':
                recommendations.append(
                    "Manager scores are low - consider manager training and feedback programs"
                )
            elif dim == 'Growth':
                recommendations.append(
                    "Growth/career development scores are low - review L&D and promotion opportunities"
                )
            elif dim == 'Wellbeing':
                recommendations.append(
                    "Wellbeing scores are low - consider work-life balance initiatives"
                )
            elif dim == 'Culture':
                recommendations.append(
                    "Culture scores are low - review company values alignment and team dynamics"
                )
            elif dim == 'Engagement':
                recommendations.append(
                    "Engagement is low - investigate workload, autonomy, and recognition programs"
                )

        return recommendations

    # =========================================================================
    # ONBOARDING ANALYSIS
    # =========================================================================

    def analyze_onboarding_trajectory(
        self,
        employee_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze onboarding survey trajectories.

        Tracks 30/60/90 day survey scores to identify early warning signs.

        Args:
            employee_id: Optional - analyze specific employee

        Returns:
            Dictionary with trajectory analysis.
        """
        if self.onboarding_df is None or self.onboarding_df.empty:
            return {
                'available': False,
                'reason': 'No supported onboarding responses matched to the employee population',
                'survey_coverage': self.survey_coverage
            }

        df = self.onboarding_df.copy()

        if employee_id:
            df = df[df['EmployeeID'] == employee_id]
            if df.empty:
                return {
                    'available': False,
                    'reason': f'No onboarding data for employee {employee_id}'
                }

        # Expected survey types
        survey_order = {'30-day': 1, '60-day': 2, '90-day': 3}

        trajectories = []

        for emp_id in df['EmployeeID'].unique():
            emp_data = df[df['EmployeeID'] == emp_id].copy()

            if 'SurveyType' not in emp_data.columns:
                continue

            # Sort by survey type
            emp_data['SurveyOrder'] = emp_data['SurveyType'].map(survey_order)
            emp_data = emp_data[emp_data['SurveyOrder'].notna()]
            if 'SurveyDate' in emp_data:
                emp_data = emp_data.sort_values('SurveyDate', kind='stable', na_position='first')
            emp_data = emp_data.drop_duplicates('SurveyType', keep='last').sort_values('SurveyOrder')

            scores = {}
            for _, row in emp_data.iterrows():
                survey_type = row.get('SurveyType', 'Unknown')
                overall = row.get('OverallScore')
                if pd.notna(overall):
                    scores[survey_type] = float(overall)

            # Calculate trajectory
            trajectory_values = list(scores.values())
            if len(trajectory_values) >= 2:
                trend = trajectory_values[-1] - trajectory_values[0]
                if trend < -0.5:
                    trajectory_direction = 'declining'
                elif trend > 0.5:
                    trajectory_direction = 'improving'
                else:
                    trajectory_direction = 'stable'
            else:
                trajectory_direction = 'insufficient_data'
                trend = None

            # Get employee context
            emp_info = self.employee_df[self.employee_df['EmployeeID'] == emp_id]
            dept = emp_info['Dept'].iloc[0] if not emp_info.empty and 'Dept' in emp_info else None

            trajectories.append({
                'EmployeeID': emp_id,
                'Dept': dept,
                'scores': scores,
                'trajectory_direction': trajectory_direction,
                'trend_change': round(trend, 2) if trend is not None else None,
                'surveys_completed': len(scores),
                'latest_score': trajectory_values[-1] if trajectory_values else None,
                'at_risk': trajectory_direction == 'declining' or (
                    trajectory_values[-1] < 3.0 if trajectory_values else False
                )
            })

        # Summary statistics
        declining = [t for t in trajectories if t['trajectory_direction'] == 'declining']
        at_risk = [t for t in trajectories if t.get('at_risk')]

        return {
            'available': True,
            'trajectories': trajectories,
            'survey_coverage': self.survey_coverage,
            'response_selection': 'latest_per_employee_and_survey_type; input_order_breaks_date_ties_or_missing_dates',
            'summary': {
                'total_employees': len(trajectories),
                'declining_count': len(declining),
                'at_risk_count': len(at_risk),
                'improving_count': len([t for t in trajectories if t['trajectory_direction'] == 'improving']),
                'avg_completion_rate': round(
                    sum(t['surveys_completed'] for t in trajectories) / len(trajectories) / 3 * 100, 1
                ) if trajectories else 0
            },
            'at_risk_employees': at_risk[:10]  # Top 10 at risk
        }

    def get_onboarding_health(self) -> Dict[str, Any]:
        """
        Get overall onboarding health metrics.

        Returns:
            Dictionary with onboarding health assessment.
        """
        if self.onboarding_df is None or self.onboarding_df.empty:
            return {'available': False, 'reason': 'No matched onboarding data available',
                    'survey_coverage': self.survey_coverage}

        df = self.onboarding_df.copy()
        min_healthy = self.onboarding_config.get('min_score_healthy', 3.5)

        # Calculate metrics by survey type
        by_survey_type = []
        if 'SurveyType' in df.columns and 'OverallScore' in df.columns:
            for survey_type in ['30-day', '60-day', '90-day']:
                type_data = df[df['SurveyType'] == survey_type]
                if not type_data.empty:
                    type_data = type_data.dropna(subset=['OverallScore'])
                    if type_data.empty:
                        continue
                    avg_score = type_data['OverallScore'].mean()
                    healthy_pct = (type_data['OverallScore'] >= min_healthy).sum() / len(type_data) * 100
                    by_survey_type.append({
                        'survey_type': survey_type,
                        'avg_score': round(avg_score, 2),
                        'responses': len(type_data),
                        'healthy_pct': round(healthy_pct, 1)
                    })

        # Calculate dimension scores
        dimension_cols = [
            'ClarityOfRole', 'ManagerSupport', 'TeamIntegration',
            'ToolsAccess', 'TrainingQuality'
        ]
        dimension_scores = []
        for col in dimension_cols:
            if col in df.columns:
                avg = df[col].mean()
                dimension_scores.append({
                    'dimension': col,
                    'avg_score': round(avg, 2) if pd.notna(avg) else None
                })

        dimension_scores.sort(key=lambda x: x['avg_score'] if x['avg_score'] else 0)

        # Identify weakest dimensions
        measured_dimensions = [d for d in dimension_scores if d['avg_score'] is not None]
        weakest = [dimension for dimension in measured_dimensions if dimension['avg_score'] < min_healthy][:2]

        return {
            'available': True,
            'by_survey_type': by_survey_type,
            'survey_coverage': self.survey_coverage,
            'response_selection': 'latest_per_employee_and_survey_type; input_order_breaks_date_ties_or_missing_dates',
            'dimension_scores': dimension_scores,
            'weakest_dimensions': weakest,
            'overall_health': ('Unavailable' if not measured_dimensions else
                               'Healthy' if all(d['avg_score'] >= min_healthy for d in measured_dimensions) else 'Needs Attention'),
            'measurement_semantics': 'observed_survey_scores_not_validated_employee_or_onboarding_health',
            'recommendations': self._generate_onboarding_recommendations(weakest)
        }

    def _generate_onboarding_recommendations(
        self,
        weakest_dimensions: List[Dict]
    ) -> List[str]:
        """Generate recommendations for onboarding improvement."""
        recommendations = []

        for dim in weakest_dimensions:
            name = dim['dimension']
            if name == 'ClarityOfRole':
                recommendations.append(
                    "Role clarity is low - ensure job descriptions and expectations are clear from day 1"
                )
            elif name == 'ManagerSupport':
                recommendations.append(
                    "Manager support scores are low - consider mandatory 1:1s during first 90 days"
                )
            elif name == 'TeamIntegration':
                recommendations.append(
                    "Team integration is weak - implement buddy programs and team introductions"
                )
            elif name == 'ToolsAccess':
                recommendations.append(
                    "Tools/access issues - review IT onboarding checklist and pre-provisioning"
                )
            elif name == 'TrainingQuality':
                recommendations.append(
                    "Training quality is low - review onboarding curriculum and delivery methods"
                )

        return recommendations

    # =========================================================================
    # EARLY WARNING DETECTION
    # =========================================================================

    def detect_early_warnings(self) -> Dict[str, Any]:
        """
        Detect employees showing early warning signs of disengagement.

        Combines eNPS trends and onboarding trajectories.

        Returns:
            Dictionary with at-risk employees and warning types.
        """
        has_dated_enps = (self.enps_df is not None and 'SurveyDate' in self.enps_df
                          and self.enps_df['SurveyDate'].notna().any())
        has_onboarding_scores = (self.onboarding_df is not None and 'OverallScore' in self.onboarding_df
                                 and self.onboarding_df['OverallScore'].notna().any())
        if not has_dated_enps and not has_onboarding_scores:
            return {
                'available': False,
                'reason': 'No matched valid responses support latest-response survey flags',
                'survey_coverage': self.survey_coverage,
                'warnings': [],
                'metric_semantics': 'observed_survey_flags_not_validated_departure_risk',
            }
        warnings = []

        # Check eNPS detractors
        if self.enps_df is not None and 'eNPSCategory' in self.enps_df.columns:
            # Get most recent survey per employee
            recent_enps = self.enps_df
            if 'SurveyDate' not in recent_enps or 'EmployeeID' not in recent_enps:
                recent_enps = recent_enps.iloc[:0].assign(EmployeeID=pd.Series(dtype=str))
            else:
                recent_enps = recent_enps.dropna(subset=['SurveyDate']).sort_values('SurveyDate', kind='stable').drop_duplicates('EmployeeID', keep='last')
            recent_enps = recent_enps.set_index('EmployeeID')
            detractors = recent_enps[recent_enps['eNPSCategory'] == 'Detractor']

            for emp_id in detractors.index:
                emp_info = self.employee_df[self.employee_df['EmployeeID'] == emp_id]
                warnings.append({
                    'EmployeeID': emp_id,
                    'Dept': emp_info['Dept'].iloc[0] if not emp_info.empty and 'Dept' in emp_info else None,
                    'warning_type': 'eNPS Detractor',
                    'severity': 'High',
                    'details': f"eNPS score: {detractors.loc[emp_id].get('eNPSScore', 'N/A')}"
                })

        # Check declining onboarding trajectories
        if self.onboarding_df is not None:
            trajectory_analysis = self.analyze_onboarding_trajectory()
            if trajectory_analysis.get('available'):
                for traj in trajectory_analysis.get('trajectories', []):
                    if not traj.get('at_risk'):
                        continue
                    # Avoid duplicates
                    if not any(w['EmployeeID'] == traj['EmployeeID'] for w in warnings):
                        warnings.append({
                            'EmployeeID': traj['EmployeeID'],
                            'Dept': traj.get('Dept'),
                            'warning_type': 'Declining Onboarding',
                            'severity': 'Medium' if traj.get('latest_score', 5) >= 2.5 else 'High',
                            'details': f"Trajectory: {traj.get('trajectory_direction')}, Latest: {traj.get('latest_score')}"
                        })

        # Sort by severity
        severity_order = {'High': 0, 'Medium': 1, 'Low': 2}
        warnings.sort(key=lambda x: severity_order.get(x['severity'], 3))

        return {
            'available': True,
            'warnings': warnings,
            'survey_coverage': self.survey_coverage,
            'metric_semantics': 'observed_survey_flags_not_validated_departure_risk',
            'summary': {
                'total_at_risk': len(warnings),
                'high_severity': len([w for w in warnings if w['severity'] == 'High']),
                'medium_severity': len([w for w in warnings if w['severity'] == 'Medium']),
                'warning_types': list(set(w['warning_type'] for w in warnings))
            },
            'recommendations': [
                "Schedule 1:1 conversations with high-severity employees",
                "Review recent changes affecting at-risk departments",
                "Consider stay interviews for employees showing warning signs"
            ] if warnings else []
        }

    # =========================================================================
    # COMPREHENSIVE ANALYSIS
    # =========================================================================

    def analyze_all(self) -> Dict[str, Any]:
        """
        Run all sentiment analyses and return comprehensive results.

        Returns:
            Dictionary with all analysis results.
        """
        results = {
            'survey_coverage': self.survey_coverage,
            'enps': {},
            'enps_trends': {},
            'enps_drivers': {},
            'onboarding': {},
            'onboarding_health': {},
            'early_warnings': {},
            'summary': {},
            'recommendations': [],
            'warnings': []
        }

        # eNPS Analysis
        try:
            results['enps'] = self.calculate_enps()
            if results['enps'].get('overall_enps') is not None:
                if results['enps']['overall_enps'] < 0:
                    results['warnings'].append(
                        f"Overall eNPS is negative ({results['enps']['overall_enps']})"
                    )
        except Exception as e:
            self.logger.error(f"eNPS analysis error: {e}")
            results['enps'] = {'available': False, 'error': str(e)}

        # eNPS Trends
        try:
            results['enps_trends'] = self.get_enps_trends()
        except Exception as e:
            self.logger.error(f"eNPS trends error: {e}")
            results['enps_trends'] = {'available': False, 'error': str(e)}

        # eNPS Drivers
        try:
            results['enps_drivers'] = self.get_enps_drivers()
            if results['enps_drivers'].get('recommendations'):
                results['recommendations'].extend(results['enps_drivers']['recommendations'])
        except Exception as e:
            self.logger.error(f"eNPS drivers error: {e}")
            results['enps_drivers'] = {'available': False, 'error': str(e)}

        # Onboarding Trajectory
        try:
            results['onboarding'] = self.analyze_onboarding_trajectory()
            if results['onboarding'].get('summary', {}).get('declining_count', 0) > 0:
                results['warnings'].append(
                    f"{results['onboarding']['summary']['declining_count']} new hires show declining onboarding scores"
                )
        except Exception as e:
            self.logger.error(f"Onboarding trajectory error: {e}")
            results['onboarding'] = {'available': False, 'error': str(e)}

        # Onboarding Health
        try:
            results['onboarding_health'] = self.get_onboarding_health()
            if results['onboarding_health'].get('recommendations'):
                results['recommendations'].extend(results['onboarding_health']['recommendations'])
        except Exception as e:
            self.logger.error(f"Onboarding health error: {e}")
            results['onboarding_health'] = {'available': False, 'error': str(e)}

        # Early Warnings
        try:
            results['early_warnings'] = self.detect_early_warnings()
            if results['early_warnings'].get('summary', {}).get('high_severity', 0) > 0:
                results['warnings'].append(
                    f"{results['early_warnings']['summary']['high_severity']} employees flagged as high-severity risk"
                )
        except Exception as e:
            self.logger.error(f"Early warnings error: {e}")
            results['early_warnings'] = {'available': False, 'error': str(e)}

        # Overall Summary
        results['summary'] = {
            'enps_available': results['enps'].get('available', False),
            'onboarding_available': results['onboarding'].get('available', False),
            'overall_enps': results['enps'].get('overall_enps'),
            'employees_at_risk': results['early_warnings'].get('summary', {}).get('total_at_risk'),
            'survey_flags_available': results['early_warnings'].get('available', False),
            'total_warnings': len(results['warnings']),
            'total_recommendations': len(results['recommendations'])
        }

        return results
