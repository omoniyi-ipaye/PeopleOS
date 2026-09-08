"""
Employee Experience Engine for PeopleOS.

Computes unified Employee Experience Index (EXI) from multiple signals:
- eNPS scores
- Onboarding survey scores (30/60/90 day)
- Pulse survey scores
- Manager satisfaction
- Career growth satisfaction
- Work-life balance
- Tenure/promotion trajectory

Single DataFrame input - all experience columns are optional.
Engine automatically detects available signals and adjusts calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from src.utils import load_config
from src.population import resolve_current_population
from src.logger import get_logger


class ExperienceEngineError(Exception):
    """Custom exception for ExperienceEngine errors."""
    pass


class ExperienceEngine:
    """
    Unified Employee Experience Analytics Engine.

    Provides:
    - Experience Index (EXI) calculation (0-100 scale)
    - Engagement segmentation (Thriving, Content, Neutral, Disengaged, Critical)
    - Experience drivers analysis
    - At-risk employee identification
    - Lifecycle stage experience tracking
    - Manager impact analysis
    """

    # Experience columns to detect (lowercase for matching)
    EXPERIENCE_COLUMNS = {
        'enps_score': 'eNPS',
        'onboarding_30d': 'Onboarding 30-day',
        'onboarding_60d': 'Onboarding 60-day',
        'onboarding_90d': 'Onboarding 90-day',
        'pulse_score': 'Pulse Survey',
        'engagementscore': 'Engagement',
        'managersatisfaction': 'Manager Satisfaction',
        'worklifebalance': 'Work-Life Balance',
        'careergrowthsatisfaction': 'Career Growth',
    }

    # Engagement segment thresholds (EXI score ranges)
    SEGMENTS = {
        'Thriving': (80, 100),
        'Content': (60, 79),
        'Neutral': (40, 59),
        'Disengaged': (20, 39),
        'Critical': (0, 19),
    }

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with unified employee DataFrame.

        Args:
            df: Employee DataFrame with optional experience columns
        """
        self.df, self.population_resolution = resolve_current_population(df)
        self.config = load_config()
        self.exp_config = self.config.get('experience', {})
        self.logger = get_logger('experience_engine')
        self.warnings: List[str] = []

        # Store original column names for later use
        self._column_map = self._build_column_map()

        # Detect available experience signals
        self._detect_available_signals()
        self._validate_data()
        self._compute_experience_index()

    def _build_column_map(self) -> Dict[str, str]:
        """Build mapping from lowercase to actual column names."""
        return {col.lower(): col for col in self.df.columns}

    def _get_column(self, lowercase_name: str) -> Optional[str]:
        """Get actual column name from lowercase version."""
        return self._column_map.get(lowercase_name)

    def _detect_available_signals(self) -> None:
        """Detect which experience columns are present."""
        cols_lower = set(self._column_map.keys())

        self.has_enps = 'enps_score' in cols_lower
        self.has_onboarding_30d = 'onboarding_30d' in cols_lower
        self.has_onboarding_60d = 'onboarding_60d' in cols_lower
        self.has_onboarding_90d = 'onboarding_90d' in cols_lower
        self.has_onboarding = any([
            self.has_onboarding_30d,
            self.has_onboarding_60d,
            self.has_onboarding_90d
        ])
        self.has_pulse = 'pulse_score' in cols_lower
        self.has_manager_satisfaction = 'managersatisfaction' in cols_lower
        self.has_engagement = 'engagementscore' in cols_lower
        self.has_work_life = 'worklifebalance' in cols_lower
        self.has_career_growth = 'careergrowthsatisfaction' in cols_lower

        # Count available survey-based signals
        self.available_survey_signals = sum([
            self.has_enps,
            self.has_onboarding,
            self.has_pulse,
            self.has_manager_satisfaction,
            self.has_engagement,
            self.has_work_life,
            self.has_career_growth,
        ])

        # Check for derived signals (always available from core columns)
        self.has_tenure = 'tenure' in cols_lower
        self.has_salary = 'salary' in cols_lower
        self.has_rating = 'lastrating' in cols_lower
        self.has_promotion_data = 'yearssincelastpromotion' in cols_lower

        if self.available_survey_signals == 0:
            self.warnings.append(
                "No experience survey columns found. Measured experience is unavailable."
            )
            self.logger.warning("No survey-based experience signals detected")

    def _validate_data(self) -> None:
        """Validate required columns exist."""
        required = ['EmployeeID']
        missing = [col for col in required if col not in self.df.columns]

        if missing:
            raise ExperienceEngineError(f"Missing required columns: {missing}")

        if len(self.df) < 10:
            self.warnings.append(
                f"Small dataset ({len(self.df)} employees). "
                "Results may not be statistically reliable."
            )

    def _compute_experience_index(self) -> None:
        """Measured responses only, using a fixed scale per signal column.

        Missing or out-of-range responses never become HRIS-derived scores.
        EngagementScore requires an explicitly configured scale; its scale
        cannot be inferred from the magnitude of each respondent's answer.
        """
        weights = self.exp_config.get('index_weights', {})
        definitions = {
            'enps': (['enps_score'], [0, 10], 'enps', .25),
            'onboarding': (['onboarding_30d', 'onboarding_60d', 'onboarding_90d'], [1, 5], 'onboarding', .15),
            'pulse': (['pulse_score'], [1, 5], 'pulse', .15),
            'manager': (['managersatisfaction'], [1, 5], 'manager_satisfaction', .15),
            'engagement': (['engagementscore'], None, 'engagement', .10),
            'work_life': (['worklifebalance'], [1, 5], 'work_life', .10),
            'career': (['careergrowthsatisfaction'], [1, 5], 'career_growth', .10),
        }
        scales = self.exp_config.get('signal_scales', {})
        weighted = pd.Series(0., index=self.df.index)
        denominator = pd.Series(0., index=self.df.index)
        components = pd.DataFrame(index=self.df.index)
        for name, (columns, default_scale, weight_key, default_weight) in definitions.items():
            normalized = []
            weight = float(weights.get(weight_key, default_weight))
            if not np.isfinite(weight) or weight < 0:
                raise ExperienceEngineError('Experience weights must be finite and non-negative')
            for column in columns:
                actual = self._get_column(column)
                if actual is None:
                    continue
                scale = scales.get(column, default_scale)
                if scale is None:
                    self.warnings.append(f'{actual} excluded: configure its signal_scales minimum and maximum.')
                    continue
                low, high = map(float, scale)
                if not np.isfinite([low, high]).all() or low >= high:
                    raise ExperienceEngineError(f'Invalid signal scale for {actual}')
                values = pd.to_numeric(self.df[actual], errors='coerce')
                valid = values.between(low, high) & np.isfinite(values)
                normalized.append((values.where(valid) - low) / (high - low) * 100)
            if normalized and weight > 0:
                component = pd.concat(normalized, axis=1).mean(axis=1)
                components[name] = component
                weighted += component.fillna(0) * weight
                denominator += component.notna() * weight
        self.df['_exi_score'] = (weighted / denominator.replace(0, np.nan)).round(1)
        self.df['_exi_components'] = [str(row.dropna().to_dict()) for _, row in components.iterrows()]
        self.respondent_count = int(self.df['_exi_score'].notna().sum())
        self.warnings.append('Composite scores are descriptive; signal coverage and configured scales must accompany comparisons.')

    def _derive_exi_from_patterns(self, idx: int) -> float:
        """
        Derive EXI from tenure and performance patterns when no survey data exists.

        Uses heuristics:
        - Good performance rating -> higher EXI
        - Moderate tenure -> higher EXI (not too new, not stagnating)
        - Recent promotion -> higher EXI
        """
        score = 50.0  # Start neutral

        # Performance impact
        if self.has_rating:
            col = self._get_column('lastrating')
            rating = self.df.at[idx, col]
            if pd.notna(rating):
                # Assume 1-5 scale
                score += (float(rating) - 3) * 10  # 3 is neutral

        # Tenure stability (sweet spot: 1-5 years)
        if self.has_tenure:
            col = self._get_column('tenure')
            tenure = self.df.at[idx, col]
            if pd.notna(tenure):
                tenure = float(tenure)
                if tenure < 0.5:
                    score -= 10  # Very new, still adjusting
                elif tenure <= 5:
                    score += 10  # Sweet spot
                elif tenure > 10:
                    score -= 5  # Potential stagnation

        # Promotion recency
        if self.has_promotion_data:
            col = self._get_column('yearssincelastpromotion')
            years_since = self.df.at[idx, col]
            if pd.notna(years_since):
                if float(years_since) < 2:
                    score += 10  # Recently promoted
                elif float(years_since) > 5:
                    score -= 10  # Long time without promotion

        return max(0, min(100, score))  # Clamp to 0-100

    # =========================================================================
    # EXPERIENCE INDEX ANALYSIS
    # =========================================================================

    def calculate_experience_index(
        self,
        group_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get Experience Index summary.

        Args:
            group_by: Optional column to segment by (Dept, Location, etc.)

        Returns:
            Dictionary with EXI metrics.
        """
        if '_exi_score' not in self.df.columns or not self.df['_exi_score'].notna().any():
            return {
                'available': False,
                'reason': 'EXI not computed'
            }

        overall_exi = self.df['_exi_score'].mean()
        exi_std = self.df['_exi_score'].std()

        result = {
            'available': True,
            'overall_exi': round(overall_exi, 1),
            'exi_std': round(exi_std, 1) if pd.notna(exi_std) else None,
            'exi_median': round(self.df['_exi_score'].median(), 1),
            'total_employees': len(self.df),
            'signals_available': self.available_survey_signals,
            'respondent_count': self.respondent_count,
            'response_coverage': self.respondent_count / len(self.df) if len(self.df) else 0.,
            'interpretation': self._interpret_exi(overall_exi),
        }

        result['benchmark'] = None

        # Calculate by group if specified
        if group_by and group_by in self.df.columns:
            by_group = []
            for group_name in self.df[group_by].dropna().unique():
                group_data = self.df[self.df[group_by] == group_name].dropna(subset=['_exi_score'])
                if group_data.empty:
                    continue
                group_exi = group_data['_exi_score'].mean()
                by_group.append({
                    'group': str(group_name),
                    'exi': round(group_exi, 1),
                    'count': len(group_data),
                    'interpretation': self._interpret_exi(group_exi),
                })
            by_group.sort(key=lambda x: x['exi'], reverse=True)
            result['by_group'] = by_group

        return result

    def _interpret_exi(self, exi: float) -> str:
        """Get human-readable interpretation of EXI score."""
        if exi >= 80:
            return 'Excellent - Employees are thriving'
        elif exi >= 60:
            return 'Good - Employees are generally satisfied'
        elif exi >= 40:
            return 'Fair - Room for improvement'
        elif exi >= 20:
            return 'Poor - Significant engagement issues'
        else:
            return 'Critical - Immediate attention needed'

    def get_employee_exi(self, employee_id: str) -> Dict[str, Any]:
        """
        Get EXI details for a specific employee.

        Args:
            employee_id: The employee ID

        Returns:
            Dictionary with employee's EXI details.
        """
        emp_data = self.df[self.df['EmployeeID'] == employee_id]

        if emp_data.empty:
            return {
                'available': False,
                'reason': f'Employee {employee_id} not found'
            }

        row = emp_data.iloc[0]
        exi = row.get('_exi_score')
        if pd.isna(exi):
            return {'available': False, 'reason': 'No valid measured response for this employee'}

        # Determine segment
        segment = 'Unknown'
        for seg_name, (low, high) in self.SEGMENTS.items():
            if low <= exi < high + 1:
                segment = seg_name
                break

        result = {
            'available': True,
            'EmployeeID': employee_id,
            'exi_score': round(exi, 1),
            'segment': segment,
            'interpretation': self._interpret_exi(exi),
            'dept': row.get('Dept'),
        }

        # Add component breakdown if available
        components_str = row.get('_exi_components')
        if components_str and components_str != 'None':
            try:
                import ast
                result['components'] = ast.literal_eval(components_str)
            except Exception:
                pass

        return result

    # =========================================================================
    # ENGAGEMENT SEGMENTATION
    # =========================================================================

    def get_engagement_segments(self) -> Dict[str, Any]:
        """
        Segment workforce by engagement level.

        Returns:
            Dictionary with segment distribution.
        """
        if '_exi_score' not in self.df.columns or not self.df['_exi_score'].notna().any():
            return {'available': False, 'reason': 'EXI not computed'}

        segments = []
        total = int(self.df['_exi_score'].notna().sum())

        for seg_name, (low, high) in self.SEGMENTS.items():
            seg_df = self.df[
                (self.df['_exi_score'] >= low) &
                ((self.df['_exi_score'] < high + 1) if high < 100 else (self.df['_exi_score'] <= 100))
            ]
            count = len(seg_df)
            segments.append({
                'segment': seg_name,
                'count': count,
                'percentage': round(count / total * 100, 1) if total > 0 else 0,
                'avg_exi': round(seg_df['_exi_score'].mean(), 1) if count > 0 else None,
                'exi_range': f'{low}-{high}',
            })

        # Calculate health indicators
        thriving_pct = next(
            (s['percentage'] for s in segments if s['segment'] == 'Thriving'), 0
        )
        critical_pct = next(
            (s['percentage'] for s in segments if s['segment'] == 'Critical'), 0
        )
        disengaged_pct = next(
            (s['percentage'] for s in segments if s['segment'] == 'Disengaged'), 0
        )

        return {
            'available': True,
            'segments': segments,
            'total_employees': total,
            'health_indicator': self._get_health_indicator(thriving_pct, critical_pct + disengaged_pct),
            'thriving_percentage': thriving_pct,
            'at_risk_percentage': round(critical_pct + disengaged_pct, 1),
            'recommendations': self._generate_segment_recommendations(segments),
        }

    def _get_health_indicator(self, thriving_pct: float, at_risk_pct: float) -> str:
        """Determine overall workforce health indicator."""
        if thriving_pct >= 30 and at_risk_pct < 15:
            return 'Healthy'
        elif thriving_pct >= 20 and at_risk_pct < 25:
            return 'Moderate'
        elif at_risk_pct >= 30:
            return 'At Risk'
        else:
            return 'Needs Attention'

    def _generate_segment_recommendations(
        self,
        segments: List[Dict]
    ) -> List[str]:
        """Generate recommendations based on segment distribution."""
        recommendations = []

        for seg in segments:
            if seg['segment'] == 'Critical' and seg['percentage'] > 5:
                recommendations.append(
                    f"{seg['count']} employees are in Critical segment - "
                    "schedule immediate 1:1 conversations"
                )
            elif seg['segment'] == 'Disengaged' and seg['percentage'] > 15:
                recommendations.append(
                    f"{seg['percentage']}% of workforce is Disengaged - "
                    "review management practices and career development programs"
                )
            elif seg['segment'] == 'Thriving' and seg['percentage'] < 15:
                recommendations.append(
                    "Low percentage of Thriving employees - "
                    "investigate barriers to engagement"
                )

        if not recommendations:
            recommendations.append(
                "Engagement distribution is healthy - maintain current practices"
            )

        return recommendations

    # =========================================================================
    # EXPERIENCE DRIVERS
    # =========================================================================

    def identify_experience_drivers(self) -> Dict[str, Any]:
        """
        Identify what factors most impact experience scores.

        Returns:
            Dictionary with driver analysis.
        """
        if '_exi_score' not in self.df.columns or not self.df['_exi_score'].notna().any():
            return {'available': False, 'reason': 'EXI not computed'}

        drivers = []

        # Analyze correlation with available numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['_exi_score', 'EmployeeID'] + [c for c in self.df if c.lower() in self.EXPERIENCE_COLUMNS]

        for col in numeric_cols:
            if col in exclude_cols:
                continue
            pairs = self.df[['_exi_score', col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pairs) < 10 or (pairs.nunique() < 2).any():
                continue
            corr = pairs['_exi_score'].corr(pairs[col])
            if pd.notna(corr):
                drivers.append({
                    'factor': col,
                    'sample_size': len(pairs),
                    'metric_semantics': 'observational_association_excluding_index_components',
                    'correlation': round(corr, 3),
                    'impact': 'High' if abs(corr) >= 0.4 else (
                        'Medium' if abs(corr) >= 0.2 else 'Low'
                    ),
                    'direction': 'Positive' if corr > 0 else 'Negative',
                })

        # Sort by absolute correlation
        drivers.sort(key=lambda x: abs(x['correlation']), reverse=True)

        # Identify top positive and negative drivers
        positive_drivers = [d for d in drivers if d['correlation'] > 0.1][:3]
        negative_drivers = [d for d in drivers if d['correlation'] < -0.1][:3]

        return {
            'available': True,
            'drivers': drivers[:10],  # Top 10
            'top_positive_drivers': positive_drivers,
            'top_negative_drivers': negative_drivers,
            'recommendations': self._generate_driver_recommendations(
                positive_drivers, negative_drivers
            ),
        }

    def _generate_driver_recommendations(
        self,
        positive: List[Dict],
        negative: List[Dict]
    ) -> List[str]:
        """Generate recommendations based on driver analysis."""
        recommendations = []

        for driver in negative[:2]:
            factor = driver['factor']
            if 'promotion' in factor.lower():
                recommendations.append(
                    "Years since promotion are associated with lower measured scores - "
                    "review career progression policies"
                )
            elif 'tenure' in factor.lower() and driver['correlation'] < -0.2:
                recommendations.append(
                    "Long-tenured employees show lower experience - "
                    "implement tenure recognition and growth programs"
                )
            elif 'salary' in factor.lower():
                recommendations.append(
                    "Salary is associated with measured scores - "
                    "review pay equity and market competitiveness"
                )

        if positive:
            top = positive[0]['factor']
            recommendations.append(
                f"'{top}' is positively associated with the measured composite - "
                "investigate confounding and validate on independent data"
            )

        return recommendations

    # =========================================================================
    # AT-RISK EMPLOYEES
    # =========================================================================

    def get_at_risk_employees(
        self,
        threshold: Optional[float] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Identify employees with low or declining experience scores.

        Args:
            threshold: EXI threshold for at-risk (default from config)
            limit: Maximum number of employees to return

        Returns:
            Dictionary with at-risk employees.
        """
        if '_exi_score' not in self.df.columns or not self.df['_exi_score'].notna().any():
            return {'available': False, 'reason': 'EXI not computed'}

        threshold = self.exp_config.get('thresholds', {}).get('at_risk_exi', 40) if threshold is None else threshold

        at_risk_df = self.df[self.df['_exi_score'] < threshold].copy()
        at_risk_df = at_risk_df.sort_values('_exi_score')

        employees = []
        for _, row in at_risk_df.head(limit).iterrows():
            risk_factors = self._identify_risk_factors(row)
            employees.append({
                'EmployeeID': row['EmployeeID'],
                'Dept': row.get('Dept'),
                'current_exi': round(row['_exi_score'], 1),
                'segment': self._get_segment(row['_exi_score']),
                'tenure': row.get('Tenure'),
                'risk_factors': risk_factors,
                'recommended_actions': self._get_recommended_actions(risk_factors),
            })

        return {
            'available': True,
            'total_at_risk': len(at_risk_df),
            'threshold_used': threshold,
            'employees': employees,
            'by_department': self._get_at_risk_by_dept(at_risk_df),
        }

    def _get_segment(self, exi: float) -> str:
        """Get segment name for EXI score."""
        for seg_name, (low, high) in self.SEGMENTS.items():
            if low <= exi < high + 1:
                return seg_name
        return 'Unknown'

    def _identify_risk_factors(self, row: pd.Series) -> List[str]:
        """Identify risk factors for an employee."""
        factors = []

        # Low EXI overall
        if row.get('_exi_score', 100) < 30:
            factors.append('Very low experience score')

        # Tenure-related
        tenure = row.get('Tenure')
        if pd.notna(tenure):
            if float(tenure) < 0.5:
                factors.append('Very new hire (< 6 months)')
            elif float(tenure) > 7:
                factors.append('Long tenure without growth signals')

        # Promotion stagnation
        years_since = row.get('YearsSinceLastPromotion')
        if pd.notna(years_since) and float(years_since) > 4:
            factors.append('No promotion in 4+ years')

        # Low individual scores
        if self.has_manager_satisfaction:
            col = self._get_column('managersatisfaction')
            val = row.get(col)
            if pd.notna(val) and float(val) < 3:
                factors.append('Low manager satisfaction')

        if self.has_work_life:
            col = self._get_column('worklifebalance')
            val = row.get(col)
            if pd.notna(val) and float(val) < 3:
                factors.append('Poor work-life balance')

        if self.has_career_growth:
            col = self._get_column('careergrowthsatisfaction')
            val = row.get(col)
            if pd.notna(val) and float(val) < 3:
                factors.append('Low career growth satisfaction')

        if not factors:
            factors.append('General low engagement')

        return factors

    def _get_recommended_actions(self, risk_factors: List[str]) -> List[str]:
        """Get recommended actions based on risk factors."""
        actions = []

        for factor in risk_factors:
            if 'manager' in factor.lower():
                actions.append('Review manager relationship and consider manager training')
            elif 'promotion' in factor.lower():
                actions.append('Discuss career path and growth opportunities')
            elif 'work-life' in factor.lower():
                actions.append('Review workload and flexibility options')
            elif 'career growth' in factor.lower():
                actions.append('Create development plan with clear milestones')
            elif 'new hire' in factor.lower():
                actions.append('Ensure proper onboarding support and check-ins')
            elif 'tenure' in factor.lower():
                actions.append('Explore new challenges or lateral moves')

        if not actions:
            actions.append('Schedule stay interview to understand concerns')

        return list(set(actions))[:3]  # Dedupe and limit

    def _get_at_risk_by_dept(self, at_risk_df: pd.DataFrame) -> List[Dict]:
        """Get at-risk employee count by department."""
        if 'Dept' not in at_risk_df.columns or at_risk_df.empty:
            return []

        by_dept = at_risk_df.groupby('Dept').size().reset_index(name='count')
        by_dept = by_dept.sort_values('count', ascending=False)

        return [
            {'department': row['Dept'], 'at_risk_count': int(row['count'])}
            for _, row in by_dept.head(10).iterrows()
        ]

    # =========================================================================
    # LIFECYCLE ANALYSIS
    # =========================================================================

    def get_lifecycle_experience(self) -> Dict[str, Any]:
        """
        Analyze experience by employee lifecycle stage.

        Stages: New Hire, Ramping, Established, Veteran

        Returns:
            Dictionary with lifecycle analysis.
        """
        if '_exi_score' not in self.df.columns or not self.df['_exi_score'].notna().any():
            return {'available': False, 'reason': 'EXI not computed'}

        if not self.has_tenure:
            return {
                'available': False,
                'reason': 'Tenure column required for lifecycle analysis'
            }

        # Get stage thresholds from config
        stages_config = self.exp_config.get('lifecycle_stages', {})
        new_hire_months = stages_config.get('new_hire_months', 6)
        ramping_months = stages_config.get('ramping_months', 12)
        established_months = stages_config.get('established_months', 36)

        tenure_col = self._get_column('tenure')

        def get_stage(tenure_years: float) -> str:
            tenure_months = tenure_years * 12
            if tenure_months < new_hire_months:
                return 'New Hire'
            elif tenure_months < ramping_months:
                return 'Ramping'
            elif tenure_months < established_months:
                return 'Established'
            else:
                return 'Veteran'

        self.df['_lifecycle_stage'] = self.df[tenure_col].apply(
            lambda x: get_stage(float(x)) if pd.notna(x) and np.isfinite(float(x)) and float(x) >= 0 else 'Unknown'
        )

        stages = []
        for stage in ['New Hire', 'Ramping', 'Established', 'Veteran', 'Unknown']:
            stage_df = self.df[self.df['_lifecycle_stage'] == stage]
            if len(stage_df) > 0:
                stages.append({
                    'stage': stage,
                    'count': len(stage_df),
                    'avg_exi': round(stage_df['_exi_score'].mean(), 1) if stage_df['_exi_score'].notna().any() else None,
                    'respondent_count': int(stage_df['_exi_score'].count()),
                    'at_risk_count': len(stage_df[stage_df['_exi_score'] < 40]),
                })

        # Identify concerning patterns
        concerns = []
        for i, stage in enumerate(stages[:-1]):
            next_stage = stages[i + 1] if i + 1 < len(stages) else None
            if next_stage and stage['avg_exi'] is not None and next_stage['avg_exi'] is not None and stage['avg_exi'] - next_stage['avg_exi'] > 10:
                concerns.append(
                    f"Observed composite is more than 10 points higher in {stage['stage']} "
                    f"than {next_stage['stage']}; cross-sectional groups do not establish a lifecycle change"
                )

        return {
            'available': True,
            'stages': stages,
            'concerns': concerns,
            'recommendations': self._generate_lifecycle_recommendations(stages),
        }

    def _generate_lifecycle_recommendations(
        self,
        stages: List[Dict]
    ) -> List[str]:
        """Generate recommendations for lifecycle patterns."""
        recommendations = []

        for stage in stages:
            if stage['avg_exi'] is None:
                continue
            if stage['stage'] == 'New Hire' and stage['avg_exi'] < 50:
                recommendations.append(
                    "New hires have low experience scores - review onboarding program"
                )
            elif stage['stage'] == 'Veteran' and stage['avg_exi'] < 50:
                recommendations.append(
                    "Veterans have low experience - consider recognition and "
                    "growth opportunities for long-tenured employees"
                )

        return recommendations

    # =========================================================================
    # MANAGER IMPACT
    # =========================================================================

    def analyze_manager_impact(self) -> Dict[str, Any]:
        """
        Analyze how managers affect team experience scores.

        Returns:
            Dictionary with manager impact analysis.
        """
        if self._get_column('managerid') is None:
            return {'available': False, 'reason': 'ManagerID is required'}
        if '_exi_score' not in self.df.columns or not self.df['_exi_score'].notna().any():
            return {'available': False, 'reason': 'EXI not computed'}

        if 'ManagerID' not in self.df.columns:
            return {
                'available': False,
                'reason': 'ManagerID column required for manager impact analysis'
            }

        # Group by manager
        manager_stats = []
        for manager_id in self.df['ManagerID'].dropna().unique():
            team = self.df[self.df['ManagerID'] == manager_id]
            respondents = team.dropna(subset=['_exi_score'])
            if len(respondents) < 2:  # Need at least 2 reports for meaningful analysis
                continue

            avg_exi = team['_exi_score'].mean()
            at_risk = len(team[team['_exi_score'] < 40])

            manager_stats.append({
                'ManagerID': str(manager_id),
                'team_size': len(team),
                'avg_team_exi': round(avg_exi, 1),
                'at_risk_count': at_risk,
                'at_risk_percentage': round(at_risk / len(respondents) * 100, 1),
                'respondent_count': len(respondents),
            })

        # Sort by EXI (lowest first to highlight concerns)
        manager_stats.sort(key=lambda x: x['avg_team_exi'])

        # Overall statistics
        overall_avg = self.df['_exi_score'].mean()
        managers_below_avg = [m for m in manager_stats if m['avg_team_exi'] < overall_avg]

        return {
            'available': True,
            'managers_analyzed': len(manager_stats),
            'overall_avg_exi': round(overall_avg, 1),
            'managers_below_average': len(managers_below_avg),
            'bottom_managers': manager_stats[:5],  # Lowest performing teams
            'top_managers': manager_stats[-5:][::-1],  # Highest performing teams
            'recommendations': [
                "Review practices of managers with high-performing teams",
                "Provide coaching support to managers with low team EXI",
                "Consider manager training focused on employee experience"
            ] if managers_below_avg else [],
        }

    # =========================================================================
    # SIGNALS STATUS
    # =========================================================================

    def get_available_signals(self) -> Dict[str, Any]:
        """
        Report which experience signals are available.

        Returns:
            Dictionary with signal availability.
        """
        signals = {
            'has_enps': self.has_enps,
            'has_onboarding': self.has_onboarding,
            'has_pulse': self.has_pulse,
            'has_manager_satisfaction': self.has_manager_satisfaction,
            'has_engagement': self.has_engagement,
            'has_work_life': self.has_work_life,
            'has_career_growth': self.has_career_growth,
            'total_signals': self.available_survey_signals,
        }

        # Count usable measured responses, including every configured signal.
        signals['coverage_percentage'] = round(
            self.respondent_count / len(self.df) * 100, 1
        ) if len(self.df) else 0.

        # Generate recommendations for improving data
        recommendations = []
        if not self.has_enps:
            recommendations.append(
                "Add eNPS_Score column to enable employee advocacy tracking"
            )
        if not self.has_pulse:
            recommendations.append(
                "Add Pulse_Score column for regular engagement pulse checks"
            )
        if not self.has_manager_satisfaction:
            recommendations.append(
                "Add ManagerSatisfaction column to track manager effectiveness"
            )
        if signals['coverage_percentage'] < 50:
            recommendations.append(
                "Increase survey response rates - less than 50% have experience data"
            )

        signals['recommendations'] = recommendations

        return signals

    # =========================================================================
    # COMPREHENSIVE ANALYSIS
    # =========================================================================

    def analyze_all(self) -> Dict[str, Any]:
        """
        Run all experience analyses and return comprehensive results.

        Returns:
            Dictionary with all analysis results.
        """
        results = {
            'experience_index': {},
            'segments': {},
            'drivers': {},
            'at_risk': {},
            'lifecycle': {},
            'manager_impact': {},
            'signals': {},
            'summary': {},
            'recommendations': [],
            'warnings': self.warnings.copy(),
        }

        # Experience Index
        try:
            results['experience_index'] = self.calculate_experience_index()
        except Exception as e:
            self.logger.error(f"Experience index error: {e}")
            results['experience_index'] = {'available': False, 'error': str(e)}

        # Engagement Segments
        try:
            results['segments'] = self.get_engagement_segments()
            if results['segments'].get('recommendations'):
                results['recommendations'].extend(results['segments']['recommendations'])
        except Exception as e:
            self.logger.error(f"Segments error: {e}")
            results['segments'] = {'available': False, 'error': str(e)}

        # Experience Drivers
        try:
            results['drivers'] = self.identify_experience_drivers()
            if results['drivers'].get('recommendations'):
                results['recommendations'].extend(results['drivers']['recommendations'])
        except Exception as e:
            self.logger.error(f"Drivers error: {e}")
            results['drivers'] = {'available': False, 'error': str(e)}

        # At-Risk Employees
        try:
            results['at_risk'] = self.get_at_risk_employees()
            if results['at_risk'].get('total_at_risk', 0) > 0:
                results['warnings'].append(
                    f"{results['at_risk']['total_at_risk']} employees flagged as at-risk"
                )
        except Exception as e:
            self.logger.error(f"At-risk error: {e}")
            results['at_risk'] = {'available': False, 'error': str(e)}

        # Lifecycle Analysis
        try:
            results['lifecycle'] = self.get_lifecycle_experience()
            if results['lifecycle'].get('recommendations'):
                results['recommendations'].extend(results['lifecycle']['recommendations'])
        except Exception as e:
            self.logger.error(f"Lifecycle error: {e}")
            results['lifecycle'] = {'available': False, 'error': str(e)}

        # Manager Impact
        try:
            results['manager_impact'] = self.analyze_manager_impact()
        except Exception as e:
            self.logger.error(f"Manager impact error: {e}")
            results['manager_impact'] = {'available': False, 'error': str(e)}

        # Available Signals
        try:
            results['signals'] = self.get_available_signals()
        except Exception as e:
            self.logger.error(f"Signals error: {e}")
            results['signals'] = {'available': False, 'error': str(e)}

        # Overall Summary
        results['summary'] = {
            'overall_exi': results['experience_index'].get('overall_exi'),
            'health_indicator': results['segments'].get('health_indicator', 'Unknown'),
            'total_employees': len(self.df),
            'at_risk_count': results['at_risk'].get('total_at_risk', 0),
            'signals_available': self.available_survey_signals,
            'total_warnings': len(results['warnings']),
            'total_recommendations': len(results['recommendations']),
        }

        # Deduplicate recommendations
        results['recommendations'] = list(set(results['recommendations']))

        return results
