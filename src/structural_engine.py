"""
Structural Analysis Engine for PeopleOS.

Analyzes organizational structure including:
- Role stagnation detection
- Span of control analysis
- Promotion velocity equity audits
"""

import pandas as pd
import numpy as np
from src.population import active_population
from typing import Dict, Any, List, Optional
from scipy import stats

from src.utils import load_config
from src.logger import get_logger


class StructuralEngine:
    """
    Engine for structural organizational analysis.

    Provides descriptive, aggregate-only structure observations:
    - Stagnation index calculation (YearsInCurrentRole / Tenure)
    - Span of control analysis (manager direct reports)
    - Unadjusted group comparisons of years since last promotion

    These measurements are not burnout, promotion-readiness, discrimination or
    causal organizational-health estimates.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize StructuralEngine with employee data.

        Args:
            df: DataFrame with employee data including Tenure, YearsInCurrentRole,
                ManagerID, JobLevel, etc.
        """
        self.df = active_population(df).copy()
        self.config = load_config()
        self.structural_config = self.config.get('structural', {})
        self.logger = get_logger('structural_engine')

        # Config values
        self.span_config = self.structural_config.get('span_of_control', {})
        self.stagnation_config = self.structural_config.get('stagnation', {})
        self.promotion_config = self.structural_config.get('promotion_equity', {})

        self._span_thresholds = self._read_span_thresholds()
        self._data_quality: Dict[str, Any] = {}
        self._reporting_quality: Dict[str, Any] = {}

        self._prepare_data()

    @staticmethod
    def _finite_nonnegative(value: Any, default: float) -> float:
        """Read a numeric threshold without allowing non-finite semantics."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        return numeric if np.isfinite(numeric) and numeric >= 0 else default

    def _read_span_thresholds(self) -> Dict[str, float]:
        """Return ordered, explicit span boundaries used by every span method."""
        optimal_min = self._finite_nonnegative(self.span_config.get('optimal_min', 4), 4)
        optimal_max = self._finite_nonnegative(self.span_config.get('optimal_max', 8), 8)
        warning = self._finite_nonnegative(self.span_config.get('warning_threshold', 12), 12)
        critical = self._finite_nonnegative(self.span_config.get('critical_threshold', 15), 15)
        if not optimal_min <= optimal_max < warning < critical:
            return {'optimal_min': 4, 'optimal_max': 8, 'warning_threshold': 12, 'critical_threshold': 15}
        return {
            'optimal_min': optimal_min,
            'optimal_max': optimal_max,
            'warning_threshold': warning,
            'critical_threshold': critical,
        }

    @staticmethod
    def _present(series: pd.Series) -> pd.Series:
        """Treat null and whitespace-only identifiers/date lexemes as missing."""
        return series.notna() & series.astype('string').str.strip().ne('')

    def _parse_dates(self) -> None:
        """Parse supported dates and retain row-level validity masks."""
        self._date_presence: Dict[str, pd.Series] = {}
        self._date_parse_failures: Dict[str, int] = {}
        for column in ('HireDate', 'PromotionDate', 'SnapshotDate'):
            if column not in self.df.columns:
                continue
            presence = self._present(self.df[column])
            parsed = pd.to_datetime(self.df[column], errors='coerce', utc=True).dt.tz_localize(None)
            self.df[column] = parsed
            self._date_presence[column] = presence
            self._date_parse_failures[column] = int((presence & parsed.isna()).sum())

        snapshot = self.df.get('SnapshotDate', pd.Series(pd.NaT, index=self.df.index))
        if snapshot.notna().any():
            observation = snapshot
        else:
            # A cross-sectional file without an explicit as-of date cannot
            # support future-date or elapsed-time reconciliation. Keep those
            # checks unavailable instead of inventing an analysis date from
            # the workforce's hire or promotion dates.
            observation = pd.Series(pd.NaT, index=self.df.index, dtype='datetime64[ns]')
        self._observation_dates = observation

        hire = self.df.get('HireDate', pd.Series(pd.NaT, index=self.df.index))
        promotion = self.df.get('PromotionDate', pd.Series(pd.NaT, index=self.df.index))
        has_snapshot = snapshot.notna()
        self._invalid_hire_date = hire.isna() & self._date_presence.get('HireDate', pd.Series(False, index=self.df.index))
        self._invalid_promotion_date = promotion.isna() & self._date_presence.get('PromotionDate', pd.Series(False, index=self.df.index))
        self._invalid_hire_date |= hire.notna() & has_snapshot & (hire > observation)
        self._invalid_promotion_date |= promotion.notna() & has_snapshot & (promotion > observation)
        self._invalid_promotion_date |= hire.notna() & promotion.notna() & (promotion < hire)

    def _record_data_quality(self) -> None:
        """Build explicit missingness/exclusion counts for aggregate consumers."""
        role_duration = self.df.get('StagnationIndex', pd.Series(np.nan, index=self.df.index))
        promotion_duration = self.df.get('YearsSinceLastPromotion', pd.Series(np.nan, index=self.df.index))
        self._data_quality = {
            'source_population': int(len(self.df)),
            'role_duration_observations': int(role_duration.notna().sum()),
            'role_duration_missing_or_invalid': int(role_duration.isna().sum()),
            'promotion_duration_observations': int(promotion_duration.notna().sum()),
            'promotion_duration_missing_or_invalid': int(promotion_duration.isna().sum()),
            'invalid_hire_dates': int(self._invalid_hire_date.sum()),
            'invalid_promotion_dates': int(self._invalid_promotion_date.sum()),
            'promotion_duration_date_mismatches': int(self._promotion_duration_mismatch.sum()),
            'date_parse_failures': {
                column: count for column, count in self._date_parse_failures.items() if count
            },
        }

    def _get_data_quality(self) -> Dict[str, Any]:
        """Return a copy of structural observation/exclusion metadata."""
        return {
            **self._data_quality,
            'reporting_links': dict(self._reporting_quality),
        }

    def _prepare_data(self) -> None:
        """Prepare data for analysis."""
        for column in ['Tenure', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'JobLevel']:
            if column in self.df:
                values = pd.to_numeric(self.df[column], errors='coerce')
                self.df[column] = values.where(np.isfinite(values) & (values >= 0))
        if 'YearsInCurrentRole' in self.df and 'Tenure' in self.df:
            valid = (self.df['Tenure'] > 0) & (self.df['YearsInCurrentRole'] <= self.df['Tenure'])
            self.df['StagnationIndex'] = (self.df['YearsInCurrentRole'] / self.df['Tenure']).where(valid)

        self._parse_dates()
        if 'StagnationIndex' in self.df.columns:
            self.df['StagnationIndex'] = self.df['StagnationIndex'].where(~self._invalid_hire_date)

        if 'YearsSinceLastPromotion' in self.df.columns:
            promotion_years = self.df['YearsSinceLastPromotion']
            promotion = self.df.get('PromotionDate', pd.Series(pd.NaT, index=self.df.index))
            expected_years = (self._observation_dates - promotion).dt.days / 365.25
            self._promotion_duration_mismatch = (
                promotion_years.notna()
                & expected_years.notna()
                & (expected_years >= 0)
                & (abs(promotion_years - expected_years) > 0.5)
            )
            self.df['YearsSinceLastPromotion'] = promotion_years.where(
                ~self._invalid_promotion_date & ~self._promotion_duration_mismatch
            )
        else:
            self._promotion_duration_mismatch = pd.Series(False, index=self.df.index)
        self._record_data_quality()

    # =========================================================================
    # STAGNATION ANALYSIS
    # =========================================================================

    def calculate_stagnation_index(self) -> pd.DataFrame:
        """
        Calculate stagnation index for all employees.

        Stagnation Index = YearsInCurrentRole / Tenure.
        Values are descriptive role-duration ratios; threshold labels are
        screening prompts and not employee performance or advancement scores.

        Returns:
            DataFrame with stagnation metrics per employee.
        """
        if 'StagnationIndex' not in self.df.columns:
            return pd.DataFrame()

        tenure_threshold = self.stagnation_config.get('tenure_threshold', 3.0)
        role_threshold = self.stagnation_config.get('role_threshold', 0.8)

        stagnation_df = self.df[['EmployeeID']].copy()
        stagnation_df['Tenure'] = self.df['Tenure']
        stagnation_df['YearsInCurrentRole'] = self.df.get('YearsInCurrentRole', 0)
        stagnation_df['StagnationIndex'] = self.df['StagnationIndex']

        # Determine stagnation category
        def categorize_stagnation(row):
            if pd.isna(row['StagnationIndex']):
                return 'Unavailable'
            if row['Tenure'] < tenure_threshold:
                return 'Too Early'
            elif row['StagnationIndex'] >= 0.9:
                return 'Critical'
            elif row['StagnationIndex'] >= role_threshold:
                return 'Warning'
            elif row['StagnationIndex'] >= 0.6:
                return 'Monitor'
            else:
                return 'Healthy'

        stagnation_df['StagnationCategory'] = stagnation_df.apply(categorize_stagnation, axis=1)

        # Add context columns
        for col in ['Dept', 'JobTitle', 'LastRating', 'Location', 'JobLevel']:
            if col in self.df.columns:
                stagnation_df[col] = self.df[col]

        # Add YearsSinceLastPromotion if available
        if 'YearsSinceLastPromotion' in self.df.columns:
            stagnation_df['YearsSinceLastPromotion'] = self.df['YearsSinceLastPromotion']

        return stagnation_df

    def identify_stagnation_hotspots(self) -> Dict[str, Any]:
        """
        Identify departments and job levels with high stagnation.

        Returns:
            Dictionary with hotspot analysis.
        """
        stagnation_df = self.calculate_stagnation_index()

        if stagnation_df.empty:
            return {
                'available': False,
                'reason': 'Missing or invalid YearsInCurrentRole and Tenure observations',
                'data_quality': self._get_data_quality(),
                'metric_semantics': 'aggregate_role_duration_screening_not_employee_performance_determination',
            }

        # Filter to employees with sufficient tenure
        tenure_threshold = self.stagnation_config.get('tenure_threshold', 3.0)
        role_threshold = self.stagnation_config.get('role_threshold', 0.8)
        eligible = stagnation_df[(stagnation_df['Tenure'] >= tenure_threshold) & stagnation_df['StagnationIndex'].notna()]

        if eligible.empty:
            return {
                'available': False,
                'reason': f'No valid employees with tenure >= {tenure_threshold} years',
                'data_quality': self._get_data_quality(),
                'metric_semantics': 'aggregate_role_duration_screening_not_employee_performance_determination',
            }

        hotspots = []

        # Analyze by department
        if 'Dept' in eligible.columns:
            eligible = eligible.copy()
            eligible['_department'] = eligible['Dept'].astype('string').str.strip().replace('', pd.NA).fillna('Unknown')
            dept_stats = eligible.groupby('_department', dropna=False).agg({
                'StagnationIndex': ['mean', 'count'],
                'EmployeeID': 'count'
            }).round(3)
            dept_stats.columns = ['avg_stagnation', 'stagnated_count', 'total']
            dept_stats = dept_stats.reset_index()

            # Count employees in warning or critical
            role_threshold = self.stagnation_config.get('role_threshold', 0.8)
            for dept in dept_stats['_department'].unique():
                dept_data = eligible[eligible['_department'] == dept]
                at_risk = len(dept_data[dept_data['StagnationIndex'] >= role_threshold])
                total = len(dept_data)
                if at_risk > 0 and total >= 5:
                    hotspots.append({
                        'type': 'department',
                        'name': dept,
                        'avg_stagnation_index': float(dept_data['StagnationIndex'].mean()),
                        'employees_at_risk': at_risk,
                        'total_employees': total,
                        'risk_pct': round(at_risk / total * 100, 1),
                        'severity': 'Critical' if at_risk / total > 0.3 else 'Warning'
                    })

        # Analyze by job level
        if 'JobLevel' in eligible.columns:
            eligible['_job_level'] = eligible['JobLevel'].astype('string').str.strip().replace('', pd.NA).fillna('Unknown')
            for level in eligible['_job_level'].unique():
                level_data = eligible[eligible['_job_level'] == level]
                role_threshold = self.stagnation_config.get('role_threshold', 0.8)
                at_risk = len(level_data[level_data['StagnationIndex'] >= role_threshold])
                total = len(level_data)
                if at_risk > 0 and total >= 5:
                    hotspots.append({
                        'type': 'job_level',
                        'name': f'Level {level}',
                        'avg_stagnation_index': float(level_data['StagnationIndex'].mean()),
                        'employees_at_risk': at_risk,
                        'total_employees': total,
                        'risk_pct': round(at_risk / total * 100, 1),
                        'severity': 'Critical' if at_risk / total > 0.3 else 'Warning'
                    })

        # Sort by risk percentage
        hotspots.sort(key=lambda x: x['risk_pct'], reverse=True)

        # Get employees at critical stagnation
        critical_employees = stagnation_df[
            stagnation_df['StagnationCategory'] == 'Critical'
        ].head(10).to_dict('records')

        return {
            'available': True,
            'hotspots': hotspots,
            'critical_employees': critical_employees,
            'summary': {
                'total_analyzed': len(eligible),
                'critical_count': len(eligible[eligible['StagnationIndex'] >= 0.9]),
                'warning_count': len(eligible[
                    (eligible['StagnationIndex'] >= role_threshold) &
                    (eligible['StagnationIndex'] < 0.9)
                ]),
                'avg_stagnation_index': round(eligible['StagnationIndex'].mean(), 3)
            },
            'population': {
                'source_population': int(len(self.df)),
                'eligible_population': int(len(eligible)),
                'coverage': round(len(eligible) / len(self.df), 4) if len(self.df) else 0.0,
            },
            'data_quality': self._get_data_quality(),
            'metric_semantics': 'aggregate_role_duration_screening_not_employee_performance_determination',
        }

    # =========================================================================
    # SPAN OF CONTROL ANALYSIS
    # =========================================================================

    def calculate_span_of_control(self) -> pd.DataFrame:
        """
        Calculate span of control (direct reports) for each manager.

        Returns:
            DataFrame with manager metrics.
        """
        if 'ManagerID' not in self.df.columns or 'EmployeeID' not in self.df.columns:
            self._reporting_quality = {
                'available': False,
                'reason': 'EmployeeID and ManagerID columns are required',
            }
            return pd.DataFrame()

        quality, valid_links = self._reporting_link_quality()
        self._reporting_quality = quality
        links = self.df.loc[valid_links]
        if links.empty:
            return pd.DataFrame()
        direct_reports = links.groupby('ManagerID').agg({
            'EmployeeID': 'count'
        }).reset_index()
        direct_reports.columns = ['ManagerID', 'DirectReports']

        # Get manager details
        columns = [c for c in ['EmployeeID', 'Dept', 'JobTitle', 'JobLevel', 'Location', 'LastRating', 'Tenure'] if c in self.df]
        managers = self.df[self.df['EmployeeID'].isin(direct_reports['ManagerID'])][columns].copy()
        managers = managers.rename(columns={'EmployeeID': 'ManagerID'})

        # Merge
        span_df = direct_reports.merge(managers, on='ManagerID', how='left')

        # Categorize span of control
        optimal_min = self._span_thresholds['optimal_min']
        optimal_max = self._span_thresholds['optimal_max']
        warning_threshold = self._span_thresholds['warning_threshold']
        critical_threshold = self._span_thresholds['critical_threshold']

        def categorize_span(reports):
            if reports < optimal_min:
                return 'Under-Leveraged'
            elif reports <= optimal_max:
                return 'Optimal'
            elif reports < warning_threshold:
                return 'Stretched'
            elif reports < critical_threshold:
                return 'Overloaded'
            else:
                return 'Critical'

        span_df['SpanCategory'] = span_df['DirectReports'].apply(categorize_span)
        span_df['MetricSemantics'] = 'recorded_reporting_span_screening_not_burnout_prediction'

        return span_df.sort_values('DirectReports', ascending=False)

    def _reporting_link_quality(self) -> tuple[Dict[str, Any], pd.Series]:
        """Validate one-hop links and identify all directed reporting cycles."""
        manager = self.df['ManagerID']
        employee = self.df['EmployeeID']
        manager_present = self._present(manager)
        employee_ids = set(employee.dropna().tolist())
        target_exists = manager.isin(employee_ids)
        self_link = manager_present & manager.eq(employee)
        dangling = manager_present & ~self_link & ~target_exists
        candidate = manager_present & ~self_link & target_exists

        parent = dict(zip(employee.loc[candidate].tolist(), manager.loc[candidate].tolist()))
        cycle_edges: set[tuple[Any, Any]] = set()
        visited: set[Any] = set()
        for start in parent:
            if start in visited:
                continue
            path: list[Any] = []
            positions: dict[Any, int] = {}
            node = start
            while node in parent and node not in visited:
                if node in positions:
                    for cycle_node in path[positions[node]:]:
                        cycle_edges.add((cycle_node, parent[cycle_node]))
                    break
                positions[node] = len(path)
                path.append(node)
                node = parent[node]
            visited.update(path)

        cycle_mask = pd.Series(False, index=self.df.index)
        for index, row in self.df.loc[candidate, ['EmployeeID', 'ManagerID']].iterrows():
            if (row['EmployeeID'], row['ManagerID']) in cycle_edges:
                cycle_mask.loc[index] = True
        valid_links = candidate & ~cycle_mask
        quality = {
            'available': True,
            'source_population': int(len(self.df)),
            'manager_observations': int(manager_present.sum()),
            'missing_manager_links': int((~manager_present).sum()),
            'self_links_excluded': int(self_link.sum()),
            'dangling_links_excluded': int(dangling.sum()),
            'cycle_links_excluded': int(cycle_mask.sum()),
            'valid_links_used': int(valid_links.sum()),
            'coverage': round(valid_links.sum() / len(self.df), 4) if len(self.df) else 0.0,
            'threshold_semantics': 'optimal_min_and_optimal_max_are_inclusive; warning_threshold_is_first_overloaded_span; critical_threshold_is_first_critical_span',
        }
        return quality, valid_links

    def _analyze_reporting_span(self) -> Dict[str, Any]:
        """
        Analyze recorded reporting span as an aggregate workload-screening signal.

        Returns:
            Dictionary with aggregate reporting-span analysis.
        """
        span_df = self.calculate_span_of_control()

        if span_df.empty:
            return {
                'available': False,
                'reason': self._reporting_quality.get('reason', 'No valid reporting links were observed'),
                'reporting_quality': self._reporting_quality,
                'metric_semantics': 'recorded_reporting_span_screening_not_burnout_prediction',
            }

        warning_threshold = self._span_thresholds['warning_threshold']

        workload_signal = span_df[span_df['DirectReports'] >= warning_threshold]

        # Get department summary
        dept_summary = []
        if 'Dept' in span_df.columns:
            for dept in span_df['Dept'].dropna().unique():
                dept_managers = span_df[span_df['Dept'] == dept]
                at_risk_in_dept = dept_managers[dept_managers['DirectReports'] >= warning_threshold]
                dept_summary.append({
                    'department': dept,
                    'total_managers': len(dept_managers),
                    'at_risk_count': len(at_risk_in_dept),
                    'avg_span': round(dept_managers['DirectReports'].mean(), 1),
                    'max_span': int(dept_managers['DirectReports'].max())
                })

        return {
            'available': True,
            'department_summary': sorted(
                dept_summary,
                key=lambda x: x['avg_span'],
                reverse=True
            ),
            'summary': {
                'total_managers': len(span_df),
                'at_risk_count': len(workload_signal),
                'avg_span': round(span_df['DirectReports'].mean(), 1),
                'max_span': int(span_df['DirectReports'].max()),
                'optimal_count': len(span_df[span_df['SpanCategory'] == 'Optimal']),
                'under_leveraged_count': len(span_df[span_df['SpanCategory'] == 'Under-Leveraged'])
            },
            'recommendations': self._generate_span_recommendations(span_df),
            'reporting_quality': self._reporting_quality,
            'metric_semantics': 'recorded_reporting_span_screening_not_burnout_prediction',
            'scientific_limits': [
                'Span thresholds are workload prompts, not observations of burnout, wellbeing or manager effectiveness.',
                'Reporting lines do not establish team health, promotion readiness, performance or causal organizational effects.',
            ],
        }

    def analyze_manager_burnout_risk(self) -> Dict[str, Any]:
        """Compatibility alias; returns structural span semantics, never burnout risk."""
        return self._analyze_reporting_span()

    def _generate_span_recommendations(self, span_df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on span of control analysis."""
        recommendations = []

        optimal_min = self._span_thresholds['optimal_min']
        warning_threshold = self._span_thresholds['warning_threshold']
        critical_threshold = self._span_thresholds['critical_threshold']

        critical_managers = span_df[span_df['DirectReports'] >= critical_threshold]
        if len(critical_managers) > 0:
            recommendations.append(
                f"{len(critical_managers)} recorded span(s) are at or above "
                f"{critical_threshold} direct reports. Review workload, responsibilities "
                "and local context before changing team design."
            )

        overloaded = span_df[
            (span_df['DirectReports'] >= warning_threshold) &
            (span_df['DirectReports'] < critical_threshold)
        ]
        if len(overloaded) > 0:
            recommendations.append(
                f"{len(overloaded)} recorded span(s) are at or above the workload prompt "
                f"of {warning_threshold} and below {critical_threshold}. Review team design "
                "with local workload evidence."
            )

        under_leveraged = span_df[span_df['SpanCategory'] == 'Under-Leveraged']
        if len(under_leveraged) > len(span_df) * 0.3:
            recommendations.append(
                f"{len(under_leveraged)} recorded span(s) have fewer than {optimal_min} direct reports. "
                "Review role design before drawing an efficiency conclusion."
            )

        return recommendations

    # =========================================================================
    # PROMOTION VELOCITY EQUITY AUDIT
    # =========================================================================

    def audit_promotion_velocity(self) -> Dict[str, Any]:
        """
        Audit promotion velocity for equity across protected groups.

        Compares observed years since last promotion. This does not estimate
        time to promotion, adjust for confounding, or establish discrimination.

        Returns:
            Dictionary with equity audit results.
        """
        protected_attrs = self.promotion_config.get('protected_attributes', ['Gender', 'Age_Group'])
        significance_level = self.promotion_config.get('significance_level', 0.05)
        min_group_size = self.promotion_config.get('min_group_size', 10)

        # Check if we have promotion data
        if 'YearsSinceLastPromotion' not in self.df.columns:
            return {
                'available': False,
                'reason': 'YearsSinceLastPromotion column not found',
                'metric_semantics': 'observational_promotion_velocity_screening_not_promotion_readiness_or_causal_discrimination_finding',
            }

        # Prepare analysis data
        analysis_df = self.df[['EmployeeID', 'YearsSinceLastPromotion']].copy()

        # Retained for signature compatibility; no adjustment is performed.
        controls = ['Tenure', 'LastRating', 'JobLevel']
        # Add protected attributes
        available_attrs = []
        for attr in protected_attrs:
            if attr in self.df.columns:
                analysis_df[attr] = self.df[attr]
                available_attrs.append(attr)

        if not available_attrs:
            return {
                'available': False,
                'reason': f'None of the protected attributes found: {protected_attrs}',
                'metric_semantics': 'observational_promotion_velocity_screening_not_promotion_readiness_or_causal_discrimination_finding',
            }

        analysis_df['YearsSinceLastPromotion'] = pd.to_numeric(analysis_df['YearsSinceLastPromotion'], errors='coerce')
        analysis_df = analysis_df.replace([np.inf, -np.inf], np.nan).dropna()
        analysis_df = analysis_df[analysis_df['YearsSinceLastPromotion'] >= 0]

        if len(analysis_df) < 30:
            return {
                'available': False,
                'reason': f'Insufficient valid promotion-duration observations (n={len(analysis_df)})',
                'data_quality': {
                    **self._get_data_quality(),
                    'promotion_analysis_population': int(len(analysis_df)),
                },
                'metric_semantics': 'observational_promotion_velocity_screening_not_promotion_readiness_or_causal_discrimination_finding',
            }

        audit_results = []

        for attr in available_attrs:
            groups = analysis_df[attr].unique()

            # Check minimum group sizes
            group_sizes = analysis_df.groupby(attr).size()
            valid_groups = group_sizes[group_sizes >= min_group_size].index.tolist()

            if len(valid_groups) < 2:
                continue

            # Perform unadjusted comparison
            attr_result = self._controlled_promotion_analysis(
                analysis_df[analysis_df[attr].isin(valid_groups)],
                attr,
                controls,
                significance_level
            )

            if attr_result:
                audit_results.append(attr_result)

        # Generate overall findings
        significant_gaps = [r for r in audit_results if r.get('significant_gap')]

        recommendations = []
        if significant_gaps:
            for gap in significant_gaps:
                recommendations.append(
                    f"Review promotion processes for {gap['attribute']}: "
                    f"{gap['finding']}"
                )
        else:
            recommendations.append(
                "No significant unadjusted group difference detected; this does not establish equity."
            )

        return {
            'available': True,
            'audit_results': audit_results,
            'significant_gaps': significant_gaps,
            'summary': {
                'employees_analyzed': len(analysis_df),
                'attributes_tested': len(audit_results),
                'significant_gaps_found': len(significant_gaps)
            },
            'recommendations': recommendations,
            'methodology': (
                "Unadjusted comparison of years since last promotion; no control variables "
                "or multiple-testing correction. This is not time to next promotion. "
                f"Exploratory significance threshold p < {significance_level}."
            ),
            'data_quality': {
                **self._get_data_quality(),
                'promotion_analysis_population': int(len(analysis_df)),
                'promotion_analysis_coverage': round(len(analysis_df) / len(self.df), 4) if len(self.df) else 0.0,
            },
            'metric_semantics': 'observational_promotion_velocity_screening_not_promotion_readiness_or_causal_discrimination_finding',
            'scientific_limits': [
                'Years since last promotion is a current-state duration, not time to next promotion or promotion readiness.',
                'Unadjusted group differences do not establish discrimination, causality or organizational health.',
            ],
        }

    def _controlled_promotion_analysis(
        self,
        df: pd.DataFrame,
        attribute: str,
        controls: List[str],
        significance_level: float
    ) -> Optional[Dict[str, Any]]:
        """Unadjusted descriptive comparison; controls are not applied."""
        try:
            groups = df[attribute].unique()

            # Calculate group statistics
            group_stats = df.groupby(attribute).agg({
                'YearsSinceLastPromotion': ['mean', 'std', 'count']
            }).round(3)
            group_stats.columns = ['mean_years', 'std_years', 'count']
            group_stats = group_stats.reset_index()

            # Simple comparison (raw means first)
            means = group_stats.set_index(attribute)['mean_years'].to_dict()

            # Get the reference group (largest group or first alphabetically)
            reference_group = group_stats.loc[group_stats['count'].idxmax(), attribute]

            # Calculate gaps
            gaps = []
            for _, row in group_stats.iterrows():
                grp = row[attribute]
                if grp != reference_group:
                    gap = row['mean_years'] - means[reference_group]
                    gaps.append({
                        'group': grp,
                        'mean_years': row['mean_years'],
                        'gap_vs_reference': round(gap, 2),
                        'count': int(row['count'])
                    })

            # Statistical test
            if len(groups) == 2:
                # Two-sample t-test
                g1, g2 = groups
                stat, p_value = stats.ttest_ind(
                    df[df[attribute] == g1]['YearsSinceLastPromotion'],
                    df[df[attribute] == g2]['YearsSinceLastPromotion'], equal_var=False
                )
            else:
                # ANOVA for multiple groups
                group_data = [df[df[attribute] == g]['YearsSinceLastPromotion'].values
                             for g in groups]
                stat, p_value = stats.f_oneway(*group_data)

            # Determine significance
            is_significant = bool(np.isfinite(p_value) and p_value < significance_level)

            # Generate finding
            max_gap = max(gaps, key=lambda x: abs(x['gap_vs_reference'])) if gaps else None

            finding = ""
            if max_gap and is_significant:
                direction = "longer" if max_gap['gap_vs_reference'] > 0 else "shorter"
                finding = (
                    f"On average, {max_gap['group']} employees have "
                    f"{abs(max_gap['gap_vs_reference']):.1f} years {direction} since last promotion "
                    f"compared to {reference_group} employees."
                )
            elif not is_significant:
                finding = f"No significant unadjusted difference in years since last promotion across {attribute} groups."

            return {
                'attribute': attribute,
                'reference_group': reference_group,
                'group_statistics': group_stats.to_dict('records'),
                'gaps': gaps,
                'p_value': round(float(p_value), 4) if np.isfinite(p_value) else None,
                'controls_applied': False,
                'multiple_testing_adjusted': False,
                'significant_gap': is_significant,
                'finding': finding
            }

        except Exception as e:
            self.logger.warning(f"Error in promotion analysis for {attribute}: {e}")
            return None

    def get_promotion_bottlenecks(self) -> Dict[str, Any]:
        """
        Identify promotion bottlenecks by department and job level.

        Returns:
            Dictionary with bottleneck analysis.
        """
        if 'YearsSinceLastPromotion' not in self.df.columns:
            return {
                'available': False,
                'reason': 'YearsSinceLastPromotion column not found',
                'metric_semantics': 'aggregate_promotion_wait_time_screening_not_promotion_readiness',
            }

        valid = self.df['YearsSinceLastPromotion'].notna()
        analysis = self.df.loc[valid].copy()
        if analysis.empty:
            return {
                'available': False,
                'reason': 'No valid YearsSinceLastPromotion observations',
                'data_quality': self._get_data_quality(),
                'metric_semantics': 'aggregate_promotion_wait_time_screening_not_promotion_readiness',
            }

        bottlenecks = []

        # Analyze by department
        if 'Dept' in analysis.columns:
            analysis['_department'] = analysis['Dept'].astype('string').str.strip().replace('', pd.NA).fillna('Unknown')
            dept_stats = analysis.groupby('_department', dropna=False).agg({
                'YearsSinceLastPromotion': ['mean', 'median', 'count'],
                'EmployeeID': 'count'
            })
            dept_stats.columns = ['avg_wait', 'median_wait', 'has_promo_data', 'total']
            dept_stats = dept_stats.reset_index()

            overall_avg = analysis['YearsSinceLastPromotion'].mean()

            for _, row in dept_stats.iterrows():
                if np.isfinite(overall_avg) and overall_avg > 0 and row['avg_wait'] > overall_avg * 1.3:
                    bottlenecks.append({
                        'type': 'department',
                        'name': row['_department'],
                        'avg_years_since_promotion': round(row['avg_wait'], 1),
                        'median_years': round(row['median_wait'], 1),
                        'employee_count': int(row['total']),
                        'above_average_by': round(
                            (row['avg_wait'] - overall_avg) / overall_avg * 100, 1
                        )
                    })

        # Analyze by job level
        if 'JobLevel' in analysis.columns:
            analysis['_job_level'] = analysis['JobLevel'].astype('string').str.strip().replace('', pd.NA).fillna('Unknown')
            level_stats = analysis.groupby('_job_level', dropna=False).agg({
                'YearsSinceLastPromotion': ['mean', 'median', 'count']
            })
            level_stats.columns = ['avg_wait', 'median_wait', 'count']
            level_stats = level_stats.reset_index()

            # Mid-level often has longest wait
            for _, row in level_stats.iterrows():
                if row['avg_wait'] > 3.0 and row['count'] >= 10:  # Waiting >3 years
                    bottlenecks.append({
                        'type': 'job_level',
                        'name': f'Level {row["_job_level"]}' if row['_job_level'] != 'Unknown' else 'Unknown',
                        'avg_years_since_promotion': round(row['avg_wait'], 1),
                        'median_years': round(row['median_wait'], 1),
                        'employee_count': int(row['count'])
                    })

        # Find employees waiting longest
        columns = [c for c in ['EmployeeID', 'Dept', 'JobLevel', 'JobTitle', 'YearsSinceLastPromotion', 'LastRating', 'Tenure'] if c in analysis]
        long_waiters = analysis.nlargest(10, 'YearsSinceLastPromotion')[columns].to_dict('records')

        return {
            'available': True,
            'bottlenecks': bottlenecks,
            'employees_waiting_longest': long_waiters,
            'summary': {
                'avg_years_since_promotion': round(
                    analysis['YearsSinceLastPromotion'].mean(), 1
                ),
                'median_years': round(
                    analysis['YearsSinceLastPromotion'].median(), 1
                ),
                'employees_over_5_years': int(
                    (analysis['YearsSinceLastPromotion'] > 5).sum()
                )
            },
            'population': {
                'source_population': int(len(self.df)),
                'eligible_population': int(len(analysis)),
                'coverage': round(len(analysis) / len(self.df), 4) if len(self.df) else 0.0,
            },
            'data_quality': self._get_data_quality(),
            'metric_semantics': 'aggregate_promotion_wait_time_screening_not_promotion_readiness',
        }

    # =========================================================================
    # COMPREHENSIVE ANALYSIS
    # =========================================================================

    def analyze_all(self) -> Dict[str, Any]:
        """
        Run all structural analyses and return comprehensive results.

        Returns:
            Dictionary with all analysis results.
        """
        results = {
            'stagnation': {},
            'span_of_control': {},
            'promotion_equity': {},
            'promotion_bottlenecks': {},
            'summary': {},
            'recommendations': [],
            'warnings': []
        }

        # Stagnation Analysis
        try:
            stagnation_df = self.calculate_stagnation_index()
            hotspots = self.identify_stagnation_hotspots()

            results['stagnation'] = {
                'available': not stagnation_df.empty,
                'employees': stagnation_df.to_dict('records') if not stagnation_df.empty else [],
                'hotspots': hotspots
            }

            if hotspots.get('available'):
                summary = hotspots.get('summary', {})
                if summary.get('critical_count', 0) > 0:
                    results['warnings'].append(
                        f"{summary['critical_count']} role-duration observations cross the critical screening threshold"
                    )
        except Exception as e:
            self.logger.error(f"Stagnation analysis error: {e}")
            results['stagnation'] = {'available': False, 'error': str(e)}

        # Span of Control Analysis
        try:
            span_analysis = self.analyze_manager_burnout_risk()
            results['span_of_control'] = span_analysis

            if span_analysis.get('available'):
                at_risk = span_analysis.get('summary', {}).get('at_risk_count', 0)
                if at_risk > 0:
                    results['warnings'].append(
                        f"{at_risk} recorded reporting spans cross the workload screening threshold"
                    )
                results['recommendations'].extend(
                    span_analysis.get('recommendations', [])
                )
        except Exception as e:
            self.logger.error(f"Span of control error: {e}")
            results['span_of_control'] = {'available': False, 'error': str(e)}

        # Promotion Equity Audit
        try:
            equity_audit = self.audit_promotion_velocity()
            results['promotion_equity'] = equity_audit

            if equity_audit.get('significant_gaps'):
                results['warnings'].append(
                    f"{len(equity_audit['significant_gaps'])} observed promotion-duration group difference(s) require review"
                )
                results['recommendations'].extend(
                    equity_audit.get('recommendations', [])
                )
        except Exception as e:
            self.logger.error(f"Promotion equity error: {e}")
            results['promotion_equity'] = {'available': False, 'error': str(e)}

        # Promotion Bottlenecks
        try:
            bottlenecks = self.get_promotion_bottlenecks()
            results['promotion_bottlenecks'] = bottlenecks

            if bottlenecks.get('bottlenecks'):
                results['warnings'].append(
                    f"{len(bottlenecks['bottlenecks'])} promotion bottleneck(s) identified"
                )
        except Exception as e:
            self.logger.error(f"Promotion bottlenecks error: {e}")
            results['promotion_bottlenecks'] = {'available': False, 'error': str(e)}

        # Overall Summary
        results['summary'] = {
            'stagnation_analysis': results['stagnation'].get('available', False),
            'span_of_control': results['span_of_control'].get('available', False),
            'promotion_equity': results['promotion_equity'].get('available', False),
            'total_warnings': len(results['warnings']),
            'total_recommendations': len(results['recommendations'])
        }

        return results
