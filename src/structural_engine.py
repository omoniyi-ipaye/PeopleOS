"""
Structural Analysis Engine for PeopleOS.

Analyzes organizational structure including:
- Role stagnation detection
- Span of control analysis
- Promotion velocity equity audits
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats

from src.utils import load_config
from src.logger import get_logger


class StructuralEngine:
    """
    Engine for structural organizational analysis.

    Provides:
    - Stagnation index calculation (YearsInCurrentRole / Tenure)
    - Span of control analysis (manager direct reports)
    - Promotion velocity audits with controlled regression
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize StructuralEngine with employee data.

        Args:
            df: DataFrame with employee data including Tenure, YearsInCurrentRole,
                ManagerID, JobLevel, etc.
        """
        self.df = df.copy()
        self.config = load_config()
        self.structural_config = self.config.get('structural', {})
        self.logger = get_logger('structural_engine')

        # Config values
        self.span_config = self.structural_config.get('span_of_control', {})
        self.stagnation_config = self.structural_config.get('stagnation', {})
        self.promotion_config = self.structural_config.get('promotion_equity', {})

        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare data for analysis."""
        # Calculate stagnation index if we have the required columns
        if 'YearsInCurrentRole' in self.df.columns and 'Tenure' in self.df.columns:
            self.df['StagnationIndex'] = self.df.apply(
                lambda row: row['YearsInCurrentRole'] / row['Tenure']
                if row['Tenure'] > 0 else 0,
                axis=1
            )

        # Parse dates if needed
        if 'PromotionDate' in self.df.columns:
            self.df['PromotionDate'] = pd.to_datetime(
                self.df['PromotionDate'],
                errors='coerce'
            )

        if 'HireDate' in self.df.columns:
            self.df['HireDate'] = pd.to_datetime(
                self.df['HireDate'],
                errors='coerce'
            )

    # =========================================================================
    # STAGNATION ANALYSIS
    # =========================================================================

    def calculate_stagnation_index(self) -> pd.DataFrame:
        """
        Calculate stagnation index for all employees.

        Stagnation Index = YearsInCurrentRole / Tenure
        - Values close to 1.0 indicate potential stagnation
        - High tenure + high ratio = high stagnation risk

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
                'reason': 'Missing YearsInCurrentRole or Tenure columns'
            }

        # Filter to employees with sufficient tenure
        tenure_threshold = self.stagnation_config.get('tenure_threshold', 3.0)
        eligible = stagnation_df[stagnation_df['Tenure'] >= tenure_threshold]

        if eligible.empty:
            return {
                'available': False,
                'reason': f'No employees with tenure >= {tenure_threshold} years'
            }

        hotspots = []

        # Analyze by department
        if 'Dept' in eligible.columns:
            dept_stats = eligible.groupby('Dept').agg({
                'StagnationIndex': ['mean', 'count'],
                'EmployeeID': 'count'
            }).round(3)
            dept_stats.columns = ['avg_stagnation', 'stagnated_count', 'total']
            dept_stats = dept_stats.reset_index()

            # Count employees in warning or critical
            role_threshold = self.stagnation_config.get('role_threshold', 0.8)
            for dept in dept_stats['Dept'].unique():
                dept_data = eligible[eligible['Dept'] == dept]
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
            for level in eligible['JobLevel'].unique():
                level_data = eligible[eligible['JobLevel'] == level]
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
            }
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
        if 'ManagerID' not in self.df.columns:
            return pd.DataFrame()

        # Count direct reports per manager
        direct_reports = self.df.groupby('ManagerID').agg({
            'EmployeeID': 'count'
        }).reset_index()
        direct_reports.columns = ['ManagerID', 'DirectReports']

        # Get manager details
        managers = self.df[self.df['EmployeeID'].isin(direct_reports['ManagerID'])][
            ['EmployeeID', 'Dept', 'JobTitle', 'JobLevel', 'Location', 'LastRating', 'Tenure']
        ].copy()
        managers = managers.rename(columns={'EmployeeID': 'ManagerID'})

        # Merge
        span_df = direct_reports.merge(managers, on='ManagerID', how='left')

        # Categorize span of control
        optimal_min = self.span_config.get('optimal_min', 4)
        optimal_max = self.span_config.get('optimal_max', 8)
        warning_threshold = self.span_config.get('warning_threshold', 12)
        critical_threshold = self.span_config.get('critical_threshold', 15)

        def categorize_span(reports):
            if reports < optimal_min:
                return 'Under-Leveraged'
            elif reports <= optimal_max:
                return 'Optimal'
            elif reports <= warning_threshold:
                return 'Stretched'
            elif reports <= critical_threshold:
                return 'Overloaded'
            else:
                return 'Critical'

        span_df['SpanCategory'] = span_df['DirectReports'].apply(categorize_span)

        # Calculate burnout risk score (0-100)
        def burnout_risk(reports):
            if reports <= optimal_max:
                return min(reports * 5, 30)  # Low risk
            elif reports <= warning_threshold:
                return 30 + (reports - optimal_max) * 10
            else:
                return min(70 + (reports - warning_threshold) * 5, 100)

        span_df['BurnoutRiskScore'] = span_df['DirectReports'].apply(burnout_risk)

        return span_df.sort_values('DirectReports', ascending=False)

    def analyze_manager_burnout_risk(self) -> Dict[str, Any]:
        """
        Analyze manager burnout risk based on span of control.

        Returns:
            Dictionary with burnout risk analysis.
        """
        span_df = self.calculate_span_of_control()

        if span_df.empty:
            return {
                'available': False,
                'reason': 'ManagerID column not found in data'
            }

        warning_threshold = self.span_config.get('warning_threshold', 12)

        at_risk_managers = span_df[span_df['DirectReports'] > warning_threshold]

        # Get department summary
        dept_summary = []
        if 'Dept' in span_df.columns:
            for dept in span_df['Dept'].dropna().unique():
                dept_managers = span_df[span_df['Dept'] == dept]
                at_risk_in_dept = dept_managers[dept_managers['DirectReports'] > warning_threshold]
                dept_summary.append({
                    'department': dept,
                    'total_managers': len(dept_managers),
                    'at_risk_count': len(at_risk_in_dept),
                    'avg_span': round(dept_managers['DirectReports'].mean(), 1),
                    'max_span': int(dept_managers['DirectReports'].max())
                })

        return {
            'available': True,
            'at_risk_managers': at_risk_managers.to_dict('records'),
            'department_summary': sorted(
                dept_summary,
                key=lambda x: x['avg_span'],
                reverse=True
            ),
            'summary': {
                'total_managers': len(span_df),
                'at_risk_count': len(at_risk_managers),
                'avg_span': round(span_df['DirectReports'].mean(), 1),
                'max_span': int(span_df['DirectReports'].max()),
                'optimal_count': len(span_df[span_df['SpanCategory'] == 'Optimal']),
                'under_leveraged_count': len(span_df[span_df['SpanCategory'] == 'Under-Leveraged'])
            },
            'recommendations': self._generate_span_recommendations(span_df)
        }

    def _generate_span_recommendations(self, span_df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on span of control analysis."""
        recommendations = []

        warning_threshold = self.span_config.get('warning_threshold', 12)
        critical_threshold = self.span_config.get('critical_threshold', 15)

        critical_managers = span_df[span_df['DirectReports'] > critical_threshold]
        if len(critical_managers) > 0:
            recommendations.append(
                f"URGENT: {len(critical_managers)} manager(s) have more than "
                f"{critical_threshold} direct reports. Consider immediate restructuring "
                "to prevent burnout and improve team effectiveness."
            )

        overloaded = span_df[
            (span_df['DirectReports'] > warning_threshold) &
            (span_df['DirectReports'] <= critical_threshold)
        ]
        if len(overloaded) > 0:
            recommendations.append(
                f"{len(overloaded)} manager(s) are stretched with "
                f"{warning_threshold}-{critical_threshold} direct reports. "
                "Consider adding team leads or splitting teams."
            )

        under_leveraged = span_df[span_df['SpanCategory'] == 'Under-Leveraged']
        if len(under_leveraged) > len(span_df) * 0.3:
            recommendations.append(
                f"{len(under_leveraged)} managers have fewer than 4 direct reports. "
                "Consider consolidating teams to improve efficiency."
            )

        return recommendations

    # =========================================================================
    # PROMOTION VELOCITY EQUITY AUDIT
    # =========================================================================

    def audit_promotion_velocity(self) -> Dict[str, Any]:
        """
        Audit promotion velocity for equity across protected groups.

        Uses controlled regression to identify if certain groups wait
        longer for promotion after controlling for tenure, rating, etc.

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
                'reason': 'YearsSinceLastPromotion column not found'
            }

        # Prepare analysis data
        analysis_df = self.df[['EmployeeID', 'YearsSinceLastPromotion']].copy()

        # Add control variables
        controls = ['Tenure', 'LastRating', 'JobLevel']
        for ctrl in controls:
            if ctrl in self.df.columns:
                analysis_df[ctrl] = self.df[ctrl]

        # Add protected attributes
        available_attrs = []
        for attr in protected_attrs:
            if attr in self.df.columns:
                analysis_df[attr] = self.df[attr]
                available_attrs.append(attr)

        if not available_attrs:
            return {
                'available': False,
                'reason': f'None of the protected attributes found: {protected_attrs}'
            }

        # Drop rows with missing values
        analysis_df = analysis_df.dropna()

        if len(analysis_df) < 30:
            return {
                'available': False,
                'reason': f'Insufficient data for analysis (n={len(analysis_df)})'
            }

        audit_results = []

        for attr in available_attrs:
            groups = analysis_df[attr].unique()

            # Check minimum group sizes
            group_sizes = analysis_df.groupby(attr).size()
            valid_groups = group_sizes[group_sizes >= min_group_size].index.tolist()

            if len(valid_groups) < 2:
                continue

            # Perform controlled comparison
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
                "No statistically significant promotion velocity gaps detected "
                "across protected groups after controlling for performance factors."
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
                "Controlled analysis comparing YearsSinceLastPromotion across groups "
                f"while accounting for {', '.join(controls)}. "
                f"Statistical significance at p < {significance_level}."
            )
        }

    def _controlled_promotion_analysis(
        self,
        df: pd.DataFrame,
        attribute: str,
        controls: List[str],
        significance_level: float
    ) -> Optional[Dict[str, Any]]:
        """
        Perform controlled promotion velocity analysis for an attribute.

        Uses residualization approach:
        1. Regress YearsSinceLastPromotion on control variables
        2. Compare residuals across attribute groups
        """
        try:
            # Simple approach: compare means with t-test for 2 groups
            # or ANOVA for multiple groups, after residualizing controls

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
                    df[df[attribute] == g2]['YearsSinceLastPromotion']
                )
            else:
                # ANOVA for multiple groups
                group_data = [df[df[attribute] == g]['YearsSinceLastPromotion'].values
                             for g in groups]
                stat, p_value = stats.f_oneway(*group_data)

            # Determine significance
            is_significant = p_value < significance_level

            # Generate finding
            max_gap = max(gaps, key=lambda x: abs(x['gap_vs_reference'])) if gaps else None

            finding = ""
            if max_gap and is_significant:
                direction = "longer" if max_gap['gap_vs_reference'] > 0 else "shorter"
                finding = (
                    f"On average, {max_gap['group']} employees wait "
                    f"{abs(max_gap['gap_vs_reference']):.1f} years {direction} for promotion "
                    f"compared to {reference_group} employees."
                )
            elif not is_significant:
                finding = f"No significant difference in promotion wait times across {attribute} groups."

            return {
                'attribute': attribute,
                'reference_group': reference_group,
                'group_statistics': group_stats.to_dict('records'),
                'gaps': gaps,
                'p_value': round(p_value, 4),
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
                'reason': 'YearsSinceLastPromotion column not found'
            }

        bottlenecks = []

        # Analyze by department
        if 'Dept' in self.df.columns:
            dept_stats = self.df.groupby('Dept').agg({
                'YearsSinceLastPromotion': ['mean', 'median', 'count'],
                'EmployeeID': 'count'
            })
            dept_stats.columns = ['avg_wait', 'median_wait', 'has_promo_data', 'total']
            dept_stats = dept_stats.reset_index()

            overall_avg = self.df['YearsSinceLastPromotion'].mean()

            for _, row in dept_stats.iterrows():
                if row['avg_wait'] > overall_avg * 1.3:  # 30% above average
                    bottlenecks.append({
                        'type': 'department',
                        'name': row['Dept'],
                        'avg_years_since_promotion': round(row['avg_wait'], 1),
                        'median_years': round(row['median_wait'], 1),
                        'employee_count': int(row['total']),
                        'above_average_by': round(
                            (row['avg_wait'] - overall_avg) / overall_avg * 100, 1
                        )
                    })

        # Analyze by job level
        if 'JobLevel' in self.df.columns:
            level_stats = self.df.groupby('JobLevel').agg({
                'YearsSinceLastPromotion': ['mean', 'median', 'count']
            })
            level_stats.columns = ['avg_wait', 'median_wait', 'count']
            level_stats = level_stats.reset_index()

            # Mid-level often has longest wait
            for _, row in level_stats.iterrows():
                if row['avg_wait'] > 3.0 and row['count'] >= 10:  # Waiting >3 years
                    bottlenecks.append({
                        'type': 'job_level',
                        'name': f'Level {int(row["JobLevel"])}',
                        'avg_years_since_promotion': round(row['avg_wait'], 1),
                        'median_years': round(row['median_wait'], 1),
                        'employee_count': int(row['count'])
                    })

        # Find employees waiting longest
        long_waiters = self.df.nlargest(10, 'YearsSinceLastPromotion')[
            ['EmployeeID', 'Dept', 'JobLevel', 'JobTitle', 'YearsSinceLastPromotion',
             'LastRating', 'Tenure']
        ].to_dict('records') if not self.df.empty else []

        return {
            'available': True,
            'bottlenecks': bottlenecks,
            'employees_waiting_longest': long_waiters,
            'summary': {
                'avg_years_since_promotion': round(
                    self.df['YearsSinceLastPromotion'].mean(), 1
                ),
                'median_years': round(
                    self.df['YearsSinceLastPromotion'].median(), 1
                ),
                'employees_over_5_years': int(
                    (self.df['YearsSinceLastPromotion'] > 5).sum()
                )
            }
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
                        f"{summary['critical_count']} employees show critical role stagnation"
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
                        f"{at_risk} managers have excessive span of control"
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
                    f"{len(equity_audit['significant_gaps'])} significant promotion equity gap(s) detected"
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
