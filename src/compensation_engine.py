"""
Compensation Analysis Engine for PeopleOS.

Provides salary benchmarking, pay equity analysis, and compensation insights.
Includes gender pay gap analysis, compa-ratio calculations, and statistical
significance testing for HR compliance and decision-making.

Key Metrics:
- Pay Equity Score: Measures salary fairness within groups
- Compa-Ratio: Individual salary vs. band midpoint (1.0 = at midpoint)
- Gender Pay Gap: Unadjusted and adjusted (controlling for job level)
- Statistical Significance: Ensures findings are not due to random chance
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from scipy import stats

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('compensation_engine')

# Minimum sample sizes for reliable analysis
MIN_SAMPLE_FOR_STATS = 5
MIN_SAMPLE_FOR_COMPARISON = 10
SIGNIFICANCE_LEVEL = 0.05  # 95% confidence


class CompensationEngineError(Exception):
    """Custom exception for compensation engine errors."""
    pass


class CompensationEngine:
    """
    Compensation analysis engine for salary and pay equity insights.

    Designed for HR professionals to make informed, compliant decisions about
    compensation fairness and market competitiveness.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize Compensation Engine.

        Args:
            df: DataFrame with employee data including Salary column.
        """
        self.df = df.copy()
        self.config = load_config()
        self.comp_config = self.config.get('compensation', {})
        self.outlier_std_threshold = self.comp_config.get('outlier_std_threshold', 2.0)
        self.equity_warning_threshold = self.comp_config.get('equity_warning_threshold', 0.15)

        # Track available columns for conditional analysis
        self.has_gender = 'Gender' in self.df.columns
        self.has_job_title = 'JobTitle' in self.df.columns
        self.has_tenure = 'Tenure' in self.df.columns
        self.has_attrition = 'Attrition' in self.df.columns

        self._validate_data()
        self.warnings: List[str] = []
        logger.info(f"CompensationEngine initialized with {len(df)} employees")

    def _validate_data(self) -> None:
        """Validate required columns exist and data quality."""
        required = ['Salary']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise CompensationEngineError(f"Missing required columns: {missing}")

        # Data quality checks
        if self.df['Salary'].isna().any():
            na_count = self.df['Salary'].isna().sum()
            logger.warning(f"{na_count} employees have missing salary data")

        if (self.df['Salary'] <= 0).any():
            invalid_count = (self.df['Salary'] <= 0).sum()
            logger.warning(f"{invalid_count} employees have invalid salary (<=0)")

    def _add_warning(self, warning: str) -> None:
        """Add a warning message for HR review."""
        self.warnings.append(warning)
        logger.warning(warning)

    def _check_sample_size(self, n: int, context: str) -> bool:
        """Check if sample size is sufficient for reliable analysis."""
        if n < MIN_SAMPLE_FOR_STATS:
            self._add_warning(
                f"{context}: Sample size ({n}) too small for reliable analysis. "
                f"Minimum recommended: {MIN_SAMPLE_FOR_STATS}"
            )
            return False
        return True

    def calculate_salary_percentiles(self) -> pd.DataFrame:
        """
        Calculate salary percentiles by department.

        Returns:
            DataFrame with percentile breakdown including IQR for spread analysis.

        HR Interpretation:
            - p25/p50/p75: Show salary distribution shape
            - IQR (p75-p25): Measures salary spread - higher = more variation
            - Spread Ratio: IQR/Median - values > 0.5 may indicate pay inconsistency
        """
        df = self.df.copy()

        if 'Dept' not in df.columns:
            self._add_warning("Dept column missing - cannot calculate department percentiles")
            return pd.DataFrame()

        # Calculate percentiles by department
        dept_percentiles = df.groupby('Dept')['Salary'].agg([
            ('p25', lambda x: np.percentile(x, 25)),
            ('p50', lambda x: np.percentile(x, 50)),
            ('p75', lambda x: np.percentile(x, 75)),
            ('min', 'min'),
            ('max', 'max'),
            ('count', 'count')
        ]).round(0).reset_index()

        # Add IQR and spread ratio for HR interpretation
        dept_percentiles['IQR'] = dept_percentiles['p75'] - dept_percentiles['p25']
        dept_percentiles['SpreadRatio'] = (
            dept_percentiles['IQR'] / dept_percentiles['p50']
        ).round(2)

        # Flag departments with high spread
        dept_percentiles['SpreadFlag'] = dept_percentiles['SpreadRatio'].apply(
            lambda x: 'High Variation' if x > 0.5 else ('Moderate' if x > 0.3 else 'Consistent')
        )

        return dept_percentiles

    def calculate_pay_equity_score(self) -> pd.DataFrame:
        """
        Calculate pay equity metrics by department.

        Uses coefficient of variation and Gini-like measures.

        Returns:
            DataFrame with equity scores per department.
        """
        df = self.df.copy()

        results = []
        for dept in df['Dept'].unique():
            dept_df = df[df['Dept'] == dept]
            salaries = dept_df['Salary'].values

            if len(salaries) < 2:
                continue

            avg_salary = np.mean(salaries)
            std_salary = np.std(salaries)
            cv = std_salary / avg_salary if avg_salary > 0 else 0

            # Calculate Gini coefficient
            gini = self._calculate_gini(salaries)

            # Equity score (inverse of CV and Gini, normalized)
            equity_score = max(0, 1 - (cv + gini) / 2)

            # Determine status
            if equity_score >= 0.8:
                status = 'Good'
            elif equity_score >= 0.6:
                status = 'Fair'
            else:
                status = 'Needs Review'

            results.append({
                'Dept': dept,
                'AvgSalary': round(avg_salary, 0),
                'StdDev': round(std_salary, 0),
                'CV': round(cv, 3),
                'Gini': round(gini, 3),
                'EquityScore': round(equity_score, 2),
                'Status': status,
                'Headcount': len(dept_df)
            })

        return pd.DataFrame(results).sort_values('EquityScore', ascending=True)

    def _calculate_gini(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for an array of values."""
        if len(values) == 0:
            return 0

        sorted_values = np.sort(values)
        n = len(values)
        cumulative = np.cumsum(sorted_values)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n

        return max(0, min(1, gini))

    def identify_salary_outliers(self) -> pd.DataFrame:
        """
        Identify employees with outlier salaries.

        Uses z-score method within each department.

        Returns:
            DataFrame with outlier employees.
        """
        df = self.df.copy()
        outliers = []

        for dept in df['Dept'].unique():
            dept_df = df[df['Dept'] == dept]

            if len(dept_df) < 3:
                continue

            mean_sal = dept_df['Salary'].mean()
            std_sal = dept_df['Salary'].std()

            if std_sal == 0:
                continue

            for _, row in dept_df.iterrows():
                z_score = (row['Salary'] - mean_sal) / std_sal

                if abs(z_score) > self.outlier_std_threshold:
                    outliers.append({
                        'EmployeeID': row['EmployeeID'],
                        'Dept': dept,
                        'Salary': row['Salary'],
                        'DeptAvg': round(mean_sal, 0),
                        'DeviationPct': round((row['Salary'] - mean_sal) / mean_sal * 100, 1),
                        'ZScore': round(z_score, 2),
                        'Flag': 'High' if z_score > 0 else 'Low'
                    })

        result_df = pd.DataFrame(outliers)
        if result_df.empty:
            return result_df
        return result_df.sort_values('ZScore', key=abs, ascending=False)

    def get_salary_bands(self) -> pd.DataFrame:
        """
        Assign salary bands to employees.

        Returns:
            DataFrame with salary band assignments.
        """
        df = self.df.copy()

        # Calculate overall percentiles
        percentiles = [0, 25, 50, 75, 100]
        boundaries = [np.percentile(df['Salary'], p) for p in percentiles]

        # Assign bands
        def get_band(salary):
            if salary <= boundaries[1]:
                return 'Band 1 (Entry)'
            elif salary <= boundaries[2]:
                return 'Band 2 (Mid)'
            elif salary <= boundaries[3]:
                return 'Band 3 (Senior)'
            else:
                return 'Band 4 (Executive)'

        df['SalaryBand'] = df['Salary'].apply(get_band)

        band_summary = df.groupby('SalaryBand').agg({
            'Salary': ['min', 'max', 'mean', 'count']
        }).round(0)

        band_summary.columns = ['Min', 'Max', 'Avg', 'Count']
        band_summary = band_summary.reset_index()

        return band_summary

    def correlate_salary_with_attrition(self) -> Optional[Dict[str, Any]]:
        """
        Calculate correlation between salary and attrition with statistical significance.

        Returns:
            Dictionary with correlation coefficient, p-value, and interpretation,
            or None if Attrition column missing.

        HR Interpretation:
            - Negative correlation: Lower salary associated with higher attrition
            - p-value < 0.05: Finding is statistically significant
        """
        if 'Attrition' not in self.df.columns:
            return None

        # Use Pearson correlation with p-value
        correlation, p_value = stats.pearsonr(
            self.df['Salary'].fillna(self.df['Salary'].median()),
            self.df['Attrition']
        )

        # Interpretation for HR
        if abs(correlation) < 0.1:
            strength = "negligible"
        elif abs(correlation) < 0.3:
            strength = "weak"
        elif abs(correlation) < 0.5:
            strength = "moderate"
        else:
            strength = "strong"

        direction = "negative" if correlation < 0 else "positive"
        significant = p_value < SIGNIFICANCE_LEVEL

        interpretation = (
            f"There is a {strength} {direction} correlation between salary and attrition. "
        )
        if significant:
            interpretation += "This finding is statistically significant."
        else:
            interpretation += "This finding is NOT statistically significant - may be due to chance."

        return {
            'correlation': round(correlation, 3),
            'p_value': round(p_value, 4),
            'is_significant': significant,
            'strength': strength,
            'direction': direction,
            'interpretation': interpretation
        }

    def calculate_gender_pay_gap(self) -> Dict[str, Any]:
        """
        Calculate gender pay gap metrics (CRITICAL for HR compliance).

        Returns:
            Dictionary with unadjusted and adjusted pay gap metrics.

        HR Interpretation:
            - Unadjusted Gap: Raw difference in average pay (affected by job mix)
            - Adjusted Gap: Difference after controlling for job level (true equity measure)
            - Gap < 5%: Generally considered acceptable
            - Gap 5-10%: Warrants investigation
            - Gap > 10%: Requires immediate action

        Legal Context:
            Equal Pay Act requires equal pay for equal work. This analysis helps
            identify potential compliance issues.
        """
        if not self.has_gender:
            return {
                'available': False,
                'reason': 'Gender column not present in data',
                'recommendation': 'Add Gender column for pay equity compliance analysis'
            }

        df = self.df.copy()
        results = {
            'available': True,
            'unadjusted': {},
            'adjusted': {},
            'by_department': [],
            'warnings': [],
            'recommendations': []
        }

        # Get gender groups (handle various formats)
        gender_col = df['Gender'].astype(str).str.strip()
        unique_genders = gender_col.unique()

        # Identify comparison groups
        male_labels = ['Male', 'M', 'male', 'MALE']
        female_labels = ['Female', 'F', 'female', 'FEMALE']

        male_mask = gender_col.isin(male_labels)
        female_mask = gender_col.isin(female_labels)

        male_salaries = df.loc[male_mask, 'Salary']
        female_salaries = df.loc[female_mask, 'Salary']

        if len(male_salaries) < MIN_SAMPLE_FOR_COMPARISON or len(female_salaries) < MIN_SAMPLE_FOR_COMPARISON:
            results['warnings'].append(
                f"Sample size too small for reliable analysis (Male: {len(male_salaries)}, Female: {len(female_salaries)})"
            )

        # UNADJUSTED GAP (raw difference)
        male_avg = male_salaries.mean() if len(male_salaries) > 0 else 0
        female_avg = female_salaries.mean() if len(female_salaries) > 0 else 0

        if male_avg > 0:
            unadjusted_gap = ((male_avg - female_avg) / male_avg) * 100
        else:
            unadjusted_gap = 0

        # Statistical test for significance
        if len(male_salaries) >= MIN_SAMPLE_FOR_STATS and len(female_salaries) >= MIN_SAMPLE_FOR_STATS:
            t_stat, p_value = stats.ttest_ind(male_salaries, female_salaries)
            is_significant = p_value < SIGNIFICANCE_LEVEL
        else:
            t_stat, p_value = None, None
            is_significant = None

        results['unadjusted'] = {
            'male_avg_salary': round(male_avg, 0),
            'female_avg_salary': round(female_avg, 0),
            'gap_percentage': round(unadjusted_gap, 1),
            'gap_amount': round(male_avg - female_avg, 0),
            't_statistic': round(t_stat, 3) if t_stat else None,
            'p_value': round(p_value, 4) if p_value else None,
            'is_significant': is_significant,
            'male_count': len(male_salaries),
            'female_count': len(female_salaries)
        }

        # ADJUSTED GAP (controlling for job level/title)
        if self.has_job_title:
            adjusted_gaps = []
            for job in df['JobTitle'].unique():
                job_df = df[df['JobTitle'] == job]
                job_male = job_df.loc[job_df['Gender'].isin(male_labels), 'Salary']
                job_female = job_df.loc[job_df['Gender'].isin(female_labels), 'Salary']

                if len(job_male) >= 2 and len(job_female) >= 2:
                    gap = ((job_male.mean() - job_female.mean()) / job_male.mean()) * 100 if job_male.mean() > 0 else 0
                    adjusted_gaps.append({
                        'job_title': job,
                        'gap_percentage': round(gap, 1),
                        'male_count': len(job_male),
                        'female_count': len(job_female)
                    })

            if adjusted_gaps:
                avg_adjusted_gap = np.mean([g['gap_percentage'] for g in adjusted_gaps])
                results['adjusted'] = {
                    'avg_gap_percentage': round(avg_adjusted_gap, 1),
                    'by_job_title': adjusted_gaps,
                    'jobs_analyzed': len(adjusted_gaps)
                }

                # Flag jobs with significant gaps
                problem_jobs = [g for g in adjusted_gaps if abs(g['gap_percentage']) > 10]
                if problem_jobs:
                    results['warnings'].append(
                        f"{len(problem_jobs)} job titles have pay gaps exceeding 10%"
                    )
        else:
            results['adjusted'] = {
                'available': False,
                'reason': 'JobTitle column needed for adjusted analysis'
            }

        # BY DEPARTMENT ANALYSIS
        if 'Dept' in df.columns:
            for dept in df['Dept'].unique():
                dept_df = df[df['Dept'] == dept]
                dept_male = dept_df.loc[dept_df['Gender'].isin(male_labels), 'Salary']
                dept_female = dept_df.loc[dept_df['Gender'].isin(female_labels), 'Salary']

                if len(dept_male) >= 2 and len(dept_female) >= 2:
                    gap = ((dept_male.mean() - dept_female.mean()) / dept_male.mean()) * 100 if dept_male.mean() > 0 else 0
                    results['by_department'].append({
                        'department': dept,
                        'gap_percentage': round(gap, 1),
                        'male_avg': round(dept_male.mean(), 0),
                        'female_avg': round(dept_female.mean(), 0),
                        'flag': 'Critical' if abs(gap) > 15 else ('Warning' if abs(gap) > 8 else 'OK')
                    })

        # RECOMMENDATIONS
        if abs(unadjusted_gap) > 15:
            results['recommendations'].append(
                "CRITICAL: Unadjusted pay gap exceeds 15%. Immediate comprehensive review required."
            )
        elif abs(unadjusted_gap) > 8:
            results['recommendations'].append(
                "WARNING: Pay gap between 8-15%. Conduct detailed analysis by role and tenure."
            )

        if 'adjusted' in results and isinstance(results['adjusted'], dict) and 'avg_gap_percentage' in results['adjusted']:
            if abs(results['adjusted']['avg_gap_percentage']) > 5:
                results['recommendations'].append(
                    "Adjusted pay gap detected after controlling for job level. Review individual cases."
                )

        if not results['recommendations']:
            results['recommendations'].append(
                "Pay equity metrics within acceptable ranges. Continue regular monitoring."
            )

        return results

    def calculate_compa_ratio(self) -> pd.DataFrame:
        """
        Calculate compa-ratio for each employee (salary vs. band midpoint).

        Returns:
            DataFrame with compa-ratios and interpretations.

        HR Interpretation:
            - Compa-ratio 1.0: At band midpoint (target)
            - Compa-ratio < 0.85: Significantly below band - review for equity
            - Compa-ratio 0.85-0.95: Below midpoint - may need adjustment
            - Compa-ratio 0.95-1.05: At market rate
            - Compa-ratio 1.05-1.15: Above midpoint - high performer or tenure
            - Compa-ratio > 1.15: Significantly above - review for compression issues
        """
        df = self.df.copy()

        # Calculate band midpoints by department (or overall if no dept)
        if 'Dept' in df.columns:
            # Department-specific midpoints
            midpoints = df.groupby('Dept')['Salary'].median().to_dict()
            df['BandMidpoint'] = df['Dept'].map(midpoints)
        else:
            # Overall midpoint
            df['BandMidpoint'] = df['Salary'].median()

        # Calculate compa-ratio
        df['CompaRatio'] = (df['Salary'] / df['BandMidpoint']).round(2)

        # Add interpretation
        def interpret_compa(ratio):
            if ratio < 0.85:
                return 'Below Band - Review'
            elif ratio < 0.95:
                return 'Below Midpoint'
            elif ratio <= 1.05:
                return 'At Market'
            elif ratio <= 1.15:
                return 'Above Midpoint'
            else:
                return 'Above Band - Review'

        df['CompaStatus'] = df['CompaRatio'].apply(interpret_compa)

        # Select relevant columns
        result_cols = ['EmployeeID', 'Salary', 'BandMidpoint', 'CompaRatio', 'CompaStatus']
        if 'Dept' in df.columns:
            result_cols.insert(1, 'Dept')
        if 'JobTitle' in df.columns:
            result_cols.insert(2, 'JobTitle')

        result = df[[c for c in result_cols if c in df.columns]].copy()

        # Add summary statistics
        below_band = len(result[result['CompaRatio'] < 0.85])
        above_band = len(result[result['CompaRatio'] > 1.15])

        if below_band > 0:
            self._add_warning(f"{below_band} employees are significantly below their salary band midpoint")
        if above_band > 0:
            self._add_warning(f"{above_band} employees are significantly above their salary band midpoint")

        return result.sort_values('CompaRatio')

    def get_salary_by_tenure(self) -> pd.DataFrame:
        """
        Analyze salary progression by tenure.

        Returns:
            DataFrame with salary by tenure bucket.
        """
        df = self.df.copy()

        df['TenureBucket'] = pd.cut(
            df['Tenure'],
            bins=[0, 1, 2, 3, 5, 10, float('inf')],
            labels=['<1yr', '1-2yr', '2-3yr', '3-5yr', '5-10yr', '10+yr']
        )

        tenure_salary = df.groupby('TenureBucket', observed=True).agg({
            'Salary': ['mean', 'median', 'min', 'max', 'count']
        }).round(0)

        tenure_salary.columns = ['Mean', 'Median', 'Min', 'Max', 'Count']
        tenure_salary = tenure_salary.reset_index()

        return tenure_salary

    def get_compensation_summary(self) -> dict:
        """
        Get overall compensation summary statistics.

        Returns:
            Dictionary with compensation metrics.
        """
        salaries = self.df['Salary']

        return {
            'total_payroll': round(salaries.sum(), 0),
            'avg_salary': round(salaries.mean(), 0),
            'median_salary': round(salaries.median(), 0),
            'min_salary': round(salaries.min(), 0),
            'max_salary': round(salaries.max(), 0),
            'salary_range': round(salaries.max() - salaries.min(), 0),
            'std_dev': round(salaries.std(), 0),
            'headcount': len(self.df)
        }

    def get_dept_salary_comparison(self) -> pd.DataFrame:
        """
        Compare salaries across departments.

        Returns:
            DataFrame with department salary comparison.
        """
        overall_avg = self.df['Salary'].mean()

        dept_comp = self.df.groupby('Dept').agg({
            'Salary': ['mean', 'median', 'count']
        }).round(0)

        dept_comp.columns = ['AvgSalary', 'MedianSalary', 'Headcount']
        dept_comp = dept_comp.reset_index()

        # Add comparison to overall
        dept_comp['VsOverall'] = ((dept_comp['AvgSalary'] - overall_avg) / overall_avg * 100).round(1)
        dept_comp['Status'] = dept_comp['VsOverall'].apply(
            lambda x: 'Above Avg' if x > 5 else ('Below Avg' if x < -5 else 'At Avg')
        )

        return dept_comp.sort_values('AvgSalary', ascending=False)

    def analyze_all(self) -> dict:
        """
        Run full compensation analysis.

        Returns:
            Dictionary with all compensation analysis results including:
            - summary: Overall compensation metrics
            - percentiles: Salary distribution by department
            - equity: Pay equity scores by department
            - gender_pay_gap: Gender-based pay analysis (if Gender column present)
            - compa_ratio: Individual salary vs. band analysis
            - outliers: Salary outliers flagged for review
            - bands: Salary band distribution
            - by_tenure: Salary progression by tenure
            - dept_comparison: Cross-department comparison
            - attrition_correlation: Salary-attrition relationship
            - warnings: Any data quality or analysis warnings
        """
        logger.info("Running full compensation analysis")

        # Clear warnings for fresh analysis
        self.warnings = []

        results = {
            'summary': self.get_compensation_summary(),
            'percentiles': self.calculate_salary_percentiles(),
            'equity': self.calculate_pay_equity_score(),
            'gender_pay_gap': self.calculate_gender_pay_gap(),
            'compa_ratio': self.calculate_compa_ratio(),
            'outliers': self.identify_salary_outliers(),
            'bands': self.get_salary_bands(),
            'by_tenure': self.get_salary_by_tenure(),
            'dept_comparison': self.get_dept_salary_comparison(),
            'attrition_correlation': self.correlate_salary_with_attrition(),
            'warnings': self.warnings.copy()
        }

        # Add reliability score based on data quality
        results['reliability'] = self._calculate_reliability_score(results)

        logger.info(f"Compensation analysis complete. Warnings: {len(self.warnings)}")
        return results

    def _calculate_reliability_score(self, results: dict) -> Dict[str, Any]:
        """Calculate overall reliability score for the analysis."""
        score = 100
        issues = []

        # Check sample size
        if results['summary']['headcount'] < MIN_SAMPLE_FOR_COMPARISON:
            score -= 30
            issues.append("Sample size below minimum for reliable statistical analysis")

        # Check for gender pay gap data
        if not results['gender_pay_gap'].get('available', False):
            score -= 10
            issues.append("Gender data not available for pay equity compliance")

        # Check for warnings
        warning_count = len(results.get('warnings', []))
        score -= min(warning_count * 5, 20)
        if warning_count > 0:
            issues.append(f"{warning_count} data quality warnings detected")

        # Determine rating
        if score >= 90:
            rating = "High"
        elif score >= 70:
            rating = "Medium"
        else:
            rating = "Low"

        return {
            'score': max(0, score),
            'rating': rating,
            'issues': issues
        }
