"""
Analytics Engine module for PeopleOS.

Provides descriptive and diagnostic analytics for HR data.
No machine learning is performed in this module.
"""

from typing import Any, Optional

import pandas as pd
import numpy as np
from scipy import stats

from src.logger import get_logger
from src.utils import safe_divide, load_config

logger = get_logger('analytics_engine')


class AnalyticsEngine:
    """
    Computes descriptive and diagnostic analytics for HR data.

    Handles headcount, turnover rates, department aggregates,
    and correlation analysis.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the AnalyticsEngine with data.

        Args:
            df: Preprocessed DataFrame with Golden Schema columns.
        """
        self.df = df
        self.config = load_config()
        self.analytics_config = self.config.get('analytics', {})
        self.high_risk_threshold = self.analytics_config.get('high_risk_dept_threshold', 0.20)
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate that required columns are present."""
        required = ['EmployeeID', 'Dept', 'Tenure', 'Salary', 'LastRating', 'Age']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            logger.warning(f"Missing columns for analytics: {missing}")
    
    def get_headcount(self) -> int:
        """
        Get total headcount.
        
        Returns:
            Total number of employees.
        """
        return len(self.df)
    
    def get_turnover_rate(self) -> Optional[float]:
        """
        Calculate turnover rate.
        
        Returns:
            Turnover rate as a decimal (e.g., 0.15 for 15%), or None if Attrition column missing.
        """
        if 'Attrition' not in self.df.columns:
            logger.warning("Attrition column not present, cannot calculate turnover rate")
            return None
        
        # Attrition should be 1 for departed, 0 for active
        attrition_count = self.df['Attrition'].sum()
        total = len(self.df)
        
        return safe_divide(attrition_count, total)
    
    def get_department_aggregates(self) -> pd.DataFrame:
        """
        Calculate department-level aggregates.
        Base averages ONLY on active employees.
        
        Returns:
            DataFrame with department statistics.
        """
        if 'Dept' not in self.df.columns:
            logger.warning("Dept column not present")
            return pd.DataFrame()
            
        # 1. Get total records and turnover per department
        base_agg = self.df.groupby('Dept').agg({
            'EmployeeID': 'count',
            'Attrition': 'mean' if 'Attrition' in self.df.columns else lambda x: 0
        }).rename(columns={'EmployeeID': 'Total_Records', 'Attrition': 'Turnover_Rate'})

        # 2. Get active employee metrics only
        active_df = self.df[self.df['Attrition'] == 0] if 'Attrition' in self.df.columns else self.df
        
        agg_funcs: dict[str, Any] = {
            'EmployeeID': 'count'
        }
        if 'Salary' in self.df.columns:
            agg_funcs['Salary'] = ['mean', 'median']
        if 'Tenure' in self.df.columns:
            agg_funcs['Tenure'] = 'mean'
        if 'LastRating' in self.df.columns:
            agg_funcs['LastRating'] = 'mean'
        if 'Age' in self.df.columns:
            agg_funcs['Age'] = 'mean'
            
        active_stats = active_df.groupby('Dept').agg(agg_funcs)
        
        # Flatten and rename active metrics
        active_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                               for col in active_stats.columns.values]
        
        active_stats = active_stats.rename(columns={
            'EmployeeID_count': 'Headcount', # This is active headcount now
            'Salary_mean': 'Avg_Salary',
            'Salary_median': 'Median_Salary',
            'Tenure_mean': 'Avg_Tenure',
            'LastRating_mean': 'Avg_Rating',
            'Age_mean': 'Avg_Age'
        })

        # 3. Merge
        dept_stats = base_agg.join(active_stats, how='left').fillna(0).reset_index()
        
        logger.info(f"Generated active-first department aggregates for {len(dept_stats)} departments")
        return dept_stats
    
    def get_correlations(self, target_column: str = 'Attrition', max_features: int = 20) -> pd.DataFrame:
        """
        Calculate correlations between numeric features and target.

        Args:
            target_column: Target column for correlation analysis.
            max_features: Maximum number of features to include (for performance).

        Returns:
            DataFrame with correlation values.
        """
        if target_column not in self.df.columns:
            logger.warning(f"Target column {target_column} not present for correlation analysis")
            return pd.DataFrame()

        # Select numeric columns only
        numeric_df = self.df.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])

        if numeric_df.empty:
            return pd.DataFrame()

        # Limit columns for wide datasets (performance optimization)
        if len(numeric_df.columns) > max_features + 1:
            # Keep target and columns with highest variance
            variances = numeric_df.var().sort_values(ascending=False)
            keep_cols = [target_column] + [c for c in variances.index if c != target_column][:max_features]
            numeric_df = numeric_df[[c for c in keep_cols if c in numeric_df.columns]]

        # Calculate correlation with target
        correlations = numeric_df.corr()[target_column].drop(target_column, errors='ignore')

        result = pd.DataFrame({
            'Feature': correlations.index,
            'Correlation': correlations.values,
            'Abs_Correlation': correlations.abs().values
        }).sort_values('Abs_Correlation', ascending=False)

        logger.info(f"Calculated correlations for {len(result)} features")
        return result
    
    def get_summary_statistics(self) -> dict:
        """
        Get overall summary statistics.
        Calculates means only for active employees.
        
        Returns:
            Dictionary with summary statistics.
        """
        active_mask = self.df['Attrition'] == 0 if 'Attrition' in self.df.columns else pd.Series([True] * len(self.df))
        active_df = self.df[active_mask]
        
        stats: dict[str, Any] = {
            'headcount': self.get_headcount(), # Total pool
            'turnover_rate': self.get_turnover_rate(),
            'department_count': self.df['Dept'].nunique() if 'Dept' in self.df.columns else 0,
            'active_count': int(active_mask.sum())
        }
        
        # Numeric column statistics (Active Only)
        numeric_cols = ['Salary', 'Tenure', 'LastRating', 'Age']
        for col in numeric_cols:
            if col in self.df.columns:
                stats[f'{col.lower()}_mean'] = float(active_df[col].mean())
                stats[f'{col.lower()}_median'] = float(active_df[col].median())
                stats[f'{col.lower()}_std'] = float(active_df[col].std())
        
        # Attrition breakdown
        if 'Attrition' in self.df.columns:
            stats['attrition_count'] = int(self.df['Attrition'].sum())
        
        # Temporal metrics (Active Only)
        # We'll need a way to filter internal methods if they use self.df
        # For now, let's keep it simple or update those methods too
        temp_stats = self.get_temporal_stats(active_only=True)
        if temp_stats:
            stats['temporal'] = temp_stats
            
        logger.info("Generated summary statistics (Active-First)")
        return stats

    def get_temporal_stats(self, active_only: bool = False) -> dict:
        """Calculate averages for temporal metrics."""
        df = self.df
        if active_only and 'Attrition' in self.df.columns:
            df = self.df[self.df['Attrition'] == 0]
            
        stats = {}
        if 'RatingVelocity' in df.columns:
            stats['avg_velocity'] = float(df['RatingVelocity'].mean())
        if 'PromotionLag' in df.columns:
            stats['avg_promo_lag'] = float(df['PromotionLag'].mean())
        if 'SalaryGrowth' in df.columns:
            stats['avg_salary_growth'] = float(df['SalaryGrowth'].mean())
        return stats
    
    def get_tenure_distribution(self) -> pd.DataFrame:
        """
        Get tenure distribution buckets.
        
        Returns:
            DataFrame with tenure buckets and counts.
        """
        if 'Tenure' not in self.df.columns:
            return pd.DataFrame()
        
        bins = [0, 1, 2, 5, 10, float('inf')]
        labels = ['<1 year', '1-2 years', '2-5 years', '5-10 years', '10+ years']
        
        self.df['Tenure_Bucket'] = pd.cut(
            self.df['Tenure'],
            bins=bins,
            labels=labels,
            right=False
        )
        
        distribution = self.df['Tenure_Bucket'].value_counts().reset_index()
        distribution.columns = ['Tenure_Range', 'Count']
        
        # Add turnover rate per bucket if possible
        if 'Attrition' in self.df.columns:
            turnover_by_tenure = self.df.groupby('Tenure_Bucket')['Attrition'].mean()
            distribution = distribution.merge(
                turnover_by_tenure.reset_index().rename(
                    columns={'Attrition': 'Turnover_Rate', 'Tenure_Bucket': 'Tenure_Range'}
                ),
                on='Tenure_Range',
                how='left'
            )
        
        return distribution
    
    def get_age_distribution(self) -> pd.DataFrame:
        """
        Get age distribution buckets.
        
        Returns:
            DataFrame with age buckets and counts.
        """
        if 'Age' not in self.df.columns:
            return pd.DataFrame()
        
        bins = [0, 25, 35, 45, 55, float('inf')]
        labels = ['Under 25', '25-34', '35-44', '45-54', '55+']
        
        self.df['Age_Bucket'] = pd.cut(
            self.df['Age'],
            bins=bins,
            labels=labels,
            right=False
        )
        
        distribution = self.df['Age_Bucket'].value_counts().reset_index()
        distribution.columns = ['Age_Range', 'Count']
        
        return distribution
    
    def get_salary_bands(self) -> pd.DataFrame:
        """
        Get salary distribution by percentile bands.
        
        Returns:
            DataFrame with salary bands and employee counts.
        """
        if 'Salary' not in self.df.columns:
            return pd.DataFrame()
        
        percentiles = [0, 0.25, 0.5, 0.75, 1.0]
        bands = []
        
        for i in range(len(percentiles) - 1):
            lower = self.df['Salary'].quantile(percentiles[i])
            upper = self.df['Salary'].quantile(percentiles[i + 1])
            count = len(self.df[(self.df['Salary'] >= lower) & (self.df['Salary'] <= upper)])
            bands.append({
                'Band': f'Q{i+1}',
                'Lower': lower,
                'Upper': upper,
                'Count': count
            })
        
        return pd.DataFrame(bands)
    
    def get_high_risk_departments(self, threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Identify departments with turnover above threshold.

        Args:
            threshold: Turnover rate threshold. If None, uses config value.

        Returns:
            DataFrame with high-risk departments.
        """
        if threshold is None:
            threshold = self.high_risk_threshold

        dept_stats = self.get_department_aggregates()

        if 'Turnover_Rate' not in dept_stats.columns:
            return pd.DataFrame()

        high_risk = dept_stats[dept_stats['Turnover_Rate'] > threshold]
        return high_risk.sort_values('Turnover_Rate', ascending=False)

    def compare_groups(self, group_col: str, metric_col: str) -> dict:
        """
        Perform statistical test to compare metric across groups.

        Uses T-test (for 2 groups) or ANOVA (for >2 groups).

        Args:
            group_col: Column defining groups (e.g., 'Gender', 'Dept').
            metric_col: Numeric metric to compare (e.g., 'Salary').

        Returns:
            Dictionary with test results:
            {
                'test_name': 'ANOVA',
                'statistic': 4.5,
                'p_value': 0.01,
                'is_significant': True,
                'interpretation': "Significant difference found..."
            }
        """
        if group_col not in self.df.columns or metric_col not in self.df.columns:
            return {'success': False, 'reason': 'Columns not found'}

        # Prepare data
        groups = self.df.dropna(subset=[group_col, metric_col]).groupby(group_col)[metric_col]
        group_list = [group.values for name, group in groups if len(group) > 5]
        group_names = [name for name, group in groups if len(group) > 5]

        if len(group_list) < 2:
            return {'success': False, 'reason': 'Not enough groups with data (>5 samples)'}

        try:
            if len(group_list) == 2:
                # T-Test
                stat, p_val = stats.ttest_ind(group_list[0], group_list[1], equal_var=False)
                test_name = "Welch's T-Test"
            else:
                # ANOVA
                stat, p_val = stats.f_oneway(*group_list)
                test_name = "One-way ANOVA"

            is_sig = p_val < 0.05
            
            interpretation = (
                f"Statistically significant difference found in {metric_col} across {group_col} ({test_name}, p={p_val:.4f})."
                if is_sig else
                f"No significant difference found in {metric_col} across {group_col} (p={p_val:.4f})."
            )

            return {
                'success': True,
                'test_name': test_name,
                'statistic': float(stat),
                'p_value': float(p_val),
                'is_significant': is_sig,
                'interpretation': interpretation,
                'groups_compared': group_names
            }

        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            return {'success': False, 'reason': str(e)}

    def get_confidence_interval(self, col: str, confidence: float = 0.95) -> Optional[tuple]:
        """
        Calculate confidence interval for the mean of a column.
        
        Args:
            col: Numeric column name.
            confidence: Confidence level (0.95 default).
            
        Returns:
            Tuple (lower, upper) or None.
        """
        if col not in self.df.columns:
            return None
            
        data = self.df[col].dropna()
        if len(data) < 2:
            return None
            
        mean = np.mean(data)
        sem = stats.sem(data)
        margin = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
        
        return float(mean - margin), float(mean + margin)
