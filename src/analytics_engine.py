"""
Analytics Engine module for PeopleOS.

Provides descriptive and diagnostic analytics for HR data.
No machine learning is performed in this module.
"""

from typing import Any, Optional

import pandas as pd

from src.logger import get_logger
from src.utils import safe_divide, format_percentage, load_config

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
        
        Returns:
            DataFrame with department statistics.
        """
        if 'Dept' not in self.df.columns:
            logger.warning("Dept column not present")
            return pd.DataFrame()
        
        agg_funcs: dict[str, Any] = {
            'EmployeeID': 'count'
        }
        
        # Add numeric column aggregations if present
        if 'Salary' in self.df.columns:
            agg_funcs['Salary'] = ['mean', 'median', 'std']
        if 'Tenure' in self.df.columns:
            agg_funcs['Tenure'] = 'mean'
        if 'LastRating' in self.df.columns:
            agg_funcs['LastRating'] = 'mean'
        if 'Age' in self.df.columns:
            agg_funcs['Age'] = 'mean'
        if 'Attrition' in self.df.columns:
            agg_funcs['Attrition'] = 'mean'  # This gives turnover rate per dept
        
        dept_stats = self.df.groupby('Dept').agg(agg_funcs)
        
        # Flatten column names
        dept_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                              for col in dept_stats.columns.values]
        
        # Rename for clarity
        dept_stats = dept_stats.rename(columns={
            'EmployeeID_count': 'Headcount',
            'Salary_mean': 'Avg_Salary',
            'Salary_median': 'Median_Salary',
            'Salary_std': 'Salary_StdDev',
            'Tenure_mean': 'Avg_Tenure',
            'LastRating_mean': 'Avg_Rating',
            'Age_mean': 'Avg_Age',
            'Attrition_mean': 'Turnover_Rate'
        })
        
        dept_stats = dept_stats.reset_index()
        logger.info(f"Generated department aggregates for {len(dept_stats)} departments")
        
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
        
        Returns:
            Dictionary with summary statistics.
        """
        stats: dict[str, Any] = {
            'headcount': self.get_headcount(),
            'turnover_rate': self.get_turnover_rate(),
            'department_count': self.df['Dept'].nunique() if 'Dept' in self.df.columns else 0
        }
        
        # Numeric column statistics
        numeric_cols = ['Salary', 'Tenure', 'LastRating', 'Age']
        for col in numeric_cols:
            if col in self.df.columns:
                stats[f'{col.lower()}_mean'] = float(self.df[col].mean())
                stats[f'{col.lower()}_median'] = float(self.df[col].median())
                stats[f'{col.lower()}_std'] = float(self.df[col].std())
        
        # Attrition breakdown
        if 'Attrition' in self.df.columns:
            stats['attrition_count'] = int(self.df['Attrition'].sum())
            stats['active_count'] = int((self.df['Attrition'] == 0).sum())
        
        # Temporal metrics
        temp_stats = self.get_temporal_stats()
        if temp_stats:
            stats['temporal'] = temp_stats
            
        logger.info("Generated summary statistics")
        return stats

    def get_temporal_stats(self) -> dict:
        """Calculate averages for temporal metrics."""
        stats = {}
        if 'RatingVelocity' in self.df.columns:
            stats['avg_velocity'] = float(self.df['RatingVelocity'].mean())
        if 'PromotionLag' in self.df.columns:
            stats['avg_promo_lag'] = float(self.df['PromotionLag'].mean())
        if 'SalaryGrowth' in self.df.columns:
            stats['avg_salary_growth'] = float(self.df['SalaryGrowth'].mean())
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
