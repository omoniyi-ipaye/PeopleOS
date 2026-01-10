"""
Succession Planning Engine for PeopleOS.

Provides succession candidate identification, readiness scoring,
and bench strength analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('succession_engine')


class SuccessionEngineError(Exception):
    """Custom exception for succession engine errors."""
    pass


class SuccessionEngine:
    """
    Succession planning engine for identifying high-potential employees
    and analyzing organizational bench strength.
    """

    def __init__(self, df: pd.DataFrame, risk_scores: Optional[pd.DataFrame] = None):
        """
        Initialize Succession Engine.

        Args:
            df: DataFrame with employee data.
            risk_scores: Optional DataFrame with attrition risk predictions.
        """
        self.df = df.copy()
        self.risk_scores = risk_scores
        self.config = load_config()
        self.succ_config = self.config.get('succession', {})

        self.high_performer_rating = self.succ_config.get('high_performer_rating', 4.0)
        self.min_tenure_years = self.succ_config.get('min_tenure_years', 2.0)
        self.readiness_weights = self.succ_config.get('readiness_weights', {
            'tenure': 0.3,
            'rating': 0.4,
            'risk_of_loss': 0.3
        })

        self._validate_data()
        logger.info(f"SuccessionEngine initialized with {len(df)} employees")

    def _validate_data(self) -> None:
        """Validate required columns exist."""
        required = ['EmployeeID', 'Dept', 'Tenure', 'LastRating']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise SuccessionEngineError(f"Missing required columns: {missing}")

    def calculate_readiness_scores(self) -> pd.DataFrame:
        """
        Calculate succession readiness scores for all employees.

        Returns:
            DataFrame with readiness scores and levels.
        """
        df = self.df.copy()

        # Normalize tenure score (0-1)
        max_tenure = df['Tenure'].max()
        df['TenureScore'] = df['Tenure'] / max_tenure if max_tenure > 0 else 0

        # Normalize rating score (assuming 1-5 scale)
        df['RatingScore'] = (df['LastRating'] - 1) / 4  # Convert 1-5 to 0-1

        # Risk of loss score (inverse - lower risk = higher score)
        if self.risk_scores is not None and not self.risk_scores.empty:
            df = df.merge(
                self.risk_scores[['EmployeeID', 'risk_score']],
                on='EmployeeID',
                how='left'
            )
            df['RiskScore'] = 1 - df['risk_score'].fillna(0.5)  # Invert: low risk = high score
        else:
            df['RiskScore'] = 0.5  # Neutral if no risk data

        # Calculate weighted readiness score
        weights = self.readiness_weights
        df['ReadinessScore'] = (
            weights.get('tenure', 0.3) * df['TenureScore'] +
            weights.get('rating', 0.4) * df['RatingScore'] +
            weights.get('risk_of_loss', 0.3) * df['RiskScore']
        )

        # Assign readiness level
        df['ReadinessLevel'] = df['ReadinessScore'].apply(self._get_readiness_level)

        result = df[['EmployeeID', 'Dept', 'Tenure', 'LastRating', 'ReadinessScore', 'ReadinessLevel']].copy()
        result['ReadinessScore'] = result['ReadinessScore'].round(2)

        return result.sort_values('ReadinessScore', ascending=False)

    def _get_readiness_level(self, score: float) -> str:
        """Map readiness score to level."""
        if score >= 0.75:
            return 'Ready Now'
        elif score >= 0.5:
            return 'Ready 1-2 Years'
        elif score >= 0.25:
            return 'Developing'
        else:
            return 'Early Career'

    def identify_high_potentials(self) -> pd.DataFrame:
        """
        Identify high-potential employees for succession.

        Returns:
            DataFrame with high-potential candidates.
        """
        df = self.df.copy()

        # Filter for high performers with sufficient tenure
        high_potentials = df[
            (df['LastRating'] >= self.high_performer_rating) &
            (df['Tenure'] >= self.min_tenure_years)
        ].copy()

        if high_potentials.empty:
            return pd.DataFrame(columns=['EmployeeID', 'Dept', 'Tenure', 'LastRating', 'PotentialLevel'])

        # Calculate potential level
        high_potentials['PotentialScore'] = (
            high_potentials['LastRating'] * 0.6 +
            np.minimum(high_potentials['Tenure'] / 10, 1) * 0.4 * 5
        )

        high_potentials['PotentialLevel'] = high_potentials['PotentialScore'].apply(
            lambda x: 'Star' if x >= 4.5 else ('High' if x >= 4.0 else 'Emerging')
        )

        # Add risk of loss if available
        if self.risk_scores is not None and not self.risk_scores.empty:
            merge_cols = ['EmployeeID', 'risk_score']
            if 'risk_category' in self.risk_scores.columns:
                merge_cols.append('risk_category')
            
            high_potentials = high_potentials.merge(
                self.risk_scores[merge_cols],
                on='EmployeeID',
                how='left'
            )
            
            # Calculate risk_category if not present
            if 'risk_category' not in high_potentials.columns:
                high_potentials['risk_category'] = high_potentials['risk_score'].apply(
                    lambda x: 'High' if x >= 0.75 else ('Medium' if x >= 0.5 else 'Low') if pd.notna(x) else 'Unknown'
                )
            
            high_potentials['AttritionRisk'] = high_potentials['risk_category'].fillna('Unknown')
        else:
            high_potentials['AttritionRisk'] = 'N/A'


        result = high_potentials[[
            'EmployeeID', 'Dept', 'Tenure', 'LastRating',
            'PotentialLevel', 'AttritionRisk'
        ]].copy()

        return result.sort_values('LastRating', ascending=False)

    def get_succession_pipeline(self) -> dict:
        """
        Get succession pipeline by department.

        Returns:
            Dictionary with pipeline counts per department.
        """
        readiness_df = self.calculate_readiness_scores()

        pipeline = {}
        for dept in self.df['Dept'].unique():
            dept_data = readiness_df[readiness_df['Dept'] == dept]

            pipeline[dept] = {
                'Ready Now': len(dept_data[dept_data['ReadinessLevel'] == 'Ready Now']),
                'Ready 1-2 Years': len(dept_data[dept_data['ReadinessLevel'] == 'Ready 1-2 Years']),
                'Developing': len(dept_data[dept_data['ReadinessLevel'] == 'Developing']),
                'Early Career': len(dept_data[dept_data['ReadinessLevel'] == 'Early Career']),
                'Total': len(dept_data)
            }

        return pipeline

    def calculate_bench_strength(self) -> pd.DataFrame:
        """
        Calculate bench strength score for each department.

        Returns:
            DataFrame with bench strength by department.
        """
        pipeline = self.get_succession_pipeline()

        results = []
        for dept, counts in pipeline.items():
            total = counts['Total']
            if total == 0:
                strength = 0
            else:
                # Weighted score: Ready Now has highest weight
                strength = (
                    counts['Ready Now'] * 1.0 +
                    counts['Ready 1-2 Years'] * 0.7 +
                    counts['Developing'] * 0.3 +
                    counts['Early Career'] * 0.1
                ) / total

            # Determine status
            if strength >= 0.6:
                status = 'Strong'
            elif strength >= 0.4:
                status = 'Adequate'
            else:
                status = 'Weak'

            results.append({
                'Dept': dept,
                'BenchStrength': round(strength, 2),
                'ReadyNow': counts['Ready Now'],
                'ReadySoon': counts['Ready 1-2 Years'],
                'Developing': counts['Developing'],
                'Total': total,
                'Status': status
            })

        return pd.DataFrame(results).sort_values('BenchStrength', ascending=True)

    def identify_critical_gaps(self) -> pd.DataFrame:
        """
        Identify departments with succession gaps.

        Returns:
            DataFrame with departments needing attention.
        """
        bench_df = self.calculate_bench_strength()

        # Flag departments with no ready successors
        gaps = bench_df[
            (bench_df['ReadyNow'] == 0) |
            (bench_df['BenchStrength'] < 0.3)
        ].copy()

        gaps['GapSeverity'] = gaps.apply(
            lambda x: 'Critical' if x['ReadyNow'] == 0 and x['ReadySoon'] == 0
            else ('High' if x['ReadyNow'] == 0 else 'Moderate'),
            axis=1
        )

        gaps['Recommendation'] = gaps['GapSeverity'].apply(
            lambda x: 'Immediate external hiring or accelerated development needed'
            if x == 'Critical' else (
                'Accelerate development pipeline' if x == 'High'
                else 'Monitor and continue development'
            )
        )

        return gaps.sort_values('GapSeverity')

    def get_retention_recommendations(self) -> list:
        """
        Generate retention recommendations for high-potential employees.

        Returns:
            List of recommendation dictionaries.
        """
        high_potentials = self.identify_high_potentials()

        if high_potentials.empty:
            return []

        recommendations = []

        # High-risk high-potentials
        if 'AttritionRisk' in high_potentials.columns:
            at_risk = high_potentials[high_potentials['AttritionRisk'] == 'High']
            for _, row in at_risk.iterrows():
                recommendations.append({
                    'EmployeeID': row['EmployeeID'],
                    'Dept': row['Dept'],
                    'Priority': 'Critical',
                    'Recommendation': 'Immediate retention intervention needed - high performer at high attrition risk',
                    'Actions': [
                        'Schedule career development discussion',
                        'Review compensation competitiveness',
                        'Explore leadership opportunities'
                    ]
                })

        # Star performers
        stars = high_potentials[high_potentials['PotentialLevel'] == 'Star']
        for _, row in stars.head(5).iterrows():
            if row['EmployeeID'] not in [r['EmployeeID'] for r in recommendations]:
                recommendations.append({
                    'EmployeeID': row['EmployeeID'],
                    'Dept': row['Dept'],
                    'Priority': 'High',
                    'Recommendation': 'Invest in star performer development',
                    'Actions': [
                        'Assign stretch assignments',
                        'Provide executive mentorship',
                        'Include in succession planning discussions'
                    ]
                })

        return recommendations[:10]  # Top 10 recommendations

    def get_9box_matrix(self) -> pd.DataFrame:
        """
        Generate 9-box matrix classification for employees.

        Returns:
            DataFrame with 9-box classifications.
        """
        df = self.df.copy()

        # Performance (LastRating) categories
        df['Performance'] = pd.cut(
            df['LastRating'],
            bins=[0, 2.5, 3.5, 5.1],
            labels=['Low', 'Medium', 'High']
        )

        # Potential (based on rating trajectory and growth signals)
        # Note: Uses RatingVelocity if available for better potential assessment
        if 'RatingVelocity' in df.columns:
            # RatingVelocity captures growth trajectory - better proxy for potential
            df['PotentialScore'] = (
                df['LastRating'] * 0.6 +
                (df['RatingVelocity'].clip(-1, 1) + 1) * 2.5 * 0.4  # Normalize -1 to 1 -> 0 to 5
            )
        else:
            # Fallback: Use pure performance as proxy for potential
            # Avoids conflating experience (tenure) with growth capacity (potential)
            df['PotentialScore'] = df['LastRating']

        df['Potential'] = pd.cut(
            df['PotentialScore'],
            bins=[0, 2.5, 3.5, 5.1],
            labels=['Low', 'Medium', 'High']
        )

        # 9-box classification
        def get_9box(row):
            perf = row['Performance']
            pot = row['Potential']

            if perf == 'High' and pot == 'High':
                return 'Stars'
            elif perf == 'High' and pot == 'Medium':
                return 'High Performers'
            elif perf == 'High' and pot == 'Low':
                return 'Solid Performers'
            elif perf == 'Medium' and pot == 'High':
                return 'High Potentials'
            elif perf == 'Medium' and pot == 'Medium':
                return 'Core Contributors'
            elif perf == 'Medium' and pot == 'Low':
                return 'Effective'
            elif perf == 'Low' and pot == 'High':
                return 'Potential Gems'
            elif perf == 'Low' and pot == 'Medium':
                return 'Inconsistent'
            else:
                return 'Underperformers'

        df['NineBox'] = df.apply(get_9box, axis=1)

        result = df[['EmployeeID', 'Dept', 'LastRating', 'Performance', 'Potential', 'NineBox']].copy()

        return result

    def get_9box_summary(self) -> pd.DataFrame:
        """
        Get summary counts for 9-box matrix.

        Returns:
            DataFrame with 9-box category counts.
        """
        nine_box = self.get_9box_matrix()

        summary = nine_box.groupby('NineBox').agg({
            'EmployeeID': 'count'
        }).reset_index()

        summary.columns = ['Category', 'Count']
        summary['Percentage'] = (summary['Count'] / summary['Count'].sum() * 100).round(1)

        # Order by typical 9-box priority
        order = ['Stars', 'High Potentials', 'High Performers', 'Potential Gems',
                 'Core Contributors', 'Solid Performers', 'Effective',
                 'Inconsistent', 'Underperformers']

        summary['SortOrder'] = summary['Category'].apply(
            lambda x: order.index(x) if x in order else 99
        )

        return summary.sort_values('SortOrder').drop(columns=['SortOrder'])

    def analyze_all(self) -> dict:
        """
        Run full succession analysis.

        Returns:
            Dictionary with all succession analysis results.
        """
        logger.info("Running full succession analysis")

        results = {
            'readiness': self.calculate_readiness_scores(),
            'high_potentials': self.identify_high_potentials(),
            'pipeline': self.get_succession_pipeline(),
            'bench_strength': self.calculate_bench_strength(),
            'gaps': self.identify_critical_gaps(),
            'recommendations': self.get_retention_recommendations(),
            'nine_box': self.get_9box_matrix(),
            'nine_box_summary': self.get_9box_summary()
        }

        logger.info("Succession analysis complete")
        return results
