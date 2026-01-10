"""
Team Dynamics Engine for PeopleOS.

Provides team health scoring, diversity analysis, and collaboration metrics.
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('team_dynamics_engine')


class TeamDynamicsEngineError(Exception):
    """Custom exception for team dynamics engine errors."""
    pass


class TeamDynamicsEngine:
    """
    Team dynamics analysis engine for team health and diversity insights.
    """

    def __init__(self, df: pd.DataFrame, nlp_data: Optional[dict] = None):
        """
        Initialize Team Dynamics Engine.

        Args:
            df: DataFrame with employee data.
            nlp_data: Optional NLP analysis results for sentiment integration.
        """
        self.df = df.copy()
        self.nlp_data = nlp_data
        self.config = load_config()
        self.team_config = self.config.get('team_dynamics', {})

        self.min_team_size = self.team_config.get('min_team_size', 3)
        self.health_warning_threshold = self.team_config.get('health_warning_threshold', 0.6)
        self.diversity_metrics = self.team_config.get('diversity_metrics', ['age', 'tenure', 'salary'])

        self._validate_data()
        logger.info(f"TeamDynamicsEngine initialized with {len(df)} employees")

    def _validate_data(self) -> None:
        """Validate required columns exist."""
        required = ['EmployeeID', 'Dept']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise TeamDynamicsEngineError(f"Missing required columns: {missing}")

    def calculate_team_health_scores(self) -> pd.DataFrame:
        """
        Calculate health scores for each team/department.

        Returns:
            DataFrame with team health metrics.
        """
        df = self.df.copy()
        results = []

        for dept in df['Dept'].unique():
            dept_df = df[df['Dept'] == dept]

            if len(dept_df) < self.min_team_size:
                continue

            health_components = {}

            # Performance health (based on ratings)
            if 'LastRating' in dept_df.columns:
                avg_rating = dept_df['LastRating'].mean()
                rating_variance = dept_df['LastRating'].std()
                # Handle NaN from std() when only 1-2 samples
                if pd.isna(rating_variance):
                    rating_variance = 0
                # Higher avg rating and lower variance = better health
                health_components['performance'] = min(avg_rating / 5, 1) * 0.7 + max(0, 1 - rating_variance / 2) * 0.3

            # Retention health (based on attrition if available)
            if 'Attrition' in dept_df.columns:
                attrition_rate = dept_df['Attrition'].mean()
                health_components['retention'] = 1 - attrition_rate

            # Tenure health (mix of experience)
            if 'Tenure' in dept_df.columns:
                avg_tenure = dept_df['Tenure'].mean()
                tenure_variance = dept_df['Tenure'].std()
                # Good tenure (2-7 years avg) with some variance
                tenure_score = 1 - abs(avg_tenure - 4.5) / 10
                health_components['tenure'] = max(0, min(1, tenure_score))

            # Sentiment health (from NLP if available)
            if self.nlp_data and 'sentiment' in self.nlp_data:
                sentiment_df = self.nlp_data['sentiment']
                if not sentiment_df.empty:
                    dept_sentiment = sentiment_df[
                        sentiment_df['EmployeeID'].isin(dept_df['EmployeeID'])
                    ]
                    if not dept_sentiment.empty:
                        health_components['sentiment'] = dept_sentiment['sentiment_score'].mean()

            # Calculate overall health score
            if health_components:
                health_score = np.mean(list(health_components.values()))
            else:
                health_score = 0.5

            # Determine status
            if health_score >= 0.75:
                status = 'Thriving'
            elif health_score >= 0.6:
                status = 'Healthy'
            elif health_score >= 0.45:
                status = 'At Risk'
            else:
                status = 'Critical'

            results.append({
                'Dept': dept,
                'Headcount': len(dept_df),
                'HealthScore': round(health_score, 2),
                'Status': status,
                'AvgRating': round(dept_df['LastRating'].mean(), 2) if 'LastRating' in dept_df.columns else None,
                'AvgTenure': round(dept_df['Tenure'].mean(), 1) if 'Tenure' in dept_df.columns else None,
                'AttritionRate': round(dept_df['Attrition'].mean() * 100, 1) if 'Attrition' in dept_df.columns else None
            })

        return pd.DataFrame(results).sort_values('HealthScore', ascending=True)

    def analyze_team_diversity(self) -> pd.DataFrame:
        """
        Analyze diversity metrics for each team.

        Returns:
            DataFrame with diversity scores.
        """
        df = self.df.copy()
        results = []

        for dept in df['Dept'].unique():
            dept_df = df[df['Dept'] == dept]

            if len(dept_df) < self.min_team_size:
                continue

            diversity_scores = {}

            # Age diversity (using coefficient of variation)
            if 'Age' in dept_df.columns and 'age' in self.diversity_metrics:
                age_cv = dept_df['Age'].std() / dept_df['Age'].mean() if dept_df['Age'].mean() > 0 else 0
                diversity_scores['AgeDiversity'] = min(age_cv / 0.3, 1)  # Normalize to 0-1

            # Tenure diversity
            if 'Tenure' in dept_df.columns and 'tenure' in self.diversity_metrics:
                tenure_cv = dept_df['Tenure'].std() / dept_df['Tenure'].mean() if dept_df['Tenure'].mean() > 0 else 0
                diversity_scores['TenureDiversity'] = min(tenure_cv / 0.5, 1)

            # Salary diversity (equity perspective - lower is better for equity)
            if 'Salary' in dept_df.columns and 'salary' in self.diversity_metrics:
                salary_cv = dept_df['Salary'].std() / dept_df['Salary'].mean() if dept_df['Salary'].mean() > 0 else 0
                diversity_scores['SalaryEquity'] = max(0, 1 - salary_cv / 0.4)

            # Overall diversity score
            if diversity_scores:
                overall = np.mean(list(diversity_scores.values()))
            else:
                overall = 0.5

            result_row = {
                'Dept': dept,
                'Headcount': len(dept_df),
                'OverallDiversity': round(overall, 2),
                **{k: round(v, 2) for k, v in diversity_scores.items()}
            }

            results.append(result_row)

        return pd.DataFrame(results).sort_values('OverallDiversity', ascending=False)

    def identify_performance_variance(self) -> pd.DataFrame:
        """
        Analyze performance variance within teams.

        Returns:
            DataFrame with performance variance metrics.
        """
        df = self.df.copy()

        if 'LastRating' not in df.columns:
            return pd.DataFrame()

        results = []

        for dept in df['Dept'].unique():
            dept_df = df[df['Dept'] == dept]

            if len(dept_df) < self.min_team_size:
                continue

            ratings = dept_df['LastRating']

            high_performers = len(dept_df[dept_df['LastRating'] >= 4.0])
            low_performers = len(dept_df[dept_df['LastRating'] < 3.0])

            # Handle NaN from std() when sample size is too small
            rating_std = ratings.std()
            if pd.isna(rating_std):
                rating_std = 0

            headcount = len(dept_df)
            results.append({
                'Dept': dept,
                'Headcount': headcount,
                'AvgRating': round(ratings.mean(), 2),
                'RatingStdDev': round(rating_std, 2),
                'HighPerformers': high_performers,
                'LowPerformers': low_performers,
                'PerformanceSpread': round(ratings.max() - ratings.min(), 1),
                'PercentHigh': round(high_performers / headcount * 100, 1) if headcount > 0 else 0,
                'PercentLow': round(low_performers / headcount * 100, 1) if headcount > 0 else 0
            })

        result_df = pd.DataFrame(results)
        result_df['Consistency'] = result_df['RatingStdDev'].apply(
            lambda x: 'High' if x < 0.5 else ('Medium' if x < 1.0 else 'Low')
        )

        return result_df.sort_values('RatingStdDev')

    def get_collaboration_indicators(self) -> dict:
        """
        Calculate collaboration indicators across the organization.

        Returns:
            Dictionary with collaboration metrics.
        """
        df = self.df.copy()

        indicators = {}

        # Cross-department ratios
        dept_counts = df['Dept'].value_counts()
        indicators['department_count'] = len(dept_counts)
        indicators['avg_team_size'] = round(dept_counts.mean(), 1)
        indicators['team_size_variance'] = round(dept_counts.std(), 1)

        # Balance score (how evenly distributed are teams)
        if len(dept_counts) > 1:
            ideal_size = len(df) / len(dept_counts)
            imbalance = sum(abs(count - ideal_size) for count in dept_counts) / len(df)
            indicators['team_balance_score'] = round(1 - imbalance / 2, 2)
        else:
            indicators['team_balance_score'] = 1.0

        # Experience distribution
        if 'Tenure' in df.columns:
            indicators['avg_org_tenure'] = round(df['Tenure'].mean(), 1)
            indicators['tenure_range'] = round(df['Tenure'].max() - df['Tenure'].min(), 1)

        # Performance distribution
        if 'LastRating' in df.columns:
            indicators['org_avg_rating'] = round(df['LastRating'].mean(), 2)
            indicators['high_performer_ratio'] = round(len(df[df['LastRating'] >= 4.0]) / len(df) * 100, 1)

        return indicators

    def flag_at_risk_teams(self) -> pd.DataFrame:
        """
        Identify teams that need attention.

        Returns:
            DataFrame with at-risk teams and reasons.
        """
        health_df = self.calculate_team_health_scores()

        at_risk = health_df[
            (health_df['HealthScore'] < self.health_warning_threshold) |
            (health_df['Status'].isin(['At Risk', 'Critical']))
        ].copy()

        if at_risk.empty:
            return pd.DataFrame(columns=['Dept', 'HealthScore', 'Status', 'RiskFactors', 'Recommendations'])

        # Add risk factors and recommendations
        def get_risk_factors(row):
            factors = []
            if row['AttritionRate'] and row['AttritionRate'] > 20:
                factors.append('High attrition')
            if row['AvgRating'] and row['AvgRating'] < 3.0:
                factors.append('Low performance')
            if row['AvgTenure'] and row['AvgTenure'] < 1.5:
                factors.append('Inexperienced team')
            if row['Headcount'] < 5:
                factors.append('Small team size')
            return ', '.join(factors) if factors else 'Multiple factors'

        def get_recommendations(row):
            recs = []
            if row['AttritionRate'] and row['AttritionRate'] > 20:
                recs.append('Conduct stay interviews')
            if row['AvgRating'] and row['AvgRating'] < 3.0:
                recs.append('Implement performance coaching')
            if row['AvgTenure'] and row['AvgTenure'] < 1.5:
                recs.append('Pair with mentors')
            return '; '.join(recs) if recs else 'Review team dynamics'

        at_risk['RiskFactors'] = at_risk.apply(get_risk_factors, axis=1)
        at_risk['Recommendations'] = at_risk.apply(get_recommendations, axis=1)

        return at_risk[['Dept', 'Headcount', 'HealthScore', 'Status', 'RiskFactors', 'Recommendations']]

    def get_team_composition(self) -> pd.DataFrame:
        """
        Analyze team composition by various dimensions.

        Returns:
            DataFrame with composition breakdown.
        """
        df = self.df.copy()
        results = []

        for dept in df['Dept'].unique():
            dept_df = df[df['Dept'] == dept]

            composition = {
                'Dept': dept,
                'Headcount': len(dept_df)
            }

            # Tenure composition
            if 'Tenure' in dept_df.columns:
                composition['New (<1yr)'] = len(dept_df[dept_df['Tenure'] < 1])
                composition['Developing (1-3yr)'] = len(dept_df[(dept_df['Tenure'] >= 1) & (dept_df['Tenure'] < 3)])
                composition['Experienced (3-5yr)'] = len(dept_df[(dept_df['Tenure'] >= 3) & (dept_df['Tenure'] < 5)])
                composition['Senior (5+yr)'] = len(dept_df[dept_df['Tenure'] >= 5])

            # Age composition
            if 'Age' in dept_df.columns:
                composition['Under30'] = len(dept_df[dept_df['Age'] < 30])
                composition['30-40'] = len(dept_df[(dept_df['Age'] >= 30) & (dept_df['Age'] < 40)])
                composition['40-50'] = len(dept_df[(dept_df['Age'] >= 40) & (dept_df['Age'] < 50)])
                composition['Over50'] = len(dept_df[dept_df['Age'] >= 50])

            results.append(composition)

        return pd.DataFrame(results)

    def get_team_summary(self) -> dict:
        """
        Get overall team dynamics summary.

        Returns:
            Dictionary with summary statistics.
        """
        health_df = self.calculate_team_health_scores()

        if health_df.empty:
            return {
                'total_teams': 0,
                'thriving_teams': 0,
                'at_risk_teams': 0,
                'avg_health_score': 0
            }

        return {
            'total_teams': len(health_df),
            'thriving_teams': len(health_df[health_df['Status'] == 'Thriving']),
            'healthy_teams': len(health_df[health_df['Status'] == 'Healthy']),
            'at_risk_teams': len(health_df[health_df['Status'].isin(['At Risk', 'Critical'])]),
            'avg_health_score': round(health_df['HealthScore'].mean(), 2),
            'min_health_score': round(health_df['HealthScore'].min(), 2),
            'max_health_score': round(health_df['HealthScore'].max(), 2)
        }

    def analyze_all(self) -> dict:
        """
        Run full team dynamics analysis.

        Returns:
            Dictionary with all team dynamics results.
        """
        logger.info("Running full team dynamics analysis")

        results = {
            'health': self.calculate_team_health_scores(),
            'diversity': self.analyze_team_diversity(),
            'performance_variance': self.identify_performance_variance(),
            'collaboration': self.get_collaboration_indicators(),
            'at_risk': self.flag_at_risk_teams(),
            'composition': self.get_team_composition(),
            'summary': self.get_team_summary()
        }

        logger.info("Team dynamics analysis complete")
        return results
