"""
Fairness and Bias Detection Engine for PeopleOS.

Provides algorithmic fairness metrics, bias detection, and
disparity analysis for HR decision-making systems.

Based on industry standards:
- Demographic parity
- Equalized odds
- Four-fifths rule (80% rule from EEOC)
- Statistical parity difference
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('fairness_engine')

# EEOC Four-Fifths Rule threshold
FOUR_FIFTHS_THRESHOLD = 0.8


class FairnessEngineError(Exception):
    """Custom exception for fairness engine errors."""
    pass


class FairnessEngine:
    """
    Fairness and bias detection engine for HR analytics.

    Analyzes ML predictions and HR outcomes for potential bias
    across protected characteristics.
    """

    def __init__(self, df: pd.DataFrame, predictions: Optional[pd.DataFrame] = None):
        """
        Initialize Fairness Engine.

        Args:
            df: DataFrame with employee data including demographic columns.
            predictions: Optional DataFrame with ML predictions (risk_score, etc.).
        """
        self.df = df.copy()
        self.predictions = predictions
        self.config = load_config()
        self.fairness_config = self.config.get('fairness', {})

        # Protected attributes to check (configurable)
        self.protected_attributes = self.fairness_config.get(
            'protected_attributes',
            ['Gender', 'Age_Group', 'Dept']  # Dept as proxy for potential systemic issues
        )

        self._identify_available_attributes()
        logger.info(f"FairnessEngine initialized. Available protected attrs: {self.available_attributes}")

    def _identify_available_attributes(self) -> None:
        """Identify which protected attributes are available in the data."""
        self.available_attributes = []
        for attr in self.protected_attributes:
            if attr in self.df.columns:
                self.available_attributes.append(attr)
            elif attr == 'Age_Group' and 'Age' in self.df.columns:
                # Create age groups if Age is available
                self.df['Age_Group'] = pd.cut(
                    self.df['Age'],
                    bins=[0, 30, 40, 50, 60, 100],
                    labels=['Under 30', '30-39', '40-49', '50-59', '60+']
                )
                self.available_attributes.append('Age_Group')

        # Also check for Location as potential proxy for bias
        if 'Location' in self.df.columns and 'Location' not in self.available_attributes:
            self.available_attributes.append('Location')

        # Check for Education level
        if 'Education' in self.df.columns and 'Education' not in self.available_attributes:
            self.available_attributes.append('Education')

    def calculate_demographic_parity(self, outcome_col: str) -> pd.DataFrame:
        """
        Calculate demographic parity across protected attributes.

        Demographic parity is satisfied when each group has the same
        positive outcome rate.

        Args:
            outcome_col: Column name for the outcome (e.g., 'Attrition', 'Promoted').

        Returns:
            DataFrame with parity metrics per attribute.
        """
        if outcome_col not in self.df.columns:
            raise FairnessEngineError(f"Outcome column '{outcome_col}' not found in data")

        results = []

        for attr in self.available_attributes:
            group_rates = self.df.groupby(attr, observed=True)[outcome_col].agg(['mean', 'count'])
            group_rates.columns = ['rate', 'count']
            group_rates = group_rates.reset_index()

            overall_rate = self.df[outcome_col].mean()

            # Calculate disparity from overall rate
            group_rates['disparity'] = (group_rates['rate'] - overall_rate).abs()
            # Safe division: when overall rate is 0, all groups have parity (ratio = 1.0)
            if overall_rate > 0:
                group_rates['parity_ratio'] = group_rates['rate'] / overall_rate
            else:
                group_rates['parity_ratio'] = 1.0
            group_rates['attribute'] = attr

            results.append(group_rates)

        if not results:
            return pd.DataFrame()

        combined = pd.concat(results, ignore_index=True)
        combined = combined.rename(columns={combined.columns[0]: 'group'})

        return combined

    def calculate_four_fifths_rule(self, outcome_col: str, favorable: bool = False) -> pd.DataFrame:
        """
        Apply the EEOC Four-Fifths (80%) Rule.

        The selection rate for any protected group should be at least
        80% of the selection rate for the group with the highest rate.

        Args:
            outcome_col: Column name for the outcome.
            favorable: If True, higher values are favorable (e.g., promotion).
                      If False, lower values are favorable (e.g., attrition = bad).

        Returns:
            DataFrame with four-fifths rule analysis.
        """
        if outcome_col not in self.df.columns:
            raise FairnessEngineError(f"Outcome column '{outcome_col}' not found in data")

        results = []

        for attr in self.available_attributes:
            group_rates = self.df.groupby(attr, observed=True)[outcome_col].mean().reset_index()
            group_rates.columns = ['group', 'rate']

            if favorable:
                reference_rate = group_rates['rate'].max()
            else:
                reference_rate = group_rates['rate'].min()

            if reference_rate == 0:
                group_rates['adverse_impact_ratio'] = 1.0
            else:
                if favorable:
                    group_rates['adverse_impact_ratio'] = group_rates['rate'] / reference_rate
                else:
                    # For unfavorable outcomes (like attrition), invert the comparison
                    # Lower rate is better, so we compare lowest to each group
                    group_rates['adverse_impact_ratio'] = reference_rate / group_rates['rate'].replace(0, 0.001)

            group_rates['passes_4_5_rule'] = group_rates['adverse_impact_ratio'] >= FOUR_FIFTHS_THRESHOLD
            group_rates['attribute'] = attr

            results.append(group_rates)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)

    def analyze_prediction_fairness(self, risk_col: str = 'risk_score') -> Dict[str, Any]:
        """
        Analyze fairness of ML risk predictions.

        Args:
            risk_col: Column name for risk predictions.

        Returns:
            Dictionary with prediction fairness metrics.
        """
        if self.predictions is None or self.predictions.empty:
            return {'error': 'No predictions available for fairness analysis'}

        # Merge predictions with employee data
        if 'EmployeeID' in self.df.columns and 'EmployeeID' in self.predictions.columns:
            merged = self.df.merge(self.predictions, on='EmployeeID', how='inner')
        else:
            # Assume same order
            merged = self.df.copy()
            merged[risk_col] = self.predictions[risk_col].values[:len(merged)]

        results = {
            'attribute_analysis': [],
            'overall_disparities': [],
            'warnings': []
        }

        for attr in self.available_attributes:
            if attr not in merged.columns:
                continue

            group_stats = merged.groupby(attr)[risk_col].agg(['mean', 'std', 'count'])
            group_stats = group_stats.reset_index()
            group_stats.columns = ['group', 'mean_risk', 'std_risk', 'count']

            overall_mean = merged[risk_col].mean()

            # Check for significant disparities
            for _, row in group_stats.iterrows():
                disparity = row['mean_risk'] - overall_mean
                if abs(disparity) > 0.1:  # 10% difference threshold
                    results['warnings'].append(
                        f"Group '{row['group']}' has {disparity:.1%} higher risk score than average. "
                        "This may indicate algorithmic bias."
                    )

            group_stats['attribute'] = attr
            results['attribute_analysis'].append(group_stats)

        if results['attribute_analysis']:
            results['attribute_analysis'] = pd.concat(results['attribute_analysis'], ignore_index=True)
        else:
            results['attribute_analysis'] = pd.DataFrame()

        return results

    def calculate_equalized_odds(self, outcome_col: str, prediction_col: str = 'predicted') -> pd.DataFrame:
        """
        Calculate equalized odds across protected attributes.

        Equalized odds requires that true positive rates and false positive
        rates are equal across protected groups.

        Args:
            outcome_col: Column name for actual outcomes.
            prediction_col: Column name for predicted outcomes.

        Returns:
            DataFrame with TPR and FPR by group.
        """
        if self.predictions is None:
            return pd.DataFrame()

        # Merge data
        if 'EmployeeID' in self.df.columns and 'EmployeeID' in self.predictions.columns:
            merged = self.df.merge(self.predictions, on='EmployeeID', how='inner')
        else:
            return pd.DataFrame()

        if outcome_col not in merged.columns or prediction_col not in merged.columns:
            return pd.DataFrame()

        results = []

        for attr in self.available_attributes:
            if attr not in merged.columns:
                continue

            for group in merged[attr].unique():
                group_data = merged[merged[attr] == group]

                # True Positive Rate (Recall)
                positives = group_data[group_data[outcome_col] == 1]
                tpr = positives[prediction_col].mean() if len(positives) > 0 else 0

                # False Positive Rate
                negatives = group_data[group_data[outcome_col] == 0]
                fpr = negatives[prediction_col].mean() if len(negatives) > 0 else 0

                results.append({
                    'attribute': attr,
                    'group': group,
                    'tpr': round(tpr, 3),
                    'fpr': round(fpr, 3),
                    'count': len(group_data)
                })

        return pd.DataFrame(results)

    def get_fairness_summary(self, outcome_col: str = 'Attrition') -> Dict[str, Any]:
        """
        Generate comprehensive fairness summary.

        Args:
            outcome_col: Primary outcome column to analyze.

        Returns:
            Dictionary with complete fairness analysis.
        """
        summary = {
            'overall_status': 'Unknown',
            'issues_found': [],
            'recommendations': [],
            'metrics': {}
        }

        try:
            # Demographic parity
            if outcome_col in self.df.columns:
                parity = self.calculate_demographic_parity(outcome_col)
                if not parity.empty:
                    summary['metrics']['demographic_parity'] = parity.to_dict('records')

                    # Check for large disparities
                    max_disparity = parity['disparity'].max()
                    if max_disparity > 0.15:
                        summary['issues_found'].append(
                            f"Significant demographic disparity detected ({max_disparity:.1%})"
                        )

                # Four-fifths rule
                four_fifths = self.calculate_four_fifths_rule(outcome_col, favorable=False)
                if not four_fifths.empty:
                    summary['metrics']['four_fifths_rule'] = four_fifths.to_dict('records')

                    violations = four_fifths[~four_fifths['passes_4_5_rule']]
                    if not violations.empty:
                        for _, row in violations.iterrows():
                            summary['issues_found'].append(
                                f"Four-fifths rule violation: {row['attribute']}='{row['group']}' "
                                f"(ratio: {row['adverse_impact_ratio']:.2f})"
                            )

            # Prediction fairness
            if self.predictions is not None:
                pred_fairness = self.analyze_prediction_fairness()
                if 'warnings' in pred_fairness:
                    summary['issues_found'].extend(pred_fairness['warnings'])
                summary['metrics']['prediction_fairness'] = pred_fairness

            # Determine overall status
            if not summary['issues_found']:
                summary['overall_status'] = 'Fair'
            elif len(summary['issues_found']) <= 2:
                summary['overall_status'] = 'Needs Attention'
            else:
                summary['overall_status'] = 'Critical'

            # Generate recommendations
            if summary['issues_found']:
                summary['recommendations'] = [
                    "Review hiring and promotion processes for potential bias",
                    "Conduct deeper statistical analysis with HR stakeholders",
                    "Consider bias mitigation techniques in ML pipeline",
                    "Document findings and create action plan for remediation"
                ]

        except Exception as e:
            logger.error(f"Fairness analysis failed: {str(e)}")
            summary['overall_status'] = 'Error'
            summary['issues_found'].append(f"Analysis failed: {str(e)}")

        return summary

    def generate_fairness_report(self, outcome_col: str = 'Attrition') -> str:
        """
        Generate human-readable fairness report.

        Args:
            outcome_col: Primary outcome column to analyze.

        Returns:
            Formatted string report.
        """
        summary = self.get_fairness_summary(outcome_col)

        report = []
        report.append("=" * 50)
        report.append("FAIRNESS & BIAS ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")

        report.append(f"Overall Status: {summary['overall_status']}")
        report.append(f"Protected Attributes Analyzed: {', '.join(self.available_attributes)}")
        report.append("")

        if summary['issues_found']:
            report.append("ISSUES DETECTED:")
            for i, issue in enumerate(summary['issues_found'], 1):
                report.append(f"  {i}. {issue}")
            report.append("")

        if summary['recommendations']:
            report.append("RECOMMENDATIONS:")
            for rec in summary['recommendations']:
                report.append(f"  - {rec}")
            report.append("")

        report.append("=" * 50)
        report.append("Note: This analysis is for guidance only. Consult HR/Legal")
        report.append("professionals before making policy decisions.")
        report.append("=" * 50)

        return "\n".join(report)

    def analyze_all(self, outcome_col: str = 'Attrition') -> Dict[str, Any]:
        """
        Run comprehensive fairness analysis.

        Args:
            outcome_col: Primary outcome column to analyze.

        Returns:
            Dictionary with all fairness analysis results.
        """
        logger.info("Running comprehensive fairness analysis")

        results = {
            'summary': self.get_fairness_summary(outcome_col),
            'report': self.generate_fairness_report(outcome_col)
        }

        if outcome_col in self.df.columns:
            results['demographic_parity'] = self.calculate_demographic_parity(outcome_col)
            results['four_fifths_rule'] = self.calculate_four_fifths_rule(outcome_col)

        if self.predictions is not None:
            results['prediction_fairness'] = self.analyze_prediction_fairness()

        logger.info(f"Fairness analysis complete. Status: {results['summary']['overall_status']}")
        return results
