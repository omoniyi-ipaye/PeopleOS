"""Outcome-disparity and predictive-fairness diagnostics for PeopleOS.

These metrics are screening signals, not legal or causal determinations of
fairness/bias. Protected-attribute analyses enforce minimum group sizes. The
four-fifths calculation is always performed on a favorable outcome rate; for
Attrition that favorable outcome is retention (1 - Attrition).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.population import resolve_current_population
from src.utils import load_config

logger = get_logger('fairness_engine')
FOUR_FIFTHS_THRESHOLD = 0.8
DEFAULT_MIN_GROUP_SIZE = 10


class FairnessEngineError(Exception):
    pass


class FairnessEngine:
    def __init__(self, df: pd.DataFrame, predictions: Optional[pd.DataFrame] = None):
        self.df, self.population_resolution = resolve_current_population(df)
        self.predictions = predictions if isinstance(predictions, pd.DataFrame) else None
        self.config = load_config()
        config = self.config.get('fairness', {})
        self.min_group_size = int(config.get('min_group_size', DEFAULT_MIN_GROUP_SIZE))
        self.protected_attributes = config.get('protected_attributes', ['Gender', 'Age_Group'])
        self.monitoring_dimensions = config.get('monitoring_dimensions', ['Dept', 'Location', 'Education'])
        self.available_protected_attributes: list[str] = []
        self.available_monitoring_dimensions: list[str] = []
        self._identify_available_attributes()
        # Compatibility alias used by existing UI/API code.
        self.available_attributes = self.available_protected_attributes + self.available_monitoring_dimensions

    def _identify_available_attributes(self) -> None:
        if 'Age_Group' in self.protected_attributes and 'Age_Group' not in self.df.columns and 'Age' in self.df.columns:
            age = pd.to_numeric(self.df['Age'], errors='coerce')
            # Keep the fairness population aligned with the canonical analytics
            # contract. Impossible ages must not silently become a protected-age
            # group such as "60+".
            age = age.where(np.isfinite(age) & age.between(1, 120))
            self.df['Age_Group'] = pd.cut(
                age,
                bins=[0, 30, 40, 50, 60, float('inf')],
                labels=['Under 30', '30-39', '40-49', '50-59', '60+'],
                right=False,
            )
        self.available_protected_attributes = [a for a in self.protected_attributes if a in self.df.columns]
        self.available_monitoring_dimensions = [a for a in self.monitoring_dimensions if a in self.df.columns and a not in self.available_protected_attributes]

    def _dimension_type(self, attr: str) -> str:
        return 'protected_attribute' if attr in self.available_protected_attributes else 'monitoring_dimension'

    def _eligible_group_rates(self, attr: str, outcome_col: str) -> tuple[pd.DataFrame, int]:
        frame = self.df[[attr, outcome_col]].dropna().copy()
        frame[outcome_col] = pd.to_numeric(frame[outcome_col], errors='coerce')
        frame = frame[frame[outcome_col].isin([0, 1])]
        grouped = frame.groupby(attr, observed=True)[outcome_col].agg(['mean', 'count']).reset_index()
        suppressed = int((grouped['count'] < self.min_group_size).sum())
        return grouped[grouped['count'] >= self.min_group_size].copy(), suppressed

    def calculate_demographic_parity(self, outcome_col: str) -> pd.DataFrame:
        """Compare observed outcome rates across eligible groups.

        For an unfavorable outcome such as Attrition, the returned `rate` remains
        the raw unfavorable rate for transparency; callers must not interpret a
        higher rate as favorable selection parity.
        """
        if outcome_col not in self.df.columns:
            raise FairnessEngineError(f"Outcome column '{outcome_col}' not found")
        rows = []
        known = pd.to_numeric(self.df[outcome_col], errors='coerce')
        known = known[known.isin([0, 1])]
        overall = float(known.mean()) if not known.empty else np.nan
        overall_known_count = int(len(known))
        for attr in self.available_attributes:
            attribute_frame = self.df[[attr, outcome_col]].copy()
            attribute_outcome = pd.to_numeric(attribute_frame[outcome_col], errors='coerce')
            attribute_known = attribute_frame[attr].notna() & attribute_outcome.isin([0, 1])
            attribute_observed_count = int(attribute_known.sum())
            eligible, suppressed = self._eligible_group_rates(attr, outcome_col)
            for _, row in eligible.iterrows():
                rows.append({
                    'group': row[attr], 'rate': float(row['mean']), 'count': int(row['count']),
                    'disparity': float(abs(row['mean'] - overall)) if np.isfinite(overall) else None,
                    # Compatibility field: this is an observed outcome-rate ratio,
                    # not a favorable-selection parity ratio unless the stored
                    # outcome itself is favorable.
                    'parity_ratio': float(row['mean'] / overall) if np.isfinite(overall) and overall > 0 else None,
                    'outcome_rate_ratio_to_overall': float(row['mean'] / overall) if np.isfinite(overall) and overall > 0 else None,
                    'attribute': attr, 'dimension_type': self._dimension_type(attr),
                    'suppressed_group_count': suppressed,
                    'overall_known_outcome_count': overall_known_count,
                    'attribute_observed_count': attribute_observed_count,
                    'attribute_coverage': (float(attribute_observed_count / overall_known_count) if overall_known_count else None),
                    'metric_semantics': f'observed_{outcome_col.lower()}_rate_disparity',
                })
        return pd.DataFrame(rows)

    def calculate_four_fifths_rule(self, outcome_col: str, favorable: bool = False) -> pd.DataFrame:
        if outcome_col not in self.df.columns:
            raise FairnessEngineError(f"Outcome column '{outcome_col}' not found")
        rows = []
        for attr in self.available_attributes:
            eligible, suppressed = self._eligible_group_rates(attr, outcome_col)
            if eligible.empty:
                continue
            # Four-fifths is a favorable-outcome comparison. When the stored
            # outcome is unfavorable (Attrition=1), convert to retention rate.
            eligible['favorable_rate'] = eligible['mean'] if favorable else (1.0 - eligible['mean'])
            reference = float(eligible['favorable_rate'].max())
            for _, row in eligible.iterrows():
                ratio = float(row['favorable_rate'] / reference) if reference > 0 else None
                rows.append({
                    'attribute': attr, 'dimension_type': self._dimension_type(attr), 'group': row[attr],
                    'rate': float(row['mean']), 'favorable_rate': float(row['favorable_rate']), 'count': int(row['count']),
                    'reference_favorable_rate': reference, 'adverse_impact_ratio': ratio,
                    'passes_4_5_rule': bool(ratio >= FOUR_FIFTHS_THRESHOLD) if ratio is not None else None,
                    'suppressed_group_count': suppressed,
                    'metric_semantics': 'four_fifths_favorable_outcome_ratio',
                })
        return pd.DataFrame(rows)

    def analyze_prediction_fairness(self, risk_col: str = 'risk_score') -> Dict[str, Any]:
        if self.predictions is None or self.predictions.empty:
            return {'available': False, 'reason': 'No predictions available', 'warnings': []}
        if 'EmployeeID' in self.df.columns and 'EmployeeID' in self.predictions.columns:
            if self.predictions['EmployeeID'].duplicated().any():
                return {'available': False, 'reason': 'Predictions must contain one row per employee', 'warnings': []}
            merged = self.df.merge(self.predictions, on='EmployeeID', how='inner')
        else:
            return {'available': False, 'reason': 'Predictions require EmployeeID alignment', 'warnings': []}
        if risk_col not in merged.columns:
            return {'available': False, 'reason': f'{risk_col} unavailable', 'warnings': []}

        warnings: list[str] = []
        rows = []
        risk = pd.to_numeric(merged[risk_col], errors='coerce')
        risk = risk.where(risk.between(0, 1))
        overall = float(risk.mean()) if risk.notna().any() else np.nan
        overall_risk_count = int(risk.notna().sum())
        for attr in self.available_attributes:
            if attr not in merged.columns:
                continue
            attr_valid = merged[attr].notna() & risk.notna()
            attribute_observed_count = int(attr_valid.sum())
            stats_df = merged.assign(_risk=risk).dropna(subset=[attr, '_risk']).groupby(attr, observed=True)['_risk'].agg(['mean', 'std', 'count']).reset_index()
            suppressed = int((stats_df['count'] < self.min_group_size).sum())
            stats_df = stats_df[stats_df['count'] >= self.min_group_size]
            for _, row in stats_df.iterrows():
                disparity = float(row['mean'] - overall)
                if abs(disparity) > .10:
                    warnings.append(f"Prediction-disparity signal for {attr}='{row[attr]}': mean risk differs from the analysed population by {disparity:+.1%}. Investigate calibration, data mix and model behavior before drawing a bias conclusion.")
                rows.append({
                    'attribute': attr, 'dimension_type': self._dimension_type(attr), 'group': row[attr],
                    'mean_risk': float(row['mean']), 'std_risk': float(row['std']) if pd.notna(row['std']) else None,
                    'count': int(row['count']), 'difference_from_overall': disparity,
                    'suppressed_group_count': suppressed,
                    'overall_risk_observations': overall_risk_count,
                    'attribute_observed_count': attribute_observed_count,
                    'attribute_coverage': (float(attribute_observed_count / overall_risk_count) if overall_risk_count else None),
                })
        return {'available': True, 'attribute_analysis': pd.DataFrame(rows), 'warnings': warnings, 'semantics': 'unadjusted_prediction_disparity_screen_not_bias_determination'}

    def calculate_equalized_odds(self, outcome_col: str, prediction_col: str = 'predicted') -> pd.DataFrame:
        if self.predictions is None or 'EmployeeID' not in self.predictions.columns or 'EmployeeID' not in self.df.columns:
            return pd.DataFrame()
        if self.predictions['EmployeeID'].duplicated().any():
            return pd.DataFrame()
        merged = self.df.merge(self.predictions, on='EmployeeID', how='inner')
        if outcome_col not in merged.columns or prediction_col not in merged.columns:
            return pd.DataFrame()
        merged[outcome_col] = pd.to_numeric(merged[outcome_col], errors='coerce')
        merged[prediction_col] = pd.to_numeric(merged[prediction_col], errors='coerce')
        merged = merged[merged[outcome_col].isin([0, 1]) & merged[prediction_col].isin([0, 1])]
        rows = []
        for attr in self.available_attributes:
            if attr not in merged.columns:
                continue
            for group, data in merged.groupby(attr, observed=True):
                if len(data) < self.min_group_size:
                    continue
                positives = data[data[outcome_col] == 1]
                negatives = data[data[outcome_col] == 0]
                positive_n = int(len(positives))
                negative_n = int(len(negatives))
                # Equalized-odds components have different denominators. A large
                # overall group does not make a TPR based on one positive outcome
                # or an FPR based on one negative outcome reliable enough to
                # present as a measured group rate.
                tpr = (
                    float(pd.to_numeric(positives[prediction_col], errors='coerce').mean())
                    if positive_n >= self.min_group_size else None
                )
                fpr = (
                    float(pd.to_numeric(negatives[prediction_col], errors='coerce').mean())
                    if negative_n >= self.min_group_size else None
                )
                rows.append({
                    'attribute': attr,
                    'dimension_type': self._dimension_type(attr),
                    'group': group,
                    'tpr': tpr,
                    'fpr': fpr,
                    'count': int(len(data)),
                    'positive_n': positive_n,
                    'negative_n': negative_n,
                    'minimum_class_size': self.min_group_size,
                    'tpr_available': tpr is not None,
                    'fpr_available': fpr is not None,
                    'metric_semantics': 'equalized_odds_rates_require_minimum_support_per_outcome_class',
                })
        return pd.DataFrame(rows)

    def get_fairness_summary(self, outcome_col: str = 'Attrition') -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            'overall_status': 'Insufficient evidence', 'issues_found': [], 'recommendations': [],
            'metrics': {}, 'minimum_group_size': self.min_group_size,
            'protected_attributes': self.available_protected_attributes,
            'monitoring_dimensions': self.available_monitoring_dimensions,
            'interpretation_boundary': 'Disparity screening is not a legal, causal, or bias determination.',
        }
        try:
            if outcome_col in self.df.columns:
                parity = self.calculate_demographic_parity(outcome_col)
                four = self.calculate_four_fifths_rule(outcome_col, favorable=False)
                summary['metrics']['outcome_disparity'] = parity.to_dict('records')
                summary['metrics']['four_fifths_rule'] = four.to_dict('records')
                protected_four = four[four['dimension_type'] == 'protected_attribute'] if not four.empty else four
                violations = protected_four[protected_four['passes_4_5_rule'] == False] if not protected_four.empty else protected_four
                for _, row in violations.iterrows():
                    summary['issues_found'].append(f"Four-fifths screening signal: {row['attribute']}='{row['group']}' favorable-outcome ratio {row['adverse_impact_ratio']:.2f}")
                valid_comparisons = protected_four.dropna(subset=['adverse_impact_ratio']) if not protected_four.empty else protected_four
                if not valid_comparisons.empty and valid_comparisons.groupby('attribute').size().max() >= 2:
                    summary['overall_status'] = 'Disparity signal detected' if not violations.empty else 'No material disparity detected in eligible groups'
            if self.predictions is not None:
                pred = self.analyze_prediction_fairness()
                summary['metrics']['prediction_fairness'] = pred
                summary['issues_found'].extend(pred.get('warnings', []))
                if pred.get('warnings') and summary['overall_status'] == 'No material disparity detected in eligible groups':
                    summary['overall_status'] = 'Prediction disparity requires review'
            if summary['issues_found']:
                summary['recommendations'] = [
                    'Validate group sample sizes, outcome definitions and data quality.',
                    'Review calibration and error rates by protected group before operational use.',
                    'Use qualified HR/legal/statistical review before consequential policy decisions.',
                ]
        except Exception as exc:
            logger.exception('Fairness analysis failed')
            summary['overall_status'] = 'Analysis error'
            summary['issues_found'].append('Fairness analysis could not be completed safely.')
            summary['error_type'] = type(exc).__name__
        return summary

    def generate_fairness_report(self, outcome_col: str = 'Attrition') -> str:
        summary = self.get_fairness_summary(outcome_col)
        lines = ['OUTCOME DISPARITY SCREENING REPORT', f"Status: {summary['overall_status']}", summary['interpretation_boundary']]
        if summary['issues_found']:
            lines.append('Signals:')
            lines.extend(f"- {item}" for item in summary['issues_found'])
        if summary['recommendations']:
            lines.append('Next checks:')
            lines.extend(f"- {item}" for item in summary['recommendations'])
        return '\n'.join(lines)

    def analyze_all(self, outcome_col: str = 'Attrition') -> Dict[str, Any]:
        result: Dict[str, Any] = {'summary': self.get_fairness_summary(outcome_col), 'report': self.generate_fairness_report(outcome_col)}
        if outcome_col in self.df.columns:
            result['demographic_parity'] = self.calculate_demographic_parity(outcome_col)
            result['four_fifths_rule'] = self.calculate_four_fifths_rule(outcome_col)
        if self.predictions is not None:
            result['prediction_fairness'] = self.analyze_prediction_fairness()
        return result
