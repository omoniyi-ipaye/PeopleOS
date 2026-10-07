"""Governed diagnostic support for retrospective predictive models.

Model Lab does not retrain, prune, optimize, or validate future-departure
accuracy. It reports bounded diagnostics for an already activated model and
keeps prospective backtesting unavailable until timestamped predictions and
mature outcomes exist.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.database import Database
from src.logger import get_logger
from src.ml_engine import MLEngine

logger = get_logger('model_lab_engine')


class ModelLabError(ValueError):
    """Model diagnostics cannot be reconciled with the active evidence."""


class ModelLabEngine:
    """Read-only diagnostics for an explicitly supplied model and snapshot."""

    def __init__(
        self,
        db: Optional[Database] = None,
        ml_engine: Optional[MLEngine] = None,
        data: Optional[pd.DataFrame] = None,
    ):
        # Database fallback is retained only for direct compatibility callers.
        # Product routes supply the verified active runtime snapshot and model.
        self.db = db
        self.ml_engine = ml_engine
        self.data = data.copy(deep=True) if isinstance(data, pd.DataFrame) else None

    def backtest_flight_risk(self, days_back: int = 90) -> dict[str, Any]:
        """Report why a valid prospective backtest is not yet available."""
        return {
            'status': 'warning',
            'metrics': None,
            'message': 'Prospective backtesting is unavailable: timestamped predictions from a model trained before the prediction date, a fixed outcome horizon, and mature follow-up are required.',
            'interpretation': 'Scoring old snapshots with a model trained on current outcomes would leak future information; it is not a valid backtest.',
        }

    def _get_interpretation(self, f1: float, recall: float) -> str:
        """Retain the legacy helper without interpreting retrospective scores as validation."""
        return 'Retrospective scores alone do not validate future departure accuracy.'

    def _analysis_data(self) -> pd.DataFrame:
        if self.data is not None:
            return self.data.copy(deep=True)
        if self.db is not None:
            frame = self.db.get_all_employees()
            return frame.copy(deep=True) if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        return pd.DataFrame()

    def analyze_feature_sensitivity(self) -> list[dict[str, Any]]:
        """Return descriptive variance/correlation flags for the active model.

        The returned ``reliability`` field is retained for API compatibility but
        is a bounded heuristic index, not measurement reliability or evidence of
        predictive accuracy.
        """
        frame = self._analysis_data()
        engine = self.ml_engine
        if frame.empty or engine is None or not getattr(engine, 'is_trained', False):
            return []

        feature_names = list(getattr(engine, 'feature_names', []))
        if not feature_names or len(feature_names) != len(set(feature_names)):
            raise ModelLabError('Active model feature evidence is unavailable or ambiguous')

        importance = engine.get_feature_importance_summary()
        if not {'feature', 'importance'}.issubset(importance.columns):
            raise ModelLabError('Active model importance evidence is unavailable')
        importance_map = {
            str(row.feature): float(row.importance)
            for row in importance.itertuples()
            if str(row.feature) in feature_names
        }
        if set(importance_map) != set(feature_names) or not np.isfinite(list(importance_map.values())).all():
            raise ModelLabError('Active model importance evidence does not match its feature contract')

        try:
            transformed = engine.preprocessor.transform(frame)
        except Exception as exc:
            raise ModelLabError('Active snapshot cannot be transformed with the active model contract') from exc
        if not set(feature_names).issubset(transformed.columns):
            raise ModelLabError('Active snapshot does not produce the active model feature contract')
        processed = transformed.loc[:, feature_names].apply(pd.to_numeric, errors='coerce')
        values = processed.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ModelLabError('Feature diagnostics require finite transformed measurements')

        correlations = processed.corr().abs()
        report: list[dict[str, Any]] = []
        for feature in feature_names:
            std = float(processed[feature].std())
            peers = correlations.loc[feature].drop(labels=[feature], errors='ignore').dropna()
            max_corr = float(peers.max()) if not peers.empty else None
            redundant_with = str(peers.idxmax()) if max_corr is not None and max_corr > 0.90 else None

            if not np.isfinite(std):
                heuristic_index = 0.0
                status = 'Insufficient observations'
                recommendation = 'Collect sufficient comparable observations before reviewing this feature.'
            elif std < 0.05:
                heuristic_index = 0.70
                status = 'Low variance observed'
                recommendation = 'Review the observed variation on independent evaluation data; no feature change was applied.'
            elif redundant_with is not None:
                heuristic_index = 0.80
                status = 'High correlation observed'
                recommendation = f'Review overlap with {redundant_with} on independent evaluation data; no feature change was applied.'
            else:
                heuristic_index = 1.0
                status = 'No heuristic flag'
                recommendation = 'No automatic change; retain only if independently validated for the intended use.'

            report.append({
                'feature': feature,
                'importance': round(importance_map[feature], 3),
                'reliability': heuristic_index,
                'status': status,
                'recommendation': recommendation,
            })

        return sorted(report, key=lambda item: item['importance'], reverse=True)

    def generate_refinement_plan(self) -> dict[str, Any]:
        """Generate a review queue without applying or promising model changes."""
        sensitivity = self.analyze_feature_sensitivity()
        low_variance = [item for item in sensitivity if item['status'] == 'Low variance observed']
        correlated = [item for item in sensitivity if item['status'] == 'High correlation observed']
        insufficient = [item for item in sensitivity if item['status'] == 'Insufficient observations']
        flagged = low_variance + correlated + insufficient

        actions = [
            f"Review {item['feature']}: {item['status'].lower()}. Validate any proposed change on independent data."
            for item in flagged
        ]
        return {
            'status': 'review_only' if sensitivity else 'insufficient_evidence',
            'suggested_actions': actions,
            'automated_features_to_prune': [],
            'metrics': {
                'noisy_features': len(low_variance) + len(insufficient),
                'redundant_dimensions': len(correlated),
                'estimated_accuracy_lift': 'Unknown; requires independent evaluation',
            },
            'reasoning': 'Variance and correlation heuristics do not establish measurement reliability or an accuracy improvement. No model, feature, threshold, or activation state was changed.',
        }
