"""Privacy-governed aggregate clustering for PeopleOS.

Clustering is exploratory cohort structure, not an employee risk score, persona,
or stable taxonomy. Cluster IDs are arbitrary labels from an unsupervised fit and
must never be exposed as individual employee assignments by this engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger
from src.population import active_population

logger = get_logger('clustering_engine')
MIN_CLUSTER_SAMPLE = 10
MIN_PRIVACY_CELL = 5
CANDIDATE_FEATURES = [
    'Salary', 'Tenure', 'LastRating', 'Age',
    'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'CompaRatio', 'EngagementScore', 'Pulse_Score', 'InterviewScore',
]


class ClusteringEngine:
    """Aggregate-only K-Means exploration over the current active workforce."""

    def __init__(self, df: pd.DataFrame):
        self.df = active_population(df).copy(deep=True)
        self._source_population_size = len(self.df)
        self.model: Optional[KMeans] = None
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []
        self.cluster_labels: Optional[np.ndarray] = None
        self.results: Dict[str, Any] = {}
        self.training_frame = pd.DataFrame()

    def _prepare_data(self) -> pd.DataFrame:
        self.feature_cols = [
            col for col in CANDIDATE_FEATURES
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col])
        ]
        if not self.feature_cols:
            raise ValueError('No approved numeric clustering features are available')
        frame = self.df[self.feature_cols].apply(pd.to_numeric, errors='coerce')
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            raise ValueError('No complete finite rows remain for clustering')
        return frame

    @staticmethod
    def _stable_scale(frame: pd.DataFrame) -> np.ndarray:
        """Scale finite values without overflowing on extreme-but-finite magnitudes."""
        normalized = pd.DataFrame(index=frame.index)
        for col in frame.columns:
            values = frame[col].to_numpy(dtype=float)
            scale = float(np.max(np.abs(values))) if len(values) else 0.0
            if not np.isfinite(scale):
                raise ValueError('Clustering feature magnitude is not finite')
            normalized[col] = values if scale == 0 else values / scale
        scaled = StandardScaler().fit_transform(normalized)
        if not np.isfinite(scaled).all():
            raise ValueError('Clustering normalization produced non-finite values')
        return scaled

    @staticmethod
    def _stable_mean(series: pd.Series) -> float:
        values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError('Cluster summary contains non-finite values')
        scale = float(np.max(np.abs(values)))
        if scale == 0:
            return 0.0
        value = float(np.mean(values / scale) * scale)
        if not np.isfinite(value):
            raise ValueError('Cluster summary mean is non-finite')
        return value

    @staticmethod
    def _python(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): ClusteringEngine._python(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [ClusteringEngine._python(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if pd.isna(value):
            return None
        return value

    def _population_metadata(self, analysis_population: int) -> Dict[str, Any]:
        source = int(self._source_population_size)
        return {
            'source_population': source,
            'analysis_population': int(analysis_population),
            'excluded_missing_or_nonfinite_features': int(source - analysis_population),
            'coverage': float(analysis_population / source) if source else 0.0,
            'population_semantics': 'current_active_employees_with_complete_finite_approved_clustering_features',
        }

    def train(self, n_clusters: int = 3, auto_tune: bool = True) -> Dict[str, Any]:
        self.results, self.model, self.cluster_labels = {}, None, None
        self.training_frame = pd.DataFrame()
        try:
            X = self._prepare_data()
            population = self._population_metadata(len(X))
            if len(X) < MIN_CLUSTER_SAMPLE:
                return {'success': False, 'reason': f'Insufficient complete-case rows for clustering ({len(X)} < {MIN_CLUSTER_SAMPLE})', 'population': population}

            distinct = len(X.drop_duplicates())
            if distinct < 2:
                return {'success': False, 'reason': 'At least two distinct observations are required', 'population': population}
            if not auto_tune and (
                isinstance(n_clusters, (bool, np.bool_))
                or not isinstance(n_clusters, (int, np.integer))
                or not 2 <= int(n_clusters) <= min(distinct, len(X) - 1)
            ):
                return {'success': False, 'reason': 'Cluster count must be an integer compatible with the analysis population', 'population': population}

            self.training_frame = self.df.loc[X.index].copy()
            X_scaled = self._stable_scale(X)

            best_n = int(n_clusters) if isinstance(n_clusters, (int, np.integer)) and not isinstance(n_clusters, (bool, np.bool_)) else 3
            best_score: Optional[float] = None
            best_model: Optional[KMeans] = None

            if auto_tune:
                max_k = min(6, distinct, len(X) - 1)
                candidates: list[tuple[float, int, KMeans]] = []
                for k in range(2, max_k + 1):
                    model = KMeans(n_clusters=k, random_state=42, n_init='auto')
                    labels = model.fit_predict(X_scaled)
                    counts = np.bincount(labels, minlength=k)
                    # Prefer models whose every cluster is privacy-publishable. If none
                    # satisfy this, we still fit the best geometry but suppress small cells.
                    score = float(silhouette_score(X_scaled, labels))
                    if np.isfinite(score):
                        candidates.append((score, k, model))
                if not candidates:
                    return {'success': False, 'reason': 'No stable clustering candidate could be fit', 'population': population}
                publishable = [item for item in candidates if np.bincount(item[2].labels_, minlength=item[1]).min() >= MIN_PRIVACY_CELL]
                best_score, best_n, best_model = max(publishable or candidates, key=lambda item: item[0])
            else:
                best_model = KMeans(n_clusters=int(n_clusters), random_state=42, n_init='auto')
                best_model.fit(X_scaled)

            if best_model is None:
                return {'success': False, 'reason': 'No clustering model could be fit safely', 'population': population}

            labels = best_model.predict(X_scaled)
            if len(labels) != len(X):
                return {'success': False, 'reason': 'Cluster assignment count does not reconcile to the analysis population', 'population': population}

            self.model = best_model
            self.cluster_labels = labels
            labeled = self.training_frame.copy()
            labeled['Cluster'] = labels
            raw_counts = labeled['Cluster'].value_counts().sort_index()
            publishable_ids = [int(cluster_id) for cluster_id, count in raw_counts.items() if int(count) >= MIN_PRIVACY_CELL]
            suppressed_ids = [int(cluster_id) for cluster_id, count in raw_counts.items() if int(count) < MIN_PRIVACY_CELL]
            suppressed_rows = int(sum(int(raw_counts.loc[cluster_id]) for cluster_id in suppressed_ids))

            cluster_counts = {str(cluster_id): int(raw_counts.loc[cluster_id]) for cluster_id in publishable_ids}
            feature_summary: Dict[str, Dict[str, float]] = {}
            top_departments: Dict[str, Dict[str, int]] = {}
            for cluster_id in publishable_ids:
                group = labeled[labeled['Cluster'] == cluster_id]
                feature_summary[str(cluster_id)] = {
                    col: self._stable_mean(group[col]) for col in self.feature_cols
                }
                if 'Dept' in group.columns:
                    counts = group['Dept'].astype('string').fillna('Unknown').value_counts()
                    # Department cells inside a cluster are independently suppressed.
                    top_departments[str(cluster_id)] = {
                        str(name): int(count) for name, count in counts.items()
                        if int(count) >= MIN_PRIVACY_CELL
                    }

            result = {
                'success': True,
                'n_clusters': int(best_n),
                'silhouette_score': float(best_score) if best_score is not None else None,
                'feature_summary': feature_summary,
                'cluster_counts': cluster_counts,
                'top_departments': top_departments,
                'suppressed_cluster_count': len(suppressed_ids),
                'suppressed_row_count': suppressed_rows,
                'minimum_privacy_cell': MIN_PRIVACY_CELL,
                'population': population,
                'features_used': list(self.feature_cols),
                'cluster_semantics': 'unsupervised_group_ids_are_arbitrary_and_not_stable_personas_or_risk_levels',
                'interpretation_boundary': 'Clusters are exploratory aggregate geometry only; they are not employee risk categories, causal groups, or stable personas.',
            }
            # Validate strict JSON finiteness before storing a publishable result.
            import json
            json.dumps(result, allow_nan=False)
            self.results = self._python(result)
            return self.results
        except Exception as exc:
            logger.exception('Clustering training failed')
            self.results, self.model, self.cluster_labels = {}, None, None
            self.training_frame = pd.DataFrame()
            return {'success': False, 'reason': f'Clustering could not be fit safely ({type(exc).__name__})'}

    def get_employee_clusters(self) -> pd.DataFrame:
        """Individual cluster membership is intentionally disabled by governance."""
        return pd.DataFrame(columns=['EmployeeID', 'Cluster', 'Cluster_Name'])
