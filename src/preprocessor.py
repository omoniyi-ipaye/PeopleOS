"""Deterministic preprocessing for PeopleOS predictive models.

All fitted statistics (imputation, clipping, categorical encodings and scaling)
are learned from the training population only and then reused unchanged for
holdout/new data. This prevents evaluation leakage and makes model runs
reproducible for a fixed dataset.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('preprocessor')


class PreprocessingError(Exception):
    pass


class Preprocessor:
    def __init__(self):
        self.config = load_config()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_metadata: dict[str, Any] = {}
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.scaling_columns: list[str] = []
        self.impute_values: dict[str, Any] = {}
        self.outlier_bounds: dict[str, tuple[float, float]] = {}
        self.dropped_columns: list[str] = []
        self.reference_date: pd.Timestamp | None = None
        # Governed training freezes the permitted raw inputs. Legacy standalone
        # preprocessing retains its existing behavior when no contract is set.
        self.input_columns: list[str] | None = None
        self._is_fitted = False

    def fit_transform(self, df: pd.DataFrame, target_column: str = 'Attrition') -> tuple[pd.DataFrame, dict]:
        frame = self._contract_input(df)
        frame = self._drop_high_null_columns(frame, threshold=0.9, fit=True)
        self._identify_column_types(frame, target_column)
        self.reference_date = self._resolve_reference_date(frame)
        frame = self._engineer_temporal_features(frame, reference_date=self.reference_date)
        # Engineered features may add numeric columns after initial identification.
        self._refresh_engineered_numeric_columns(frame, target_column)
        frame = self._impute_missing(frame, fit=True)
        frame = self._cap_outliers(frame, fit=True)
        frame = self._encode_categorical(frame, target_column, fit=True)
        frame = self._scale_features(frame, target_column, fit=True)
        self._is_fitted = True
        self.feature_metadata = {
            'numeric_columns': list(self.numeric_columns),
            'categorical_columns': list(self.categorical_columns),
            'scaling_columns': list(self.scaling_columns),
            'label_encoders': list(self.label_encoders.keys()),
            'processed_columns': list(frame.columns),
            'dropped_columns': list(self.dropped_columns),
            'reference_date': self.reference_date.isoformat() if self.reference_date is not None else None,
            'fit_scope': 'training_population',
            'input_columns': self.input_columns,
        }
        logger.info("Preprocessing fitted on %s rows. Shape: %s", len(df), frame.shape)
        return frame, self.feature_metadata

    def transform(self, df: pd.DataFrame, target_column: str = 'Attrition') -> pd.DataFrame:
        if not self._is_fitted:
            raise PreprocessingError('Preprocessor must be fitted before transform')
        frame = self._contract_input(df)
        frame = frame.drop(columns=[c for c in self.dropped_columns if c in frame.columns], errors='ignore')
        frame = self._engineer_temporal_features(frame, reference_date=self.reference_date)
        for col in self.numeric_columns + self.categorical_columns:
            if col not in frame:
                frame[col] = float('nan')
        frame = self._impute_missing(frame, fit=False)
        frame = self._cap_outliers(frame, fit=False)
        frame = self._encode_categorical(frame, target_column, fit=False)
        frame = self._scale_features(frame, target_column, fit=False)
        return frame

    def _contract_input(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.input_columns is None:
            return df.copy()
        if not df.columns.is_unique:
            raise PreprocessingError('Predictive inputs require unique column names')
        return df.loc[:, [c for c in self.input_columns if c in df.columns]].copy()

    def _drop_high_null_columns(self, df: pd.DataFrame, threshold: float = 0.9, fit: bool = False) -> pd.DataFrame:
        if fit:
            ratios = df.isna().mean()
            self.dropped_columns = ratios[ratios > threshold].index.tolist()
        if self.dropped_columns:
            logger.warning("Dropping columns with >%s%% nulls: %s", threshold * 100, self.dropped_columns)
        return df.drop(columns=[c for c in self.dropped_columns if c in df.columns], errors='ignore')

    def _identify_column_types(self, df: pd.DataFrame, target_column: str) -> None:
        self.numeric_columns = []
        self.categorical_columns = []
        metadata = {target_column.lower(), 'employeeid', 'employee_id', 'created_at', 'updated_at', 'is_active', 'snapshotdate', 'terminationdate', 'exitdate', 'terminationreason', 'exitreason', 'employmentstatus', 'status', 'attrition', 'hiredate', 'promotiondate', 'ratinghistory', 'performancetext', 'managerid', 'employeenumber', 'name', 'email'}
        for col in df.columns:
            if col.lower() in metadata:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_columns.append(col)
            elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
                self.categorical_columns.append(col)
        # Preserve bounded/raw meaning for Age and LastRating; other continuous features may be scaled.
        self.scaling_columns = [c for c in self.numeric_columns if c not in {'Age', 'LastRating'}]

    def _refresh_engineered_numeric_columns(self, df: pd.DataFrame, target_column: str) -> None:
        for col in ('RatingVelocity', 'PromotionLag', 'SalaryGrowth'):
            if col in df.columns and col != target_column and col not in self.numeric_columns:
                self.numeric_columns.append(col)
                if col not in self.scaling_columns:
                    self.scaling_columns.append(col)

    def _resolve_reference_date(self, df: pd.DataFrame) -> pd.Timestamp:
        if 'SnapshotDate' in df.columns:
            dates = pd.to_datetime(df['SnapshotDate'], errors='coerce', utc=True)
            if dates.notna().any():
                return dates.max().tz_localize(None)
        # HireDate is data-derived and stable for a fixed dataset. Use the latest
        # observed date as the analysis as-of fallback instead of wall-clock time.
        if 'HireDate' in df.columns:
            dates = pd.to_datetime(df['HireDate'], errors='coerce', utc=True)
            if dates.notna().any():
                return dates.max().tz_localize(None)
        return pd.Timestamp('1970-01-01')

    def _engineer_temporal_features(self, df: pd.DataFrame, reference_date: pd.Timestamp | None) -> pd.DataFrame:
        frame = df.copy()
        if 'RatingHistory' in frame.columns:
            def calc_velocity(history):
                if not isinstance(history, str) or not history:
                    return 0.0
                try:
                    ratings = [float(r.strip()) for r in history.split(',') if r.strip()]
                    recent = ratings[-3:]
                    return (recent[-1] - recent[0]) / (len(recent) - 1) if len(recent) >= 2 else 0.0
                except (TypeError, ValueError):
                    return 0.0
            frame['RatingVelocity'] = frame['RatingHistory'].apply(calc_velocity)

        if 'PromotionDate' in frame.columns:
            as_of = reference_date or pd.Timestamp('1970-01-01')
            promotion = pd.to_datetime(frame['PromotionDate'], errors='coerce', utc=True).dt.tz_localize(None)
            months = (as_of.year - promotion.dt.year) * 12 + (as_of.month - promotion.dt.month)
            frame['PromotionLag'] = months.fillna(0).clip(lower=0).astype(float)

        if {'Salary', 'StartingSalary', 'Tenure'}.issubset(frame.columns):
            start = pd.to_numeric(frame['StartingSalary'], errors='coerce')
            current = pd.to_numeric(frame['Salary'], errors='coerce')
            tenure = pd.to_numeric(frame['Tenure'], errors='coerce')
            valid = (start > 0) & (tenure > 0)
            growth = pd.Series(0.0, index=frame.index)
            growth.loc[valid] = ((current.loc[valid] - start.loc[valid]) / start.loc[valid]) / tenure.loc[valid]
            frame['SalaryGrowth'] = growth.replace([float('inf'), float('-inf')], 0).fillna(0)
        return frame

    def _impute_missing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        frame = df.copy()
        for col in self.numeric_columns:
            if col not in frame.columns:
                continue
            numeric = pd.to_numeric(frame[col], errors='coerce').replace([float('inf'), float('-inf')], float('nan'))
            if fit:
                median = numeric.median()
                self.impute_values[col] = float(median) if pd.notna(median) else 0.0
            frame[col] = numeric.fillna(self.impute_values.get(col, 0.0))
        for col in self.categorical_columns:
            if col not in frame.columns:
                continue
            if fit:
                mode = frame[col].dropna().astype(str).mode()
                self.impute_values[col] = mode.iloc[0] if not mode.empty else '__UNKNOWN__'
            frame[col] = frame[col].fillna(self.impute_values.get(col, '__UNKNOWN__')).astype(str)
        return frame

    def _cap_outliers(self, df: pd.DataFrame, fit: bool, iqr_multiplier: float = 1.5) -> pd.DataFrame:
        frame = df.copy()
        for col in self.numeric_columns:
            if col not in frame.columns:
                continue
            if fit:
                q1 = frame[col].quantile(0.25)
                q3 = frame[col].quantile(0.75)
                iqr = q3 - q1
                lower = float(q1 - iqr_multiplier * iqr)
                upper = float(q3 + iqr_multiplier * iqr)
                self.outlier_bounds[col] = (lower, upper)
            bounds = self.outlier_bounds.get(col)
            if bounds:
                frame[col] = frame[col].clip(lower=bounds[0], upper=bounds[1])
        return frame

    def _encode_categorical(self, df: pd.DataFrame, target_column: str, fit: bool) -> pd.DataFrame:
        frame = df.copy()
        for col in self.categorical_columns:
            if col not in frame.columns:
                continue
            values = frame[col].astype(str)
            if fit:
                encoder = LabelEncoder()
                # Explicit unknown class avoids silently mapping a novel value to a real category.
                encoder.fit(pd.concat([values, pd.Series(['__UNKNOWN__'])], ignore_index=True))
                self.label_encoders[col] = encoder
            encoder = self.label_encoders.get(col)
            if encoder is not None:
                known = set(encoder.classes_)
                values = values.where(values.isin(known), '__UNKNOWN__')
                frame[col] = encoder.transform(values)

        # Target encoding is intentionally narrow; callers should normalize Attrition before training.
        if target_column in frame.columns and frame[target_column].dtype == 'object':
            if fit:
                encoder = LabelEncoder()
                encoder.fit(frame[target_column].astype(str))
                self.label_encoders[target_column] = encoder
            encoder = self.label_encoders.get(target_column)
            if encoder is not None:
                values = frame[target_column].astype(str)
                unknown = ~values.isin(set(encoder.classes_))
                if unknown.any():
                    raise PreprocessingError(f'Unknown target label(s) in transform: {sorted(values[unknown].unique())}')
                frame[target_column] = encoder.transform(values)
        return frame

    def _scale_features(self, df: pd.DataFrame, target_column: str, fit: bool) -> pd.DataFrame:
        frame = df.copy()
        cols = [c for c in self.scaling_columns if c in frame.columns and c != target_column]
        if not cols:
            return frame
        if fit:
            frame[cols] = self.scaler.fit_transform(frame[cols])
        else:
            frame[cols] = self.scaler.transform(frame[cols])
        return frame

    def get_feature_metadata(self) -> dict:
        return self.feature_metadata
