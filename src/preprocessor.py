"""
Preprocessor module for PeopleOS.

Handles data preprocessing including imputation, encoding, scaling,
and outlier handling.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('preprocessor')


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass


class Preprocessor:
    """
    Data preprocessing pipeline for PeopleOS.
    
    Handles missing values, encoding, scaling, and outlier detection.
    """
    
    def __init__(self):
        """Initialize the Preprocessor with configuration."""
        self.config = load_config()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler = StandardScaler()
        self.feature_metadata: dict[str, Any] = {}
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.scaling_columns: list[str] = []
    
    def fit_transform(self, df: pd.DataFrame, target_column: str = 'Attrition') -> tuple[pd.DataFrame, dict]:
        """
        Fit and transform the data through the preprocessing pipeline.
        
        Args:
            df: Input DataFrame.
            target_column: Name of the target column for prediction.
            
        Returns:
            Tuple of (processed DataFrame, feature metadata).
        """
        df = df.copy()
        
        # Step 1: Drop columns with >90% missing values
        df = self._drop_high_null_columns(df, threshold=0.9)
        
        # Step 2: Identify column types
        self._identify_column_types(df, target_column)
        
        # Step 2.5: Engineer temporal features
        df = self._engineer_temporal_features(df)
        
        # Step 3: Impute missing values
        df = self._impute_missing(df)
        
        # Step 4: Detect and cap outliers (IQR method)
        df = self._cap_outliers(df)
        
        # Step 5: Encode categorical variables
        df = self._encode_categorical(df, target_column)
        
        # Step 6: Scale numeric features
        df = self._scale_features(df, target_column)
        
        # Store feature metadata
        self.feature_metadata = {
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'scaling_columns': self.scaling_columns,
            'label_encoders': list(self.label_encoders.keys()),
            'processed_columns': list(df.columns)
        }
        
        logger.info(f"Preprocessing complete. Shape: {df.shape}")
        return df, self.feature_metadata
    
    def transform(self, df: pd.DataFrame, target_column: str = 'Attrition') -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame.
            target_column: Name of the target column.
            
        Returns:
            Transformed DataFrame.
        """
        df = df.copy()
        
        # Apply same transformations (using fitted encoders/scaler)
        df = self._identify_column_types_transform(df, target_column)
        df = self._engineer_temporal_features(df)
        df = self._impute_missing(df)
        df = self._cap_outliers(df)
        df = self._encode_categorical(df, target_column, fit=False)
        df = self._scale_features(df, target_column, fit=False)
        
        return df
    
    def _drop_high_null_columns(self, df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
        """
        Drop columns with null ratio above threshold.
        
        Args:
            df: Input DataFrame.
            threshold: Maximum allowed null ratio.
            
        Returns:
            DataFrame with high-null columns removed.
        """
        null_ratios = df.isna().sum() / len(df)
        cols_to_drop = null_ratios[null_ratios > threshold].index.tolist()
        
        if cols_to_drop:
            logger.warning(f"Dropping columns with >{threshold*100}% nulls: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        return df
    
    def _identify_column_types(self, df: pd.DataFrame, target_column: str) -> None:
        """
        Identify numeric and categorical columns.
        
        Args:
            df: Input DataFrame.
            target_column: Name of target column to exclude.
        """
        self.numeric_columns = []
        self.categorical_columns = []
        
        for col in df.columns:
            if col == target_column or col == 'EmployeeID':
                continue
            
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                self.numeric_columns.append(col)
            elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
                self.categorical_columns.append(col)
        
        # Columns to scale (continuous numeric features)
        self.scaling_columns = [col for col in self.numeric_columns 
                                if col not in ['Age', 'LastRating']]  # These are bounded already
        
        logger.info(f"Numeric columns: {self.numeric_columns}")
        logger.info(f"Categorical columns: {self.categorical_columns}")

    def _identify_column_types_transform(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Helper to ensure engineered columns are tracked during transform."""
        # Simple check for the features we know we engineer
        for col in ['RatingVelocity', 'PromotionLag', 'SalaryGrowth']:
            if col not in self.numeric_columns:
                # We'll add them if they're about to be engineered
                pass 
        return df

    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RatingVelocity, PromotionLag, and SalaryGrowth.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with new features.
        """
        # RatingVelocity: slope of last 3 ratings
        if 'RatingHistory' in df.columns:
            def calc_velocity(history):
                if not isinstance(history, str) or not history:
                    return 0.0
                try:
                    ratings = [float(r.strip()) for r in history.split(',') if r.strip()]
                    if len(ratings) < 2:
                        return 0.0
                    recent = ratings[-3:]
                    if len(recent) < 2:
                        return 0.0
                    return (recent[-1] - recent[0]) / (len(recent) - 1)
                except Exception:
                    return 0.0
            
            df['RatingVelocity'] = df['RatingHistory'].apply(calc_velocity)
            if 'RatingVelocity' not in self.numeric_columns:
                self.numeric_columns.append('RatingVelocity')
            if 'RatingVelocity' not in self.scaling_columns:
                self.scaling_columns.append('RatingVelocity')

        # PromotionLag: months since last promotion
        if 'PromotionDate' in df.columns:
            def calc_lag(promo_date):
                try:
                    p_date = pd.to_datetime(promo_date)
                    now = pd.Timestamp.now()
                    return (now.year - p_date.year) * 12 + (now.month - p_date.month)
                except Exception:
                    return 0.0
            
            df['PromotionLag'] = df['PromotionDate'].apply(calc_lag)
            df['PromotionLag'] = df['PromotionLag'].clip(lower=0)
            if 'PromotionLag' not in self.numeric_columns:
                self.numeric_columns.append('PromotionLag')
            if 'PromotionLag' not in self.scaling_columns:
                self.scaling_columns.append('PromotionLag')

        # SalaryGrowth: annualized increase %
        if 'Salary' in df.columns and 'StartingSalary' in df.columns and 'Tenure' in df.columns:
            def calc_growth(row):
                try:
                    start = float(row['StartingSalary'])
                    current = float(row['Salary'])
                    tenure = float(row['Tenure'])
                    if start <= 0 or tenure <= 0:
                        return 0.0
                    return ((current - start) / start) / tenure
                except Exception:
                    return 0.0
            
            df['SalaryGrowth'] = df.apply(calc_growth, axis=1)
            if 'SalaryGrowth' not in self.numeric_columns:
                self.numeric_columns.append('SalaryGrowth')
            if 'SalaryGrowth' not in self.scaling_columns:
                self.scaling_columns.append('SalaryGrowth')
        
        return df
    
    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values.
        
        Strategy: median for numeric, mode for categorical.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with imputed values.
        """
        # Numeric: median
        for col in self.numeric_columns:
            if col in df.columns and df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Imputed {col} with median: {median_val}")
        
        # Categorical: mode
        for col in self.categorical_columns:
            if col in df.columns and df[col].isna().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                    logger.info(f"Imputed {col} with mode: {mode_val[0]}")
        
        return df
    
    def _cap_outliers(self, df: pd.DataFrame, iqr_multiplier: float = 1.5) -> pd.DataFrame:
        """
        Cap outliers using IQR method.
        
        Args:
            df: Input DataFrame.
            iqr_multiplier: Multiplier for IQR bounds.
            
        Returns:
            DataFrame with capped outliers.
        """
        for col in self.numeric_columns:
            if col not in df.columns:
                continue
            
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - (iqr_multiplier * iqr)
            upper_bound = q3 + (iqr_multiplier * iqr)
            
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers_count > 0:
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Capped {outliers_count} outliers in {col}")
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, target_column: str, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables using LabelEncoder.
        
        Args:
            df: Input DataFrame.
            target_column: Target column name.
            fit: Whether to fit new encoders.
            
        Returns:
            DataFrame with encoded categoricals.
        """
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            
            if fit:
                encoder = LabelEncoder()
                # Handle unseen values by adding a placeholder
                df[col] = df[col].astype(str)
                df[col] = encoder.fit_transform(df[col])
                self.label_encoders[col] = encoder
            else:
                if col in self.label_encoders:
                    encoder = self.label_encoders[col]
                    df[col] = df[col].astype(str)
                    # Handle unseen values
                    known_classes = set(encoder.classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_classes else encoder.classes_[0]
                    )
                    df[col] = encoder.transform(df[col])
        
        # Encode target if present and categorical
        if target_column in df.columns and df[target_column].dtype == 'object':
            if fit:
                encoder = LabelEncoder()
                df[target_column] = encoder.fit_transform(df[target_column].astype(str))
                self.label_encoders[target_column] = encoder
            elif target_column in self.label_encoders:
                df[target_column] = self.label_encoders[target_column].transform(
                    df[target_column].astype(str)
                )
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, target_column: str, fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features using StandardScaler.
        
        Args:
            df: Input DataFrame.
            target_column: Target column to exclude from scaling.
            fit: Whether to fit the scaler.
            
        Returns:
            DataFrame with scaled features.
        """
        cols_to_scale = [col for col in self.scaling_columns 
                         if col in df.columns and col != target_column]
        
        if not cols_to_scale:
            return df
        
        if fit:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        else:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        logger.info(f"Scaled columns: {cols_to_scale}")
        return df
    
    def get_feature_metadata(self) -> dict:
        """
        Get metadata about processed features.
        
        Returns:
            Dictionary with feature metadata.
        """
        return self.feature_metadata
