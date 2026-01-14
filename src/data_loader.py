"""
Data Loader module for PeopleOS.

Handles loading of CSV, JSON, and SQLite data with fuzzy column matching
and comprehensive validation. Supports persistent database storage.
"""

import json
import sqlite3
from difflib import get_close_matches
from typing import Optional, Dict, Any

import pandas as pd

from src.logger import get_logger, sanitize_for_logging
from src.utils import load_config, get_error_message, get_file_extension

logger = get_logger('data_loader')


# Golden Schema - the standard column names
GOLDEN_SCHEMA = {
    'required': [
        'EmployeeID', 'Dept', 'Tenure', 'Salary', 'LastRating', 'Age', 'Gender',
        'JobTitle', 'Location', 'HireDate', 'ManagerID'
    ],
    'optional': [
        'Attrition', 'PerformanceText', 'RatingHistory', 'PromotionDate', 
        'StartingSalary', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'PromotionCount', 'InterviewScore', 'AssessmentScore', 'HireSource',
        'SnapshotDate', 'Country', 'JobLevel', 'CompaRatio', 
        'PriorExperienceYears', 'ManagerChangeCount'
    ]
}

# Common variations for fuzzy matching
COLUMN_ALIASES = {
    'employeeid': ['emp_id', 'employee_id', 'id', 'empid', 'emp_no', 'employee_no', 'staff_id'],
    'snapshotdate': ['date', 'month', 'snapshot_date', 'period', 'as_of_date'],
    'dept': ['department', 'dept_name', 'department_name', 'division', 'team'],
    'tenure': ['years_of_service', 'yos', 'experience', 'years_employed', 'service_years'],
    'salary': ['compensation', 'pay', 'wage', 'annual_salary', 'base_salary', 'income'],
    'lastrating': ['rating', 'performance_rating', 'perf_rating', 'review_score', 'performance_score', 'last_review'],
    'age': ['employee_age', 'years_old'],
    'gender': ['sex'],
    'jobtitle': ['title', 'role', 'position', 'job_role'],
    'location': ['office', 'site', 'city', 'region', 'work_location'],
    'hiredate': ['date_of_hire', 'hired_at', 'joined_date', 'join_date'],
    'managerid': ['manager_id', 'manager', 'supervisor_id', 'reports_to'],
    'attrition': ['left', 'departed', 'terminated', 'resigned', 'churned', 'turnover', 'attrition_flag'],
    'performancetext': ['performance_review', 'review_text', 'feedback', 'comments', 'performance_notes'],
    'ratinghistory': ['historical_ratings', 'rating_history', 'past_ratings', 'performance_history'],
    'promotiondate': ['last_promotion_date', 'date_of_promotion', 'last_promo_date'],
    'promotioncount': ['promo_count', 'number_of_promotions'],
    'yearssincelastpromotion': ['years_since_promotion', 'time_since_promotion', 'promotion_lag'],
    'yearsincurrentrole': ['years_in_role', 'role_tenure', 'time_in_role'],
    'startingsalary': ['start_salary', 'initial_salary', 'hiring_salary', 'base_pay_start'],
    'interviewscore': ['interview_avg', 'interview_rating'],
    'assessmentscore': ['test_score', 'aptitude_score'],
    'hiresource': ['source', 'recruitment_source', 'hiring_channel', 'referral_source']
}


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class DataLoader:
    """
    Handles loading and validation of HR data from multiple formats.
    
    Supports CSV, JSON, and SQLite formats with fuzzy column matching
    to the Golden Schema.
    """
    
    def __init__(self):
        """Initialize the DataLoader with configuration."""
        self.config = load_config()
        self.data_config = self.config.get('data', {})
        self.min_rows = self.data_config.get('min_rows', 50)
        self.max_rows = self.data_config.get('max_rows', 50000)
        self.allowed_formats = self.data_config.get('allowed_formats', ['csv', 'json', 'sqlite'])
        self.column_mapping: dict[str, str] = {}
        self.validation_warnings: list[str] = []
        self.features_enabled = {
            'predictive': True,
            'nlp': True
        }
    
    def load(self, file_path: str, table_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from a file with validation.
        
        Args:
            file_path: Path to the data file.
            table_name: Name of table for SQLite files.
            
        Returns:
            Validated and mapped DataFrame.
            
        Raises:
            DataValidationError: If validation fails.
        """
        self.validation_warnings = []
        
        # Check file extension
        ext = get_file_extension(file_path)
        if ext not in self.allowed_formats:
            raise DataValidationError(
                get_error_message('file_load_failed')
            )
        
        # Load raw data
        try:
            if ext == 'csv':
                df = self._load_csv(file_path)
            elif ext == 'json':
                df = self._load_json(file_path)
            elif ext in ('sqlite', 'db', 'sqlite3'):
                df = self._load_sqlite(file_path, table_name)
            else:
                raise DataValidationError(get_error_message('file_load_failed'))
        except Exception as e:
            logger.error(f"Failed to load file: {type(e).__name__}")
            raise DataValidationError(get_error_message('file_load_failed'))
        
        # Validate row count
        if len(df) < self.min_rows:
            raise DataValidationError(
                get_error_message('insufficient_data', count=len(df))
            )
        
        # Truncate if too large
        if len(df) > self.max_rows:
            logger.warning(f"Dataset truncated from {len(df)} to {self.max_rows} rows")
            df = df.head(self.max_rows)
            self.validation_warnings.append(f"Dataset truncated to {self.max_rows} rows")
        
        # Map columns to Golden Schema
        df = self._map_columns(df)
        
        # Validate required columns
        self._validate_required_columns(df)
        
        # Validate data quality
        df = self._validate_data_quality(df)
        
        logger.info(f"Successfully loaded {len(df)} rows")
        return df
    
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        return pd.read_csv(file_path)
    
    def _load_json(self, file_path: str) -> pd.DataFrame:
        """Load data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both array and object with 'data' key
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            return pd.DataFrame(data['data'])
        else:
            return pd.DataFrame([data])
    
    def _load_sqlite(self, file_path: str, table_name: Optional[str] = None) -> pd.DataFrame:
        """Load data from SQLite database."""
        conn = sqlite3.connect(file_path)
        
        if table_name is None:
            # Get first table
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            if not tables:
                conn.close()
                raise DataValidationError("No tables found in SQLite database")
            table_name = tables[0][0]
        
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    
    def _fuzzy_match_column(self, column: str) -> Optional[str]:
        """
        Match a column name to the Golden Schema using fuzzy matching.
        
        Args:
            column: The column name to match.
            
        Returns:
            Matched Golden Schema column name or None.
        """
        col_lower = column.lower().replace(' ', '_').replace('-', '_')
        
        # Direct match
        for golden_col in GOLDEN_SCHEMA['required'] + GOLDEN_SCHEMA['optional']:
            if col_lower == golden_col.lower():
                return golden_col
        
        # Alias match
        for golden_col, aliases in COLUMN_ALIASES.items():
            if col_lower in aliases or col_lower == golden_col:
                # Convert back to proper case
                for gc in GOLDEN_SCHEMA['required'] + GOLDEN_SCHEMA['optional']:
                    if gc.lower() == golden_col:
                        return gc
        
        # Fuzzy match using difflib
        all_golden = [c.lower() for c in GOLDEN_SCHEMA['required'] + GOLDEN_SCHEMA['optional']]
        matches = get_close_matches(col_lower, all_golden, n=1, cutoff=0.6)
        
        if matches:
            for gc in GOLDEN_SCHEMA['required'] + GOLDEN_SCHEMA['optional']:
                if gc.lower() == matches[0]:
                    return gc
        
        return None
    
    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map DataFrame columns to the Golden Schema.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with mapped column names.
        """
        self.column_mapping = {}
        rename_map = {}
        mapped_targets = set()
        
        # Step 1: Direct matches first
        for col in df.columns:
            # Strip whitespace to handle dirty CSV headers
            col_clean = col.strip()
            col_lower = col_clean.lower().replace(' ', '_').replace('-', '_')
            for golden_col in GOLDEN_SCHEMA['required'] + GOLDEN_SCHEMA['optional']:
                if col_lower == golden_col.lower():
                    self.column_mapping[col] = golden_col
                    rename_map[col] = golden_col
                    mapped_targets.add(golden_col)
                    break
        
        # Step 2: Alias and fuzzy matches for remaining columns
        for col in df.columns:
            if col in rename_map:
                continue
                
            matched = self._fuzzy_match_column(col)
            # Only map if target is NOT already taken
            if matched and matched not in mapped_targets:
                self.column_mapping[col] = matched
                rename_map[col] = matched
                mapped_targets.add(matched)
        
        df = df.rename(columns=rename_map)
        
        # Log column mapping
        if self.column_mapping:
            logger.info(f"Column mapping: {sanitize_for_logging(self.column_mapping)}")
        
        return df
    
    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that all required columns are present.
        
        Args:
            df: DataFrame to validate.
            
        Raises:
            DataValidationError: If required columns are missing.
        """
        missing = [col for col in GOLDEN_SCHEMA['required'] if col not in df.columns]
        
        if missing:
            raise DataValidationError(
                get_error_message('missing_columns', columns=', '.join(missing))
            )
        
        # Check optional columns and disable features if missing
        if 'Attrition' not in df.columns:
            self.features_enabled['predictive'] = False
            self.validation_warnings.append("Attrition column missing - predictive analytics disabled")
            logger.warning("Attrition column missing - predictive analytics disabled")
        
        if 'PerformanceText' not in df.columns:
            self.features_enabled['nlp'] = False
            self.validation_warnings.append("PerformanceText column missing - NLP features disabled")
            logger.warning("PerformanceText column missing - NLP features disabled")
    
    def _validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data quality and clean issues.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Cleaned DataFrame.
            
        Raises:
            DataValidationError: If critical validation fails.
        """
        # Check for duplicate EmployeeIDs
        # Only error if SnapshotDate is NOT present (SnapshotDate implies historical history)
        if 'SnapshotDate' not in df.columns:
            duplicates = df['EmployeeID'].duplicated().sum()
            if duplicates > 0:
                raise DataValidationError(
                    get_error_message('duplicate_ids', count=duplicates)
                )
        
        # Check for empty columns (>90% null)
        for col in df.columns:
            null_ratio = df[col].isna().sum() / len(df)
            if null_ratio > 0.9:
                logger.warning(f"Column {col} has >90% null values, excluding")
                self.validation_warnings.append(f"Column '{col}' excluded (>90% null)")
                df = df.drop(columns=[col])
        
        # Check and handle negative values in numeric columns
        numeric_cols = ['Salary', 'Tenure', 'Age']
        negative_mask = pd.Series([False] * len(df))
        
        for col in numeric_cols:
            if col in df.columns:
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                col_negative = df[col] < 0
                negative_mask = negative_mask | col_negative
        
        negative_count = negative_mask.sum()
        if negative_count > 0:
            df = df[~negative_mask]
            self.validation_warnings.append(f"Removed {negative_count} rows with negative values")
            logger.warning(f"Removed {negative_count} rows with negative values")
        
        # Check remaining row count
        if len(df) < self.min_rows:
            raise DataValidationError(
                get_error_message('insufficient_data', count=len(df))
            )
        
        return df
    
    def get_column_mapping_report(self) -> dict:
        """
        Get a report of column mappings.

        Returns:
            Dictionary with original -> mapped column names.
        """
        return {
            'mappings': self.column_mapping,
            'warnings': self.validation_warnings,
            'features_enabled': self.features_enabled
        }

    def load_from_database(self) -> Optional[pd.DataFrame]:
        """
        Load employee data from the persistent database.

        Returns:
            DataFrame with all employees, or None if database is empty
            or persistence is disabled.
        """
        persistence_config = self.config.get('persistence', {})
        if not persistence_config.get('enabled', True):
            logger.info("Persistence is disabled, skipping database load")
            return None

        try:
            from src.database import get_database
            db = get_database()

            if not db.has_data():
                logger.info("Database is empty")
                return None

            df = db.get_all_employees()

            if df.empty:
                return None

            # Set feature flags based on columns present
            if 'Attrition' in df.columns and df['Attrition'].notna().any():
                self.features_enabled['predictive'] = True
            else:
                self.features_enabled['predictive'] = False

            if 'PerformanceText' in df.columns and df['PerformanceText'].notna().any():
                self.features_enabled['nlp'] = True
            else:
                self.features_enabled['nlp'] = False

            logger.info(f"Loaded {len(df)} employees from database")
            return df

        except Exception as e:
            logger.error(f"Failed to load from database: {e}")
            return None

    def load_and_merge(
        self,
        file_path: str,
        file_name: str = "upload",
        table_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load data from file and merge with existing database.

        This is the main method for incremental data loading.
        Data is validated, then merged with existing records
        (auto-update for existing EmployeeIDs).

        Args:
            file_path: Path to the data file.
            file_name: Original file name for history tracking.
            table_name: Name of table for SQLite files.

        Returns:
            Dictionary with:
            - 'df': The merged DataFrame
            - 'merge_result': Result of the merge operation
            - 'report': Column mapping report
        """
        # First load and validate the file
        df = self.load(file_path, table_name)

        persistence_config = self.config.get('persistence', {})
        if not persistence_config.get('enabled', True):
            # Persistence disabled - just return the loaded data
            return {
                'df': df,
                'merge_result': None,
                'report': self.get_column_mapping_report()
            }

        # Merge with database
        try:
            from src.merge_engine import get_merge_engine
            merge_engine = get_merge_engine()
            merge_result = merge_engine.execute_merge(df, file_name)

            # Reload from database to get the complete merged dataset
            from src.database import get_database
            db = get_database()
            merged_df = db.get_all_employees()

            # Update feature flags
            if 'Attrition' in merged_df.columns and merged_df['Attrition'].notna().any():
                self.features_enabled['predictive'] = True

            if 'PerformanceText' in merged_df.columns and merged_df['PerformanceText'].notna().any():
                self.features_enabled['nlp'] = True

            logger.info(f"Merge complete: {merge_result.get_summary_text()}")

            return {
                'df': merged_df,
                'merge_result': merge_result,
                'report': self.get_column_mapping_report()
            }

        except Exception as e:
            logger.error(f"Merge failed: {e}")
            # Fall back to just the uploaded data
            return {
                'df': df,
                'merge_result': None,
                'report': self.get_column_mapping_report()
            }

    def is_persistence_enabled(self) -> bool:
        """Check if persistence is enabled in config."""
        return self.config.get('persistence', {}).get('enabled', True)

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the persistent database.

        Returns:
            Dictionary with database statistics.
        """
        if not self.is_persistence_enabled():
            return {'enabled': False}

        try:
            from src.database import get_database
            db = get_database()

            return {
                'enabled': True,
                'has_data': db.has_data(),
                'employee_count': db.get_employee_count(),
                'recent_uploads': db.get_upload_history(limit=5)
            }
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {'enabled': True, 'error': str(e)}
