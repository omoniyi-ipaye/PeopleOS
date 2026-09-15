"""Validated workforce data ingestion for PeopleOS.

Column mapping is conservative for identity/outcome/pay fields. Snapshot datasets
are preserved as histories instead of being collapsed into the legacy one-row-
per-employee SQLite store; current-state resolution happens explicitly in the
runtime population layer.
"""

from __future__ import annotations

import json
import sqlite3
import numpy as np
from difflib import get_close_matches
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from src.logger import get_logger, sanitize_for_logging
from src.population import normalize_attrition
from src.data_contract import normalize_measurements, is_numeric_measurement
from src.utils import get_error_message, get_file_extension, load_config

logger = get_logger('data_loader')

GOLDEN_SCHEMA = {
    'required': ['EmployeeID', 'Dept', 'Tenure', 'Salary', 'LastRating', 'Age', 'Gender', 'JobTitle', 'Location', 'HireDate', 'ManagerID'],
    'optional': ['Attrition', 'PerformanceText', 'RatingHistory', 'PromotionDate', 'StartingSalary', 'YearsInCurrentRole',
                 'YearsSinceLastPromotion', 'PromotionCount', 'InterviewScore', 'AssessmentScore', 'HireSource', 'SnapshotDate',
                 'Country', 'JobLevel', 'CompaRatio', 'PriorExperienceYears', 'ManagerChangeCount',
                 'PayFrequency', 'PayPeriod', 'Currency']
}

COLUMN_ALIASES = {
    'payfrequency': ['pay_frequency', 'salary_frequency', 'salaryfrequency'],
    'payperiod': ['pay_period', 'salary_period', 'salaryperiod', 'salary_basis', 'pay_basis'],
    'currency': ['salary_currency', 'salarycurrency', 'pay_currency', 'currency_code'],
    'employeeid': ['emp_id', 'employee_id', 'id', 'empid', 'emp_no', 'employee_no', 'staff_id'],
    'snapshotdate': ['date', 'month', 'snapshot_date', 'period', 'as_of_date'],
    'dept': ['department', 'dept_name', 'department_name', 'division', 'team'],
    'tenure': ['years_of_service', 'yos', 'experience', 'years_employed', 'service_years'],
    'salary': ['compensation', 'pay', 'wage', 'annual_salary', 'base_salary', 'income'],
    'lastrating': ['rating', 'performance_rating', 'perf_rating', 'review_score', 'performance_score', 'last_review'],
    'age': ['employee_age', 'years_old'], 'gender': ['sex'],
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
    'hiresource': ['source', 'recruitment_source', 'hiring_channel', 'referral_source'],
}

CRITICAL_MAPPING_FIELDS = {'EmployeeID', 'Salary', 'Attrition', 'HireDate', 'SnapshotDate', 'Gender', 'Age', 'ManagerID',
                           'PayFrequency', 'PayPeriod', 'Currency'}


class DataValidationError(Exception):
    pass


class DataLoader:
    def __init__(self):
        self.config = load_config()
        self.data_config = self.config.get('data', {})
        self.min_rows = self.data_config.get('min_rows', 50)
        self.max_rows = self.data_config.get('max_rows', 50000)
        self.allowed_formats = self.data_config.get('allowed_formats', ['csv', 'json', 'sqlite'])
        self.column_mapping: dict[str, str] = {}
        self.mapping_details: dict[str, dict[str, Any]] = {}
        self.validation_warnings: list[str] = []
        self.features_enabled = {'predictive': True, 'nlp': True}

    def _reset_load_state(self) -> None:
        self.validation_warnings = []
        self.column_mapping = {}
        self.mapping_details = {}
        self.features_enabled = {'predictive': True, 'nlp': True}

    def _read_frame(self, file_path: str, table_name: Optional[str] = None) -> pd.DataFrame:
        """Read a source without mapping, validation, or runtime activation."""
        ext = get_file_extension(file_path)
        if ext not in self.allowed_formats:
            raise DataValidationError(get_error_message('file_load_failed'))
        try:
            if ext == 'csv':
                # Preserve lexemes before aliases are mapped; measurements convert below.
                return pd.read_csv(file_path, dtype=str, keep_default_na=False, na_values=[''])
            if ext == 'json':
                with open(file_path, 'r') as handle:
                    payload = json.load(handle)
                return pd.DataFrame(
                    payload if isinstance(payload, list)
                    else payload.get('data', [payload]) if isinstance(payload, dict)
                    else payload
                )
            if ext in ('sqlite', 'db', 'sqlite3'):
                return self._load_sqlite(file_path, table_name)
            raise DataValidationError(get_error_message('file_load_failed'))
        except DataValidationError:
            raise
        except Exception as exc:
            logger.error('Failed to load file: %s', type(exc).__name__)
            raise DataValidationError(get_error_message('file_load_failed')) from exc

    def load(
        self,
        file_path: str,
        table_name: Optional[str] = None,
        *,
        column_mapping: Optional[Mapping[str, Optional[str]]] = None,
        mapping_methods: Optional[Mapping[str, str]] = None,
    ) -> pd.DataFrame:
        self._reset_load_state()
        df = self._read_frame(file_path, table_name)

        if len(df) < self.min_rows:
            raise DataValidationError(get_error_message('insufficient_data', count=len(df)))
        if len(df) > self.max_rows:
            raise DataValidationError(
                f'Dataset contains {len(df)} rows; the supported maximum is {self.max_rows}. '
                'Import was rejected to avoid analyzing an incomplete population.'
            )

        df = self._map_columns(df, explicit_mapping=column_mapping, mapping_methods=mapping_methods)
        self._validate_required_columns(df)
        df = self._validate_data_quality(df)
        logger.info('Successfully loaded %s rows', len(df))
        return df

    def _load_sqlite(self, file_path: str, table_name: Optional[str] = None) -> pd.DataFrame:
        with sqlite3.connect(file_path) as conn:
            if table_name is None:
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
                if not tables:
                    raise DataValidationError('No tables found in SQLite database')
                table_name = tables[0][0]
            # table_name comes from SQLite metadata or a caller-controlled parameter;
            # quote identifiers rather than interpolating executable SQL fragments.
            safe_name = table_name.replace('"', '""')
            return pd.read_sql_query(f'SELECT * FROM "{safe_name}"', conn)

    def _canonical_match(self, normalized: str) -> tuple[Optional[str], str, float]:
        all_fields = GOLDEN_SCHEMA['required'] + GOLDEN_SCHEMA['optional']
        for canonical in all_fields:
            if normalized == canonical.lower():
                return canonical, 'exact', 1.0
        for key, aliases in COLUMN_ALIASES.items():
            if normalized == key or normalized in aliases:
                canonical = next((field for field in all_fields if field.lower() == key), None)
                return canonical, 'alias', 0.96
        return None, 'unmapped', 0.0

    def _canonical_from_alias(self, normalized: str) -> Optional[str]:
        return self._canonical_match(normalized)[0]

    def _mapping_detail(
        self,
        source: str,
        target: Optional[str],
        method: str,
        confidence: float,
    ) -> None:
        status = 'mapped'
        if target is None:
            status = 'needs_review' if method == 'ambiguous' else 'unmapped'
        elif method in {'similarity', 'llm'}:
            status = 'needs_review'
        self.mapping_details[source] = {
            'source': source,
            'target': target,
            'method': method,
            'confidence': max(0.0, min(1.0, float(confidence))),
            'required': target in GOLDEN_SCHEMA['required'] if target else False,
            'status': status,
        }

    def _fuzzy_match_detail(self, column: str) -> tuple[Optional[str], str, float]:
        normalized = column.strip().lower().replace(' ', '_').replace('-', '_')
        explicit, method, confidence = self._canonical_match(normalized)
        if explicit:
            return explicit, method, confidence
        # Critical identity/pay/outcome fields are never inferred from weak spelling similarity.
        noncritical = [
            f for f in GOLDEN_SCHEMA['required'] + GOLDEN_SCHEMA['optional']
            if f not in CRITICAL_MAPPING_FIELDS
        ]
        candidates = {f.lower(): f for f in noncritical}
        matches = get_close_matches(normalized, list(candidates), n=2, cutoff=0.88)
        if len(matches) == 1:
            mapped = candidates[matches[0]]
            self.validation_warnings.append(
                f"Column '{column}' was conservatively mapped to '{mapped}' by name similarity"
            )
            return mapped, 'similarity', 0.88
        if matches:
            self.validation_warnings.append(
                f"Column '{column}' was not mapped because the match was ambiguous"
            )
            return None, 'ambiguous', 0.0
        return None, 'unmapped', 0.0

    def _fuzzy_match_column(self, column: str) -> Optional[str]:
        return self._fuzzy_match_detail(column)[0]

    def _map_columns(
        self,
        df: pd.DataFrame,
        *,
        explicit_mapping: Optional[Mapping[str, Optional[str]]] = None,
        mapping_methods: Optional[Mapping[str, str]] = None,
    ) -> pd.DataFrame:
        source_columns = [str(column) for column in df.columns]
        if len(source_columns) != len(set(source_columns)):
            raise DataValidationError('The file contains duplicate column names. Rename them before importing.')

        rename_map: dict[str, str] = {}
        mapped_targets: set[str] = set()

        if explicit_mapping is not None:
            unknown_sources = sorted(set(str(key) for key in explicit_mapping) - set(source_columns))
            if unknown_sources:
                raise DataValidationError(
                    'The mapping refers to columns that are not in this file: ' + ', '.join(unknown_sources)
                )
            allowed_targets = set(GOLDEN_SCHEMA['required'] + GOLDEN_SCHEMA['optional'])
            for col in source_columns:
                requested = explicit_mapping.get(col)
                # A canonical column omitted from a partial mapping remains safe as-is.
                if col not in explicit_mapping and col in allowed_targets:
                    requested = col
                if requested is None or str(requested).strip() == '':
                    self._mapping_detail(col, None, 'unmapped', 0.0)
                    continue
                target = str(requested).strip()
                if target not in allowed_targets:
                    raise DataValidationError(
                        f"Column '{col}' cannot be mapped to '{target}'. Choose a PeopleOS field from the import review."
                    )
                if target in mapped_targets:
                    raise DataValidationError(f"Multiple columns map to '{target}'; provide one unambiguous source column")
                rename_map[col] = target
                mapped_targets.add(target)
                method = str((mapping_methods or {}).get(col, 'user_confirmed'))
                confidence = 1.0 if method in {'user_confirmed', 'exact'} else 0.9
                self.column_mapping[col] = target
                self._mapping_detail(col, target, method, confidence)
            if self.column_mapping:
                logger.info('Column mapping: %s', sanitize_for_logging(self.column_mapping))
            return df.rename(columns=rename_map)

        for col in df.columns:
            source = str(col)
            normalized = source.strip().lower().replace(' ', '_').replace('-', '_')
            mapped, method, confidence = self._canonical_match(normalized)
            if mapped and mapped in mapped_targets:
                raise DataValidationError(f"Multiple columns map to '{mapped}'; provide one unambiguous source column")
            if mapped and mapped not in mapped_targets:
                rename_map[col] = mapped
                self.column_mapping[source] = mapped
                mapped_targets.add(mapped)
                self._mapping_detail(source, mapped, method, confidence)
        for col in df.columns:
            if col in rename_map:
                continue
            source = str(col)
            mapped, method, confidence = self._fuzzy_match_detail(source)
            if mapped and mapped not in mapped_targets:
                rename_map[col] = mapped
                self.column_mapping[source] = mapped
                mapped_targets.add(mapped)
                self._mapping_detail(source, mapped, method, confidence)
            else:
                self._mapping_detail(source, mapped, method, confidence)
        if self.column_mapping:
            logger.info('Column mapping: %s', sanitize_for_logging(self.column_mapping))
        return df.rename(columns=rename_map)

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        missing = [col for col in GOLDEN_SCHEMA['required'] if col not in df.columns]
        if missing:
            raise DataValidationError(get_error_message('missing_columns', columns=', '.join(missing)))
        if 'Attrition' not in df.columns:
            self.features_enabled['predictive'] = False
            self.validation_warnings.append('Attrition column missing - predictive analytics disabled')
        if 'PerformanceText' not in df.columns:
            self.features_enabled['nlp'] = False
            self.validation_warnings.append('PerformanceText column missing - NLP features disabled')

    def _validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        for column in ('EmployeeID', 'ManagerID'):
            if column in frame:
                frame[column] = frame[column].astype('string').str.strip().replace('', pd.NA)
        if 'EmployeeID' not in frame or frame['EmployeeID'].isna().any():
            raise DataValidationError('Every workforce row requires a nonempty EmployeeID')
        if 'SnapshotDate' not in frame.columns:
            duplicates = int(frame['EmployeeID'].duplicated().sum())
            if duplicates:
                raise DataValidationError(get_error_message('duplicate_ids', count=duplicates))
        else:
            parsed = pd.to_datetime(frame['SnapshotDate'], errors='coerce', utc=True)
            invalid = int(parsed.isna().sum())
            if invalid:
                raise DataValidationError(
                    f'{invalid} snapshot row(s) have a missing or invalid SnapshotDate; '
                    'correct the dates before importing so the current employee state can be determined'
                )
            duplicate_snapshots = int(frame.assign(_snapshot=parsed).duplicated(['EmployeeID', '_snapshot']).sum())
            if duplicate_snapshots:
                raise DataValidationError(f'Duplicate EmployeeID + SnapshotDate rows found: {duplicate_snapshots}')
            frame['SnapshotDate'] = parsed

        for col in list(frame.columns):
            null_ratio = frame[col].isna().mean()
            if null_ratio > 0.9 and col not in set(GOLDEN_SCHEMA['required'] + GOLDEN_SCHEMA['optional']):
                self.validation_warnings.append(f"Column '{col}' excluded (>90% null)")
                frame = frame.drop(columns=[col])

        self._validate_pay_basis(frame)
        original = frame
        frame = normalize_measurements(frame)
        for col in frame:
            if not is_numeric_measurement(col):
                continue
            invalid = frame[col].isna() | ~np.isfinite(frame[col])
            if col in ('Salary', 'StartingSalary', 'Tenure', 'Age'):
                invalid |= frame[col] < 0
            supplied = original[col].notna() & original[col].astype('string').str.strip().ne('')
            count = int((invalid & supplied).sum())
            frame.loc[invalid, col] = float('nan')
            if count:
                self.validation_warnings.append(
                    f'{count} invalid {col} measurement(s) marked missing; employee rows preserved'
                )

        if 'Attrition' in frame.columns:
            normalized = normalize_attrition(frame['Attrition'])
            unknown_count = int((normalized.isna() & frame['Attrition'].notna()).sum())
            frame['Attrition'] = normalized
            if unknown_count:
                self.features_enabled['predictive'] = False
                self.validation_warnings.append(f'{unknown_count} Attrition value(s) could not be normalized; predictive training disabled')
            elif normalized.dropna().nunique() < 2:
                self.features_enabled['predictive'] = False
                self.validation_warnings.append('Attrition contains fewer than two observed classes; predictive training disabled')

        if len(frame) < self.min_rows:
            raise DataValidationError(get_error_message('insufficient_data', count=len(frame)))
        return frame

    def _validate_pay_basis(self, frame: pd.DataFrame) -> None:
        """Accept annual comparable salary only; never guess conversion factors or FX."""
        basis_columns = [col for col in ('PayFrequency', 'PayPeriod') if col in frame]
        annual = {'annual', 'annually', 'year', 'yearly', 'per year', 'per annum', 'annualized', 'annualised'}
        for col in basis_columns:
            values = frame[col].astype('string').str.strip().str.lower()
            if not values.isin(annual).all():
                raise DataValidationError(
                    f'{col} must explicitly declare annual salary for every row. '
                    'Convert all Salary and StartingSalary amounts to an annual basis before importing; '
                    'monthly, hourly, mixed, missing and unknown pay periods cannot be compared safely'
                )
        if not basis_columns:
            self.validation_warnings.append(
                'Pay basis not supplied: Salary and StartingSalary are interpreted as annual amounts; '
                'confirm or supply PayPeriod=annual before using pay comparisons'
            )
        if 'Currency' in frame:
            values = frame['Currency'].astype('string').str.strip().str.upper()
            if values.isna().any() or not values.str.fullmatch('[A-Z]{3}').all() or values.nunique() != 1:
                raise DataValidationError(
                    'Currency must contain one shared three-letter currency code for every row. '
                    'Convert salaries to one reporting currency before importing; automatic FX conversion is unsupported'
                )
            frame['Currency'] = values
        else:
            self.validation_warnings.append(
                'Currency not supplied: pay comparisons assume one shared currency; '
                'confirm or supply Currency before interpreting compensation results'
            )

    def get_column_mapping_report(self) -> dict:
        return {
            'mappings': self.column_mapping,
            'details': list(self.mapping_details.values()),
            'warnings': self.validation_warnings,
            'features_enabled': self.features_enabled,
        }

    def load_from_database(self) -> Optional[pd.DataFrame]:
        if not self.config.get('persistence', {}).get('enabled', True):
            return None
        try:
            from src.database import get_database
            db = get_database()
            if not db.has_data():
                return None
            df = db.get_all_employees()
            if df.empty:
                return None
            self.features_enabled = {
                'predictive': 'Attrition' in df.columns and normalize_attrition(df['Attrition']).dropna().nunique() == 2,
                'nlp': 'PerformanceText' in df.columns and df['PerformanceText'].notna().any(),
            }
            if 'Attrition' in df.columns:
                df['Attrition'] = normalize_attrition(df['Attrition'])
            return df
        except Exception as exc:
            logger.error('Failed to load from database: %s', exc)
            return None

    def load_and_merge(self, file_path: str, file_name: str = 'upload', table_name: Optional[str] = None) -> Dict[str, Any]:
        df = self.load(file_path, table_name)
        persistence_enabled = self.config.get('persistence', {}).get('enabled', True)
        # Legacy persistence is one-row-per-employee and cannot safely preserve a
        # snapshot history. Fail closed by analysing the validated upload directly.
        if 'SnapshotDate' in df.columns:
            self.validation_warnings.append('Snapshot history is analysed directly; legacy employee persistence is bypassed to preserve temporal rows')
            return {'df': df, 'merge_result': None, 'report': self.get_column_mapping_report()}
        if not persistence_enabled:
            return {'df': df, 'merge_result': None, 'report': self.get_column_mapping_report()}
        try:
            from src.merge_engine import get_merge_engine
            merge_result = get_merge_engine().execute_merge(df, file_name)
            from src.database import get_database
            merged_df = get_database().get_all_employees()
            self.features_enabled['predictive'] = 'Attrition' in merged_df.columns and normalize_attrition(merged_df['Attrition']).dropna().nunique() == 2
            self.features_enabled['nlp'] = 'PerformanceText' in merged_df.columns and merged_df['PerformanceText'].notna().any()
            if 'Attrition' in merged_df.columns:
                merged_df['Attrition'] = normalize_attrition(merged_df['Attrition'])
            return {'df': merged_df, 'merge_result': merge_result, 'report': self.get_column_mapping_report()}
        except Exception as exc:
            logger.error('Merge failed: %s', exc)
            return {'df': df, 'merge_result': None, 'report': self.get_column_mapping_report()}

    def is_persistence_enabled(self) -> bool:
        return self.config.get('persistence', {}).get('enabled', True)

    def get_database_stats(self) -> Dict[str, Any]:
        if not self.is_persistence_enabled():
            return {'enabled': False}
        try:
            from src.database import get_database
            db = get_database()
            return {'enabled': True, 'has_data': db.has_data(), 'employee_count': db.get_employee_count(), 'recent_uploads': db.get_upload_history(limit=5)}
        except Exception as exc:
            logger.error('Failed to get database stats: %s', exc)
            return {'enabled': True, 'error': str(exc)}
