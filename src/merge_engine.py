"""Governed merge preview and persistence coordination for PeopleOS.

MergeEngine is an operational HR workflow. It may identify which employee record
changed, but preview details are restricted to database-supported fields and raw
values for sensitive fields are redacted. Missing incoming values never erase a
known persisted value unless PeopleOS gains an explicit clear-field contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from numbers import Real
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.database import Database, get_database
from src.logger import get_logger
from src.population import resolve_current_population

logger = get_logger('merge_engine')

GOVERNED_PERSISTED_FIELDS = (
    'Dept', 'Tenure', 'Salary', 'LastRating', 'Age', 'Attrition', 'Gender',
    'JobTitle', 'Location', 'Country', 'HireDate', 'ManagerID',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'PromotionCount',
    'InterviewScore', 'AssessmentScore', 'HireSource',
    'InterviewScore_Technical', 'InterviewScore_Cultural',
    'InterviewScore_Curiosity', 'InterviewScore_Communication',
    'InterviewScore_Leadership', 'PerformanceText', 'RatingHistory',
    'PromotionDate', 'StartingSalary', 'eNPS_Score', 'Pulse_Score',
    'ManagerSatisfaction', 'WorkLifeBalance', 'CareerGrowthSatisfaction',
    'Onboarding_30d', 'Onboarding_60d', 'Onboarding_90d', 'JobLevel',
    'CompaRatio', 'PriorExperienceYears', 'ManagerChangeCount',
)

NUMERIC_FIELDS = {
    'Tenure', 'Salary', 'LastRating', 'Age', 'Attrition', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'PromotionCount', 'InterviewScore',
    'AssessmentScore', 'InterviewScore_Technical', 'InterviewScore_Cultural',
    'InterviewScore_Curiosity', 'InterviewScore_Communication',
    'InterviewScore_Leadership', 'StartingSalary', 'eNPS_Score', 'Pulse_Score',
    'ManagerSatisfaction', 'WorkLifeBalance', 'CareerGrowthSatisfaction',
    'Onboarding_30d', 'Onboarding_60d', 'Onboarding_90d', 'JobLevel',
    'CompaRatio', 'PriorExperienceYears', 'ManagerChangeCount',
}

REDACTED_PREVIEW_FIELDS = {
    'Gender', 'HireDate', 'ManagerID', 'InterviewScore', 'AssessmentScore',
    'InterviewScore_Technical', 'InterviewScore_Cultural',
    'InterviewScore_Curiosity', 'InterviewScore_Communication',
    'InterviewScore_Leadership', 'PerformanceText', 'RatingHistory',
    'PromotionDate', 'eNPS_Score', 'Pulse_Score', 'ManagerSatisfaction',
    'WorkLifeBalance', 'CareerGrowthSatisfaction', 'Onboarding_30d',
    'Onboarding_60d', 'Onboarding_90d',
}


def _json_safe_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, 'item') and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, Real):
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        return numeric
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


@dataclass
class FieldChange:
    field_name: str
    old_value: Any
    new_value: Any
    values_redacted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'field': self.field_name,
            'old': None if self.values_redacted else _json_safe_scalar(self.old_value),
            'new': None if self.values_redacted else _json_safe_scalar(self.new_value),
            'values_redacted': bool(self.values_redacted),
        }


@dataclass
class EmployeeChange:
    employee_id: str
    change_type: str
    changes: List[FieldChange] = field(default_factory=list)
    dept: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'employee_id': str(self.employee_id),
            'change_type': self.change_type,
            'dept': _json_safe_scalar(self.dept),
            'changes': [change.to_dict() for change in self.changes],
        }


@dataclass
class MergeResult:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    total: int = 0
    employee_changes: List[EmployeeChange] = field(default_factory=list)
    file_name: str = ''
    timestamp: str = ''

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'added': int(self.added),
            'updated': int(self.updated),
            'unchanged': int(self.unchanged),
            'skipped': int(self.skipped),
            'total': int(self.total),
            'file_name': str(self.file_name),
            'timestamp': str(self.timestamp),
        }

    def get_summary_text(self) -> str:
        parts: list[str] = []
        if self.added > 0:
            parts.append(f"{self.added} new employee{'s' if self.added != 1 else ''} added")
        if self.updated > 0:
            parts.append(f"{self.updated} employee{'s' if self.updated != 1 else ''} updated")
        if self.unchanged > 0:
            parts.append(f'{self.unchanged} unchanged')
        if self.skipped > 0:
            parts.append(f'{self.skipped} skipped (invalid or superseded data)')
        return ', '.join(parts) if parts else 'No changes made'


class MergeEngine:
    """Preview and execute governed employee-record merges."""

    COMPARE_FIELDS = [
        ('Dept', 'dept'), ('Tenure', 'tenure'), ('Salary', 'salary'),
        ('LastRating', 'last_rating'), ('Age', 'age'), ('Attrition', 'attrition'),
    ]

    def __init__(self, database: Optional[Database] = None):
        self.db = database or get_database()

    @staticmethod
    def _prepare_merge_frame(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or 'EmployeeID' not in df:
            return pd.DataFrame(columns=df.columns if isinstance(df, pd.DataFrame) else [])
        frame = df.copy(deep=True)
        valid = frame['EmployeeID'].notna() & frame['EmployeeID'].astype(str).str.strip().ne('')
        frame = frame.loc[valid].copy()
        frame['EmployeeID'] = frame['EmployeeID'].astype(str).str.strip()
        for column in NUMERIC_FIELDS.intersection(frame.columns):
            numeric = pd.to_numeric(frame[column], errors='coerce')
            frame[column] = numeric.where(np.isfinite(numeric), np.nan)
        frame, _ = resolve_current_population(frame)
        return frame

    @staticmethod
    def _incoming_has_value(new: pd.Series, field_name: str) -> bool:
        if field_name not in new.index:
            return False
        value = new.get(field_name)
        try:
            return not bool(pd.isna(value))
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _values_equal(old_value: Any, new_value: Any) -> bool:
        try:
            old_null = bool(pd.isna(old_value))
        except (TypeError, ValueError):
            old_null = False
        try:
            new_null = bool(pd.isna(new_value))
        except (TypeError, ValueError):
            new_null = False
        if old_null or new_null:
            return old_null and new_null
        if isinstance(old_value, Real) and isinstance(new_value, Real):
            old_numeric, new_numeric = float(old_value), float(new_value)
            if not np.isfinite(old_numeric) or not np.isfinite(new_numeric):
                return False
            return abs(old_numeric - new_numeric) < 0.001
        if isinstance(old_value, pd.Timestamp) or isinstance(new_value, pd.Timestamp):
            try:
                return pd.Timestamp(old_value) == pd.Timestamp(new_value)
            except Exception:
                return False
        return str(old_value) == str(new_value)

    def _detect_changes(self, existing: pd.Series, new: pd.Series) -> List[FieldChange]:
        changes: list[FieldChange] = []
        for field_name in GOVERNED_PERSISTED_FIELDS:
            if not self._incoming_has_value(new, field_name):
                continue
            old_value = existing.get(field_name)
            new_value = new.get(field_name)
            if self._values_equal(old_value, new_value):
                continue
            redacted = field_name in REDACTED_PREVIEW_FIELDS
            changes.append(FieldChange(
                field_name=field_name,
                old_value=None if redacted else old_value,
                new_value=None if redacted else new_value,
                values_redacted=redacted,
            ))
        return changes

    def preview_merge(self, df: pd.DataFrame) -> MergeResult:
        source_total = len(df) if isinstance(df, pd.DataFrame) else 0
        result = MergeResult(total=source_total, timestamp=datetime.now().isoformat())
        frame = self._prepare_merge_frame(df)
        result.skipped = source_total - len(frame)

        existing = self.db.get_all_employees()
        existing_map = {
            str(row['EmployeeID']): row for _, row in existing.iterrows()
        } if not existing.empty and 'EmployeeID' in existing else {}

        for _, row in frame.iterrows():
            employee_id = str(row['EmployeeID'])
            dept = row.get('Dept', 'Unknown')
            if employee_id not in existing_map:
                result.added += 1
                result.employee_changes.append(EmployeeChange(employee_id, 'added', dept=dept))
                continue
            changes = self._detect_changes(existing_map[employee_id], row)
            if changes:
                result.updated += 1
                result.employee_changes.append(EmployeeChange(employee_id, 'updated', changes, dept))
            else:
                result.unchanged += 1
                result.employee_changes.append(EmployeeChange(employee_id, 'unchanged', dept=dept))
        return result

    @staticmethod
    def _coalesce_update_row(row: pd.Series, existing: pd.Series) -> Dict[str, Any]:
        output: Dict[str, Any] = {'EmployeeID': str(row['EmployeeID'])}
        if 'SnapshotDate' in row.index and pd.notna(row.get('SnapshotDate')):
            output['SnapshotDate'] = row.get('SnapshotDate')
        for field_name in GOVERNED_PERSISTED_FIELDS:
            incoming_present = field_name in row.index
            incoming_value = row.get(field_name) if incoming_present else None
            try:
                incoming_missing = (not incoming_present) or bool(pd.isna(incoming_value))
            except (TypeError, ValueError):
                incoming_missing = not incoming_present
            output[field_name] = existing.get(field_name) if incoming_missing else incoming_value
        return output

    def _write_frame(self, frame: pd.DataFrame, changed_ids: set[str], existing: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or not changed_ids:
            return pd.DataFrame(columns=['EmployeeID'])
        existing_map = {
            str(row['EmployeeID']): row for _, row in existing.iterrows()
        } if not existing.empty and 'EmployeeID' in existing else {}
        rows: list[Dict[str, Any]] = []
        for _, row in frame.iterrows():
            employee_id = str(row['EmployeeID'])
            if employee_id not in changed_ids:
                continue
            if employee_id in existing_map:
                rows.append(self._coalesce_update_row(row, existing_map[employee_id]))
            else:
                payload: Dict[str, Any] = {'EmployeeID': employee_id}
                if 'SnapshotDate' in row.index and pd.notna(row.get('SnapshotDate')):
                    payload['SnapshotDate'] = row.get('SnapshotDate')
                for field_name in GOVERNED_PERSISTED_FIELDS:
                    if field_name in row.index:
                        payload[field_name] = row.get(field_name)
                rows.append(payload)
        return pd.DataFrame(rows)

    def execute_merge(self, df: pd.DataFrame, file_name: str = 'upload') -> MergeResult:
        result = self.preview_merge(df)
        result.file_name = str(file_name)
        frame = self._prepare_merge_frame(df)
        existing = self.db.get_all_employees()
        changed_ids = {
            item.employee_id for item in result.employee_changes
            if item.change_type in {'added', 'updated'}
        }
        writes = self._write_frame(frame, changed_ids, existing)
        db_result = self.db.upsert_employees(writes, str(file_name))
        result.added = int(db_result.get('added', 0))
        result.updated = int(db_result.get('updated', 0))
        result.skipped += int(db_result.get('skipped', 0))
        logger.info('Merge completed: %s', result.get_summary_text())
        return result

    @staticmethod
    def _validated_threshold(value: Any, name: str) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f'{name} must be a finite non-negative number')
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0:
            raise ValueError(f'{name} must be a finite non-negative number')
        return numeric

    def get_significant_changes(
        self,
        result: MergeResult,
        salary_threshold: float = 0.10,
        rating_threshold: float = 0.5,
    ) -> List[EmployeeChange]:
        salary_threshold = self._validated_threshold(salary_threshold, 'salary_threshold')
        rating_threshold = self._validated_threshold(rating_threshold, 'rating_threshold')
        significant: list[EmployeeChange] = []
        for employee_change in result.employee_changes:
            if employee_change.change_type != 'updated':
                continue
            is_significant = False
            for change in employee_change.changes:
                if change.field_name == 'Salary':
                    try:
                        old_salary = float(change.old_value)
                        new_salary = float(change.new_value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(old_salary) and np.isfinite(new_salary) and old_salary > 0:
                        if abs(new_salary - old_salary) / old_salary >= salary_threshold:
                            is_significant = True
                elif change.field_name == 'LastRating':
                    try:
                        old_rating = float(change.old_value)
                        new_rating = float(change.new_value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(old_rating) and np.isfinite(new_rating):
                        if abs(new_rating - old_rating) >= rating_threshold:
                            is_significant = True
                elif change.field_name in {'Attrition', 'Dept'}:
                    is_significant = True
            if is_significant:
                significant.append(employee_change)
        return significant


def get_merge_engine() -> MergeEngine:
    return MergeEngine()
