"""
Merge Engine module for PeopleOS.

Handles merging of uploaded data with existing database records,
providing summaries of what was added, updated, or changed.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd

from src.database import get_database, Database
from src.logger import get_logger

logger = get_logger('merge_engine')


@dataclass
class FieldChange:
    """Represents a change to a single field."""
    field_name: str
    old_value: Any
    new_value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            'field': self.field_name,
            'old': self.old_value,
            'new': self.new_value
        }


@dataclass
class EmployeeChange:
    """Represents changes to a single employee."""
    employee_id: str
    change_type: str  # 'added', 'updated', 'unchanged'
    changes: List[FieldChange] = field(default_factory=list)
    dept: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'employee_id': self.employee_id,
            'change_type': self.change_type,
            'dept': self.dept,
            'changes': [c.to_dict() for c in self.changes]
        }


@dataclass
class MergeResult:
    """Result of a merge operation."""
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    total: int = 0
    employee_changes: List[EmployeeChange] = field(default_factory=list)
    file_name: str = ""
    timestamp: str = ""

    def to_summary_dict(self) -> Dict[str, Any]:
        """Get a summary dictionary for display."""
        return {
            'added': self.added,
            'updated': self.updated,
            'unchanged': self.unchanged,
            'skipped': self.skipped,
            'total': self.total,
            'file_name': self.file_name,
            'timestamp': self.timestamp
        }

    def get_summary_text(self) -> str:
        """Get a human-readable summary."""
        parts = []
        if self.added > 0:
            parts.append(f"{self.added} new employee{'s' if self.added != 1 else ''} added")
        if self.updated > 0:
            parts.append(f"{self.updated} employee{'s' if self.updated != 1 else ''} updated")
        if self.unchanged > 0:
            parts.append(f"{self.unchanged} unchanged")
        if self.skipped > 0:
            parts.append(f"{self.skipped} skipped (invalid data)")

        if not parts:
            return "No changes made"

        return ", ".join(parts)


class MergeEngine:
    """
    Handles merging of uploaded data with existing database records.

    Uses auto-update strategy: new data automatically overwrites
    existing data for the same EmployeeID.
    """

    # Fields to compare for change detection
    COMPARE_FIELDS = [
        ('Dept', 'dept'),
        ('Tenure', 'tenure'),
        ('Salary', 'salary'),
        ('LastRating', 'last_rating'),
        ('Age', 'age'),
        ('Attrition', 'attrition')
    ]

    def __init__(self, database: Optional[Database] = None):
        """
        Initialize merge engine.

        Args:
            database: Optional database instance. If not provided,
                     uses the singleton instance.
        """
        self.db = database or get_database()

    def preview_merge(self, df: pd.DataFrame) -> MergeResult:
        """
        Preview what changes would occur without committing.

        Args:
            df: DataFrame with new employee data.

        Returns:
            MergeResult with detailed change information.
        """
        result = MergeResult(
            total=len(df),
            timestamp=datetime.now().isoformat()
        )

        # Get existing employees
        existing_df = self.db.get_all_employees()
        existing_ids = set(existing_df['EmployeeID'].astype(str).tolist()) if not existing_df.empty else set()

        for _, row in df.iterrows():
            raw_emp_id = row.get('EmployeeID')
            # Handle None, NaN, and empty string
            if pd.isna(raw_emp_id) or raw_emp_id == '' or raw_emp_id is None:
                result.skipped += 1
                continue
            emp_id = str(raw_emp_id)

            dept = row.get('Dept', 'Unknown')

            if emp_id in existing_ids:
                # Check what would change
                existing_row = existing_df[existing_df['EmployeeID'].astype(str) == emp_id].iloc[0]
                changes = self._detect_changes(existing_row, row)

                if changes:
                    result.updated += 1
                    result.employee_changes.append(EmployeeChange(
                        employee_id=emp_id,
                        change_type='updated',
                        changes=changes,
                        dept=dept
                    ))
                else:
                    result.unchanged += 1
                    result.employee_changes.append(EmployeeChange(
                        employee_id=emp_id,
                        change_type='unchanged',
                        dept=dept
                    ))
            else:
                result.added += 1
                result.employee_changes.append(EmployeeChange(
                    employee_id=emp_id,
                    change_type='added',
                    dept=dept
                ))

        return result

    def _detect_changes(self, existing: pd.Series, new: pd.Series) -> List[FieldChange]:
        """
        Detect changes between existing and new data for an employee.

        Args:
            existing: Existing employee row.
            new: New employee row.

        Returns:
            List of field changes.
        """
        changes = []

        for df_field, db_field in self.COMPARE_FIELDS:
            old_val = existing.get(df_field)
            new_val = new.get(df_field)

            # Handle NaN comparisons
            old_is_null = pd.isna(old_val)
            new_is_null = pd.isna(new_val)

            if old_is_null and new_is_null:
                continue
            elif old_is_null != new_is_null:
                changes.append(FieldChange(
                    field_name=df_field,
                    old_value=None if old_is_null else old_val,
                    new_value=None if new_is_null else new_val
                ))
            elif not self._values_equal(old_val, new_val):
                changes.append(FieldChange(
                    field_name=df_field,
                    old_value=old_val,
                    new_value=new_val
                ))

        return changes

    def _values_equal(self, old_val: Any, new_val: Any) -> bool:
        """
        Compare two values, handling numeric precision.

        Args:
            old_val: Old value.
            new_val: New value.

        Returns:
            True if values are effectively equal.
        """
        # Handle numeric comparison with tolerance
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            return abs(float(old_val) - float(new_val)) < 0.001

        # String comparison
        return str(old_val) == str(new_val)

    def execute_merge(self, df: pd.DataFrame, file_name: str = "upload") -> MergeResult:
        """
        Execute the merge, persisting changes to database.

        Args:
            df: DataFrame with new employee data.
            file_name: Name of the uploaded file.

        Returns:
            MergeResult with details of what changed.
        """
        # First preview to get detailed changes
        result = self.preview_merge(df)
        result.file_name = file_name

        # Execute the actual upsert
        db_result = self.db.upsert_employees(df, file_name)

        # Update counts from actual database operation
        result.added = db_result['added']
        result.updated = db_result['updated']
        result.skipped = db_result['skipped']

        logger.info(f"Merge completed: {result.get_summary_text()}")

        return result

    def get_significant_changes(
        self,
        result: MergeResult,
        salary_threshold: float = 0.10,
        rating_threshold: float = 0.5
    ) -> List[EmployeeChange]:
        """
        Get list of significant changes that HR might want to review.

        Args:
            result: MergeResult from a merge operation.
            salary_threshold: Percentage change in salary to flag (default 10%).
            rating_threshold: Change in rating to flag (default 0.5).

        Returns:
            List of employee changes that are significant.
        """
        significant = []

        for emp_change in result.employee_changes:
            if emp_change.change_type != 'updated':
                continue

            is_significant = False
            for change in emp_change.changes:
                if change.field_name == 'Salary':
                    old_sal = float(change.old_value or 0)
                    new_sal = float(change.new_value or 0)
                    if old_sal > 0:
                        pct_change = abs(new_sal - old_sal) / old_sal
                        if pct_change >= salary_threshold:
                            is_significant = True

                elif change.field_name == 'LastRating':
                    old_rating = float(change.old_value or 0)
                    new_rating = float(change.new_value or 0)
                    if abs(new_rating - old_rating) >= rating_threshold:
                        is_significant = True

                elif change.field_name == 'Attrition':
                    # Any change in attrition status is significant
                    if change.old_value != change.new_value:
                        is_significant = True

                elif change.field_name == 'Dept':
                    # Department changes are significant
                    is_significant = True

            if is_significant:
                significant.append(emp_change)

        return significant


def get_merge_engine() -> MergeEngine:
    """
    Get a MergeEngine instance.

    Returns:
        MergeEngine instance.
    """
    return MergeEngine()
