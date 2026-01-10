"""
Database module for PeopleOS.

Provides SQLite-based persistent storage for employee data,
upload history, and historical snapshots.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

import pandas as pd

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('database')


class Database:
    """
    SQLite database manager for PeopleOS.

    Handles persistent storage of employee data with:
    - Auto-update on duplicate EmployeeIDs
    - Historical snapshots for trend analysis
    - Upload history tracking
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            db_path: Optional custom database path. If not provided,
                    uses path from config.yaml.
        """
        config = load_config()
        persistence_config = config.get('persistence', {})

        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = Path(persistence_config.get('database_path', 'data/peopleos.db'))

        self.keep_history = persistence_config.get('keep_history', True)
        self.history_retention_days = persistence_config.get('history_retention_days', 365)

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_schema()

        logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Main employees table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    employee_id TEXT PRIMARY KEY,
                    dept TEXT,
                    tenure REAL,
                    salary REAL,
                    last_rating REAL,
                    age INTEGER,
                    attrition INTEGER,
                    gender TEXT,
                    job_title TEXT,
                    location TEXT,
                    hire_date TEXT,
                    manager_id TEXT,
                    years_in_current_role REAL,
                    years_since_last_promotion REAL,
                    promotion_count INTEGER,
                    interview_score REAL,
                    assessment_score REAL,
                    hire_source TEXT,
                    interview_score_technical REAL,
                    interview_score_cultural REAL,
                    interview_score_curiosity REAL,
                    interview_score_communication REAL,
                    interview_score_leadership REAL,
                    performance_text TEXT,
                    rating_history TEXT,
                    promotion_date TEXT,
                    starting_salary REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 1
                )
            """)

            # Auto-migration: Add missing columns if table already exists
            new_columns = [
                ('gender', 'TEXT'),
                ('job_title', 'TEXT'),
                ('location', 'TEXT'),
                ('hire_date', 'TEXT'),
                ('manager_id', 'TEXT'),
                ('years_in_current_role', 'REAL'),
                ('years_since_last_promotion', 'REAL'),
                ('promotion_count', 'INTEGER'),
                ('interview_score', 'REAL'),
                ('assessment_score', 'REAL'),
                ('hire_source', 'TEXT'),
                ('interview_score_technical', 'REAL'),
                ('interview_score_cultural', 'REAL'),
                ('interview_score_curiosity', 'REAL'),
                ('interview_score_communication', 'REAL'),
                ('interview_score_leadership', 'REAL')
            ]

            cursor.execute("PRAGMA table_info(employees)")
            existing_columns = [row[1] for row in cursor.fetchall()]

            for col_name, col_type in new_columns:
                if col_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE employees ADD COLUMN {col_name} {col_type}")
                        logger.info(f"Added column {col_name} to employees table")
                    except Exception as e:
                        logger.error(f"Failed to add column {col_name}: {e}")

            # Upload history for audit trail
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS upload_history (
                    upload_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_name TEXT,
                    rows_added INTEGER DEFAULT 0,
                    rows_updated INTEGER DEFAULT 0,
                    rows_skipped INTEGER DEFAULT 0,
                    total_rows INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'completed',
                    error_message TEXT
                )
            """)

            # Historical snapshots for trend analysis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS employee_snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id TEXT,
                    snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    dept TEXT,
                    tenure REAL,
                    salary REAL,
                    last_rating REAL,
                    age INTEGER,
                    attrition INTEGER,
                    hire_source TEXT,
                    interview_score REAL,
                    upload_id INTEGER,
                    FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
                    FOREIGN KEY (upload_id) REFERENCES upload_history(upload_id)
                )
            """)

            # Auto-migration for snapshots too
            snapshot_new_columns = [
                ('hire_source', 'TEXT'),
                ('interview_score', 'REAL')
            ]
            cursor.execute("PRAGMA table_info(employee_snapshots)")
            existing_snapshot_cols = [row[1] for row in cursor.fetchall()]
            for col_name, col_type in snapshot_new_columns:
                if col_name not in existing_snapshot_cols:
                    try:
                        cursor.execute(f"ALTER TABLE employee_snapshots ADD COLUMN {col_name} {col_type}")
                    except Exception:
                        pass

            # Index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_employee
                ON employee_snapshots(employee_id, snapshot_date)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_employees_dept
                ON employees(dept)
            """)

            logger.info("Database schema initialized")

    def get_employee_count(self) -> int:
        """Get total count of active employees."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM employees WHERE is_active = 1")
            return cursor.fetchone()[0]

    def get_all_employees(self) -> pd.DataFrame:
        """
        Retrieve all active employees as a DataFrame.

        Returns:
            DataFrame with all employee data in the expected format.
        """
        with self._get_connection() as conn:
            query = """
                SELECT
                    employee_id as EmployeeID,
                    dept as Dept,
                    tenure as Tenure,
                    salary as Salary,
                    last_rating as LastRating,
                    age as Age,
                    attrition as Attrition,
                    gender as Gender,
                    job_title as JobTitle,
                    location as Location,
                    hire_date as HireDate,
                    manager_id as ManagerID,
                    years_in_current_role as YearsInCurrentRole,
                    years_since_last_promotion as YearsSinceLastPromotion,
                    promotion_count as PromotionCount,
                    interview_score as InterviewScore,
                    assessment_score as AssessmentScore,
                    hire_source as HireSource,
                    interview_score_technical as InterviewScore_Technical,
                    interview_score_cultural as InterviewScore_Cultural,
                    interview_score_curiosity as InterviewScore_Curiosity,
                    interview_score_communication as InterviewScore_Communication,
                    interview_score_leadership as InterviewScore_Leadership,
                    performance_text as PerformanceText,
                    rating_history as RatingHistory,
                    promotion_date as PromotionDate,
                    starting_salary as StartingSalary,
                    created_at,
                    updated_at
                FROM employees
                WHERE is_active = 1
                ORDER BY employee_id
            """
            df = pd.read_sql_query(query, conn)

            # Convert attrition to nullable integer type to preserve NULL values
            if 'Attrition' in df.columns:
                # Use nullable Int64 type to preserve NULL while keeping integer semantics
                df['Attrition'] = df['Attrition'].astype('Int64')

            return df

    def upsert_employees(
        self,
        df: pd.DataFrame,
        file_name: str = "manual_upload"
    ) -> Dict[str, int]:
        """
        Insert or update employees from a DataFrame.

        Uses auto-update strategy: new data overwrites existing for same EmployeeID.
        Creates historical snapshots before updating.

        Args:
            df: DataFrame with employee data (must have EmployeeID column).
            file_name: Name of the uploaded file for history tracking.

        Returns:
            Dictionary with counts: {'added': n, 'updated': n, 'skipped': n, 'total': n}
        """
        if df is None or df.empty:
            return {'added': 0, 'updated': 0, 'skipped': 0, 'total': 0}

        # Normalize column names to match schema
        column_map = {
            'EmployeeID': 'employee_id',
            'Dept': 'dept',
            'Tenure': 'tenure',
            'Salary': 'salary',
            'LastRating': 'last_rating',
            'Age': 'age',
            'Attrition': 'attrition',
            'Gender': 'gender',
            'JobTitle': 'job_title',
            'Location': 'location',
            'HireDate': 'hire_date',
            'ManagerID': 'manager_id',
            'YearsInCurrentRole': 'years_in_current_role',
            'YearsSinceLastPromotion': 'years_since_last_promotion',
            'PromotionCount': 'promotion_count',
            'InterviewScore': 'interview_score',
            'AssessmentScore': 'assessment_score',
            'HireSource': 'hire_source',
            'InterviewScore_Technical': 'interview_score_technical',
            'InterviewScore_Cultural': 'interview_score_cultural',
            'InterviewScore_Curiosity': 'interview_score_curiosity',
            'InterviewScore_Communication': 'interview_score_communication',
            'InterviewScore_Leadership': 'interview_score_leadership',
            'PerformanceText': 'performance_text',
            'RatingHistory': 'rating_history',
            'PromotionDate': 'promotion_date',
            'StartingSalary': 'starting_salary'
        }

        added = 0
        updated = 0
        skipped = 0

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create upload history record
            cursor.execute(
                "INSERT INTO upload_history (file_name, total_rows) VALUES (?, ?)",
                (file_name, len(df))
            )
            upload_id = cursor.lastrowid

            for _, row in df.iterrows():
                try:
                    raw_emp_id = row.get('EmployeeID')
                    # Handle None, NaN, and empty string
                    if pd.isna(raw_emp_id) or raw_emp_id == '' or raw_emp_id is None:
                        skipped += 1
                        continue
                    emp_id = str(raw_emp_id)

                    # Check if employee exists
                    cursor.execute(
                        "SELECT employee_id FROM employees WHERE employee_id = ?",
                        (emp_id,)
                    )
                    existing = cursor.fetchone()

                    data = {col_db: row.get(col_schema) for col_schema, col_db in column_map.items()}
                    # Specific type handling for Attrition
                    if pd.notna(row.get('Attrition')):
                        data['attrition'] = int(row.get('Attrition'))
                    else:
                        data['attrition'] = None

                    if existing:
                        # Create snapshot before updating (if history enabled)
                        if self.keep_history:
                            self._create_snapshot(cursor, emp_id, upload_id)

                        # Update existing employee
                        update_fields = [f"{col} = ?" for col in data.keys() if col != 'employee_id']
                        update_sql = f"""
                            UPDATE employees SET 
                                {", ".join(update_fields)},
                                updated_at = CURRENT_TIMESTAMP,
                                is_active = 1
                            WHERE employee_id = ?
                        """
                        update_params = [data[col] for col in data.keys() if col != 'employee_id'] + [emp_id]
                        
                        # Handle SQL conversion for pandas types if needed
                        update_params = [
                            v.item() if hasattr(v, 'item') and not isinstance(v, (str, bytes)) else v 
                            for v in update_params
                        ]
                        
                        cursor.execute(update_sql, update_params)
                        updated += 1
                    else:
                        # Insert new employee
                        cols = list(data.keys())
                        placeholders = ['?'] * len(cols)
                        insert_sql = f"""
                            INSERT INTO employees ({", ".join(cols)})
                            VALUES ({", ".join(placeholders)})
                        """
                        insert_params = [data[col] for col in cols]
                        
                        # Handle SQL conversion for pandas types if needed
                        insert_params = [
                            v.item() if hasattr(v, 'item') and not isinstance(v, (str, bytes)) else v 
                            for v in insert_params
                        ]
                        
                        cursor.execute(insert_sql, insert_params)
                        added += 1

                        # Create initial snapshot for new employee
                        if self.keep_history:
                            self._create_snapshot(cursor, emp_id, upload_id)

                except Exception as e:
                    logger.warning(f"Error processing employee {emp_id}: {e}")
                    skipped += 1

            # Update upload history
            cursor.execute("""
                UPDATE upload_history SET
                    rows_added = ?,
                    rows_updated = ?,
                    rows_skipped = ?,
                    status = 'completed'
                WHERE upload_id = ?
            """, (added, updated, skipped, upload_id))

        result = {
            'added': added,
            'updated': updated,
            'skipped': skipped,
            'total': added + updated
        }

        logger.info(f"Upsert completed: {result}")
        return result

    def _create_snapshot(self, cursor, employee_id: str, upload_id: int) -> None:
        """Create a historical snapshot of an employee's current state."""
        cursor.execute("""
            INSERT INTO employee_snapshots (
                employee_id, dept, tenure, salary, last_rating, age, attrition, 
                hire_source, interview_score, upload_id
            )
            SELECT
                employee_id, dept, tenure, salary, last_rating, age, attrition,
                hire_source, interview_score, ?
            FROM employees
            WHERE employee_id = ?
        """, (upload_id, employee_id))

    def get_employee_history(self, employee_id: str) -> pd.DataFrame:
        """
        Get historical snapshots for a specific employee.

        Args:
            employee_id: The employee's ID.

        Returns:
            DataFrame with historical data sorted by date.
        """
        with self._get_connection() as conn:
            query = """
                SELECT
                    snapshot_date,
                    dept as Dept,
                    tenure as Tenure,
                    salary as Salary,
                    last_rating as LastRating,
                    age as Age,
                    attrition as Attrition
                FROM employee_snapshots
                WHERE employee_id = ?
                ORDER BY snapshot_date ASC
            """
            return pd.read_sql_query(query, conn, params=(employee_id,))

    def get_upload_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent upload history.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of upload history records.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    upload_id,
                    upload_date,
                    file_name,
                    rows_added,
                    rows_updated,
                    rows_skipped,
                    total_rows,
                    status
                FROM upload_history
                ORDER BY upload_date DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_headcount_over_time(self, months: int = 12) -> pd.DataFrame:
        """
        Get headcount changes over time from snapshots.

        Args:
            months: Number of months to look back.

        Returns:
            DataFrame with date and headcount columns.
        """
        with self._get_connection() as conn:
            query = """
                SELECT
                    DATE(snapshot_date) as date,
                    COUNT(DISTINCT employee_id) as headcount
                FROM employee_snapshots
                WHERE snapshot_date >= DATE('now', '-' || ? || ' months')
                GROUP BY DATE(snapshot_date)
                ORDER BY date ASC
            """
            return pd.read_sql_query(query, conn, params=(months,))

    def get_salary_progression(self, employee_id: str) -> pd.DataFrame:
        """
        Get salary changes over time for an employee.

        Args:
            employee_id: The employee's ID.

        Returns:
            DataFrame with date and salary columns.
        """
        with self._get_connection() as conn:
            query = """
                SELECT
                    DATE(snapshot_date) as date,
                    salary
                FROM employee_snapshots
                WHERE employee_id = ?
                ORDER BY snapshot_date ASC
            """
            return pd.read_sql_query(query, conn, params=(employee_id,))

    def soft_delete_employee(self, employee_id: str) -> bool:
        """
        Mark an employee as inactive (soft delete).

        Args:
            employee_id: The employee's ID.

        Returns:
            True if successful.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE employees SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE employee_id = ?",
                (employee_id,)
            )
            return cursor.rowcount > 0

    def cleanup_old_snapshots(self) -> int:
        """
        Remove snapshots older than retention period.

        Returns:
            Number of snapshots deleted.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM employee_snapshots
                WHERE snapshot_date < DATE('now', '-' || ? || ' days')
            """, (self.history_retention_days,))

            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old snapshots")
            return deleted

    def has_data(self) -> bool:
        """Check if the database has any employee data."""
        return self.get_employee_count() > 0

    def clear_all_data(self) -> None:
        """Clear all data from the database (use with caution!)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM employee_snapshots")
            cursor.execute("DELETE FROM upload_history")
            cursor.execute("DELETE FROM employees")
            logger.warning("All database data cleared")


# Singleton instance
_database_instance: Optional[Database] = None


def get_database() -> Database:
    """
    Get the singleton database instance.

    Returns:
        Database instance.
    """
    global _database_instance
    if _database_instance is None:
        _database_instance = Database()
    return _database_instance


def reset_database_instance() -> None:
    """Reset the singleton instance (mainly for testing)."""
    global _database_instance
    _database_instance = None
