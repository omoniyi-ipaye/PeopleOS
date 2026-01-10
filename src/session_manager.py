"""
Session Manager module for PeopleOS.

Handles saving and loading of analysis sessions for persistence.
Sessions are stored locally in the sessions/ directory.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('session_manager')


class SessionManager:
    """
    Manages saving and loading of analysis sessions.

    Sessions include the uploaded data and analytics results.
    """

    def __init__(self, sessions_dir: Optional[str] = None):
        """
        Initialize the SessionManager with configuration.

        Args:
            sessions_dir: Optional custom directory for sessions (for testing).
        """
        self.config = load_config()
        self.sessions_config = self.config.get('sessions', {})

        if sessions_dir:
            self.sessions_dir = Path(sessions_dir)
        else:
            self.sessions_dir = Path(self.sessions_config.get('directory', 'sessions'))

        self.max_saved = self.sessions_config.get('max_saved', 10)

        # Ensure sessions directory exists
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure the sessions directory exists."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def save_session(
        self,
        session_name: str,
        raw_data: pd.DataFrame,
        analytics_data: dict,
        ml_data: dict,
        features_enabled: dict
    ) -> str:
        """
        Save an analysis session to disk.

        Args:
            session_name: User-provided name for the session.
            raw_data: The uploaded DataFrame.
            analytics_data: Dictionary with analytics results.
            ml_data: Dictionary with ML results (without model).
            features_enabled: Dictionary of enabled features.

        Returns:
            Path to the saved session file.
        """
        session_id = self._generate_session_id()
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_name)
        filename = f"{session_id}_{safe_name}.json"
        filepath = self.sessions_dir / filename

        # Prepare session data
        session_data = {
            'session_id': session_id,
            'session_name': session_name,
            'created_at': datetime.now().isoformat(),
            'features_enabled': features_enabled,
            'analytics_summary': self._extract_analytics_summary(analytics_data),
            'ml_summary': self._extract_ml_summary(ml_data),
            'row_count': len(raw_data) if raw_data is not None else 0
        }

        # Save metadata
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

        # Save raw data as CSV
        data_filepath = self.sessions_dir / f"{session_id}_{safe_name}_data.csv"
        if raw_data is not None:
            raw_data.to_csv(data_filepath, index=False)

        logger.info(f"Session saved: {session_name} ({session_id})")

        # Cleanup old sessions
        self._cleanup_old_sessions()

        return str(filepath)

    def _extract_analytics_summary(self, analytics_data: dict) -> dict:
        """Extract serializable summary from analytics data."""
        if not analytics_data:
            return {}

        summary = {}

        # Copy simple values
        for key in ['headcount', 'department_count', 'turnover_rate', 'tenure_mean']:
            if key in analytics_data:
                value = analytics_data[key]
                if pd.notna(value):
                    summary[key] = float(value) if isinstance(value, (int, float)) else value

        return summary

    def _extract_ml_summary(self, ml_data: dict) -> dict:
        """Extract serializable summary from ML data."""
        if not ml_data:
            return {}

        summary = {}

        if 'metrics' in ml_data:
            summary['metrics'] = ml_data['metrics']

        if 'risk_distribution' in ml_data:
            summary['risk_distribution'] = ml_data['risk_distribution']

        if 'model_trained' in ml_data:
            summary['model_trained'] = ml_data['model_trained']

        return summary

    def load_session(self, session_path: str) -> dict:
        """
        Load a session from disk.

        Args:
            session_path: Path to the session JSON file.

        Returns:
            Dictionary with session data and DataFrame.
        """
        filepath = Path(session_path)

        if not filepath.exists():
            logger.error(f"Session file not found: {session_path}")
            return {}

        # Load metadata
        with open(filepath, 'r') as f:
            session_data = json.load(f)

        # Load raw data
        data_path = filepath.with_name(filepath.stem + "_data.csv")
        if data_path.exists():
            session_data['raw_data'] = pd.read_csv(data_path)
        else:
            session_data['raw_data'] = None

        logger.info(f"Session loaded: {session_data.get('session_name', 'Unknown')}")
        return session_data

    def list_sessions(self) -> list[dict]:
        """
        List all saved sessions.

        Returns:
            List of session metadata dictionaries.
        """
        sessions = []

        for filepath in sorted(self.sessions_dir.glob("*.json"), reverse=True):
            # Skip data files
            if "_data" in filepath.stem:
                continue

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    data['filepath'] = str(filepath)
                    sessions.append(data)
            except Exception as e:
                logger.warning(f"Failed to read session file {filepath}: {e}")

        return sessions[:self.max_saved]

    def delete_session(self, session_path: str) -> bool:
        """
        Delete a session.

        Args:
            session_path: Path to the session JSON file.

        Returns:
            True if deleted successfully.
        """
        filepath = Path(session_path)

        if not filepath.exists():
            return False

        try:
            # Delete metadata file
            filepath.unlink()

            # Delete data file if exists
            data_path = filepath.with_name(filepath.stem + "_data.csv")
            if data_path.exists():
                data_path.unlink()

            logger.info(f"Session deleted: {session_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False

    def _cleanup_old_sessions(self) -> None:
        """Remove sessions beyond max_saved limit."""
        sessions = self.list_sessions()

        if len(sessions) > self.max_saved:
            for session in sessions[self.max_saved:]:
                self.delete_session(session.get('filepath', ''))


def get_session_manager() -> SessionManager:
    """
    Get a SessionManager instance.

    Returns:
        SessionManager instance.
    """
    return SessionManager()
