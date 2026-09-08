"""Collaboration network availability contract.

Department membership and reporting lines do not measure collaboration,
influence or isolation. Reporting structure is available through StructuralEngine.
No synthetic collaboration edges or employee influence rankings are generated.
"""
from typing import Any
import pandas as pd


class NetworkEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.graph = None

    def get_key_influencers(self, limit: int = 10) -> list[dict[str, Any]]:
        return []

    def get_isolated_employees(self, limit: int = 10) -> list[dict[str, Any]]:
        return []

    def get_network_summary(self) -> dict[str, Any]:
        return {'success': False, 'available': False,
                'reason': 'Measured collaboration relationships are required; department and reporting data cannot establish influence or isolation.'}
