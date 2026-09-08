"""Causal inference availability contract.

Employee snapshots alone do not identify intervention effects. The former
correlation fallback and fixed confidence numbers have been removed. A causal
implementation needs an explicit identification design, treatment timing,
pre-treatment covariates, diagnostics and independent validation before use.
"""
from typing import Any, Optional
import pandas as pd


class CausalEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def estimate_intervention_effect(self, treatment: str, outcome: str = 'Attrition',
                                     confounders: Optional[list[str]] = None) -> dict[str, Any]:
        return {'success': False, 'available': False, 'estimated_effect': None,
                'confidence_interval': None,
                'reason': 'Causal effects are unavailable without a validated identification design; correlations are not treatment effects.'}

    def get_intervention_recommendations(self) -> list[dict[str, Any]]:
        return []
