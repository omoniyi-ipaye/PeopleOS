"""
Model Lab Engine for PeopleOS.

Provides modular automated validation, backtesting, and refinement
for predictive models (Flight Risk and Retention).
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from src.logger import get_logger
from src.database import Database
from src.ml_engine import MLEngine

logger = get_logger('model_lab_engine')

class ModelLabEngine:
    """
    Automated laboratory for predictive model validation and refinement.
    
    Separates the monitoring and optimization logic from the core prediction engines.
    """

    def __init__(self, db: Optional[Database] = None):
        """Initialize the Model Lab Engine."""
        self.db = db or Database()
        self.ml_engine = MLEngine()

    def backtest_flight_risk(self, days_back: int = 90) -> Dict[str, Any]:
        """
        Compare historical flight risk predictions against current attrition status.
        Uses retroactive scoring (scoring old data with current model).
        """
        return {
            'status': 'warning',
            'metrics': None,
            'message': 'Prospective backtesting is unavailable: timestamped predictions from a model trained before the prediction date, a fixed outcome horizon, and mature follow-up are required.',
            'interpretation': 'Scoring old snapshots with a model trained on current outcomes would leak future information; it is not a valid backtest.'
        }

    def _get_interpretation(self, f1: float, recall: float) -> str:
        """Provide human-friendly interpretation of backtest results."""
        return 'Retrospective scores alone do not validate future departure accuracy.'

    def analyze_feature_sensitivity(self) -> List[Dict[str, Any]]:
        """
        Analyze which features are 'noisy' or contributing to model instability.
        """
        logger.info("Analyzing feature sensitivity and data quality")
        
        # 1. Get current data and importance
        df = self.db.get_all_employees()
        if df.empty:
            return []
        
        # 2. Train baseline if needed to get importance
        if not self.ml_engine.is_trained:
            return []
            
        importance_df = self.ml_engine.get_feature_importance_summary()
        
        # 3. Analyze Data Metrics (Nulls, Variance, Correlation)
        quality_report = []
        
        # Get processed data for analysis
        processed_df = self.ml_engine.preprocessor.transform(df).reindex(columns=self.ml_engine.feature_names)
        corr_matrix = processed_df.corr(numeric_only=True).abs()
        
        for feat in self.ml_engine.feature_names:
            # Importance
            feat_imp = importance_df[importance_df['feature'] == feat]['importance'].values[0] if feat in importance_df['feature'].values else 0
            
            # Variance (Std Dev)
            std = processed_df[feat].std() if feat in processed_df.columns else 0
            
            # Redundancy (High Correlation with other features)
            others = corr_matrix[feat].sort_values(ascending=False)[1:2]
            max_corr = others.values[0] if not others.empty else 0
            redundant_with = others.index[0] if max_corr > 0.85 else None
            
            # Reliability Score (0.0 - 1.0)
            reliability = 1.0
            if std < 0.05: reliability -= 0.3  # Too little variance (constant)
            if max_corr > 0.9: reliability -= 0.2  # Highly redundant
            
            quality_report.append({
                "feature": feat,
                "importance": round(feat_imp, 3),
                "reliability": round(max(0, reliability), 2),
                "status": "Stable" if reliability > 0.8 else "Noisy" if reliability > 0.5 else "Redundant",
                "recommendation": f"Remove redundant feature (overlaps with {redundant_with})" if redundant_with else "Maintain"
            })
            
        return sorted(quality_report, key=lambda x: x['importance'], reverse=True)

    def generate_refinement_plan(self) -> Dict[str, Any]:
        """
        Generate automated suggestions to improve model accuracy.
        """
        sensitivity = self.analyze_feature_sensitivity()
        
        to_drop = [f['feature'] for f in sensitivity if f['reliability'] < 0.6]
        critical_high_imp = [f['feature'] for f in sensitivity if f['importance'] > 0.1 and f['reliability'] < 0.8]
        
        plan = {
            "status": "review_only" if sensitivity else "insufficient_evidence",
            "suggested_actions": [],
            "automated_features_to_prune": [],
            "metrics": {
                "noisy_features": len(to_drop),
                "redundant_dimensions": len([f for f in sensitivity if f['status'] == 'Redundant']),
                "estimated_accuracy_lift": "Unknown; requires independent evaluation"
            }
        }
        
        if to_drop:
            plan["suggested_actions"].append(f"Prune {len(to_drop)} low-reliability features: {', '.join(to_drop)}")
        
        if critical_high_imp:
             plan["suggested_actions"].append(f"Enhance data quality for: {', '.join(critical_high_imp)} (High impact but noisy)")
             
        plan["reasoning"] = "Feature heuristics do not establish an accuracy improvement. Validate any change on independent data before activation."
        
        return plan
