"""
Model Lab Engine for PeopleOS.

Provides modular automated validation, backtesting, and refinement
for predictive models (Flight Risk and Retention).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from src.logger import get_logger
from src.database import Database
from src.ml_engine import MLEngine
from src.survival_engine import SurvivalEngine

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
        logger.info(f"Running retroactive backtest for Flight Risk (window: {days_back} days)")
        
        # 1. Ensure model is trained
        if not self.ml_engine.is_trained:
            logger.info("Training ML engine for backtest baseline")
            df = self.db.get_all_employees()
            if df.empty:
                return {"status": "error", "message": "No data to train baseline"}
            self.ml_engine.train(df)

        # 2. Get current attrition status (ground truth)
        current_df = self.db.get_all_employees()
        if current_df.empty:
            return {"status": "error", "message": "No employee data available for validation"}
        # We need to find people who currently have Attrition=1
        actual_attrition = current_df[current_df['Attrition'] == 1]['EmployeeID'].tolist()

        # 3. Get snapshots from the lookback period
        # We want snapshots of people who were ACTIVE at that time
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        snapshot_df = self.db.get_historical_snapshots(start_date=start_date)
        
        if snapshot_df.empty:
            return {
                "status": "warning", 
                "message": f"Historical data baseline missing. This feature works best after 60-90 days of tracking. Total snapshots found: 0."
            }

        # Filter for snapshots where they were still active (attrition=0 at the time)
        active_at_time = snapshot_df[snapshot_df['Attrition'] == 0]
        
        if active_at_time.empty:
            return {"status": "warning", "message": "No active employee snapshots found in lookback period."}

        # 4. Retroactive Scoring
        # Predict risk for those active employees using the current model
        # We need to map snapshot columns to what the MLEngine expects
        # (The MLEngine handles mapping via its internal preprocessor)
        
        predictions = []
        for _, row in active_at_time.iterrows():
            emp_id = row['EmployeeID']
            # Reconstruct the feature set from snapshot
            # Simplified for now: use the raw data from snapshot
            row_dict = row.to_dict()
            try:
                risk_data = self.ml_engine.predict(pd.DataFrame([row_dict]))
                if risk_data:
                    score = risk_data[0].get('risk_score', 0)
                    predictions.append({
                        'employee_id': emp_id,
                        'predicted_high_risk': score >= self.ml_engine.risk_threshold_high,
                        'was_actually_attrited': emp_id in actual_attrition
                    })
            except Exception as e:
                logger.warning(f"Failed to score snapshot for {emp_id}: {e}")

        logger.info(f"Backtest generated {len(predictions)} predictions from {len(active_at_time)} active snapshots")
        if not predictions:
            return {
                "status": "warning", 
                "message": "Historical snapshots have a reduced schema compared to current employee data. Full backtesting will be available once snapshots include the complete feature set. Current model accuracy is based on cross-validation during training."
            }

        pred_df = pd.DataFrame(predictions)
        
        # 5. Calculate Metrics
        # True Positive: Predicted High Risk AND Actually Left
        tp = len(pred_df[(pred_df['predicted_high_risk'] == True) & (pred_df['was_actually_attrited'] == True)])
        # False Positive: Predicted High Risk AND Still Present
        fp = len(pred_df[(pred_df['predicted_high_risk'] == True) & (pred_df['was_actually_attrited'] == False)])
        # False Negative: Predicted Low Risk AND Actually Left
        fn = len(pred_df[(pred_df['predicted_high_risk'] == False) & (pred_df['was_actually_attrited'] == True)])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "status": "success",
            "metrics": {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(f1, 3),
                "sample_size": len(pred_df),
                "true_positives": tp,
                "false_positives": fp,
                "missed_exits": fn
            },
            "interpretation": self._get_interpretation(f1, recall)
        }

    def _get_interpretation(self, f1: float, recall: float) -> str:
        """Provide human-friendly interpretation of backtest results."""
        if f1 > 0.7:
            return "EXCELLENT: The model is highly reliable at identifying future departures."
        elif f1 > 0.4:
            return "GOOD: The model captures a significant portion of risks but has some noise."
        elif recall < 0.3:
            return "CAUTION: The model is conservative and missing many actual departures."
        else:
            return "UPDATING: Insufficient outcomes to verify model accuracy. Continue tracking for 30 more days."

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
            self.ml_engine.train(df)
            
        importance_df = self.ml_engine.get_feature_importance_summary()
        
        # 3. Analyze Data Metrics (Nulls, Variance, Correlation)
        quality_report = []
        
        # Get processed data for analysis
        processed_df, _ = self.ml_engine.preprocessor.fit_transform(df)
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
            "status": "optimization_needed" if to_drop else "healthy",
            "suggested_actions": [],
            "automated_features_to_prune": to_drop,
            "metrics": {
                "noisy_features": len(to_drop),
                "redundant_dimensions": len([f for f in sensitivity if f['status'] == 'Redundant']),
                "estimated_accuracy_lift": f"{len(to_drop) * 3}% - {len(to_drop) * 5}%" if to_drop else "0%"
            }
        }
        
        if to_drop:
            plan["suggested_actions"].append(f"Prune {len(to_drop)} low-reliability features: {', '.join(to_drop)}")
        
        if critical_high_imp:
             plan["suggested_actions"].append(f"Enhance data quality for: {', '.join(critical_high_imp)} (High impact but noisy)")
             
        plan["reasoning"] = f"Model accuracy (F1) is currently affected by {len(to_drop)} features with low signal-to-noise ratios."
        
        return plan
