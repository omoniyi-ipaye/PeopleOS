"""
Causal Inference Engine for PeopleOS.

Uses DoWhy library to estimate causal effects of HR interventions.
Answers "What-If" questions like:
- "If we increase salary by 10%, will turnover decrease?"
- "Does a promotion reduce departure probability?"
"""

from typing import Dict, Any, Optional, List
import pandas as pd
from src.logger import get_logger

logger = get_logger('causal_engine')

# Try to import DoWhy, fallback gracefully
try:
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logger.warning("DoWhy library not available. Causal inference will use heuristic fallback.")


class CausalEngine:
    """
    Engine for estimating causal effects of HR interventions.
    
    Uses DoWhy when available, otherwise falls back to
    correlation-based heuristics with appropriate caveats.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Causal Engine.
        
        Args:
            df: DataFrame with employee data including outcome variable.
        """
        self.df = df.copy()
        self._prepare_data()
        
    def _prepare_data(self) -> None:
        """Prepare data for causal analysis."""
        # Ensure binary outcome
        if 'Attrition' in self.df.columns:
            self.df['Attrition'] = self.df['Attrition'].astype(int)
        
        # Create useful derived variables
        if 'Salary' in self.df.columns:
            median_salary = self.df['Salary'].median()
            self.df['HighSalary'] = (self.df['Salary'] > median_salary).astype(int)
        
        if 'LastRating' in self.df.columns:
            self.df['HighPerformer'] = (self.df['LastRating'] >= 4.0).astype(int)
    
    def estimate_intervention_effect(
        self,
        treatment: str,
        outcome: str = 'Attrition',
        confounders: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Estimate the causal effect of a treatment on an outcome.
        
        Args:
            treatment: Treatment variable (e.g., 'HighSalary', 'Promotion').
            outcome: Outcome variable (default: 'Attrition').
            confounders: List of confounding variables to control for.
            
        Returns:
            Dictionary with:
            - estimated_effect: The causal effect size
            - confidence_interval: (lower, upper) bounds
            - interpretation: HR-friendly explanation
            - method: The estimation method used
        """
        # Validate inputs
        if treatment not in self.df.columns:
            return {
                'success': False,
                'reason': f"Treatment variable '{treatment}' not found in data."
            }
        
        if outcome not in self.df.columns:
            return {
                'success': False,
                'reason': f"Outcome variable '{outcome}' not found in data."
            }
        
        # Default confounders based on HR domain knowledge
        if confounders is None:
            confounders = self._get_default_confounders(treatment)
            
        if DOWHY_AVAILABLE:
            return self._estimate_with_dowhy(treatment, outcome, confounders)
        else:
            return self._estimate_heuristic(treatment, outcome, confounders)
    
    def _get_default_confounders(self, treatment: str) -> List[str]:
        """Get default confounders based on treatment variable."""
        # Common HR confounders
        potential = ['Age', 'Tenure', 'Dept', 'JobLevel', 'Gender', 'Location']
        
        # Don't include treatment itself as confounder
        available = [c for c in potential if c in self.df.columns and c != treatment]
        
        # Limit to available numeric/categorical columns
        return available[:5]  # Max 5 confounders for stability
    
    def _estimate_with_dowhy(
        self,
        treatment: str,
        outcome: str,
        confounders: List[str]
    ) -> Dict[str, Any]:
        """Use DoWhy for proper causal inference."""
        try:
            # Build causal graph specification
            # Simple DAG: Confounders -> Treatment -> Outcome
            #             Confounders -> Outcome
            
            # Create causal model
            model = CausalModel(
                data=self.df,
                treatment=treatment,
                outcome=outcome,
                common_causes=confounders
            )
            
            # Identify causal effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            # Estimate using propensity score matching
            causal_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_matching"
            )
            
            effect = float(causal_estimate.value)
            
            # Refutation test for robustness (optional, can be slow)
            # refutation = model.refute_estimate(identified_estimand, causal_estimate, method_name="placebo_treatment_refuter")
            
            return self._format_result(
                effect=effect,
                treatment=treatment,
                outcome=outcome,
                method='DoWhy (Propensity Score Matching)',
                confidence=0.85  # Approximate
            )
            
        except Exception as e:
            logger.error(f"DoWhy estimation failed: {e}")
            # Fallback to heuristic
            return self._estimate_heuristic(treatment, outcome, confounders)
    
    def _estimate_heuristic(
        self,
        treatment: str,
        outcome: str,
        confounders: List[str]
    ) -> Dict[str, Any]:
        """
        Heuristic estimation when DoWhy is unavailable.
        
        Uses difference-in-means adjusted by observable characteristics.
        """
        try:
            # Ensure treatment is binary
            if self.df[treatment].nunique() > 2:
                # Convert to binary (above/below median)
                median_val = self.df[treatment].median()
                treatment_binary = (self.df[treatment] > median_val).astype(int)
            else:
                treatment_binary = self.df[treatment].astype(int)
            
            # Simple difference in means
            treated = self.df[treatment_binary == 1][outcome].mean()
            control = self.df[treatment_binary == 0][outcome].mean()
            
            raw_effect = treated - control
            
            # Adjust for confounders using regression if possible
            try:
                from sklearn.linear_model import LinearRegression
                
                # Control for confounders
                X = pd.get_dummies(self.df[confounders + [treatment]], drop_first=True)
                X = X.fillna(0)
                y = self.df[outcome]
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Find the treatment coefficient
                treatment_col = [c for c in X.columns if treatment in c]
                if treatment_col:
                    coef_idx = list(X.columns).index(treatment_col[0])
                    adjusted_effect = float(model.coef_[coef_idx])
                else:
                    adjusted_effect = raw_effect
                    
            except Exception:
                adjusted_effect = raw_effect
            
            return self._format_result(
                effect=adjusted_effect,
                treatment=treatment,
                outcome=outcome,
                method='Heuristic (Adjusted Difference)',
                confidence=0.6  # Lower confidence for heuristic
            )
            
        except Exception as e:
            logger.error(f"Heuristic estimation failed: {e}")
            return {
                'success': False,
                'reason': str(e)
            }
    
    def _format_result(
        self,
        effect: float,
        treatment: str,
        outcome: str,
        method: str,
        confidence: float
    ) -> Dict[str, Any]:
        """Format the causal inference result with HR-friendly interpretation."""
        
        # Determine direction and magnitude
        direction = "increases" if effect > 0 else "decreases"
        magnitude = abs(effect)
        
        # Create HR-friendly interpretation
        if outcome == 'Attrition':
            if effect < -0.05:
                interpretation = f"✅ **Positive Impact**: {treatment} reduces departure risk by {magnitude*100:.1f}%."
                action = "Consider expanding this policy to at-risk employees."
            elif effect > 0.05:
                interpretation = f"⚠️ **Negative Impact**: {treatment} increases departure risk by {magnitude*100:.1f}%."
                action = "Review this policy for unintended consequences."
            else:
                interpretation = f"➡️ **Neutral Impact**: {treatment} has minimal effect on departures (±{magnitude*100:.1f}%)."
                action = "Focus resources on other intervention areas."
        else:
            interpretation = f"{treatment} {direction} {outcome} by {magnitude:.2f} units."
            action = "Review implications and consider strategic adjustments."
        
        return {
            'success': True,
            'treatment': treatment,
            'outcome': outcome,
            'estimated_effect': round(effect, 4),
            'effect_percentage': round(effect * 100, 2),
            'direction': direction,
            'confidence_score': confidence,
            'method': method,
            'interpretation': interpretation,
            'recommended_action': action,
            'caveats': [
                "Causal estimates assume no unmeasured confounders.",
                "Effect may vary across employee segments.",
                "Historical patterns may not predict future results."
            ]
        }
    
    def get_intervention_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate a list of potential intervention recommendations.
        
        Tests common HR interventions and ranks by estimated impact.
        """
        # Define common interventions to test
        interventions = [
            {'treatment': 'HighSalary', 'name': 'Competitive Compensation'},
            {'treatment': 'HighPerformer', 'name': 'Performance Recognition'},
        ]
        
        # Add more if columns exist
        if 'YearsSinceLastPromotion' in self.df.columns:
            median_promo = self.df['YearsSinceLastPromotion'].median()
            self.df['RecentPromotion'] = (self.df['YearsSinceLastPromotion'] < median_promo).astype(int)
            interventions.append({'treatment': 'RecentPromotion', 'name': 'Career Progression'})
        
        results = []
        for intervention in interventions:
            if intervention['treatment'] in self.df.columns:
                result = self.estimate_intervention_effect(intervention['treatment'])
                if result.get('success'):
                    results.append({
                        'intervention': intervention['name'],
                        **result
                    })
        
        # Sort by absolute effect size
        results.sort(key=lambda x: abs(x.get('estimated_effect', 0)), reverse=True)
        
        return results
