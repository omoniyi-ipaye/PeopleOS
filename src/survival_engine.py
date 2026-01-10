"""
Survival Analysis Engine for PeopleOS.

Provides probabilistic flight risk modeling using survival analysis techniques:
- Kaplan-Meier survival curves for tenure analysis
- Cox Proportional Hazards model for identifying risk factors
- Cohort-based flight risk predictions

Key Insights Generated:
- "Engineers in UK with >2yrs tenure and no promotion in 18mo have 65%
  probability of leaving in next quarter"
- Hazard ratios show which factors increase/decrease attrition risk
- Survival curves show when employees are most likely to leave

Mathematical Foundation:
- Cox Model: h(t|X) = h0(t) * exp(beta1*X1 + ... + betap*Xp)
- Kaplan-Meier: Non-parametric survival probability estimation
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta

from src.logger import get_logger
from src.utils import load_config

logger = get_logger('survival_engine')

# Minimum sample sizes for reliable analysis
MIN_SAMPLE_FOR_SURVIVAL = 30
MIN_SAMPLE_FOR_COX = 50
MIN_EVENTS_FOR_MODEL = 10  # Minimum attrition events


class SurvivalEngineError(Exception):
    """Custom exception for survival engine errors."""
    pass


class SurvivalEngine:
    """
    Survival analysis engine for probabilistic attrition modeling.

    Uses Kaplan-Meier survival curves and Cox Proportional Hazards
    regression to model time-to-event (attrition) with covariates.

    HR Interpretation:
    - Survival probability: Chance an employee stays until time t
    - Hazard ratio > 1: Factor INCREASES attrition risk
    - Hazard ratio < 1: Factor DECREASES attrition risk
    - Median survival: Half of similar employees leave by this time
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize Survival Engine.

        Args:
            df: DataFrame with employee data including:
                Required:
                - EmployeeID: Unique identifier
                - Tenure: Time in company (years) - the time variable
                - Attrition: Event indicator (1=left, 0=still employed)

                Recommended covariates:
                - Salary, CompaRatio: Compensation factors
                - YearsSinceLastPromotion: Career progression
                - YearsInCurrentRole: Role stagnation
                - LastRating: Performance
                - Age, Dept, Location: Demographics
        """
        self.df = df.copy()
        self.config = load_config()
        self.surv_config = self.config.get('survival', {})

        # Load configuration
        self.min_sample_size = self.surv_config.get('min_sample_size', MIN_SAMPLE_FOR_SURVIVAL)
        self.time_horizons = self.surv_config.get('time_horizon_months', [3, 6, 12, 24])
        self.risk_high = self.surv_config.get('risk_categories', {}).get('high', 0.65)
        self.risk_medium = self.surv_config.get('risk_categories', {}).get('medium', 0.35)

        # Cox model covariates - Dynamically identify from Golden Schema
        from src.data_loader import GOLDEN_SCHEMA
        all_potential_features = GOLDEN_SCHEMA['required'] + GOLDEN_SCHEMA['optional']
        
        # We only want numeric columns that aren't the target or ID
        exclude_cols = ['EmployeeID', 'Attrition', 'Tenure', 'HireDate', 'PromotionDate', 'RatingHistory', 'PerformanceText', 'Gender', 'Dept', 'Location', 'JobTitle', 'ManagerID', 'HireSource']
        
        # Check available columns
        self.has_attrition = 'Attrition' in self.df.columns
        
        self.available_covariates = [
            col for col in self.df.columns 
            if col in all_potential_features 
            and col not in exclude_cols
            and pd.api.types.is_numeric_dtype(self.df[col])
        ]
        
        logger.info(f"Dynamically identified {len(self.available_covariates)} covariates for Cox model: {self.available_covariates}")

        self.warnings: List[str] = []
        
        # Model state
        self.km_fitter = None
        self.cox_model = None
        self.cox_fitted = False

        self._validate_data()
        
        # Pre-fit models
        self.fit_kaplan_meier()
        self.fit_cox_proportional_hazards()

        logger.info(f"SurvivalEngine initialized with {len(df)} employees, "
                   f"Attrition available: {self.has_attrition}")

    def _validate_data(self) -> None:
        """Validate required columns exist and data quality."""
        required = ['EmployeeID', 'Tenure']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise SurvivalEngineError(f"Missing required columns: {missing}")

        if not self.has_attrition:
            self._add_warning(
                "Attrition column not found. Survival analysis requires attrition data. "
                "Only descriptive tenure analysis will be available."
            )
        else:
            # Check for sufficient events
            n_events = self.df['Attrition'].sum()
            if n_events < MIN_EVENTS_FOR_MODEL:
                self._add_warning(
                    f"Only {n_events} attrition events found. Need at least {MIN_EVENTS_FOR_MODEL} "
                    "for reliable Cox model. Results may be unstable."
                )

        # Check tenure quality
        if self.df['Tenure'].isna().any():
            na_count = self.df['Tenure'].isna().sum()
            self._add_warning(f"{na_count} employees have missing tenure data")

    def _add_warning(self, warning: str) -> None:
        """Add a warning message for HR review."""
        self.warnings.append(warning)
        logger.warning(warning)

    def _check_sample_size(self, n: int, context: str) -> bool:
        """Check if sample size is sufficient for reliable analysis."""
        if n < self.min_sample_size:
            self._add_warning(
                f"{context}: Sample size ({n}) too small for reliable survival analysis. "
                f"Minimum recommended: {self.min_sample_size}"
            )
            return False
        return True

    def fit_kaplan_meier(self, segment_by: Optional[str] = None) -> Dict[str, Any]:
        """
        Fit Kaplan-Meier survival curves.

        The Kaplan-Meier estimator is a non-parametric statistic used to
        estimate the survival function from lifetime data. It shows the
        probability of an employee remaining at the company over time.

        Args:
            segment_by: Optional column to segment analysis (e.g., 'Dept', 'Location')

        Returns:
            Dictionary with survival curves and statistics:
            {
                'overall': {
                    'survival_function': [...],  # Time points and probabilities
                    'median_survival_time': 4.2,  # Median tenure before attrition
                    'confidence_intervals': [...]
                },
                'segments': {...}  # If segment_by provided
            }

        HR Interpretation:
        - Steep initial drop: High early attrition (onboarding issues)
        - Plateau after 2-3 years: Stable period
        - Drop at 5-7 years: Common "7-year itch"
        """
        try:
            from lifelines import KaplanMeierFitter
        except ImportError:
            self._add_warning("lifelines library not installed. Install with: pip install lifelines")
            return {'available': False, 'reason': 'lifelines library not installed'}

        if not self.has_attrition:
            return {
                'available': False,
                'reason': 'Attrition column required for Kaplan-Meier analysis'
            }

        df = self.df.copy()

        # Convert tenure to months for more granular analysis
        df['tenure_months'] = df['Tenure'] * 12

        results = {
            'available': True,
            'overall': {},
            'segments': {},
            'interpretation': []
        }

        # Fit overall survival curve
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=df['tenure_months'],
            event_observed=df['Attrition'],
            label='Overall'
        )
        self.km_fitter = kmf

        # Extract survival function
        survival_func = kmf.survival_function_

        results['overall'] = {
            'survival_function': [
                {
                    'time_months': float(t),
                    'time_years': round(float(t) / 12, 2),
                    'survival_probability': round(float(p), 4),
                    'at_risk': int(kmf.event_table.loc[:t, 'at_risk'].iloc[-1]) if t <= kmf.event_table.index.max() else 0
                }
                for t, p in zip(survival_func.index, survival_func['Overall'])
            ],
            'median_survival_months': float(kmf.median_survival_time_) if kmf.median_survival_time_ != np.inf else None,
            'median_survival_years': round(float(kmf.median_survival_time_) / 12, 2) if kmf.median_survival_time_ != np.inf else None,
            'mean_survival_months': float(kmf.survival_function_.sum().iloc[0]) if isinstance(kmf.survival_function_.sum(), pd.Series) else float(kmf.survival_function_.sum()),
            'confidence_intervals': {
                'lower': kmf.confidence_interval_survival_function_.iloc[:, 0].tolist()[:20],  # Sample
                'upper': kmf.confidence_interval_survival_function_.iloc[:, 1].tolist()[:20]
            }
        }

        # Add time-point survival probabilities
        for months in [6, 12, 24, 36, 60]:
            if months <= survival_func.index.max():
                idx = survival_func.index[survival_func.index <= months].max()
                prob = float(survival_func.loc[idx, 'Overall'])
                results['overall'][f'survival_at_{months}mo'] = round(prob, 3)

        # Segment analysis if requested
        if segment_by and segment_by in df.columns:
            unique_segments = df[segment_by].unique()

            for segment in unique_segments:
                segment_df = df[df[segment_by] == segment]

                if len(segment_df) < 10 or segment_df['Attrition'].sum() < 3:
                    continue  # Skip small segments

                seg_kmf = KaplanMeierFitter()
                seg_kmf.fit(
                    durations=segment_df['tenure_months'],
                    event_observed=segment_df['Attrition'],
                    label=str(segment)
                )

                seg_survival = seg_kmf.survival_function_

                results['segments'][str(segment)] = {
                    'survival_function': [
                        {
                            'time_months': float(t),
                            'survival_probability': round(float(p), 4)
                        }
                        for t, p in list(zip(seg_survival.index, seg_survival.iloc[:, 0]))[:20]  # Limit points
                    ],
                    'median_survival_months': float(seg_kmf.median_survival_time_) if seg_kmf.median_survival_time_ != np.inf else None,
                    'sample_size': len(segment_df),
                    'events': int(segment_df['Attrition'].sum())
                }

        # Generate interpretations
        if results['overall'].get('median_survival_years'):
            median = results['overall']['median_survival_years']
            results['interpretation'].append(
                f"Half of employees who leave do so within {median:.1f} years."
            )

        if 'survival_at_12mo' in results['overall']:
            prob_12mo = results['overall']['survival_at_12mo']
            attrition_12mo = round((1 - prob_12mo) * 100, 1)
            results['interpretation'].append(
                f"First-year attrition: {attrition_12mo}% of employees leave within 12 months."
            )

        return results

    def fit_cox_proportional_hazards(self) -> Dict[str, Any]:
        """
        Fit Cox Proportional Hazards model.

        The Cox model relates the time-to-event (attrition) to covariates:
        h(t|X) = h0(t) * exp(beta1*X1 + ... + betap*Xp)

        Where:
        - h(t|X) is the hazard (instantaneous risk) at time t given covariates X
        - h0(t) is the baseline hazard
        - beta coefficients indicate impact of each covariate

        Returns:
            Dictionary with model results:
            {
                'coefficients': {
                    'Salary': {'coef': -0.23, 'hazard_ratio': 0.79, 'p_value': 0.01,
                              'interpretation': 'Higher salary reduces attrition by 21%'}
                },
                'concordance_index': 0.72,  # Model fit metric (0.5=random, 1=perfect)
                'log_likelihood_ratio_test': {...}
            }

        HR Interpretation:
        - Hazard Ratio > 1: Factor INCREASES risk of leaving
        - Hazard Ratio < 1: Factor DECREASES risk of leaving
        - Hazard Ratio = 1: No effect
        - p-value < 0.05: Finding is statistically significant
        """
        try:
            from lifelines import CoxPHFitter
        except ImportError:
            self._add_warning("lifelines library not installed")
            return {'available': False, 'reason': 'lifelines library not installed'}

        if not self.has_attrition:
            return {
                'available': False,
                'reason': 'Attrition column required for Cox model'
            }

        if len(self.available_covariates) == 0:
            return {
                'available': False,
                'reason': f'No Cox covariates available. Check that your data includes numeric columns like Salary, Age, YearsInCurrentRole, etc.'
            }

        df = self.df.copy()

        # Prepare data for Cox model
        cox_df = df[['Tenure', 'Attrition'] + self.available_covariates].copy()

        # Handle missing values
        cox_df = cox_df.dropna()

        if len(cox_df) < MIN_SAMPLE_FOR_COX:
            return {
                'available': False,
                'reason': f'Insufficient data after removing missing values ({len(cox_df)} < {MIN_SAMPLE_FOR_COX})'
            }

        n_events = cox_df['Attrition'].sum()
        if n_events < MIN_EVENTS_FOR_MODEL:
            return {
                'available': False,
                'reason': f'Insufficient attrition events ({n_events} < {MIN_EVENTS_FOR_MODEL})'
            }

        # Normalize numeric covariates for stable fitting
        for col in self.available_covariates:
            if cox_df[col].dtype in [np.float64, np.int64]:
                mean_val = cox_df[col].mean()
                std_val = cox_df[col].std()
                if std_val > 0:
                    cox_df[f'{col}_norm'] = (cox_df[col] - mean_val) / std_val

        # Identify normalized columns
        norm_cols = [c for c in cox_df.columns if c.endswith('_norm')]
        if not norm_cols:
            norm_cols = self.available_covariates

        # Fit Cox model
        try:
            cph = CoxPHFitter()
            cph.fit(
                cox_df[['Tenure', 'Attrition'] + norm_cols],
                duration_col='Tenure',
                event_col='Attrition'
            )
            self.cox_model = cph
            self.cox_fitted = True
        except Exception as e:
            logger.error(f"Cox model fitting failed: {e}")
            return {
                'available': False,
                'reason': f'Model fitting failed: {str(e)}'
            }

        # Extract results
        summary = cph.summary

        results = {
            'available': True,
            'coefficients': {},
            'model_metrics': {
                'concordance_index': round(cph.concordance_index_, 3),
                'log_likelihood': round(cph.log_likelihood_, 2),
                'aic': round(cph.AIC_partial_, 2),
                'sample_size': len(cox_df),
                'events': int(n_events)
            },
            'covariates_used': self.available_covariates,
            'recommendations': []
        }

        # Process each coefficient
        for covariate in summary.index:
            # Get original covariate name (remove _norm suffix)
            orig_name = covariate.replace('_norm', '')

            coef = float(summary.loc[covariate, 'coef'])
            hazard_ratio = float(summary.loc[covariate, 'exp(coef)'])
            p_value = float(summary.loc[covariate, 'p'])
            ci_lower = float(summary.loc[covariate, 'exp(coef) lower 95%'])
            ci_upper = float(summary.loc[covariate, 'exp(coef) upper 95%'])

            # Generate HR interpretation
            is_significant = p_value < 0.05

            if hazard_ratio > 1:
                pct_increase = round((hazard_ratio - 1) * 100, 1)
                direction = 'increases'
                effect = f"increases attrition risk by {pct_increase}%"
            else:
                pct_decrease = round((1 - hazard_ratio) * 100, 1)
                direction = 'decreases'
                effect = f"decreases attrition risk by {pct_decrease}%"

            interpretation = f"One standard deviation increase in {orig_name} {effect}"
            if not is_significant:
                interpretation += " (NOT statistically significant)"

            results['coefficients'][orig_name] = {
                'feature': orig_name,
                'coefficient': round(coef, 4),
                'hazard_ratio': round(hazard_ratio, 3),
                'p_value': round(p_value, 4),
                'is_significant': is_significant,
                'ci_lower': round(ci_lower, 3),
                'ci_upper': round(ci_upper, 3),
                'direction': direction,
                'interpretation': interpretation
            }

            # Generate recommendations for significant factors
            if is_significant:
                if hazard_ratio > 1.2:
                    results['recommendations'].append(
                        f"HIGH RISK FACTOR: {orig_name} significantly increases attrition. "
                        f"Review employees with high {orig_name} values for intervention."
                    )
                elif hazard_ratio < 0.8:
                    results['recommendations'].append(
                        f"PROTECTIVE FACTOR: {orig_name} reduces attrition. "
                        f"Consider strategies to increase {orig_name} for at-risk employees."
                    )

        # Model quality interpretation
        c_index = results['model_metrics']['concordance_index']
        if c_index >= 0.7:
            quality = "good"
        elif c_index >= 0.6:
            quality = "moderate"
        else:
            quality = "weak"

        results['model_metrics']['quality_interpretation'] = (
            f"Concordance index of {c_index:.2f} indicates {quality} predictive ability. "
            f"(0.5 = random, 0.7+ = good, 0.8+ = excellent)"
        )

        return results

    def predict_survival_probability(
        self,
        employee_ids: Optional[List[str]] = None,
        time_horizon_months: int = 12
    ) -> pd.DataFrame:
        """
        Predict survival probability for specific employees.

        Uses the fitted Cox model to predict individual survival probabilities.

        Args:
            employee_ids: List of employee IDs to predict (None = all)
            time_horizon_months: Prediction horizon in months

        Returns:
            DataFrame with survival predictions including:
            - EmployeeID
            - survival_prob_3mo, survival_prob_6mo, survival_prob_12mo
            - risk_category (High/Medium/Low)
            - key_risk_factors

        HR Action:
        - High Risk (>65%): Immediate stay interview, compensation review
        - Medium Risk (35-65%): Career development discussion
        - Low Risk (<35%): Standard engagement activities
        """
        if not self.cox_fitted:
            # Try to fit the model first
            cox_result = self.fit_cox_proportional_hazards()
            if not cox_result.get('available', False):
                return pd.DataFrame()

        if self.cox_model is None:
            return pd.DataFrame()

        df = self.df.copy()

        if employee_ids:
            df = df[df['EmployeeID'].isin(employee_ids)]

        if len(df) == 0:
            return pd.DataFrame()

        # Prepare features for prediction
        cox_df = df[['EmployeeID', 'Tenure', 'Attrition'] + self.available_covariates].copy()
        cox_df = cox_df.dropna()

        if len(cox_df) == 0:
            return pd.DataFrame()

        # Normalize features (same as training)
        original_df = self.df.copy()
        for col in self.available_covariates:
            if original_df[col].dtype in [np.float64, np.int64]:
                mean_val = original_df[col].mean()
                std_val = original_df[col].std()
                if std_val > 0:
                    cox_df[f'{col}_norm'] = (cox_df[col] - mean_val) / std_val

        norm_cols = [c for c in cox_df.columns if c.endswith('_norm')]

        # Get survival predictions
        try:
            # Predict survival function for each employee
            predictions = []

            for _, row in cox_df.iterrows():
                emp_data = row[norm_cols].to_frame().T

                # Get survival function
                surv_func = self.cox_model.predict_survival_function(emp_data)

                # Extract probabilities at different time points
                pred = {'EmployeeID': row['EmployeeID']}

                # Current tenure in months
                current_tenure_months = row['Tenure'] * 12

                # Predict for different horizons (from current tenure)
                for months in [3, 6, 12, 24]:
                    target_time = current_tenure_months + months
                    target_time_years = target_time / 12

                    # Find closest time point in survival function
                    if target_time_years <= surv_func.index.max():
                        closest_idx = surv_func.index[surv_func.index <= target_time_years].max()
                        prob = float(surv_func.loc[closest_idx].iloc[0])
                    else:
                        prob = float(surv_func.iloc[-1, 0])

                    pred[f'survival_{months}mo'] = round(prob, 3)
                    pred[f'attrition_risk_{months}mo'] = round(1 - prob, 3)

                # Calculate overall risk category based on 12-month attrition risk
                risk_12mo = pred.get('attrition_risk_12mo', 0)

                if risk_12mo >= self.risk_high:
                    pred['risk_category'] = 'High'
                elif risk_12mo >= self.risk_medium:
                    pred['risk_category'] = 'Medium'
                else:
                    pred['risk_category'] = 'Low'

                # Add current data for context
                pred['current_tenure_years'] = round(row['Tenure'], 1)
                pred['current_rating'] = row.get('LastRating', None)

                # Generate explanations
                pred['risk_factors'] = self._explain_risk(row, norm_cols)

                predictions.append(pred)

            result_df = pd.DataFrame(predictions)
            return result_df.sort_values('attrition_risk_12mo', ascending=False)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return pd.DataFrame()

    def _explain_risk(self, emp_row: pd.Series, norm_cols: List[str]) -> List[Dict[str, Any]]:
        """Generate human-readable explanations for an employee's risk profile."""
        if not self.cox_fitted or self.cox_model is None:
            return []

        factors = []
        summary = self.cox_model.summary

        for norm_col in norm_cols:
            col = norm_col.replace('_norm', '')
            if col not in summary.index and norm_col not in summary.index:
                continue
            
            idx = norm_col if norm_col in summary.index else col
            beta = summary.loc[idx, 'coef']
            norm_val = emp_row[norm_col]
            
            # Contribution to hazard log-ratio
            contribution = beta * norm_val
            
            if abs(contribution) < 0.1: # Skip negligible factors
                continue
                
            impact_multiplier = np.exp(contribution)
            
            # Determine impact level
            if abs(contribution) > 0.5:
                impact = "High"
            elif abs(contribution) > 0.2:
                impact = "Medium"
            else:
                impact = "Low"
                
            direction = "Increase Risk" if contribution > 0 else "Decrease Risk"
            
            # Generate description
            orig_val = emp_row[col]
            
            description = ""
            if col == 'Salary':
                if contribution > 0:
                    description = f"Salary is below company average for this role, contributing to {round((impact_multiplier-1)*100)}% higher relative risk."
                else:
                    description = f"Competitive salary (above company average) is a strong retention factor, reducing relative risk by {round((1-impact_multiplier)*100)}%."
            elif col == 'YearsSinceLastPromotion':
                if contribution > 0:
                    description = f"Stagnation in role ({round(orig_val, 1)} years since last promotion) increases departure probability."
                else:
                    description = "Recent promotion demonstrates clear career growth path."
            elif col == 'LastRating':
                if contribution > 0:
                     description = "Recent performance score indicates potential disengagement or mismatch."
                else:
                     description = "High performance score correlates with higher organizational commitment."
            elif col == 'YearsInCurrentRole':
                if contribution > 0:
                    description = f"Extended time in current role ({round(orig_val, 1)} years) may lead to role stagnation."
                else:
                    description = "Relatively new to role, still in growth phase."
            elif col == 'Age':
                 # Usually older employees have lower risk (negative beta)
                 if contribution > 0:
                     description = "Life stage or career level characteristics correlate with higher mobility."
                 else:
                     description = "Seniority and life stage correlate with higher organizational stability."
            else:
                if contribution > 0:
                    description = f"Factor {col} ({round(orig_val, 2)}) is currently increasing attrition risk."
                else:
                    description = f"Factor {col} ({round(orig_val, 2)}) is currently acting as a retention driver."

            factors.append({
                'factor': col,
                'impact': impact,
                'direction': direction,
                'score': round(contribution, 3),
                'description': description
            })
        
        # Sort by absolute contribution
        return sorted(factors, key=lambda x: abs(x['score']), reverse=True)

    def generate_cohort_insights(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate human-readable insights for filtered employee cohorts.

        This is the key output for HR stakeholders - translating statistical
        findings into actionable language.

        Args:
            filters: Dictionary of filters, e.g.:
                {'Dept': 'Engineering', 'Location': 'UK', 'tenure_min': 2}

        Returns:
            Dictionary with cohort analysis:
            {
                'cohort_description': 'Engineers in UK with >2yrs tenure',
                'cohort_size': 45,
                'median_survival_time': 2.3,
                'quarterly_attrition_probability': 0.65,
                'key_risk_factors': [...],
                'narrative': 'Engineers in UK with >2yrs tenure and no promotion
                             in 18mo have 65% probability of leaving in next quarter'
            }
        """
        df = self.df.copy()

        # Apply filters
        description_parts = []

        if filters:
            if 'Dept' in filters:
                df = df[df['Dept'] == filters['Dept']]
                description_parts.append(f"{filters['Dept']} department")

            if 'Location' in filters:
                df = df[df['Location'] == filters['Location']]
                description_parts.append(f"in {filters['Location']}")

            if 'tenure_min' in filters:
                df = df[df['Tenure'] >= filters['tenure_min']]
                description_parts.append(f">{filters['tenure_min']} years tenure")

            if 'tenure_max' in filters:
                df = df[df['Tenure'] <= filters['tenure_max']]
                description_parts.append(f"<{filters['tenure_max']} years tenure")

            if 'years_since_promotion_min' in filters and 'YearsSinceLastPromotion' in df.columns:
                df = df[df['YearsSinceLastPromotion'] >= filters['years_since_promotion_min']]
                description_parts.append(f"no promotion in {filters['years_since_promotion_min']}+ years")

            if 'rating_max' in filters and 'LastRating' in df.columns:
                df = df[df['LastRating'] <= filters['rating_max']]
                description_parts.append(f"rating ≤{filters['rating_max']}")

        cohort_description = "Employees" + (" - " + ", ".join(description_parts) if description_parts else "")

        result = {
            'cohort_description': cohort_description,
            'cohort_size': len(df),
            'filters_applied': filters or {},
        }

        if len(df) < 10:
            result['warning'] = "Cohort too small for reliable analysis"
            return result

        if not self.has_attrition:
            result['warning'] = "Attrition data not available"
            return result

        # Calculate cohort statistics
        n_left = df['Attrition'].sum()
        attrition_rate = n_left / len(df) if len(df) > 0 else 0
        avg_tenure = df['Tenure'].mean()

        result.update({
            'attrition_count': int(n_left),
            'attrition_rate': round(attrition_rate, 3),
            'avg_tenure_years': round(avg_tenure, 1),
        })

        # Fit Kaplan-Meier for this cohort
        try:
            from lifelines import KaplanMeierFitter

            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=df['Tenure'] * 12,  # Convert to months
                event_observed=df['Attrition']
            )

            # Get survival at key timepoints
            for months in [3, 6, 12]:
                if months <= kmf.survival_function_.index.max():
                    idx = kmf.survival_function_.index[kmf.survival_function_.index <= months].max()
                    survival_prob = float(kmf.survival_function_.loc[idx].iloc[0])
                    result[f'survival_probability_{months}mo'] = round(survival_prob, 3)
                    result[f'attrition_probability_{months}mo'] = round(1 - survival_prob, 3)

            # Median survival time
            if kmf.median_survival_time_ != np.inf:
                result['median_survival_months'] = round(float(kmf.median_survival_time_), 1)
                result['median_survival_years'] = round(float(kmf.median_survival_time_) / 12, 2)

        except Exception as e:
            logger.warning(f"Cohort survival analysis failed: {e}")

        # Generate key risk factors for this cohort
        key_factors = []

        if 'YearsSinceLastPromotion' in df.columns:
            avg_years_no_promo = df['YearsSinceLastPromotion'].mean()
            if avg_years_no_promo > 2:
                key_factors.append(f"Average {avg_years_no_promo:.1f} years since last promotion")

        if 'CompaRatio' in df.columns:
            avg_compa = df['CompaRatio'].mean()
            if avg_compa < 0.95:
                key_factors.append(f"Below-market compensation (compa-ratio: {avg_compa:.2f})")

        if 'LastRating' in df.columns:
            avg_rating = df['LastRating'].mean()
            if avg_rating < 3.0:
                key_factors.append(f"Low average performance rating ({avg_rating:.1f})")
            elif avg_rating > 4.0:
                key_factors.append(f"High performers at risk (avg rating: {avg_rating:.1f})")

        result['key_risk_factors'] = key_factors

        # Generate narrative
        narrative_parts = [cohort_description]

        if 'attrition_probability_12mo' in result:
            prob = result['attrition_probability_12mo']
            pct = round(prob * 100)
            narrative_parts.append(f"have a {pct}% probability of leaving in the next 12 months")

        if key_factors:
            narrative_parts.append(f"Key risk factors: {'; '.join(key_factors)}")

        result['narrative'] = ". ".join(narrative_parts) + "."

        return result

    def get_hazard_over_time(self) -> Dict[str, Any]:
        """
        Get baseline hazard function over time.

        Shows when attrition risk is highest during the employee lifecycle.
        Useful for understanding the "7-year itch" or "18-month cliff" patterns.

        Returns:
            Dictionary with hazard function data and key findings.

        HR Interpretation:
        - Peak hazard at 12-18 months: Onboarding/culture fit issues
        - Peak hazard at 3-4 years: Career progression concerns
        - Peak hazard at 7+ years: Long-tenure flight (often to leadership roles elsewhere)
        """
        if not self.cox_fitted:
            cox_result = self.fit_cox_proportional_hazards()
            if not cox_result.get('available', False):
                return {'available': False, 'reason': cox_result.get('reason', 'Cox model not fitted')}

        if self.cox_model is None:
            return {'available': False, 'reason': 'Cox model not available'}

        try:
            # Get baseline hazard
            baseline_hazard = self.cox_model.baseline_hazard_
            baseline_cumulative = self.cox_model.baseline_cumulative_hazard_
            baseline_survival = self.cox_model.baseline_survival_

            result = {
                'available': True,
                'hazard_over_time': [
                    {
                        'time_years': round(float(t), 2),
                        'baseline_hazard': round(float(h), 4),
                        'cumulative_hazard': round(float(baseline_cumulative.loc[t].iloc[0]), 4),
                        'survival': round(float(baseline_survival.loc[t].iloc[0]), 4)
                    }
                    for t, h in zip(baseline_hazard.index, baseline_hazard.iloc[:, 0])
                    if t <= 15  # Limit to 15 years
                ],
                'risk_periods': []
            }

            # Identify high-risk periods
            hazard_values = baseline_hazard.iloc[:, 0]
            mean_hazard = hazard_values.mean()

            for i, (time, hazard) in enumerate(zip(baseline_hazard.index, hazard_values)):
                if hazard > mean_hazard * 1.5 and time <= 10:  # 50% above average
                    result['risk_periods'].append({
                        'time_years': round(float(time), 1),
                        'relative_risk': round(float(hazard / mean_hazard), 2),
                        'interpretation': f"Attrition risk peaks around year {time:.1f}"
                    })

            return result

        except Exception as e:
            logger.error(f"Hazard analysis failed: {e}")
            return {'available': False, 'reason': str(e)}

    def get_at_risk_employees(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get the employees with highest attrition risk.

        Args:
            top_n: Number of high-risk employees to return

        Returns:
            DataFrame with high-risk employees and their risk factors.
        """
        predictions = self.predict_survival_probability()

        if predictions.empty:
            return pd.DataFrame()

        # Filter to high and medium risk
        at_risk = predictions[predictions['risk_category'].isin(['High', 'Medium'])]

        # Add additional context from main dataframe
        if not at_risk.empty:
            context_cols = ['Dept', 'Location', 'JobTitle', 'YearsSinceLastPromotion', 'CompaRatio']
            available_context = [c for c in context_cols if c in self.df.columns]

            if available_context:
                context_df = self.df[['EmployeeID'] + available_context]
                at_risk = at_risk.merge(context_df, on='EmployeeID', how='left')

        return at_risk.head(top_n)

    def analyze_all(self) -> Dict[str, Any]:
        """
        Run complete survival analysis.

        Returns:
            Dictionary with all survival analysis results:
            - kaplan_meier: Survival curves
            - cox_model: Hazard ratios and model metrics
            - cohort_insights: Example cohort analyses
            - at_risk_employees: Top employees at risk
            - recommendations: Actionable insights
            - warnings: Data quality warnings
        """
        logger.info("Running full survival analysis")

        # Clear warnings for fresh analysis
        self.warnings = []

        results = {
            'kaplan_meier': self.fit_kaplan_meier(),
            'kaplan_meier_by_dept': self.fit_kaplan_meier(segment_by='Dept'),
            'cox_model': self.fit_cox_proportional_hazards(),
            'hazard_over_time': self.get_hazard_over_time(),
            'cohort_insights': [],
            'at_risk_employees': [],
            'summary': {},
            'recommendations': [],
            'warnings': []
        }

        # Generate cohort insights for key segments
        cohort_filters = [
            {'Dept': 'Engineering'},
            {'Dept': 'Sales'},
            {'tenure_min': 2, 'years_since_promotion_min': 1.5},
        ]

        for filters in cohort_filters:
            insight = self.generate_cohort_insights(filters)
            if insight.get('cohort_size', 0) >= 10:
                results['cohort_insights'].append(insight)

        # Get all predictions for full summary stats (not truncated)
        all_predictions = self.predict_survival_probability()
        
        # Get truncated at-risk employees for the list view
        at_risk_df = self.get_at_risk_employees()
        if not at_risk_df.empty:
            results['at_risk_employees'] = at_risk_df.to_dict('records')

        # Generate summary
        results['summary'] = {
            'total_employees': len(self.df),
            'attrition_available': self.has_attrition,
            'attrition_count': int(self.df['Attrition'].sum()) if self.has_attrition else None,
            'overall_attrition_rate': round(self.df['Attrition'].mean(), 3) if self.has_attrition else None,
            'cox_model_fitted': self.cox_fitted,
            'covariates_used': self.available_covariates,
            'high_risk_count': len(all_predictions[all_predictions['risk_category'] == 'High']) if not all_predictions.empty else 0,
            'medium_risk_count': len(all_predictions[all_predictions['risk_category'] == 'Medium']) if not all_predictions.empty else 0,
            'median_tenure': results['kaplan_meier']['overall'].get('median_survival_years'),
            'avg_12mo_risk': 1 - results['kaplan_meier']['overall'].get('survival_at_12mo', 1)
        }

        # Generate recommendations based on findings
        if results['cox_model'].get('available'):
            results['recommendations'].extend(results['cox_model'].get('recommendations', []))

        if results['summary'].get('high_risk_count', 0) > 0:
            results['recommendations'].append(
                f"PRIORITY: {results['summary']['high_risk_count']} employees identified as high attrition risk. "
                "Consider immediate stay interviews and career development discussions."
            )

        # Add warnings
        results['warnings'] = self.warnings.copy()

        logger.info(f"Survival analysis complete. Warnings: {len(self.warnings)}")
        return results
