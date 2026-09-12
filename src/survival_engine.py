"""Cohort-level survival analysis for PeopleOS.

Kaplan-Meier and Cox outputs are descriptive/inferential cohort evidence. They
require a critical data assumption: Tenure must mean time-to-exit for departed
employees and censoring time for active employees. PeopleOS does not use this
engine to rank employees or claim unconditional cohort survival is an
individual's probability of leaving in the next N months.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.population import normalize_attrition, resolve_current_population
from src.utils import load_config

logger = get_logger('survival_engine')
MIN_SAMPLE_FOR_SURVIVAL = 30
MIN_SAMPLE_FOR_COX = 50
MIN_EVENTS_FOR_MODEL = 10
MIN_PRIVACY_CELL = 5
SAFE_SEGMENT_DIMENSIONS = {'Dept', 'Location', 'JobLevel'}
SAFE_COX_COVARIATES = {
    'Salary', 'YearsSinceLastPromotion', 'YearsInCurrentRole', 'LastRating', 'Age', 'CompaRatio',
}


class SurvivalEngineError(Exception):
    pass


def _validated_min_sample(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise SurvivalEngineError(f'minimum survival sample size must be an integer of at least {MIN_SAMPLE_FOR_SURVIVAL}')
    numeric = int(value)
    if numeric < MIN_SAMPLE_FOR_SURVIVAL:
        raise SurvivalEngineError(f'minimum survival sample size must be an integer of at least {MIN_SAMPLE_FOR_SURVIVAL}')
    return numeric


def _finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


class SurvivalEngine:
    def __init__(self, df: pd.DataFrame):
        current, self.population_resolution = resolve_current_population(df)
        self.df = current.copy(deep=True)
        self._current_population_size = len(self.df)
        self.config = load_config()
        cfg = self.config.get('survival', {})
        self.min_sample_size = _validated_min_sample(cfg.get('min_sample_size', MIN_SAMPLE_FOR_SURVIVAL))
        self.has_attrition = 'Attrition' in self.df.columns
        self.warnings: List[str] = [
            'Survival estimates assume Tenure is time-to-exit for departed employees and censoring time for active employees. Validate this source-system meaning before decision use.'
        ]
        self.km_fitter = None
        self.cox_model = None
        self.cox_fitted = False
        self.available_covariates: list[str] = []
        self.excluded_invalid_duration = 0
        self.excluded_unknown_outcome = 0
        self._prepare_data()
        self.fit_kaplan_meier()
        self.fit_cox_proportional_hazards()

    def _population_metadata(self) -> Dict[str, Any]:
        source = int(self._current_population_size)
        analysis = int(len(self.df))
        return {
            'source_population': source,
            'analysis_population': analysis,
            'excluded_invalid_duration': int(self.excluded_invalid_duration),
            'excluded_unknown_outcome': int(self.excluded_unknown_outcome),
            'coverage': float(analysis / source) if source else 0.0,
            'population_semantics': 'current_employee_rows_with_valid_duration_and_known_event_or_censoring_status',
        }

    def _prepare_data(self) -> None:
        if 'EmployeeID' not in self.df.columns or 'Tenure' not in self.df.columns:
            raise SurvivalEngineError('EmployeeID and Tenure are required for survival analysis')
        self.df['Tenure'] = pd.to_numeric(self.df['Tenure'], errors='coerce')
        if self.has_attrition:
            self.df['Attrition'] = normalize_attrition(self.df['Attrition'])
        invalid_duration = ~np.isfinite(self.df['Tenure']) | (self.df['Tenure'] < 0)
        self.excluded_invalid_duration = int(invalid_duration.sum())
        if self.excluded_invalid_duration:
            self.warnings.append(f'Excluded {self.excluded_invalid_duration} row(s) with invalid survival duration')
        self.df = self.df.loc[~invalid_duration].copy()
        if self.has_attrition:
            unknown = self.df['Attrition'].isna()
            self.excluded_unknown_outcome = int(unknown.sum())
            if self.excluded_unknown_outcome:
                self.warnings.append(f'Excluded {self.excluded_unknown_outcome} row(s) with unknown Attrition outcome from survival fitting')
                self.df = self.df.loc[~unknown].copy()
            self.df['Attrition'] = self.df['Attrition'].astype(int)
        exclude = {
            'EmployeeID', 'Attrition', 'Tenure', 'HireDate', 'PromotionDate', 'RatingHistory',
            'PerformanceText', 'Gender', 'Dept', 'Location', 'JobTitle', 'ManagerID', 'HireSource',
            'SnapshotDate', 'is_active', 'TerminationDate', 'ExitDate', 'NationalID', 'National_ID',
        }
        configured = self.config.get('survival', {}).get('cox_covariates', [])
        if not isinstance(configured, list) or any(not isinstance(c, str) for c in configured):
            raise SurvivalEngineError('Cox covariates must be configured as a list of column names')
        configured = list(dict.fromkeys(configured))
        self.available_covariates = [
            c for c in configured
            if c in self.df and c in SAFE_COX_COVARIATES and c not in exclude and pd.api.types.is_numeric_dtype(self.df[c])
        ]
        if self.has_attrition and int(self.df['Attrition'].sum()) < MIN_EVENTS_FOR_MODEL:
            self.warnings.append(f'Only {int(self.df["Attrition"].sum())} observed attrition events; Cox estimates may be unstable.')

    def _km_result(self, frame: pd.DataFrame, label: str) -> Dict[str, Any]:
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit(frame['Tenure'] * 12, event_observed=frame['Attrition'], label=label)
        sf = kmf.survival_function_.iloc[:, 0]
        timeline = sf.index.to_numpy(dtype=float)
        values = sf.to_numpy(dtype=float)
        lower = kmf.confidence_interval_survival_function_.iloc[:, 0].to_numpy(dtype=float)
        upper = kmf.confidence_interval_survival_function_.iloc[:, 1].to_numpy(dtype=float)
        if not (np.isfinite(timeline).all() and np.isfinite(values).all() and np.isfinite(lower).all() and np.isfinite(upper).all()):
            raise SurvivalEngineError('Kaplan-Meier calculation produced non-finite values')
        restricted_mean = float(np.sum(values[:-1] * np.diff(timeline)))
        if not np.isfinite(restricted_mean) or restricted_mean < 0:
            raise SurvivalEngineError('Restricted mean survival time is invalid')
        points = []
        for t, probability in zip(timeline, values):
            event_rows = kmf.event_table.index[kmf.event_table.index <= t]
            at_risk = int(kmf.event_table.loc[event_rows[-1], 'at_risk']) if len(event_rows) else len(frame)
            points.append({
                'time_months': float(t),
                'time_years': round(float(t) / 12, 2),
                'survival_probability': round(float(probability), 4),
                'at_risk': at_risk if at_risk >= MIN_PRIVACY_CELL else None,
                'at_risk_suppressed': bool(at_risk < MIN_PRIVACY_CELL),
            })
        median = kmf.median_survival_time_
        result: Dict[str, Any] = {
            'survival_function': points,
            'median_survival_months': None if np.isinf(median) else float(median),
            'median_survival_years': None if np.isinf(median) else round(float(median) / 12, 2),
            'mean_survival_months': restricted_mean,
            'restricted_mean_horizon_months': float(timeline[-1]),
            'mean_semantics': 'restricted_mean_through_last_observed_duration',
            'confidence_intervals': {'lower': lower.tolist(), 'upper': upper.tolist()},
            'semantics': 'cohort_survival_from_employment_origin_not_individual_future_probability',
        }
        for months in (3, 6, 12, 24, 36, 60):
            result[f'survival_at_{months}mo'] = round(float(kmf.predict(months)), 3) if months <= timeline[-1] else None
        return result

    def fit_kaplan_meier(self, segment_by: Optional[str] = None) -> Dict[str, Any]:
        population = self._population_metadata()
        if not self.has_attrition:
            return {'available': False, 'reason': 'Attrition is required for Kaplan-Meier analysis', 'population': population}
        if self._current_population_size and self.excluded_unknown_outcome == self._current_population_size - self.excluded_invalid_duration:
            return {'available': False, 'reason': 'No known Attrition outcomes remain for event/censoring classification', 'population': population}
        if len(self.df) < self.min_sample_size:
            return {'available': False, 'reason': f'Insufficient sample size ({len(self.df)} < {self.min_sample_size})', 'population': population}
        events = int(self.df['Attrition'].sum())
        censored = int(len(self.df) - events)
        if events < MIN_PRIVACY_CELL or censored < MIN_PRIVACY_CELL:
            return {
                'available': False,
                'reason': 'Insufficient event/censor support for a privacy-safe cohort survival estimate',
                'population': population,
                'outcome_counts_suppressed': True,
            }
        try:
            overall = self._km_result(self.df, 'Overall')
            results: Dict[str, Any] = {
                'available': True,
                'overall': overall,
                'segments': {},
                'population': population,
                'suppressed_segment_count': 0,
                'interpretation': [
                    'Survival probabilities are cohort estimates measured from employment origin; they are not next-period individual forecasts.'
                ],
            }
            self.km_fitter = True
            if segment_by:
                if segment_by not in SAFE_SEGMENT_DIMENSIONS or segment_by not in self.df.columns:
                    results['segment_unavailable_reason'] = 'Requested segmentation is unavailable or not approved for survival output'
                    return results
                suppressed = 0
                for segment, group in self.df.groupby(segment_by, observed=True, dropna=False):
                    segment_events = int(group['Attrition'].sum())
                    segment_censored = int(len(group) - segment_events)
                    if len(group) < 10 or segment_events < MIN_PRIVACY_CELL or segment_censored < MIN_PRIVACY_CELL:
                        suppressed += 1
                        continue
                    segment_result = self._km_result(group, str(segment))
                    results['segments'][str(segment)] = {
                        'segment_name': str(segment),
                        'median_survival_months': segment_result['median_survival_months'],
                        'sample_size': len(group),
                        'events': segment_events,
                        'survival_function': [
                            {'time_months': p['time_months'], 'survival_probability': p['survival_probability']}
                            for p in segment_result['survival_function']
                        ],
                    }
                results['suppressed_segment_count'] = suppressed
            return results
        except ImportError:
            return {'available': False, 'reason': 'lifelines library not installed', 'population': population}
        except Exception as exc:
            logger.exception('Kaplan-Meier fitting failed')
            return {'available': False, 'reason': f'Kaplan-Meier fitting failed ({type(exc).__name__})', 'population': population}

    @staticmethod
    def _stable_zscore(values: pd.Series) -> Optional[np.ndarray]:
        array = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
        if len(array) == 0 or not np.isfinite(array).all():
            return None
        scale = float(np.max(np.abs(array)))
        if not np.isfinite(scale) or scale == 0:
            return None
        scaled = array / scale
        mean = float(np.mean(scaled))
        centered = scaled - mean
        std = float(np.sqrt(np.sum(centered * centered) / max(len(centered) - 1, 1)))
        if not np.isfinite(std) or std <= 0:
            return None
        result = centered / std
        return result if np.isfinite(result).all() else None

    def fit_cox_proportional_hazards(self) -> Dict[str, Any]:
        self.cox_fitted, self.cox_model = False, None
        if not self.has_attrition:
            return {'available': False, 'reason': 'Attrition is required for Cox analysis'}
        if not self.available_covariates:
            return {'available': False, 'reason': 'No numeric covariates are available'}
        try:
            from lifelines import CoxPHFitter
            from lifelines.statistics import proportional_hazard_test
        except ImportError:
            return {'available': False, 'reason': 'lifelines library not installed'}

        raw = self.df[['Tenure', 'Attrition'] + self.available_covariates].copy().replace([np.inf, -np.inf], np.nan).dropna()
        cox_population = {
            'survival_analysis_population': int(len(self.df)),
            'cox_model_population': int(len(raw)),
            'excluded_missing_covariates': int(len(self.df) - len(raw)),
            'coverage': float(len(raw) / len(self.df)) if len(self.df) else 0.0,
            'population_semantics': 'survival_analysis_rows_with_complete_configured_cox_covariates',
        }
        cox_events = int(raw['Attrition'].sum()) if len(raw) else 0
        cox_censored = int(len(raw) - cox_events)
        if len(raw) < MIN_SAMPLE_FOR_COX or cox_events < MIN_EVENTS_FOR_MODEL or cox_censored < MIN_EVENTS_FOR_MODEL:
            return {
                'available': False,
                'reason': 'Insufficient rows or event/censor support for stable Cox estimation',
                'population': cox_population,
            }
        cox = raw[['Tenure', 'Attrition']].copy()
        norm_cols: list[str] = []
        for col in self.available_covariates:
            normalized = self._stable_zscore(raw[col])
            if normalized is not None:
                name = f'{col}_norm'
                cox[name] = normalized
                norm_cols.append(name)
        if not norm_cols:
            return {
                'available': False,
                'reason': 'No variable covariates remain after stable normalization',
                'population': cox_population,
            }
        try:
            from lifelines.exceptions import ConvergenceWarning
            import warnings as pywarnings
            with pywarnings.catch_warnings(record=True) as convergence_warnings:
                pywarnings.simplefilter('always', ConvergenceWarning)
                cph = CoxPHFitter()
                cph.fit(cox[['Tenure', 'Attrition'] + norm_cols], duration_col='Tenure', event_col='Attrition')
            if any(isinstance(item.message, ConvergenceWarning) for item in convergence_warnings):
                return {
                    'available': False,
                    'reason': 'Cox model emitted a convergence warning and was withheld as unstable',
                    'population': cox_population,
                }
            ph_test = proportional_hazard_test(
                cph, cox[['Tenure', 'Attrition'] + norm_cols], time_transform='rank'
            )
            coefficients: Dict[str, Any] = {}
            for norm_col in norm_cols:
                row = cph.summary.loc[norm_col]
                feature = norm_col.removesuffix('_norm')
                required = [row['coef'], row['exp(coef)'], row['p'], row['exp(coef) lower 95%'], row['exp(coef) upper 95%']]
                if not all(_finite_number(v) for v in required):
                    self.cox_fitted, self.cox_model = False, None
                    return {
                        'available': False,
                        'reason': 'Cox model produced non-finite coefficient evidence and was withheld',
                        'population': cox_population,
                    }
                hr = float(row['exp(coef)'])
                p_value = float(row['p'])
                delta = abs(hr - 1) * 100
                direction = 'higher' if hr > 1 else 'lower'
                coefficients[feature] = {
                    'feature': feature,
                    'coefficient': round(float(row['coef']), 4),
                    'hazard_ratio': round(hr, 3),
                    'p_value': round(p_value, 4),
                    'is_significant': bool(p_value < .05),
                    'ci_lower': round(float(row['exp(coef) lower 95%']), 3),
                    'ci_upper': round(float(row['exp(coef) upper 95%']), 3),
                    'direction': 'increases' if hr > 1 else 'decreases',
                    'interpretation': (
                        f'One-standard-deviation difference in {feature} is associated with approximately {delta:.1f}% {direction} relative hazard; this is not a causal effect.'
                        + (' The association is not statistically significant.' if p_value >= .05 else '')
                    ),
                }
            metric_values = [cph.concordance_index_, cph.log_likelihood_, cph.AIC_partial_]
            if not all(_finite_number(v) for v in metric_values):
                return {
                    'available': False,
                    'reason': 'Cox model produced non-finite model-fit evidence and was withheld',
                    'population': cox_population,
                }
            violated = [
                idx.replace('_norm', '')
                for idx, p in ph_test.summary['p'].items()
                if _finite_number(p) and float(p) < .05
            ]
            if violated:
                self.warnings.append(
                    f'Proportional-hazards assumption screening flagged: {", ".join(violated)}. Interpret Cox ratios cautiously.'
                )
            self.cox_model = cph
            self.cox_fitted = True
            return {
                'available': True,
                'coefficients': coefficients,
                'population': cox_population,
                'model_metrics': {
                    'concordance_index': round(float(cph.concordance_index_), 3),
                    'log_likelihood': round(float(cph.log_likelihood_), 2),
                    'aic': round(float(cph.AIC_partial_), 2),
                    'sample_size': len(cox),
                    'events': cox_events,
                    'censored': cox_censored,
                    'quality_interpretation': (
                        f'In-sample concordance is {cph.concordance_index_:.2f}. This is model-fit context, not held-out predictive validation.'
                    ),
                },
                'covariates_used': [c.removesuffix('_norm') for c in norm_cols],
                'recommendations': [
                    'Use significant hazard-ratio associations to form aggregate investigation hypotheses; do not treat them as causes or employee-level action rules.'
                ],
            }
        except Exception as exc:
            logger.warning('Cox model fitting unavailable: %s', exc)
            self.cox_fitted = False
            self.cox_model = None
            return {
                'available': False,
                'reason': f'Cox model could not be fit safely ({type(exc).__name__})',
                'population': cox_population,
            }

    def get_hazard_over_time(self) -> Dict[str, Any]:
        if not self.cox_fitted or self.cox_model is None:
            return {'available': False, 'reason': 'Cox model not available'}
        try:
            hazard = self.cox_model.baseline_hazard_.iloc[:, 0]
            cumulative = self.cox_model.baseline_cumulative_hazard_.iloc[:, 0]
            survival = self.cox_model.baseline_survival_.iloc[:, 0]
            points = []
            for t in hazard.index:
                if float(t) > 15:
                    continue
                values = [t, hazard.loc[t], cumulative.loc[t], survival.loc[t]]
                if not all(_finite_number(v) for v in values):
                    return {'available': False, 'reason': 'Hazard curve produced non-finite evidence and was withheld'}
                points.append({
                    'time_years': round(float(t), 2),
                    'baseline_hazard': round(float(hazard.loc[t]), 4),
                    'cumulative_hazard': round(float(cumulative.loc[t]), 4),
                    'survival': round(float(survival.loc[t]), 4),
                })
            return {
                'available': True,
                'hazard_over_time': points,
                'risk_periods': [],
                'semantics': 'estimated_baseline_hazard_not_employee_specific_risk_period',
            }
        except Exception as exc:
            return {'available': False, 'reason': f'Hazard curve unavailable ({type(exc).__name__})'}

    def generate_cohort_insights(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        frame = self.df.copy()
        parts = []
        filters = filters or {}
        required_columns = {
            'Dept': 'Dept', 'Location': 'Location', 'tenure_min': 'Tenure',
            'tenure_max': 'Tenure', 'years_since_promotion_min': 'YearsSinceLastPromotion',
        }
        if any(key not in required_columns or required_columns[key] not in frame for key in filters):
            return {'cohort_size': 0, 'filters_applied': {}, 'warning': 'Requested cohort filter is unavailable'}
        for key in ('tenure_min', 'tenure_max', 'years_since_promotion_min'):
            if key in filters:
                value = filters[key]
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or value < 0:
                    return {'cohort_size': 0, 'filters_applied': {}, 'warning': 'Cohort durations must be finite non-negative numbers'}
        if filters.get('tenure_min', 0) > filters.get('tenure_max', float('inf')):
            return {'cohort_size': 0, 'filters_applied': {}, 'warning': 'Cohort duration range is reversed'}
        for key in ('Dept', 'Location'):
            if key in filters and key in frame.columns:
                frame = frame[frame[key] == filters[key]]
                parts.append(f'{key}={filters[key]}')
        if 'tenure_min' in filters:
            frame = frame[frame['Tenure'] >= filters['tenure_min']]
            parts.append(f'tenure ≥ {filters["tenure_min"]}y')
        if 'tenure_max' in filters:
            frame = frame[frame['Tenure'] <= filters['tenure_max']]
            parts.append(f'tenure ≤ {filters["tenure_max"]}y')
        if 'years_since_promotion_min' in filters and 'YearsSinceLastPromotion' in frame.columns:
            frame = frame[pd.to_numeric(frame['YearsSinceLastPromotion'], errors='coerce') >= filters['years_since_promotion_min']]
            parts.append(f'years since promotion ≥ {filters["years_since_promotion_min"]}')
        description = 'Employees' + (f" ({', '.join(parts)})" if parts else '')
        result: Dict[str, Any] = {
            'cohort_description': description,
            'cohort_size': len(frame),
            'filters_applied': filters,
        }
        if len(frame) < 10:
            result['warning'] = 'Cohort too small for reliable survival description'
            return result
        if not self.has_attrition:
            result['warning'] = 'Attrition unavailable'
            return result
        events = int(frame['Attrition'].sum())
        censored = int(len(frame) - events)
        result['avg_tenure_years'] = round(float(frame['Tenure'].mean()), 1)
        if events < MIN_PRIVACY_CELL or censored < MIN_PRIVACY_CELL:
            result.update({
                'outcome_counts_suppressed': True,
                'attrition_count': None,
                'attrition_rate': None,
                'survival_probability_3mo': None,
                'survival_probability_6mo': None,
                'survival_probability_12mo': None,
                'median_survival_months': None,
                'median_survival_years': None,
                'warning': 'Outcome cells are below the privacy support floor; event counts, rates and survival estimates are withheld.',
                'key_risk_factors': [],
            })
            return result
        result['outcome_counts_suppressed'] = False
        result['attrition_count'] = events
        result['attrition_rate'] = round(float(frame['Attrition'].mean()), 3)
        try:
            km = self._km_result(frame, 'Cohort')
            for months in (3, 6, 12):
                result[f'survival_probability_{months}mo'] = km.get(f'survival_at_{months}mo')
            result['median_survival_months'] = km.get('median_survival_months')
            result['median_survival_years'] = km.get('median_survival_years')
            p12 = result.get('survival_probability_12mo')
            result['narrative'] = (
                f'{description}. Estimated cohort retention through month 12 from employment origin is {p12 * 100:.1f}%.'
                if p12 is not None else
                f'{description}. Cohort survival estimate available with limited horizon.'
            )
        except Exception:
            result['warning'] = 'Cohort survival estimate unavailable'
        result['key_risk_factors'] = []
        return result

    def predict_survival_probability(self, employee_ids: Optional[List[str]] = None, time_horizon_months: int = 12) -> pd.DataFrame:
        """Individual survival prediction is intentionally disabled by governance."""
        return pd.DataFrame()

    def get_at_risk_employees(self, top_n: int = 20) -> pd.DataFrame:
        return pd.DataFrame()

    def analyze_all(self) -> Dict[str, Any]:
        km = self.fit_kaplan_meier()
        km_dept = self.fit_kaplan_meier(segment_by='Dept')
        cox = self.fit_cox_proportional_hazards()
        hazard = self.get_hazard_over_time()
        cohorts = []
        for filters in ({'Dept': 'Engineering'}, {'Dept': 'Sales'}, {'tenure_min': 2, 'years_since_promotion_min': 1.5}):
            insight = self.generate_cohort_insights(filters)
            if insight.get('cohort_size', 0) >= 10:
                cohorts.append(insight)
        overall = km.get('overall', {}) if km.get('available') else {}
        survival12 = overall.get('survival_at_12mo')
        events = int(self.df['Attrition'].sum()) if self.has_attrition else None
        censored = int(len(self.df) - events) if events is not None else None
        outcome_supported = bool(events is not None and events >= MIN_PRIVACY_CELL and censored is not None and censored >= MIN_PRIVACY_CELL)
        summary = {
            'total_employees': len(self.df),
            'population': self._population_metadata(),
            'attrition_available': self.has_attrition,
            'attrition_count': events if outcome_supported else None,
            'overall_attrition_rate': round(float(self.df['Attrition'].mean()), 3) if self.has_attrition and outcome_supported else None,
            'outcome_counts_suppressed': bool(self.has_attrition and not outcome_supported),
            'cox_model_fitted': self.cox_fitted,
            'covariates_used': self.available_covariates,
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'median_tenure': overall.get('median_survival_years'),
            'avg_12mo_risk': round(1 - survival12, 3) if survival12 is not None and outcome_supported else None,
        }
        recommendations = list(cox.get('recommendations', [])) if cox.get('available') else []
        return {
            'kaplan_meier': km,
            'kaplan_meier_by_dept': km_dept,
            'cox_model': cox,
            'hazard_over_time': hazard,
            'cohort_insights': cohorts,
            'at_risk_employees': [],
            'summary': summary,
            'recommendations': recommendations,
            'warnings': list(dict.fromkeys(self.warnings + [
                'The 12-month compatibility risk field is cumulative cohort attrition through month 12 from employment origin, not an individual next-12-month probability.'
            ])),
        }
