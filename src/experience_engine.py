"""Measured employee-experience analytics with aggregate-first safety semantics.

PeopleOS computes a configurable 0-100 composite only from explicit measured
experience responses. HRIS proxy fields do not become sentiment. Individual
experience ranking, manager ranking and undersized cohort/cell disclosure are
blocked at the engine boundary as well as at the API boundary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.population import resolve_current_population
from src.utils import load_config


class ExperienceEngineError(Exception):
    pass


MIN_AGGREGATE_SUPPORT = 10


def _identifier_like(name: str) -> bool:
    token = ''.join(ch.lower() for ch in str(name) if ch.isalnum() or ch == '_')
    return token == 'id' or token.endswith('id') or '_id' in token or 'identifier' in token


def _finite_range(value: Any, *, name: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite and between {low} and {high}') from exc
    if not np.isfinite(number) or number < low or number > high:
        raise ValueError(f'{name} must be finite and between {low} and {high}')
    return number


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a positive integer')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if not np.isfinite(number) or number < 1 or not number.is_integer():
        raise ValueError(f'{name} must be a positive integer')
    return int(number)


class ExperienceEngine:
    EXPERIENCE_COLUMNS = {
        'enps_score': 'eNPS',
        'onboarding_30d': 'Onboarding 30-day',
        'onboarding_60d': 'Onboarding 60-day',
        'onboarding_90d': 'Onboarding 90-day',
        'pulse_score': 'Pulse Survey',
        'engagementscore': 'Engagement',
        'managersatisfaction': 'Manager Satisfaction',
        'worklifebalance': 'Work-Life Balance',
        'careergrowthsatisfaction': 'Career Growth',
    }

    SEGMENTS = {
        'Very high score band': (80.0, 100.0),
        'High score band': (60.0, 80.0),
        'Mid score band': (40.0, 60.0),
        'Low score band': (20.0, 40.0),
        'Very low score band': (0.0, 20.0),
    }

    def __init__(self, df: pd.DataFrame):
        self.df, self.population_resolution = resolve_current_population(df)
        self.config = load_config()
        self.exp_config = self.config.get('experience', {})
        self.logger = get_logger('experience_engine')
        self.warnings: List[str] = []
        self._column_map = self._build_column_map()
        self._detect_available_signals()
        self._validate_data()
        self._compute_experience_index()

    def _build_column_map(self) -> Dict[str, str]:
        buckets: Dict[str, List[str]] = {}
        for col in self.df.columns:
            buckets.setdefault(str(col).lower(), []).append(str(col))
        collisions = {key: names for key, names in buckets.items() if len(names) > 1}
        if collisions:
            detail = ', '.join(f'{key}: {names}' for key, names in sorted(collisions.items()))
            raise ExperienceEngineError(f'Ambiguous case-insensitive column names: {detail}')
        return {key: names[0] for key, names in buckets.items()}

    def _get_column(self, lowercase_name: str) -> Optional[str]:
        return self._column_map.get(lowercase_name)

    def _detect_available_signals(self) -> None:
        cols = set(self._column_map)
        self.has_enps = 'enps_score' in cols
        self.has_onboarding_30d = 'onboarding_30d' in cols
        self.has_onboarding_60d = 'onboarding_60d' in cols
        self.has_onboarding_90d = 'onboarding_90d' in cols
        self.has_onboarding = any((self.has_onboarding_30d, self.has_onboarding_60d, self.has_onboarding_90d))
        self.has_pulse = 'pulse_score' in cols
        self.has_manager_satisfaction = 'managersatisfaction' in cols
        self.has_engagement = 'engagementscore' in cols
        self.has_work_life = 'worklifebalance' in cols
        self.has_career_growth = 'careergrowthsatisfaction' in cols
        self.available_survey_signals = sum((
            self.has_enps, self.has_onboarding, self.has_pulse,
            self.has_manager_satisfaction, self.has_engagement,
            self.has_work_life, self.has_career_growth,
        ))
        self.has_tenure = 'tenure' in cols
        self.has_salary = 'salary' in cols
        self.has_rating = 'lastrating' in cols
        self.has_promotion_data = 'yearssincelastpromotion' in cols
        if self.available_survey_signals == 0:
            self.warnings.append('No experience survey columns found. Measured experience is unavailable.')

    def _validate_data(self) -> None:
        if 'EmployeeID' not in self.df.columns:
            raise ExperienceEngineError("Missing required columns: ['EmployeeID']")
        if len(self.df) < MIN_AGGREGATE_SUPPORT:
            self.warnings.append(f'Small dataset ({len(self.df)} employees). Aggregate results may be unstable.')

    def _compute_experience_index(self) -> None:
        weights = self.exp_config.get('index_weights', {})
        definitions = {
            'enps': (['enps_score'], [0, 10], 'enps', .25),
            'onboarding': (['onboarding_30d', 'onboarding_60d', 'onboarding_90d'], [1, 5], 'onboarding', .15),
            'pulse': (['pulse_score'], [1, 5], 'pulse', .15),
            'manager': (['managersatisfaction'], [1, 5], 'manager_satisfaction', .15),
            'engagement': (['engagementscore'], None, 'engagement', .10),
            'work_life': (['worklifebalance'], [1, 5], 'work_life', .10),
            'career': (['careergrowthsatisfaction'], [1, 5], 'career_growth', .10),
        }
        scales = self.exp_config.get('signal_scales', {})
        weighted = pd.Series(0.0, index=self.df.index)
        denominator = pd.Series(0.0, index=self.df.index)
        components = pd.DataFrame(index=self.df.index)
        self.signal_observation_counts: Dict[str, int] = {}
        for name, (columns, default_scale, weight_key, default_weight) in definitions.items():
            weight = float(weights.get(weight_key, default_weight))
            if not np.isfinite(weight) or weight < 0:
                raise ExperienceEngineError('Experience weights must be finite and non-negative')
            normalized: List[pd.Series] = []
            for column in columns:
                actual = self._get_column(column)
                if actual is None:
                    continue
                scale = scales.get(column, default_scale)
                if scale is None:
                    self.warnings.append(f'{actual} excluded: configure its signal_scales minimum and maximum.')
                    continue
                try:
                    low, high = map(float, scale)
                except (TypeError, ValueError) as exc:
                    raise ExperienceEngineError(f'Invalid signal scale for {actual}') from exc
                if not np.isfinite([low, high]).all() or low >= high:
                    raise ExperienceEngineError(f'Invalid signal scale for {actual}')
                values = pd.to_numeric(self.df[actual], errors='coerce')
                valid = values.between(low, high) & np.isfinite(values)
                normalized.append((values.where(valid) - low) / (high - low) * 100)
            if normalized and weight > 0:
                component = pd.concat(normalized, axis=1).mean(axis=1)
                components[name] = component
                self.signal_observation_counts[name] = int(component.notna().sum())
                weighted += component.fillna(0) * weight
                denominator += component.notna() * weight
        self.df['_exi_score'] = (weighted / denominator.replace(0, np.nan)).round(1)
        self.df['_exi_components'] = [str(row.dropna().to_dict()) for _, row in components.iterrows()]
        self.respondent_count = int(self.df['_exi_score'].notna().sum())
        self.warnings.append('Composite scores are descriptive; signal coverage and configured scales must accompany comparisons.')

    def _interpret_exi(self, exi: float) -> str:
        if exi >= 80:
            return 'Very high configured experience score band'
        if exi >= 60:
            return 'High configured experience score band'
        if exi >= 40:
            return 'Mid configured experience score band'
        if exi >= 20:
            return 'Low configured experience score band'
        return 'Very low configured experience score band'

    def _segment_mask(self, values: pd.Series, low: float, high: float) -> pd.Series:
        if high == 100.0:
            return (values >= low) & (values <= high)
        return (values >= low) & (values < high)

    def _get_segment(self, exi: float) -> str:
        for name, (low, high) in self.SEGMENTS.items():
            if exi >= low and (exi <= high if high == 100.0 else exi < high):
                return name
        return 'Unknown'

    def calculate_experience_index(self, group_by: Optional[str] = None) -> Dict[str, Any]:
        measured = self.df['_exi_score'].dropna()
        if measured.empty:
            return {'available': False, 'reason': 'EXI not computed'}
        result: Dict[str, Any] = {
            'available': True,
            'overall_exi': round(float(measured.mean()), 1),
            'exi_std': round(float(measured.std(ddof=1)), 1) if len(measured) > 1 else None,
            'exi_median': round(float(measured.median()), 1),
            'total_employees': int(len(self.df)),
            'signals_available': int(self.available_survey_signals),
            'respondent_count': int(self.respondent_count),
            'response_coverage': float(self.respondent_count / len(self.df)) if len(self.df) else 0.0,
            'interpretation': 'Configured weighted composite of available measured experience signals.',
            'benchmark': None,
        }
        if group_by and group_by in self.df.columns and not _identifier_like(group_by):
            groups: List[Dict[str, Any]] = []
            for group_name, group in self.df.groupby(group_by, dropna=False):
                respondents = group['_exi_score'].dropna()
                if len(respondents) < MIN_AGGREGATE_SUPPORT:
                    continue
                groups.append({
                    'group': 'Unknown' if pd.isna(group_name) else str(group_name),
                    'exi': round(float(respondents.mean()), 1),
                    'count': int(len(respondents)),
                    'interpretation': 'Group mean of the configured measured-signal composite.',
                })
            groups.sort(key=lambda row: row['exi'], reverse=True)
            result['by_group'] = groups
        return result

    def get_employee_exi(self, employee_id: str) -> Dict[str, Any]:
        emp = self.df[self.df['EmployeeID'] == employee_id]
        if emp.empty:
            return {'available': False, 'reason': f'Employee {employee_id} not found'}
        row = emp.iloc[0]
        exi = row.get('_exi_score')
        if pd.isna(exi):
            return {'available': False, 'reason': 'No valid measured response for this employee'}
        result = {
            'available': True,
            'EmployeeID': employee_id,
            'exi_score': round(float(exi), 1),
            'segment': self._get_segment(float(exi)),
            'interpretation': self._interpret_exi(float(exi)),
            'dept': row.get('Dept'),
        }
        raw = row.get('_exi_components')
        if raw and raw != 'None':
            try:
                import ast
                result['components'] = ast.literal_eval(raw)
            except Exception:
                pass
        return result

    def _apply_segment_suppression(self, rows: List[Dict[str, Any]]) -> bool:
        primary = [i for i, row in enumerate(rows) if 0 < int(row['_raw_count']) < MIN_AGGREGATE_SUPPORT]
        suppress = set(primary)
        if len(primary) == 1:
            candidates = [(int(row['_raw_count']), i) for i, row in enumerate(rows) if i not in suppress and int(row['_raw_count']) > 0]
            if candidates:
                suppress.add(min(candidates)[1])
        for i, row in enumerate(rows):
            row['suppressed'] = i in suppress
            if row['suppressed']:
                row['count'] = None
                row['percentage'] = None
                row['avg_exi'] = None
            row.pop('_raw_count', None)
        return bool(suppress)

    def get_engagement_segments(self) -> Dict[str, Any]:
        measured = self.df['_exi_score'].dropna()
        if measured.empty:
            return {'available': False, 'reason': 'EXI not computed'}
        total = int(len(measured))
        rows: List[Dict[str, Any]] = []
        for name, (low, high) in self.SEGMENTS.items():
            values = measured[self._segment_mask(measured, low, high)]
            count = int(len(values))
            rows.append({'segment': name, '_raw_count': count, 'count': count, 'percentage': round(count / total * 100, 1) if total else 0.0, 'avg_exi': round(float(values.mean()), 1) if count else None, 'exi_range': f'{low:g}-{high:g}', 'suppressed': False})
        suppression_applied = self._apply_segment_suppression(rows)
        if suppression_applied:
            high_pct = low_pct = None
        else:
            high_pct = round(sum(float(row['percentage'] or 0) for row in rows if row['segment'] in {'High score band', 'Very high score band'}), 1)
            low_pct = round(sum(float(row['percentage'] or 0) for row in rows if row['segment'] in {'Low score band', 'Very low score band'}), 1)
        return {'available': True, 'segments': rows, 'total_employees': total, 'health_indicator': 'Measured score distribution', 'thriving_percentage': high_pct, 'at_risk_percentage': low_pct, 'suppression_applied': suppression_applied, 'recommendations': ['Treat score bands as descriptive aggregate monitoring, not diagnoses of employee engagement.']}

    def identify_experience_drivers(self) -> Dict[str, Any]:
        if not self.df['_exi_score'].notna().any():
            return {'available': False, 'reason': 'EXI not computed'}
        experience_cols = {c.lower() for c in self.df.columns if c.lower() in self.EXPERIENCE_COLUMNS}
        drivers: List[Dict[str, Any]] = []
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if col == '_exi_score' or col.lower() in experience_cols or _identifier_like(col):
                continue
            pairs = self.df[['_exi_score', col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pairs) < MIN_AGGREGATE_SUPPORT or (pairs.nunique() < 2).any():
                continue
            corr = float(pairs['_exi_score'].corr(pairs[col]))
            if not np.isfinite(corr):
                continue
            drivers.append({'factor': col, 'sample_size': int(len(pairs)), 'metric_semantics': 'observational_association_excluding_index_components', 'correlation': round(corr, 3), 'impact': 'High' if abs(corr) >= .4 else 'Medium' if abs(corr) >= .2 else 'Low', 'direction': 'Positive' if corr > 0 else 'Negative' if corr < 0 else 'None'})
        drivers.sort(key=lambda row: abs(row['correlation']), reverse=True)
        positive = [d for d in drivers if d['correlation'] > .1][:3]
        negative = [d for d in drivers if d['correlation'] < -.1][:3]
        return {'available': True, 'drivers': drivers[:10], 'top_positive_drivers': positive, 'top_negative_drivers': negative, 'recommendations': ['Observed correlations are not causal drivers; validate confounding and stability before intervention.']}

    def _get_at_risk_by_dept(self, low_score_df: pd.DataFrame) -> List[Dict[str, Any]]:
        if 'Dept' not in low_score_df.columns or low_score_df.empty:
            return []
        rows = []
        for dept, group in low_score_df.groupby('Dept', dropna=False):
            if len(group) < MIN_AGGREGATE_SUPPORT:
                continue
            rows.append({'department': 'Unknown' if pd.isna(dept) else str(dept), 'at_risk_count': int(len(group))})
        rows.sort(key=lambda row: row['at_risk_count'], reverse=True)
        return rows[:10]

    def get_at_risk_employees(self, threshold: Optional[float] = None, limit: int = 20) -> Dict[str, Any]:
        if not self.df['_exi_score'].notna().any():
            return {'available': False, 'reason': 'EXI not computed'}
        threshold = self.exp_config.get('thresholds', {}).get('at_risk_exi', 40) if threshold is None else threshold
        threshold = _finite_range(threshold, name='threshold', low=0, high=100)
        _positive_integer(limit, name='limit')
        low = self.df[self.df['_exi_score'] < threshold].copy()
        low_count = int(len(low))
        suppressed = 0 < low_count < MIN_AGGREGATE_SUPPORT
        return {'available': True, 'total_at_risk': None if suppressed else low_count, 'threshold_used': threshold, 'employees': None, 'by_department': [] if suppressed else self._get_at_risk_by_dept(low), 'suppressed': suppressed, 'metric_semantics': 'aggregate_count_below_configured_composite_threshold_not_employee_risk_prediction'}

    def get_lifecycle_experience(self) -> Dict[str, Any]:
        if not self.df['_exi_score'].notna().any():
            return {'available': False, 'reason': 'EXI not computed'}
        if not self.has_tenure:
            return {'available': False, 'reason': 'Tenure column required for lifecycle analysis'}
        cfg = self.exp_config.get('lifecycle_stages', {})
        new_hire = float(cfg.get('new_hire_months', 6)); ramping = float(cfg.get('ramping_months', 12)); established = float(cfg.get('established_months', 36))
        if not np.isfinite([new_hire, ramping, established]).all() or not (0 <= new_hire <= ramping <= established):
            raise ExperienceEngineError('Lifecycle stage thresholds must be finite, ordered, and non-negative')
        tenure = pd.to_numeric(self.df[self._get_column('tenure')], errors='coerce'); months = tenure * 12
        stage = pd.Series('Unknown', index=self.df.index, dtype='object'); valid = tenure.notna() & np.isfinite(tenure) & (tenure >= 0)
        stage.loc[valid & (months < new_hire)] = 'New Hire'; stage.loc[valid & (months >= new_hire) & (months < ramping)] = 'Ramping'; stage.loc[valid & (months >= ramping) & (months < established)] = 'Established'; stage.loc[valid & (months >= established)] = 'Veteran'
        stages = []
        for name in ['New Hire', 'Ramping', 'Established', 'Veteran', 'Unknown']:
            group = self.df[stage == name]; respondents = group['_exi_score'].dropna()
            if len(respondents) < MIN_AGGREGATE_SUPPORT:
                continue
            low_count = int((respondents < 40).sum()); low_suppressed = 0 < low_count < MIN_AGGREGATE_SUPPORT
            stages.append({'stage': name, 'count': int(len(group)), 'avg_exi': round(float(respondents.mean()), 1), 'respondent_count': int(len(respondents)), 'at_risk_count': None if low_suppressed else low_count, 'at_risk_suppressed': low_suppressed})
        return {'available': True, 'stages': stages, 'concerns': [], 'recommendations': ['Lifecycle differences are cross-sectional descriptive comparisons, not longitudinal stage effects.']}

    def analyze_manager_impact(self) -> Dict[str, Any]:
        return {'available': False, 'reason': 'Manager-level experience ranking is disabled at the aggregate engine boundary.', 'managers_analyzed': 0, 'overall_avg_exi': None, 'managers_below_average': 0, 'bottom_managers': None, 'top_managers': None, 'recommendations': []}

    def get_available_signals(self) -> Dict[str, Any]:
        coverage = round(self.respondent_count / len(self.df) * 100, 1) if len(self.df) else 0.0
        recommendations = []
        if not self.has_enps: recommendations.append('Add eNPS_Score column to enable measured advocacy tracking')
        if not self.has_pulse: recommendations.append('Add Pulse_Score column for measured pulse responses')
        if not self.has_manager_satisfaction: recommendations.append('Add ManagerSatisfaction column to measure manager experience')
        if coverage < 50: recommendations.append('Increase survey response coverage before drawing broad workforce conclusions')
        return {'has_enps': self.has_enps, 'has_onboarding': self.has_onboarding, 'has_pulse': self.has_pulse, 'has_manager_satisfaction': self.has_manager_satisfaction, 'has_engagement': self.has_engagement, 'has_work_life': self.has_work_life, 'has_career_growth': self.has_career_growth, 'total_signals': int(self.available_survey_signals), 'coverage_percentage': coverage, 'recommendations': recommendations}

    def analyze_all(self) -> Dict[str, Any]:
        index = self.calculate_experience_index(); segments = self.get_engagement_segments(); drivers = self.identify_experience_drivers(); low_score = self.get_at_risk_employees(); lifecycle = self.get_lifecycle_experience(); manager = self.analyze_manager_impact(); signals = self.get_available_signals()
        recommendations = []
        for section in (segments, drivers, lifecycle): recommendations.extend(section.get('recommendations', []) or [])
        recommendations = list(dict.fromkeys(recommendations))
        return {'experience_index': index, 'segments': segments, 'drivers': drivers, 'at_risk': low_score, 'lifecycle': lifecycle, 'manager_impact': manager, 'signals': signals, 'summary': {'overall_exi': index.get('overall_exi'), 'health_indicator': segments.get('health_indicator', 'Unavailable'), 'total_employees': int(len(self.df)), 'at_risk_count': None if low_score.get('suppressed') else low_score.get('total_at_risk'), 'signals_available': int(self.available_survey_signals), 'total_warnings': len(self.warnings), 'total_recommendations': len(recommendations)}, 'recommendations': recommendations, 'warnings': list(dict.fromkeys(self.warnings))}
