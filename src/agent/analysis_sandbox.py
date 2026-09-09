"""Governed downstream analysis for PeopleOS.

This is deliberately not a Python/shell sandbox. AI may request one of a small
set of typed analytical operations; deterministic code validates the columns,
population and minimum support before returning aggregate results.
"""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from scipy.stats import pearsonr

from src.population import active_population, resolve_current_population
from src.serialization import json_safe

MIN_GROUP_SIZE = 5
MIN_CORRELATION_PAIRS = 10
MAX_GROUPS = 20

_BLOCKED_COLUMN = re.compile(
    r"(?:^|_)(employee_?id|manager_?id|name|first_?name|last_?name|email|phone|address|"
    r"ssn|social_?security|passport|national_?id|nie|dni)(?:$|_)", re.I
)


class AnalysisSpec(BaseModel):
    operation: Literal['group_summary', 'compare_groups', 'correlation', 'crosstab']
    population: Literal['active', 'current'] = 'active'
    group_by: Optional[str] = None
    second_group_by: Optional[str] = None
    measure: Optional[str] = None
    second_measure: Optional[str] = None
    statistic: Literal['count', 'mean', 'median', 'sum', 'rate'] = 'count'
    group_a: Optional[str] = None
    group_b: Optional[str] = None
    max_groups: int = Field(default=12, ge=2, le=MAX_GROUPS)

    @model_validator(mode='after')
    def validate_shape(self):
        if self.operation == 'group_summary' and not self.group_by:
            raise ValueError('group_summary requires group_by')
        if self.operation == 'compare_groups' and not (self.group_by and self.measure and self.group_a is not None and self.group_b is not None):
            raise ValueError('compare_groups requires group_by, measure, group_a and group_b')
        if self.operation == 'correlation' and not (self.measure and self.second_measure):
            raise ValueError('correlation requires measure and second_measure')
        if self.operation == 'crosstab' and not (self.group_by and self.second_group_by):
            raise ValueError('crosstab requires group_by and second_group_by')
        if self.statistic != 'count' and self.operation == 'group_summary' and not self.measure:
            raise ValueError('this group summary statistic requires measure')
        return self


class GovernedAnalysisSandbox:
    def __init__(self, frame: pd.DataFrame):
        self.source = frame.copy()

    def run(self, spec: AnalysisSpec) -> dict[str, Any]:
        frame = self._population(spec.population)
        if frame.empty:
            return self._unavailable(spec, 'The requested population is empty.')
        for column in [spec.group_by, spec.second_group_by, spec.measure, spec.second_measure]:
            if column is not None:
                self._validate_column(frame, column)
        if spec.operation == 'group_summary':
            return self._group_summary(frame, spec)
        if spec.operation == 'compare_groups':
            return self._compare_groups(frame, spec)
        if spec.operation == 'correlation':
            return self._correlation(frame, spec)
        return self._crosstab(frame, spec)

    def _population(self, population: str) -> pd.DataFrame:
        if population == 'active':
            return active_population(self.source)
        current, _ = resolve_current_population(self.source)
        return current

    @staticmethod
    def _validate_column(frame: pd.DataFrame, column: str) -> None:
        if column not in frame.columns:
            raise ValueError(f"Unknown analysis column: {column}")
        normalized = re.sub(r'[^a-z0-9]+', '_', column.lower()).strip('_')
        if _BLOCKED_COLUMN.search(normalized):
            raise ValueError(f"Identifier-like column is not available for downstream analysis: {column}")

    @staticmethod
    def _numeric(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
        return numeric

    def _group_summary(self, frame: pd.DataFrame, spec: AnalysisSpec) -> dict[str, Any]:
        group = frame[spec.group_by].fillna('Unknown').astype(str).str.strip().replace('', 'Unknown')
        working = frame.assign(_group=group)
        rows = []
        suppressed = 0
        for label, part in working.groupby('_group', dropna=False):
            eligible = len(part)
            if eligible < MIN_GROUP_SIZE:
                suppressed += 1
                continue
            if spec.statistic == 'count':
                value, measured, excluded = eligible, eligible, 0
            else:
                values = self._numeric(part[spec.measure])
                valid = values.dropna()
                measured, excluded = len(valid), eligible - len(valid)
                if measured < MIN_GROUP_SIZE:
                    suppressed += 1
                    continue
                if spec.statistic == 'mean': value = float(valid.mean())
                elif spec.statistic == 'median': value = float(valid.median())
                elif spec.statistic == 'sum': value = float(valid.sum())
                else:
                    unique = set(valid.unique().tolist())
                    if not unique <= {0, 1}:
                        raise ValueError('rate requires a binary 0/1 measure')
                    value = float(valid.mean())
            rows.append({'group': str(label), 'value': value, 'measured_count': measured, 'eligible_count': eligible, 'excluded_count': excluded})
        rows.sort(key=lambda row: (-row['eligible_count'], row['group']))
        rows = rows[:spec.max_groups]
        if not rows:
            return self._unavailable(spec, f'No groups meet the minimum support of {MIN_GROUP_SIZE}.')
        return self._result(spec, {'groups': rows, 'suppressed_groups': suppressed, 'minimum_group_size': MIN_GROUP_SIZE}, len(frame))

    def _compare_groups(self, frame: pd.DataFrame, spec: AnalysisSpec) -> dict[str, Any]:
        labels = frame[spec.group_by].fillna('Unknown').astype(str).str.strip().replace('', 'Unknown')
        values = self._numeric(frame[spec.measure])
        results = []
        for wanted in [spec.group_a, spec.group_b]:
            mask = labels == str(wanted)
            eligible = int(mask.sum())
            valid = values[mask].dropna()
            if eligible < MIN_GROUP_SIZE or len(valid) < MIN_GROUP_SIZE:
                return self._unavailable(spec, f'Each compared group needs at least {MIN_GROUP_SIZE} measured records.')
            value = float(valid.median()) if spec.statistic == 'median' else float(valid.mean())
            results.append({'group': str(wanted), 'value': value, 'measured_count': len(valid), 'eligible_count': eligible, 'excluded_count': eligible - len(valid)})
        return self._result(spec, {'groups': results, 'difference_b_minus_a': results[1]['value'] - results[0]['value'], 'minimum_group_size': MIN_GROUP_SIZE}, len(frame))

    def _correlation(self, frame: pd.DataFrame, spec: AnalysisSpec) -> dict[str, Any]:
        left = self._numeric(frame[spec.measure])
        right = self._numeric(frame[spec.second_measure])
        paired = pd.DataFrame({'left': left, 'right': right}).dropna()
        n = len(paired)
        if n < MIN_CORRELATION_PAIRS:
            return self._unavailable(spec, f'Correlation needs at least {MIN_CORRELATION_PAIRS} valid pairs.')
        if paired.left.nunique() < 2 or paired.right.nunique() < 2:
            return self._unavailable(spec, 'Correlation is undefined for a constant measure.')
        correlation, p_value = pearsonr(paired.left, paired.right)
        if not np.isfinite(correlation) or not np.isfinite(p_value):
            return self._unavailable(spec, 'Correlation could not be estimated reliably.')
        return self._result(spec, {'correlation': float(correlation), 'p_value': float(p_value), 'paired_observations': n, 'excluded_count': len(frame) - n, 'minimum_pairs': MIN_CORRELATION_PAIRS, 'causal': False}, len(frame))

    def _crosstab(self, frame: pd.DataFrame, spec: AnalysisSpec) -> dict[str, Any]:
        left = frame[spec.group_by].fillna('Unknown').astype(str).str.strip().replace('', 'Unknown')
        right = frame[spec.second_group_by].fillna('Unknown').astype(str).str.strip().replace('', 'Unknown')
        counts = pd.crosstab(left, right, dropna=False)
        row_order = counts.sum(axis=1).sort_values(ascending=False).head(spec.max_groups).index
        col_order = counts.sum(axis=0).sort_values(ascending=False).head(spec.max_groups).index
        cells = []
        suppressed = 0
        for row in row_order:
            for column in col_order:
                count = int(counts.loc[row, column])
                if 0 < count < MIN_GROUP_SIZE:
                    suppressed += 1
                    value = None
                else:
                    value = count
                cells.append({'row': str(row), 'column': str(column), 'count': value})
        return self._result(spec, {'cells': cells, 'suppressed_cells': suppressed, 'minimum_cell_size': MIN_GROUP_SIZE}, len(frame))

    @staticmethod
    def _result(spec: AnalysisSpec, output: dict[str, Any], population_count: int) -> dict[str, Any]:
        return json_safe({'available': True, 'operation': spec.operation, 'population': spec.population, 'population_count': population_count, 'spec': spec.model_dump(), 'output': output, 'semantics': 'Deterministic derived aggregate; not a causal conclusion or employment recommendation.'})

    @staticmethod
    def _unavailable(spec: AnalysisSpec, reason: str) -> dict[str, Any]:
        return {'available': False, 'operation': spec.operation, 'population': spec.population, 'spec': spec.model_dump(), 'reason': reason}
