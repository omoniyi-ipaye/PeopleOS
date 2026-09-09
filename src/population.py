"""Canonical workforce population and outcome contracts for PeopleOS.

Current-state analytics must operate on one latest observation per employee.
Historical snapshot rows are preserved separately for explicitly longitudinal
analysis. Attrition is normalized to a deterministic binary outcome so every
engine uses the same meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


_TRUE_ATTRITION = {"1", "true", "yes", "y", "left", "departed", "terminated", "resigned", "attrited"}
_FALSE_ATTRITION = {"0", "false", "no", "n", "active", "employed", "retained", "stayed"}


@dataclass(frozen=True)
class PopulationResolution:
    source_rows: int
    current_rows: int
    unique_employees: int
    snapshot_history: bool
    as_of_date: Optional[str]


def normalize_attrition(series: pd.Series) -> pd.Series:
    """Return nullable Int64 Attrition where 1=departed and 0=active.

    Unknown labels remain missing rather than being silently coerced to active.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype("Int64")

    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    valid_numeric = numeric.isin([0, 1])
    result.loc[valid_numeric] = numeric.loc[valid_numeric].astype("Int64")

    unresolved = result.isna() & series.notna()
    if unresolved.any():
        normalized = series.loc[unresolved].astype(str).str.strip().str.lower()
        result.loc[normalized[normalized.isin(_TRUE_ATTRITION)].index] = 1
        result.loc[normalized[normalized.isin(_FALSE_ATTRITION)].index] = 0
    return result


def _reject_conflicting_snapshot_ties(frame: pd.DataFrame, parsed: pd.Series) -> pd.DataFrame:
    """Collapse exact duplicate snapshot rows and reject ambiguous conflicting ties.

    If an employee has multiple different records for the same valid SnapshotDate,
    choosing one based on source-row order would make current-state analytics
    non-deterministic. Exact duplicates are harmless and may be collapsed; conflicting
    duplicates fail closed so the source data can be corrected explicitly.
    """
    working = frame.assign(_parsed_snapshot=parsed)
    valid = working['_parsed_snapshot'].notna()
    if not valid.any():
        return frame

    key_cols = ['EmployeeID', '_parsed_snapshot']
    duplicate_key = working.loc[valid].duplicated(subset=key_cols, keep=False)
    if duplicate_key.any():
        duplicate_rows = working.loc[valid].loc[duplicate_key]
        value_cols = [column for column in frame.columns if column != 'SnapshotDate']
        conflicts: list[tuple[str, str]] = []
        for (employee_id, snapshot), group in duplicate_rows.groupby(key_cols, dropna=False, sort=False):
            comparable = group[value_cols].copy()
            # Treat matching missing values as equal and compare row content independent of source order.
            normalized = comparable.astype('string').fillna('<NA>')
            if len(normalized.drop_duplicates()) > 1:
                conflicts.append((str(employee_id), pd.Timestamp(snapshot).isoformat()))
        if conflicts:
            preview = ', '.join(f'{employee}@{snapshot}' for employee, snapshot in conflicts[:5])
            raise ValueError(
                'Conflicting employee records share the same SnapshotDate; current state is ambiguous. '
                f'Correct the source data before analysis. Conflicts: {preview}'
            )

    # Exact duplicate rows do not carry additional information and can be removed
    # before current-state resolution. Keep the first only for byte-identical content.
    return frame.drop_duplicates().copy()


def resolve_current_population(df: pd.DataFrame) -> tuple[pd.DataFrame, PopulationResolution]:
    """Resolve one current row per employee, using latest SnapshotDate when present."""
    frame = df.copy()
    source_rows = len(frame)
    snapshot_history = "SnapshotDate" in frame.columns
    as_of_date: Optional[str] = None

    if snapshot_history:
        parsed = pd.to_datetime(frame["SnapshotDate"], errors="coerce", utc=True)
        frame = _reject_conflicting_snapshot_ties(frame, parsed)
        parsed = pd.to_datetime(frame["SnapshotDate"], errors="coerce", utc=True)
        frame["SnapshotDate"] = parsed
        if parsed.notna().any():
            as_of_date = parsed.max().date().isoformat()
        # Stable ordering means invalid/missing dates remain older than valid dates.
        frame = frame.assign(_snapshot_sort=parsed)
        frame = frame.sort_values(["EmployeeID", "_snapshot_sort"], na_position="first", kind="stable")
        frame = frame.drop_duplicates(subset=["EmployeeID"], keep="last").drop(columns=["_snapshot_sort"])
    elif "EmployeeID" in frame.columns:
        frame = frame.drop_duplicates(subset=["EmployeeID"], keep="last")

    if "Attrition" in frame.columns:
        frame["Attrition"] = normalize_attrition(frame["Attrition"])

    resolution = PopulationResolution(
        source_rows=source_rows,
        current_rows=len(frame),
        unique_employees=int(frame["EmployeeID"].nunique()) if "EmployeeID" in frame.columns else len(frame),
        snapshot_history=snapshot_history,
        as_of_date=as_of_date,
    )
    return frame.reset_index(drop=True), resolution


def active_population(df: pd.DataFrame) -> pd.DataFrame:
    """Return the current active population; unknown Attrition is excluded when present."""
    current, _ = resolve_current_population(df)
    if "Attrition" not in current.columns:
        return current
    return current[current["Attrition"] == 0].copy().reset_index(drop=True)


def observed_attrition_share(df: pd.DataFrame) -> Optional[float]:
    """Share of current rows whose observed Attrition outcome is 1.

    This is intentionally not called a period turnover rate. A true turnover
    rate requires a defined observation period and denominator/exposure model.
    """
    current, _ = resolve_current_population(df)
    if "Attrition" not in current.columns:
        return None
    known = current["Attrition"].dropna()
    if known.empty:
        return None
    return float(known.mean())