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


def resolve_current_population(df: pd.DataFrame) -> tuple[pd.DataFrame, PopulationResolution]:
    """Resolve one current row per employee, using latest SnapshotDate when present."""
    frame = df.copy()
    source_rows = len(frame)
    snapshot_history = "SnapshotDate" in frame.columns
    as_of_date: Optional[str] = None

    if snapshot_history:
        parsed = pd.to_datetime(frame["SnapshotDate"], errors="coerce", utc=True)
        frame["SnapshotDate"] = parsed
        if parsed.notna().any():
            as_of_date = parsed.max().date().isoformat()
        # Stable ordering means invalid/missing dates remain older than valid dates.
        frame = frame.assign(_snapshot_sort=parsed)
        frame = frame.sort_values(["EmployeeID", "_snapshot_sort"], na_position="first")
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
