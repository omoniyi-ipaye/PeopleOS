"""Shared ingestion/storage contract for numeric workforce measurements.

Unlisted columns are preserved as categorical/text lexemes. In particular,
numeric-looking departments, locations and identifiers are not measurements.
"""
import pandas as pd


NUMERIC_MEASUREMENT_COLUMNS = frozenset({
    'Salary', 'Tenure', 'Age', 'LastRating', 'StartingSalary', 'InterviewScore',
    'AssessmentScore', 'PriorExperienceYears', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'PromotionCount', 'CompaRatio',
    'ManagerChangeCount', 'PotentialRating', 'eNPS_Score', 'Pulse_Score',
    'ManagerSatisfaction', 'WorkLifeBalance', 'CareerGrowthSatisfaction',
})


def is_numeric_measurement(column: str) -> bool:
    return column in NUMERIC_MEASUREMENT_COLUMNS or column.startswith('InterviewScore_')


def normalize_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert declared measurements only; do not filter rows or impute values."""
    frame = frame.copy()
    for column in frame:
        if is_numeric_measurement(column):
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
    return frame
