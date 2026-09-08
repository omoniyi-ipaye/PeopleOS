"""Geographic counts reconcile to the active workspace population."""
from typing import Any
import pandas as pd
from fastapi import APIRouter, Depends
from api.dependencies import AppState
from api.routes.analytics import require_data
from src.population import active_population

router = APIRouter(prefix='/api/geo', tags=['Geographic'])


def _distribution(state: AppState) -> list[dict[str, Any]]:
    frame = active_population(state.raw_df)
    countries = frame.get('Country', pd.Series('Unknown', index=frame.index)).astype('string').str.strip()
    countries = countries.fillna('Unknown').replace({'': 'Unknown', 'unknown': 'Unknown', 'remote': 'Remote'})
    counts = countries.value_counts()
    return [{'country': str(country), 'count': int(count),
             'percentage': round(count / len(frame) * 100, 1),
             'mapped': country not in {'Unknown', 'Remote'},
             'population': 'current_active_workspace_employees'} for country, count in counts.items()]


@router.get('/distribution')
async def get_geo_distribution(state: AppState = Depends(require_data)):
    # Unknown and remote records stay visible so the denominator reconciles.
    return _distribution(state)


@router.get('/summary')
async def get_geo_summary(state: AppState = Depends(require_data)):
    rows = _distribution(state)
    total = sum(row['count'] for row in rows)
    remote = sum(row['count'] for row in rows if row['country'] == 'Remote')
    unknown = sum(row['count'] for row in rows if row['country'] == 'Unknown')
    return {'total_employees': total, 'countries_represented': sum(row['mapped'] for row in rows),
            'remote_workers': remote, 'unknown_country_count': unknown,
            'remote_percentage': round(remote / total * 100, 1) if total else 0,
            'semantics': 'Country field labels only; Remote is not inferred from other location fields.'}
