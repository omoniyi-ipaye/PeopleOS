"""Known-answer API checks for explicit salary units and durable restoration."""
import pandas as pd
import pytest

from tests.test_runtime_integrity import runtime, roster


def send(runtime, frame, **declarations):
    return runtime.client.post('/api/upload', data=declarations,
        files={'file': ('pay.csv', frame.to_csv(index=False).encode(), 'text/csv')})


def test_undeclared_pay_preserves_population_but_never_becomes_analytic_evidence(runtime):
    response = send(runtime, roster())
    assert response.status_code == 200, response.text
    assert response.json()['rows_loaded'] == 20
    assert response.json()['features_enabled']['compensation'] is False
    assert runtime.state.raw_df.Salary.isna().all()
    assert runtime.state.historical_df.Salary.eq(60000).all()
    summary = runtime.client.get('/api/analytics/summary').json()
    assert summary['active_count'] == 18
    assert runtime.state.runtime_provenance['pay_basis_confirmed'] is False


def test_confirmed_pay_is_persisted_and_survives_dataset_restore(runtime):
    response = send(runtime, roster(), salary_basis='annual', salary_currency='eur')
    assert response.status_code == 200, response.text
    dataset_id = response.json()['dataset_id']
    assert response.json()['features_enabled']['compensation'] is True
    assert runtime.state.raw_df.Salary.mean() == 60000
    assert runtime.state.historical_df.PayPeriod.eq('annual').all()
    assert runtime.state.historical_df.Currency.eq('EUR').all()
    assert send(runtime, roster('B')).status_code == 200
    restored = runtime.client.post(f'/api/platform/workspaces/local/datasets/{dataset_id}/activate')
    assert restored.status_code == 200, restored.text
    assert runtime.state.raw_df.Salary.mean() == 60000
    assert runtime.state.features_enabled['compensation'] is True
    assert runtime.client.get('/api/upload/status').json()['reporting_currency'] == 'EUR'


@pytest.mark.parametrize('declarations', [{'salary_basis': 'annual'}, {'salary_currency': 'EUR'}])
def test_partial_confirmation_does_not_enable_pay(runtime, declarations):
    response = send(runtime, roster(), **declarations)
    assert response.status_code == 200, response.text
    assert runtime.state.raw_df.Salary.isna().all()


def test_file_metadata_enables_pay_without_redundant_confirmation(runtime):
    frame = roster().assign(PayPeriod='annual', Currency='EUR')
    response = send(runtime, frame)
    assert response.status_code == 200, response.text
    assert runtime.state.raw_df.Salary.mean() == 60000


@pytest.mark.parametrize('frame,declarations', [
    (roster().assign(PayPeriod='monthly', Currency='EUR'), {'salary_basis':'annual'}),
    (roster().assign(PayPeriod='annual', Currency='USD'), {'salary_currency':'EUR'}),
    (roster(), {'salary_currency':'EURO'}),
])
def test_confirmation_cannot_override_conflicting_or_invalid_source(runtime, frame, declarations):
    response = send(runtime, frame, **declarations)
    assert response.status_code == 400
    assert runtime.state.raw_df is None


def test_old_artifact_remains_blocked_on_restore(runtime):
    response = send(runtime, roster())
    dataset_id = response.json()['dataset_id']
    assert send(runtime, roster('B'), salary_basis='annual', salary_currency='EUR').status_code == 200
    restored = runtime.client.post(f'/api/platform/workspaces/local/datasets/{dataset_id}/activate')
    assert restored.status_code == 200, restored.text
    assert runtime.state.raw_df.Salary.isna().all()
    assert len(runtime.state.raw_df) == 20
    assert runtime.client.get('/api/upload/status').json()['reporting_currency'] is None


def test_eur_scenario_assumptions_never_label_pay_as_dollars():
    from src.scenario_engine import ScenarioEngine
    scenario = ScenarioEngine(roster().assign(PayPeriod='annual', Currency='EUR')).simulate_headcount_change(
        change_type='expansion', target={'scope':'all'}, change_count=2)
    assert any('Average salary: 60,000' in assumption for assumption in scenario.assumptions)
    assert all('$' not in assumption for assumption in scenario.assumptions)


@pytest.mark.parametrize('metric', ['salary', 'Salary', 'avg_salary'])
def test_historical_forecast_cannot_bypass_pay_confirmation(metric):
    from src.forecasting_engine import ForecastingEngine
    history = pd.concat([roster().assign(SnapshotDate=date) for date in pd.date_range('2024-01-01', periods=12, freq='MS')])
    engine = ForecastingEngine(history)
    assert engine.forecast_metric(metric, 1)['success'] is False
    count = engine.forecast_metric('headcount', 1)
    assert count['success'] is True
    assert count['forecast'][0]['value'] == 18
    confirmed = ForecastingEngine(history.assign(PayPeriod='annual', Currency='EUR'))
    assert confirmed.forecast_metric(metric, 1)['forecast'][0]['value'] == 60000
