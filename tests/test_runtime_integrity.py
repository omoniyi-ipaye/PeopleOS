"""Route-level acceptance of dataset identity, activation and recovery contracts."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request


def roster(prefix='A', n=20):
    return pd.DataFrame({'EmployeeID':[f'{prefix}{i:04d}' for i in range(n)], 'Dept':'001',
        'Salary':60000., 'Tenure':2., 'LastRating':4., 'Age':30, 'Gender':'Female',
        'JobTitle':'Analyst', 'JobLevel':'L03', 'Location':'Madrid', 'HireDate':'2024-01-01', 'ManagerID':f'{prefix}0000',
        'Attrition':[1,1]+[0]*(n-2)})


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setenv('PEOPLEOS_HOME', str(tmp_path/'home'))
    monkeypatch.setenv('PEOPLEOS_WORKSPACE_REGISTRY', str(tmp_path/'workspace.json'))
    from api.runtime_registry import WorkspaceRuntimeState
    from api.dependencies import get_app_state
    from api.routes import upload, platform, analytics, intelligence, sentiment
    from src.platform import runtime_loader
    from src.platform.workspace import WorkspaceStore
    from src.platform.jobs import JobStore
    from src.platform.model_lifecycle import ModelLifecycleService
    from src.analytics_engine import AnalyticsEngine
    from src.sentiment_engine import SentimentEngine
    from src import database
    monkeypatch.setattr(database, '_database_instance', None)
    state = WorkspaceRuntimeState('local')
    state.data_loader.min_rows = 1
    original_initialize=runtime_loader._initialize_read_only_engines
    def initialize(candidate):
        candidate.analytics_engine = AnalyticsEngine(candidate.raw_df)
        candidate.sentiment_engine = SentimentEngine(candidate.raw_df, candidate.enps_df, candidate.onboarding_df)
    monkeypatch.setattr(runtime_loader, '_initialize_read_only_engines', initialize)
    store = WorkspaceStore()
    for module in (upload, platform, intelligence):
        monkeypatch.setattr(module, '_store', store)
    for module in (platform, intelligence):
        monkeypatch.setattr(module, 'require_permission', lambda *a: SimpleNamespace(actor_id='owner'))
    monkeypatch.setattr(platform, '_jobs', JobStore(str(tmp_path/'jobs.json')))
    monkeypatch.setattr(ModelLifecycleService, '_runtime_artifacts', {})
    app=FastAPI()
    app.dependency_overrides[get_app_state]=lambda:state
    for module in (upload,platform,analytics,intelligence,sentiment):
        app.include_router(module.router)
    from api import integrity
    monkeypatch.setattr(integrity, 'get_workspace_state', lambda request:state)
    app.middleware('http')(integrity.evidence_snapshot_guard)
    with TestClient(app) as client:
        yield SimpleNamespace(state=state, store=store, client=client, platform=platform, upload=upload, loader=runtime_loader, original_initialize=original_initialize)


def upload(runtime, frame):
    response=runtime.client.post('/api/upload', files={'file':('workforce.csv',frame.to_csv(index=False).encode(),'text/csv')})
    assert response.status_code==200,response.text
    return response.json()['dataset_id']


def test_http_dataset_switch_restores_exact_population_and_provenance(runtime):
    a=upload(runtime,roster('A'))
    response=runtime.client.get('/api/analytics/summary')
    assert response.status_code==200,response.text
    assert response.json()['active_count']==18
    assert response.headers['X-PeopleOS-Dataset']==a
    b=upload(runtime,roster('B',30))
    assert len(runtime.state.raw_df)==30  # Upload is a snapshot, not an implicit SQLite merge.
    assert runtime.state.raw_df.EmployeeID.str.startswith('B').all()
    assert runtime.client.post(f'/api/platform/workspaces/local/datasets/{a}/activate').status_code==200
    assert runtime.state.raw_df.EmployeeID.str.startswith('A').all()
    assert runtime.state.raw_df.Dept.eq('001').all()
    assert runtime.state.raw_df.JobLevel.eq('L03').all()
    assert runtime.state.runtime_provenance['dataset_id']==a
    assert runtime.store.get_workspace('local').active_dataset_id==a
    stale=runtime.client.post('/api/intelligence/investigate',json={'question':'How many employees?', 'dataset_version':b})
    assert stale.status_code==409,stale.text
    assert runtime.store.get_workspace('local').sessions==[]


def test_failed_upload_persistence_keeps_previous_runtime_and_selection(runtime,monkeypatch):
    a=upload(runtime,roster('A'))
    previous=dict(runtime.state.runtime_provenance)
    monkeypatch.setattr(runtime.upload,'save_dataset_artifact',lambda *a:(_ for _ in ()).throw(OSError('disk unavailable')))
    response=runtime.client.post('/api/upload',files={'file':('B.csv',roster('B').to_csv(index=False).encode(),'text/csv')})
    assert response.status_code==400
    assert runtime.state.runtime_provenance==previous
    assert runtime.store.get_workspace('local').active_dataset_id==a
    assert runtime.state.raw_df.EmployeeID.str.startswith('A').all()


def test_failed_dataset_prepare_keeps_active_selection(runtime,monkeypatch):
    a=upload(runtime,roster('A'))
    b=upload(runtime,roster('B'))
    monkeypatch.setattr(runtime.platform,'prepare_dataframe',lambda *a,**kw:(_ for _ in ()).throw(ValueError('bad transform')))
    response=runtime.client.post(f'/api/platform/workspaces/local/datasets/{a}/activate')
    assert response.status_code==409
    assert runtime.store.get_workspace('local').active_dataset_id==b
    assert runtime.state.runtime_provenance['dataset_id']==b


def test_reactivation_reconciles_model_registry(runtime):
    from src.platform.workspace import ModelState
    a=upload(runtime,roster())
    model=runtime.store.create_model(workspace_id='local',dataset_id=a)
    runtime.store.update_model('local',model.model_id,state=ModelState.CANDIDATE)
    runtime.store.activate_model('local',model.model_id)
    response=runtime.client.post(f'/api/platform/workspaces/local/datasets/{a}/activate')
    assert response.status_code==200,response.text
    workspace=runtime.store.get_workspace('local')
    assert workspace.active_model_id is None
    assert workspace.models[0].state==ModelState.RETIRED
    assert runtime.state.ml_engine is None


def test_retrain_recreates_missing_artifact_after_restart(runtime,monkeypatch):
    from src.platform.model_lifecycle import ModelLifecycleService
    import src.model_training as training
    upload(runtime,roster())
    seen=[]
    def fit(frame):
        seen.append(frame.EmployeeID.tolist())
        return SimpleNamespace(metrics={},engine=SimpleNamespace(is_trained=True))
    monkeypatch.setattr(training,'train_attrition_model',fit)
    request=Request({'type':'http','headers':[(b'idempotency-key',b'retrain-fixture')]})
    first=asyncio.run(runtime.platform.train_model('local',request,state=runtime.state))
    reused=asyncio.run(runtime.platform.train_model('local',request,state=runtime.state))
    assert reused['reused'] is True and len(seen)==1
    ModelLifecycleService._runtime_artifacts.clear()
    retrained=asyncio.run(runtime.platform.train_model('local',request,state=runtime.state))
    assert retrained['reused'] is False and len(seen)==2
    assert retrained['model']['model_id']!=first['model']['model_id']
    assert retrained['model']['dataset_id']==first['model']['dataset_id']


def test_valid_model_activation_commits_exact_active_outputs(runtime):
    from src.platform.workspace import ModelState
    from src.platform.model_lifecycle import ModelLifecycleService
    a=upload(runtime,roster())
    model=runtime.store.create_model(workspace_id='local',dataset_id=a)
    runtime.store.update_model('local',model.model_id,state=ModelState.CANDIDATE)
    engine=SimpleNamespace(is_trained=True,feature_names=['Salary'],risk_threshold_medium=.5,
        preprocessor=SimpleNamespace(transform=lambda frame,**kw:frame[['Salary']]),
        predict_risk=lambda x:np.full(len(x),.25),get_risk_category=lambda score:'Low')
    metrics={'training_current_fingerprint':runtime.state.runtime_provenance['current_fingerprint']}
    ModelLifecycleService._runtime_artifacts[model.model_id]=SimpleNamespace(engine=engine,metrics=metrics)
    response=runtime.client.post(f'/api/platform/workspaces/local/models/{model.model_id}/activate')
    assert response.status_code==200,response.text
    assert len(runtime.state.risk_scores)==18
    assert runtime.state.model_provenance['model_id']==model.model_id
    assert runtime.store.get_workspace('local').active_model_id==model.model_id
    from src.platform.provenance import validated_risk_scores
    assert validated_risk_scores(runtime.state).risk_category.eq('Low').all()


def test_mutated_runtime_fails_http_evidence_guard(runtime):
    upload(runtime,roster())
    runtime.state.raw_df.loc[0,'Salary']=999999
    response=runtime.client.get('/api/analytics/summary')
    assert response.status_code==409
    assert 'changed after activation' in response.json()['detail']


def test_nonlocal_activation_cannot_replace_local_runtime(runtime):
    a=upload(runtime,roster())
    other=runtime.store.register_dataset(workspace_id='other',source_name='other',content_hash='other',row_count=1,columns=['EmployeeID'])
    response=runtime.client.post(f'/api/platform/workspaces/other/datasets/{other.dataset_id}/activate')
    assert response.status_code==409
    assert runtime.state.runtime_provenance['dataset_id']==a


def test_sentiment_api_retains_excluded_population_and_unknown_flags(runtime):
    upload(runtime,roster())
    survey='EmployeeID,SurveyDate,eNPSScore\nOTHER,2026-01-01,10\n'
    response=runtime.client.post('/api/sentiment/upload/enps',files={'file':('survey.csv',survey,'text/csv')})
    assert response.status_code==200,response.text
    assert response.json()['survey_coverage']['enps']['unmatched_rows']==1
    response=runtime.client.get('/api/sentiment/analysis')
    assert response.status_code==200,response.text
    assert response.json()['survey_coverage']['enps']['unmatched_rows']==1
    assert response.json()['summary']['employees_at_risk'] is None
    assert response.json()['summary']['survey_flags_available'] is False


def test_reset_failure_does_not_clear_selected_dataset(runtime,monkeypatch):
    a=upload(runtime,roster())
    def failure():
        raise OSError('database unavailable')
    monkeypatch.setattr(runtime.state,'reset',failure)
    with pytest.raises(OSError):
        asyncio.run(runtime.upload.reset_data(state=runtime.state))
    assert runtime.store.get_workspace('local').active_dataset_id==a
    assert runtime.state.runtime_provenance['dataset_id']==a


def test_session_cannot_be_relabelled_after_dataset_switch(runtime):
    a=upload(runtime,roster('A'))
    session=runtime.store.open_session(workspace_id='local',dataset_id=a)
    b=upload(runtime,roster('B'))
    response=runtime.client.post('/api/intelligence/investigate',json={
        'question':'How many employees?', 'session_id':session.session_id, 'dataset_version':b})
    assert response.status_code==409
    saved=runtime.store.get_workspace('local').sessions[0]
    assert saved.dataset_id==a and saved.request_ids==[]


def test_retired_training_result_is_not_reused(runtime,monkeypatch):
    import src.model_training as training
    from src.platform.workspace import ModelState
    upload(runtime,roster())
    monkeypatch.setattr(training,'train_attrition_model',lambda frame:SimpleNamespace(metrics={},engine=SimpleNamespace(is_trained=True)))
    request=Request({'type':'http','headers':[]})
    first=asyncio.run(runtime.platform.train_model('local',request,state=runtime.state))
    runtime.store.update_model('local',first['model']['model_id'],state=ModelState.RETIRED)
    second=asyncio.run(runtime.platform.train_model('local',request,state=runtime.state))
    assert second['reused'] is False
    assert second['model']['model_id']!=first['model']['model_id']


def test_observed_outcome_fairness_is_available_without_training(runtime):
    upload(runtime,roster())
    runtime.original_initialize(runtime.state)
    assert runtime.state.ml_engine is None
    assert runtime.state.fairness_engine is not None
    result=runtime.state.fairness_engine.calculate_demographic_parity('Attrition')
    assert not result.empty
