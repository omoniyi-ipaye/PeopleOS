"""Known-answer tests at engine, API, export and presentation boundaries."""
from types import SimpleNamespace
import io
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def workforce(n=40):
    return pd.DataFrame({'EmployeeID':[f'E{i}' for i in range(n)], 'Dept':'A', 'Tenure':2.,
                         'LastRating':4., 'Salary':100., 'Age':30., 'Attrition':0})


def client_for(route, dependency, state):
    app=FastAPI(); app.include_router(route.router)
    app.dependency_overrides[dependency]=lambda:state
    return TestClient(app)


def test_analytics_sparse_distributions_are_valid_json_and_reconcile():
    from src.analytics_engine import AnalyticsEngine
    from api.routes import analytics
    frame=workforce(3); frame['Tenure']=[2.,-1.,np.nan]; frame['Age']=[30.,999.,np.nan]
    frame['LastRating']=[4.,99.,np.nan]
    engine=AnalyticsEngine(frame)
    with client_for(analytics, analytics.require_data, SimpleNamespace(analytics_engine=engine)) as client:
        response=client.get('/api/analytics/distributions')
        assert response.status_code==200, response.text
        assert sum(x['count'] for x in response.json()['tenure'])==3
        assert sum(x['count'] for x in response.json()['age'])==3
        assert next(x for x in response.json()['tenure'] if x['tenure_range']=='Unknown')['count']==2
        summary=client.get('/api/analytics/summary').json()
        assert summary['lastrating_mean']==4 and summary['age_mean']==30 and summary['tenure_mean']==2


def test_compensation_empty_tenure_buckets_are_nullable():
    from src.compensation_engine import CompensationEngine
    from api.routes import compensation
    with client_for(compensation, compensation.require_compensation, SimpleNamespace(compensation_engine=CompensationEngine(workforce()))) as client:
        response=client.get('/api/compensation/by-tenure')
        assert response.status_code==200, response.text
        assert sum(row['count'] for row in response.json())==40
        assert all(row['mean'] is None for row in response.json() if row['count']==0)


@pytest.mark.parametrize('endpoint', ['/api/fairness/four-fifths','/api/fairness/analysis'])
def test_fairness_renderer_uses_favorable_selection_and_reference(endpoint):
    from src.fairness_engine import FairnessEngine
    from api.routes import fairness
    frame=workforce();frame['Gender']=['Male']*20+['Female']*20
    frame['Attrition']=[1]*4+[0]*16+[1]*10+[0]*10
    with client_for(fairness, fairness.require_fairness, SimpleNamespace(fairness_engine=FairnessEngine(frame))) as client:
        data=client.get(endpoint).json(); rows=data if isinstance(data,list) else data['four_fifths']
        female=next(row for row in rows if row['attribute']=='Gender' and row['group']=='Female')
        assert female['selection_rate']==.5 and female['reference_rate']==.8
        assert female['ratio']==pytest.approx(.625)
        assert female['status']=='Screening signal'


def test_fairness_undefined_ratios_stay_unavailable():
    from src.fairness_engine import FairnessEngine
    from api.routes import fairness
    frame=workforce().assign(Gender=['Male']*20+['Female']*20, Attrition=1)
    with client_for(fairness, fairness.require_fairness, SimpleNamespace(fairness_engine=FairnessEngine(frame))) as client:
        response=client.get('/api/fairness/four-fifths')
        assert response.status_code==200,response.text
        assert all(row['ratio'] is None and row['passes_rule'] is None for row in response.json())
    frame['Attrition']=0
    with client_for(fairness, fairness.require_fairness, SimpleNamespace(fairness_engine=FairnessEngine(frame))) as client:
        response=client.get('/api/fairness/demographic-parity')
        assert response.status_code==200,response.text
        assert all(row['parity_ratio'] is None for row in response.json()['results'])


def test_export_preserves_zero_and_does_not_invent_missing_metrics():
    from src.export import export_risk_report
    workbook=export_risk_report({'headcount':5,'turnover_rate':0.}, {'metrics':{'accuracy':.8}})
    summary=pd.read_excel(io.BytesIO(workbook),sheet_name='Summary',keep_default_na=False)
    values=dict(zip(summary['Metric'],summary['Value']))
    assert values['Observed Attrition Share']=='0.0%'
    assert values['Model Precision']=='Unavailable'


def test_merge_result_categories_reconcile_and_optional_changes_are_detected(tmp_path):
    from src.database import Database
    from src.merge_engine import MergeEngine
    db=Database(db_path=str(tmp_path/'merge.db')); engine=MergeEngine(db)
    frame=workforce(2).assign(Location='Madrid')
    engine.execute_merge(frame)
    frame.loc[0,'Location']='Paris'
    preview=engine.preview_merge(frame)
    assert preview.updated==1 and preview.unchanged==1
    result=engine.execute_merge(frame)
    assert result.added+result.updated+result.unchanged+result.skipped==result.total==2
    assert result.updated==1 and result.unchanged==1


def test_structural_missing_and_impossible_role_duration_is_unavailable():
    from src.structural_engine import StructuralEngine
    frame=workforce(5);frame['Tenure']=[0.,3.,3.,-1.,4.];frame['YearsInCurrentRole']=[0.,np.nan,6.,1.,2.]
    scores=StructuralEngine(frame).calculate_stagnation_index()
    assert scores['StagnationIndex'].isna().sum()==4
    assert set(scores.iloc[:4]['StagnationCategory'])=={'Unavailable'}
    assert scores.iloc[4]['StagnationIndex']==.5


def test_structural_reporting_lines_need_no_optional_columns_and_exclude_self_links():
    from src.structural_engine import StructuralEngine
    frame=workforce(4).assign(ManagerID=['E0','E0','E0','unknown'])
    spans=StructuralEngine(frame).calculate_span_of_control()
    manager=spans[spans['ManagerID']=='E0'].iloc[0]
    assert manager['DirectReports']==2
    assert not spans['ManagerID'].eq('unknown').any()


def test_experience_lifecycle_missing_responses_are_nullable_and_associations_need_pairs():
    from src.experience_engine import ExperienceEngine
    frame=workforce(12);frame['Pulse_Score']=[1.,5.]+[np.nan]*10;frame['Tenure']=[.2,.3]+[5.]*10
    engine=ExperienceEngine(frame)
    assert engine.identify_experience_drivers()['drivers']==[]
    stages=engine.get_lifecycle_experience()['stages']
    veteran=next(row for row in stages if row['stage']=='Veteran')
    assert veteran['avg_exi'] is None and veteran['respondent_count']==0
    assert not engine.get_employee_exi('E2')['available']


@pytest.mark.parametrize('classes, probabilities, expected', [([0], [[1],[1]], [0,0]), ([1], [[1],[1]], [1,1]), ([1,0], [[.7,.3],[.2,.8]], [.7,.2])])
def test_ml_scores_identify_positive_class_without_inventing_confidence(classes, probabilities, expected):
    from src.ml_engine import MLEngine
    engine=MLEngine();engine.is_trained=True
    engine.model=SimpleNamespace(classes_=classes,predict_proba=lambda X:probabilities)
    result=engine.predict_risk_with_confidence(pd.DataFrame({'x':[1,2]}))
    assert result['risk_score'].tolist()==expected
    assert result[['ci_lower','ci_upper','confidence_level']].isna().all().all()
    assert engine.get_risk_category(np.nan)=='Unavailable'


@pytest.mark.parametrize('probabilities', [[[np.nan,0]], [[-.1,1.1]], [[.2,.2]]])
def test_ml_rejects_invalid_probability_contract(probabilities):
    from src.ml_engine import MLEngine, MLEngineError
    engine=MLEngine();engine.is_trained=True
    engine.model=SimpleNamespace(classes_=[0,1],predict_proba=lambda X:probabilities)
    with pytest.raises(MLEngineError): engine.predict_risk(pd.DataFrame({'x':[1]}))


@pytest.mark.parametrize('modern', [True,False])
def test_shap_multioutput_shapes_preserve_positive_class_feature_alignment(modern):
    from src.ml_engine import MLEngine
    engine=MLEngine();engine.is_trained=True;engine.feature_names=['a','b']
    engine.model=SimpleNamespace(classes_=np.array([0,1]))
    values=np.array([[[-.1,.1],[.2,-.2]]]) if modern else [np.array([[-.1,.2]]),np.array([[.1,-.2]])]
    engine.shap_explainer=SimpleNamespace(shap_values=lambda X:values)
    result=engine.get_risk_drivers(0,pd.DataFrame({'a':[1],'b':[2]}))
    assert {row['feature']:row['contribution'] for row in result}=={'a':.1,'b':-.2}
    assert all(row['output_units']=='raw_model_output' for row in result)


def nlp_with_response(response):
    import json
    from src.nlp_engine import NLPEngine
    return NLPEngine(SimpleNamespace(is_available=True,model='fixture',client=SimpleNamespace(generate=lambda **kwargs:{'response':json.dumps(response)})))


def test_nlp_topics_validate_topic_schema_and_never_invent_prevalence():
    from api.routes.nlp import TopicInfo
    engine=nlp_with_response([{'name':'Support','description':'Team support','sentiment':'Positive','prevalence':'99%'}])
    topics=engine.extract_topics(workforce(2).assign(PerformanceText='Support from team'))
    assert len(topics)==1 and topics[0]['prevalence'] is None
    assert TopicInfo(**topics[0]).sample_size==2


def test_nlp_sentiment_label_cannot_contradict_score_and_skill_matching_has_boundaries():
    from src.nlp_engine import NLPEngineError
    engine=nlp_with_response([{'EmployeeID':'E0','sentiment_score':.9,'sentiment_label':'Negative'}])
    with pytest.raises(NLPEngineError): engine._analyze_sentiment_batch(['Great work'],['E0'])
    assert engine._count_skills_in_texts(['Exposure to Rust','Trustworthy analyst'],{'technical_skills':['Rust']})=={'Rust':1}


def test_clustering_geometry_ids_and_failed_retrain_clear_previous_results():
    from src.clustering_engine import ClusteringEngine
    from sklearn.metrics import adjusted_rand_score
    frame=workforce(20);frame['Salary']=[100]*10+[1000]*10;frame['Tenure']=[1]*10+[10]*10
    engine=ClusteringEngine(frame)
    result=engine.train(n_clusters=2,auto_tune=False)
    assert result['success'] and sum(result['cluster_counts'].values())==20
    labels=engine.get_employee_clusters()
    assert labels.EmployeeID.tolist()==frame.EmployeeID.tolist()
    assert adjusted_rand_score([0]*10+[1]*10,labels.Cluster)==1
    engine.df['Salary']=100;engine.df['Tenure']=1
    assert not engine.train()['success']
    assert engine.get_employee_clusters().empty and engine.model is None


def test_survival_cannot_claim_ignored_cohort_filters_or_reuse_stale_cox_model():
    from src.survival_engine import SurvivalEngine
    engine=SurvivalEngine(workforce().drop(columns='Dept'))
    assert engine.generate_cohort_insights({'Dept':'A'})['cohort_size']==0
    assert engine.generate_cohort_insights({'tenure_min':5,'tenure_max':2})['filters_applied']=={}
    engine.cox_fitted=True;engine.cox_model=object();engine.available_covariates=[]
    assert not engine.fit_cox_proportional_hazards()['available']
    assert not engine.cox_fitted and engine.cox_model is None


def test_perfect_onboarding_is_not_described_as_low_and_driver_pairs_are_required():
    from src.sentiment_engine import SentimentEngine
    enps=pd.DataFrame({'EmployeeID':['E0','E1'],'eNPSScore':[1,10],'ManagerScore':['1','5']})
    onboarding=pd.DataFrame({'EmployeeID':['E0'],'SurveyType':['30-day'],'OverallScore':[5], 'ManagerSupport':[5]})
    engine=SentimentEngine(workforce(),enps,onboarding)
    assert engine.get_enps_drivers()['drivers']==[]
    assert engine.get_onboarding_health()['recommendations']==[]
    assert engine.detect_early_warnings()['warnings']==[]  # No dated survey evidence.


def test_forecast_fractional_horizon_and_scenario_missing_scope_rejected():
    from src.forecasting_engine import ForecastingEngine
    from src.scenario_engine import ScenarioEngine,ScenarioEngineError
    assert not ForecastingEngine(workforce()).forecast_metric('headcount',1.5)['success']
    engine=ScenarioEngine(workforce())
    with pytest.raises(ScenarioEngineError):engine.simulate_compensation_change('percentage',{'scope':'department'},5)
    with pytest.raises(ScenarioEngineError):engine.simulate_headcount_change('expansion',{'scope':'all'},change_count=-2)


def test_numeric_interpretations_do_not_invent_industry_benchmarks_or_forward_accuracy():
    from src.insight_interpreter import InsightInterpreter
    interpreter=InsightInterpreter()
    assert 'Observed attrition share is 0.0%' in interpreter.interpret_metric('turnover_rate',0)
    assert 'enough data' in interpreter.interpret_metric('headcount',np.nan)
    assert 'lower watch' not in ' '.join(interpreter.get_key_takeaways({'turnover_rate':0}))


def test_causal_and_network_engines_cannot_bypass_unavailable_api_boundaries():
    from src.causal_engine import CausalEngine
    from src.network_engine import NetworkEngine
    causal=CausalEngine(workforce())
    assert causal.estimate_intervention_effect('Salary')['estimated_effect'] is None
    assert causal.get_intervention_recommendations()==[]
    network=NetworkEngine(workforce().assign(ManagerID='E0'))
    assert network.graph is None and not network.get_network_summary()['available']
    assert network.get_key_influencers()==network.get_isolated_employees()==[]


def test_qoh_unmeasured_sources_have_no_winner_or_roi_recommendation():
    from src.quality_of_hire_engine import QualityOfHireEngine
    frame=workforce().assign(HireSource='Agency',LastRating=np.nan,Attrition=pd.NA,Tenure=np.nan)
    engine=QualityOfHireEngine(frame)
    result=engine.get_hiring_insights()
    assert result['top_sources']==[] and result['roi_analysis']=={}
    assert not any('EXPAND' in item for item in result['recommendations'])


def test_legacy_team_renderer_preserves_unavailable_score(monkeypatch):
    import ui.components as components
    from contextlib import nullcontext
    rendered=[]
    monkeypatch.setattr(components.st,'columns',lambda n:[nullcontext() for _ in range(n)])
    monkeypatch.setattr(components.st,'markdown',lambda value,**kwargs:rendered.append(value))
    components.render_team_health_cards(pd.DataFrame([{'Dept':'A','HealthScore':None,'Status':'Unavailable','Headcount':2}]))
    assert 'Unavailable' in ''.join(rendered) and 'nan%' not in ''.join(rendered)


def test_predictive_api_uses_current_active_ids_and_derived_categories():
    from src.ml_engine import MLEngine
    from api.routes import predictions
    frame=workforce(3);frame.loc[2,'Attrition']=1
    metrics={'accuracy':.8,'precision':.7,'recall':.6,'f1':.65,'best_model':'fixture','train_size':80,'test_size':20}
    scores=pd.DataFrame({'EmployeeID':['E0','E1'],'risk_score':[.9,.1],'risk_category':['Low']*2})
    state=SimpleNamespace(risk_scores=scores,raw_df=frame,ml_engine=MLEngine(),model_metrics=metrics)
    from src.platform.provenance import frame_fingerprint
    state.ml_engine.is_trained=True
    state.runtime_provenance={'workspace_id':'local','dataset_id':'fixture','generation':'one','current_fingerprint':frame_fingerprint(frame)}
    state.model_provenance={**state.runtime_provenance,'model_id':'model-fixture'}
    with client_for(predictions,predictions.require_predictions,state) as client:
        response=client.get('/api/predictions/risk')
        assert response.status_code==200,response.text
        assert response.json()['distribution']['total']==2
        assert response.json()['distribution']['high_risk']==1
        state.risk_scores=scores.iloc[:1]
        assert client.get('/api/predictions/risk').status_code==409
        state.risk_scores=pd.concat([scores,scores.iloc[:1]])
        assert client.get('/api/predictions/risk').status_code==409


def test_geo_keeps_unknown_and_remote_records_in_population_counts():
    from api.routes import geo
    from api.routes.analytics import require_data
    frame=workforce(4).assign(Country=['UK','Remote',None,'US'])
    frame.loc[3,'Attrition']=1
    with client_for(geo,require_data,SimpleNamespace(raw_df=frame)) as client:
        result=client.get('/api/geo/distribution')
        assert result.status_code==200,result.text
        assert sum(row['count'] for row in result.json())==3


def test_scenario_elasticity_uses_proportional_changes_not_percentage_points():
    from src.scenario_engine import ScenarioEngine
    engine=ScenarioEngine(workforce(100))
    engine.scenario_config=dict(engine.scenario_config,assumed_baseline_turnover=.2,assumed_compensation_elasticity=.5)
    result=engine.simulate_compensation_change('percentage',{'scope':'all'},10)
    # .5 elasticity * .10 proportional raise * .20 assumed baseline = .01 rate change.
    assert result.turnover_change==1.0
    assert result.projected_turnover_rate==19.0
    assert result.cost_impact.total_cost==1000
    assert result.cost_impact.total_benefit==pytest.approx(150, abs=1e-9)
    assert result.cost_impact.net_impact==-850
