"""Cycle 013 forensic contracts for SurvivalEngine."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import src.survival_engine as survival_module
from src.survival_engine import SurvivalEngine, SurvivalEngineError


def workforce(n=60):
    return pd.DataFrame({
        'EmployeeID':[f'E{i}' for i in range(n)],
        'Tenure':np.linspace(0.5,6.0,n),
        'Attrition':([1,0,0,0,0]*((n+4)//5))[:n],
        'Dept':(['Engineering']*(n//2)+['Sales']*(n-n//2)),
        'Location':(['Madrid','Berlin']*((n+1)//2))[:n],
        'ManagerID':([f'M{i//10}' for i in range(n)]),
        'Age':np.linspace(25,55,n),
        'LastRating':np.linspace(2,5,n),
    })


def test_known_answer_restricted_mean_and_supported_horizons():
    frame=pd.DataFrame({'EmployeeID':[f'E{i}' for i in range(60)],'Tenure':[1.0]*30+[2.0]*30,'Attrition':[1]*30+[0]*30})
    result=SurvivalEngine(frame).fit_kaplan_meier()['overall']
    assert result['mean_survival_months']==pytest.approx(18)
    assert result['survival_at_12mo']==.5
    assert result['survival_at_60mo'] is None
    assert result['restricted_mean_horizon_months']==24


def test_survival_curve_and_confidence_interval_arrays_align():
    result=SurvivalEngine(workforce()).fit_kaplan_meier()['overall']
    assert len(result['survival_function'])==len(result['confidence_intervals']['lower'])==len(result['confidence_intervals']['upper'])
    probs=[p['survival_probability'] for p in result['survival_function']]
    assert all(0 <= p <= 1 for p in probs)
    assert all(b <= a for a,b in zip(probs,probs[1:]))


def test_source_frame_is_not_mutated():
    frame=workforce(); before=frame.copy(deep=True)
    SurvivalEngine(frame).analyze_all()
    pd.testing.assert_frame_equal(frame,before)


def test_individual_survival_and_at_risk_rankings_are_disabled():
    engine=SurvivalEngine(workforce())
    assert engine.predict_survival_probability(['E1'],12).empty
    assert engine.get_at_risk_employees(10).empty
    assert engine.analyze_all()['at_risk_employees']==[]


@pytest.mark.parametrize('bad',[0,1,-1,1.5,True,False,np.nan,np.inf,-np.inf,'10'])
def test_invalid_minimum_sample_configuration_fails_closed(monkeypatch,bad):
    monkeypatch.setattr(survival_module,'load_config',lambda: {'survival':{'min_sample_size':bad}})
    with pytest.raises(SurvivalEngineError): SurvivalEngine(workforce())


def test_population_exclusions_are_structurally_disclosed():
    frame=workforce()
    frame.loc[0,'Tenure']=np.nan
    frame.loc[1,'Attrition']=pd.NA
    result=SurvivalEngine(frame).fit_kaplan_meier()
    assert result['available'] is True
    pop=result['population']
    assert pop['source_population']==60
    assert pop['analysis_population']==58
    assert pop['excluded_invalid_duration']==1
    assert pop['excluded_unknown_outcome']==1
    assert pop['coverage']==pytest.approx(58/60)


def test_all_unknown_outcomes_fail_closed_without_false_censoring():
    frame=workforce(); frame['Attrition']=pd.NA
    result=SurvivalEngine(frame).fit_kaplan_meier()
    assert result['available'] is False
    assert 'outcome' in result['reason'].lower() or 'attrition' in result['reason'].lower()


def test_invalid_durations_never_become_valid_survival_time():
    frame=workforce(); frame.loc[:30,'Tenure']=[np.nan,-1,np.inf]*10+[np.nan]
    result=SurvivalEngine(frame).fit_kaplan_meier()
    assert result['available'] is False
    assert 'sample' in result['reason'].lower() or 'duration' in result['reason'].lower()


def test_identifier_like_segmentation_is_rejected():
    engine=SurvivalEngine(workforce())
    result=engine.fit_kaplan_meier(segment_by='ManagerID')
    assert result['available'] is True
    assert result['segments']=={}
    assert result.get('segment_unavailable_reason')


def test_supported_department_segmentation_obeys_privacy_floor_for_events_and_nonevents():
    frame=workforce(60)
    frame['Dept']=['TinyEvents']*20+['Supported']*40
    frame.loc[:19,'Attrition']=0
    frame.loc[:2,'Attrition']=1  # 3 event rows in a 20-person cohort: below privacy floor 5.
    frame.loc[20:39,'Attrition']=1
    result=SurvivalEngine(frame).fit_kaplan_meier(segment_by='Dept')
    assert 'TinyEvents' not in result['segments']
    assert 'Supported' in result['segments']
    assert result['suppressed_segment_count'] >= 1


def test_cohort_insight_suppresses_small_event_cell_instead_of_exposing_exact_count():
    frame=workforce(60); frame['Dept']='Engineering'; frame['Attrition']=0; frame.loc[:2,'Attrition']=1
    result=SurvivalEngine(frame).generate_cohort_insights({'Dept':'Engineering'})
    assert result['cohort_size']==60
    assert result.get('outcome_counts_suppressed') is True
    assert result.get('attrition_count') is None
    assert result.get('attrition_rate') is None


def test_arbitrary_or_identifier_filter_is_rejected():
    engine=SurvivalEngine(workforce())
    result=engine.generate_cohort_insights({'ManagerID':'M1'})
    assert result['cohort_size']==0
    assert 'unavailable' in result['warning'].lower()


def test_cohort_numeric_filter_domain_is_strict():
    engine=SurvivalEngine(workforce())
    for bad in [True,False,np.nan,np.inf,-1,'2']:
        result=engine.generate_cohort_insights({'tenure_min':bad})
        assert result['cohort_size']==0
    assert engine.generate_cohort_insights({'tenure_min':4,'tenure_max':2})['cohort_size']==0


def test_cox_uses_only_explicitly_allowlisted_covariates(monkeypatch):
    data=workforce(100)
    data['NumericEmployeeKey']=np.arange(len(data))
    monkeypatch.setattr(survival_module,'load_config',lambda: {'survival':{'min_sample_size':30,'cox_covariates':['Age','LastRating','ManagerID','EmployeeID','NumericEmployeeKey']}})
    engine=SurvivalEngine(data)
    assert set(engine.available_covariates)=={'Age','LastRating'}


def test_cox_convergence_warning_fails_closed(monkeypatch):
    monkeypatch.setattr(survival_module,'load_config',lambda: {'survival':{'min_sample_size':30,'cox_covariates':['Age']}})
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceWarning
    original_fit=CoxPHFitter.fit
    def warning_fit(self,*args,**kwargs):
        warnings.warn('forced convergence warning',ConvergenceWarning)
        return original_fit(self,*args,**kwargs)
    monkeypatch.setattr(CoxPHFitter,'fit',warning_fit)
    result=SurvivalEngine(workforce(120)).fit_cox_proportional_hazards()
    assert result['available'] is False
    assert 'convergence' in result['reason'].lower()


def test_cox_extreme_finite_covariates_never_emit_nonfinite_output(monkeypatch):
    monkeypatch.setattr(survival_module,'load_config',lambda: {'survival':{'min_sample_size':30,'cox_covariates':['Age']}})
    frame=workforce(120)
    # Extreme absolute magnitude with meaningful relative variation.
    frame['Age']=np.linspace(1e300,1.1e300,len(frame))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result=SurvivalEngine(frame).fit_cox_proportional_hazards()
    json.dumps(result,allow_nan=False)
    assert not any('overflow' in str(w.message).lower() for w in caught)
    if result.get('available'):
        for row in result['coefficients'].values():
            assert all(np.isfinite(float(row[k])) for k in ['coefficient','hazard_ratio','p_value','ci_lower','ci_upper'])


def test_hazard_curve_is_json_finite_when_available(monkeypatch):
    monkeypatch.setattr(survival_module,'load_config',lambda: {'survival':{'min_sample_size':30,'cox_covariates':['Age','LastRating']}})
    engine=SurvivalEngine(workforce(120))
    result=engine.get_hazard_over_time()
    json.dumps(result,allow_nan=False)
    if result.get('available'):
        for p in result['hazard_over_time']:
            assert 0 <= p['survival'] <= 1
            assert p['baseline_hazard'] >= 0 and p['cumulative_hazard'] >= 0


def test_analyze_all_is_json_finite_and_labels_12mo_field_as_cohort_cumulative():
    result=SurvivalEngine(workforce()).analyze_all()
    json.dumps(result,allow_nan=False)
    assert any('cumulative cohort attrition' in w.lower() for w in result['warnings'])


def test_row_order_does_not_change_kaplan_meier_result():
    frame=workforce(100)
    a=SurvivalEngine(frame).fit_kaplan_meier()
    b=SurvivalEngine(frame.sample(frac=1,random_state=13)).fit_kaplan_meier()
    assert a==b
