"""Second-pass semantic and privacy contracts for Cycle 013."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

import src.survival_engine as survival_module
from src.survival_engine import SurvivalEngine


def frame(n=100):
    return pd.DataFrame({
        'EmployeeID':[f'E{i}' for i in range(n)],
        'Tenure':np.round(np.linspace(.5,8,n)*12)/12,
        'Attrition':([1]*20+[0]*(n-20)),
        'Dept':['Engineering']*(n//2)+['Sales']*(n-n//2),
        'Age':np.linspace(25,60,n),
        'LastRating':np.linspace(2,5,n),
    })


def test_overall_km_requires_supported_event_and_censor_cells():
    data=frame(100); data['Attrition']=0; data.loc[:2,'Attrition']=1
    result=SurvivalEngine(data).fit_kaplan_meier()
    assert result['available'] is False
    assert 'event' in result['reason'].lower() or 'support' in result['reason'].lower()


def test_overall_km_requires_supported_censor_cell_too():
    data=frame(100); data['Attrition']=1; data.loc[:2,'Attrition']=0
    result=SurvivalEngine(data).fit_kaplan_meier()
    assert result['available'] is False
    assert 'censor' in result['reason'].lower() or 'support' in result['reason'].lower()


def test_cox_reports_model_population_coverage(monkeypatch):
    monkeypatch.setattr(survival_module,'load_config',lambda:{'survival':{'min_sample_size':30,'cox_covariates':['Age','LastRating']}})
    data=frame(120); data.loc[:9,'Age']=np.nan
    result=SurvivalEngine(data).fit_cox_proportional_hazards()
    if result.get('available'):
        pop=result['population']
        assert pop['survival_analysis_population']==120
        assert pop['cox_model_population']==110
        assert pop['excluded_missing_covariates']==10
        assert pop['coverage']==pytest.approx(110/120)
        json.dumps(result,allow_nan=False)


def test_duplicate_configured_covariates_are_deduplicated(monkeypatch):
    monkeypatch.setattr(survival_module,'load_config',lambda:{'survival':{'min_sample_size':30,'cox_covariates':['Age','Age','LastRating','Age']}})
    engine=SurvivalEngine(frame(120))
    assert engine.available_covariates==['Age','LastRating']


def test_cox_requires_both_outcome_classes_to_have_support(monkeypatch):
    monkeypatch.setattr(survival_module,'load_config',lambda:{'survival':{'min_sample_size':30,'cox_covariates':['Age']}})
    data=frame(120); data['Attrition']=1; data.loc[:2,'Attrition']=0
    result=SurvivalEngine(data).fit_cox_proportional_hazards()
    assert result['available'] is False
    assert 'censor' in result['reason'].lower() or 'outcome' in result['reason'].lower() or 'support' in result['reason'].lower()


def test_tail_at_risk_counts_are_suppressed_below_privacy_floor():
    result=SurvivalEngine(frame(100)).fit_kaplan_meier()
    assert result['available']
    tail=[p for p in result['overall']['survival_function'] if p.get('at_risk_suppressed')]
    assert tail
    assert all(p['at_risk'] is None for p in tail)
