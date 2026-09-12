from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import src.survival_engine as survival_module
from src.survival_engine import SurvivalEngine

def ck(name,ok,actual=None,expected=None): return {'name':name,'passed':bool(ok),'actual':actual,'expected':expected}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--rows',type=int,default=250000); p.add_argument('--output',default='survival-engine-cycle013-stress.json'); a=p.parse_args()
    n=a.rows; rng=np.random.default_rng(1313)
    # HRIS tenure is typically stored at day/month precision. Monthly resolution
    # keeps 250k population scale while avoiding an artificial 250k-point KM timeline.
    tenure=np.round(np.clip(rng.gamma(shape=2.5,scale=1.5,size=n),1/12,15)*12)/12
    attrition=rng.choice([0,1],n,p=[.82,.18])
    dept=rng.choice(['Engineering','Sales','Operations','People'],n,p=[.4,.3,.2,.1])
    frame=pd.DataFrame({'EmployeeID':[f'E{i}' for i in range(n)],'Tenure':tenure,'Attrition':attrition,'Dept':dept,'Age':rng.integers(20,66,n),'LastRating':rng.uniform(1,5,n)})
    before=frame.copy(deep=True)
    # This stress lane targets KM/population scale. Cox stability is independently
    # challenged in the forensic suite at a size appropriate for model fitting.
    survival_module.load_config=lambda: {'survival':{'min_sample_size':30,'cox_covariates':[]}}
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        engine=SurvivalEngine(frame); km=engine.fit_kaplan_meier(); segmented=engine.fit_kaplan_meier(segment_by='Dept')
        shuffled=SurvivalEngine(frame.sample(frac=1,random_state=17)).fit_kaplan_meier()
    elapsed=time.perf_counter()-started
    overall=km.get('overall',{}); points=overall.get('survival_function',[]); probs=[p['survival_probability'] for p in points]
    checks=[]
    checks.append(ck('km_available',km.get('available') is True))
    checks.append(ck('population_exact',km.get('population',{}).get('analysis_population')==n,km.get('population',{}).get('analysis_population'),n))
    checks.append(ck('survival_unit_interval',all(0<=float(x)<=1 for x in probs)))
    checks.append(ck('survival_monotone',all(b<=x for x,b in zip(probs,probs[1:]))))
    checks.append(ck('ci_aligned',len(points)==len(overall.get('confidence_intervals',{}).get('lower',[]))==len(overall.get('confidence_intervals',{}).get('upper',[]))))
    checks.append(ck('rmst_finite_nonnegative',np.isfinite(overall.get('mean_survival_months',np.nan)) and overall.get('mean_survival_months',-1)>=0))
    checks.append(ck('segments_supported',set(segmented.get('segments',{}))==set(frame.Dept.unique())))
    checks.append(ck('no_individual_rankings',engine.get_at_risk_employees().empty and engine.predict_survival_probability(['E1']).empty))
    checks.append(ck('source_unchanged',frame.equals(before)))
    checks.append(ck('no_overflow_warnings',not any('overflow' in str(w.message).lower() for w in caught)))
    json.dumps(km,allow_nan=False); json.dumps(segmented,allow_nan=False); checks.append(ck('json_finite',True))
    checks.append(ck('row_order_invariant',km==shuffled))
    checks.append(ck('tail_at_risk_privacy',all((p.get('at_risk') is None) == bool(p.get('at_risk_suppressed')) for p in points if p.get('at_risk_suppressed'))))
    passed=sum(c['passed'] for c in checks); result={'cycle':'PEOPLEOS-QUALITY-013','engine':'SurvivalEngine','rows':n,'elapsed_seconds':round(elapsed,4),'checks':checks,'passed_count':passed,'check_count':len(checks),'passed':passed==len(checks)}
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)); print(json.dumps(result,indent=2,allow_nan=False))
    if not result['passed']: raise SystemExit(1)
if __name__=='__main__': main()
