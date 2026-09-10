from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.forecasting_engine import ForecastingEngine

def ck(name,ok,actual=None,expected=None): return {'name':name,'passed':bool(ok),'actual':actual,'expected':expected}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--rows-per-month',type=int,default=10000); p.add_argument('--months',type=int,default=24); p.add_argument('--output',default='forecasting-engine-cycle012-stress.json'); a=p.parse_args()
    frames=[]
    for j,period in enumerate(pd.period_range('2023-01',periods=a.months,freq='M')):
        n=a.rows_per_month + j*10
        frames.append(pd.DataFrame({'EmployeeID':[f'E{i}' for i in range(n)],'SnapshotDate':period.to_timestamp(how='end').date().isoformat(),'Attrition':0,'Salary':80000.0+j*250,'PayPeriod':'annual','Currency':'EUR'}))
    frame=pd.concat(frames,ignore_index=True); before=frame.copy(deep=True)
    started=time.perf_counter(); head=ForecastingEngine(frame).forecast_metric('headcount',periods=12); salary=ForecastingEngine(frame).forecast_metric('salary',periods=12); elapsed=time.perf_counter()-started
    shuffled=ForecastingEngine(frame.sample(frac=1,random_state=12)).forecast_metric('headcount',periods=12)
    checks=[]
    checks.append(ck('headcount_success',head.get('success') is True))
    checks.append(ck('salary_success',salary.get('success') is True))
    checks.append(ck('history_months_exact',len(head.get('history',[]))==a.months,len(head.get('history',[])),a.months))
    checks.append(ck('last_headcount_exact',head['history'][-1]['value']==a.rows_per_month+(a.months-1)*10,head['history'][-1]['value'],a.rows_per_month+(a.months-1)*10))
    checks.append(ck('salary_history_exact',all(abs(r['value']-(80000+i*250))<1e-9 for i,r in enumerate(salary.get('history',[])))))
    checks.append(ck('forecast_lengths_exact',len(head.get('forecast',[]))==12 and len(salary.get('forecast',[]))==12))
    checks.append(ck('headcount_nonnegative_finite',all(np.isfinite(r['value']) and r['value']>=0 for r in head.get('forecast',[]))))
    checks.append(ck('salary_positive_finite',all(np.isfinite(r['value']) and r['value']>0 for r in salary.get('forecast',[]))))
    checks.append(ck('row_order_invariant',head==shuffled))
    checks.append(ck('source_unchanged',frame.equals(before)))
    json.dumps(head,allow_nan=False); json.dumps(salary,allow_nan=False)
    checks.append(ck('json_finite',True))
    passed=sum(x['passed'] for x in checks); result={'cycle':'PEOPLEOS-QUALITY-012','engine':'ForecastingEngine','source_rows':len(frame),'months':a.months,'rows_per_month_start':a.rows_per_month,'elapsed_seconds':round(elapsed,4),'checks':checks,'passed_count':passed,'check_count':len(checks),'passed':passed==len(checks)}
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)); print(json.dumps(result,indent=2,allow_nan=False))
    if not result['passed']: raise SystemExit(1)
if __name__=='__main__': main()
