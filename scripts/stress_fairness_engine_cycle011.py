from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.fairness_engine import FairnessEngine


def check(name, condition, actual=None, expected=None):
    return {"name": name, "passed": bool(condition), "actual": actual, "expected": expected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=250000)
    parser.add_argument('--output', default='fairness-engine-cycle011-stress.json')
    args = parser.parse_args()
    n = args.rows
    rng = np.random.default_rng(1111)
    ids = np.array([f'E{i}' for i in range(n)], dtype=object)
    gender = rng.choice(['Female','Male','Nonbinary',None], n, p=[.46,.46,.06,.02])
    dept = rng.choice(['Engineering','Sales','Operations','People'], n)
    age = rng.integers(18, 70, n)
    attrition = rng.choice([0,1,np.nan], n, p=[.78,.17,.05])
    frame = pd.DataFrame({'EmployeeID':ids,'Gender':gender,'Dept':dept,'Age':age,'Attrition':attrition})
    predictions = pd.DataFrame({'EmployeeID':ids,'risk_score':rng.uniform(0,1,n),'predicted':rng.integers(0,2,n)})
    before = frame.copy(deep=True)
    started = time.perf_counter()
    engine = FairnessEngine(frame, predictions)
    four = engine.calculate_four_fifths_rule('Attrition')
    parity = engine.calculate_demographic_parity('Attrition')
    pred = engine.analyze_prediction_fairness()
    odds = engine.calculate_equalized_odds('Attrition')
    elapsed = time.perf_counter()-started
    checks=[]
    checks.append(check('prediction_population_exact', pred.get('population_size') == n, pred.get('population_size'), n))
    checks.append(check('prediction_observations_exact', pred.get('overall_risk_observations') == n, pred.get('overall_risk_observations'), n))
    checks.append(check('four_fifths_finite_or_null', all(pd.isna(v) or np.isfinite(float(v)) for v in four.get('adverse_impact_ratio', []))))
    checks.append(check('parity_finite_or_null', all(pd.isna(v) or np.isfinite(float(v)) for v in parity.get('outcome_rate_ratio_to_overall', []))))
    checks.append(check('prediction_means_in_unit_interval', all(0 <= float(v) <= 1 for v in pred['attribute_analysis'].get('mean_risk', []))))
    checks.append(check('equalized_odds_rates_valid', all(pd.isna(v) or 0 <= float(v) <= 1 for c in ['tpr','fpr'] for v in odds.get(c, []))))
    checks.append(check('no_reconstructable_class_counts', all(not bool(r.get('class_counts_suppressed')) or (pd.isna(r.get('positive_n')) and pd.isna(r.get('negative_n'))) for _,r in odds.iterrows())))
    checks.append(check('source_rows_unchanged', frame.equals(before), len(frame), len(before)))
    shuffled = FairnessEngine(frame, predictions.sample(frac=1, random_state=9)).analyze_prediction_fairness()['attribute_analysis']
    cols=['attribute','group','mean_risk','count','difference_from_overall']
    a=pred['attribute_analysis'][cols].sort_values(['attribute','group']).reset_index(drop=True)
    b=shuffled[cols].sort_values(['attribute','group']).reset_index(drop=True)
    checks.append(check('prediction_row_order_invariant', a.equals(b)))
    passed=sum(c['passed'] for c in checks)
    result={'cycle':'PEOPLEOS-QUALITY-011','engine':'FairnessEngine','rows':n,'elapsed_seconds':round(elapsed,4),'checks':checks,'passed_count':passed,'check_count':len(checks),'passed':passed==len(checks)}
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
