#!/usr/bin/env python3
"""Pinned public fixtures plus independent arithmetic and negative controls.

IBM's HR fixture is synthetic, not evidence of performance on real employees.
Waltons is a survival mechanics benchmark, not workforce validation.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from lifelines.datasets import load_waltons
from src.analytics_engine import AnalyticsEngine
from src.compensation_engine import CompensationEngine
from src.model_training import binary_metrics, train_attrition_model
from src.platform.model_lifecycle import ModelEvaluationPolicy
from src.survival_engine import SurvivalEngine

IBM_COMMIT = '13287d5f717dc978eda249aef4665e04c7cec8b0'
IBM_URL = f'https://raw.githubusercontent.com/IBM/employee-attrition-aif360/{IBM_COMMIT}/data/emp_attrition.csv'
IBM_SHA256 = 'a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659'


def run(ibm_data):
    content = Path(ibm_data).read_bytes() if ibm_data else urlopen(IBM_URL, timeout=60).read()
    assert hashlib.sha256(content).hexdigest() == IBM_SHA256, 'Public fixture changed'
    from io import BytesIO
    source = pd.read_csv(BytesIO(content), encoding='utf-8-sig')
    data = source.rename(columns={'EmployeeNumber': 'EmployeeID', 'Department': 'Dept',
                                  'YearsAtCompany': 'Tenure', 'PerformanceRating': 'LastRating',
                                  'JobRole': 'JobTitle'}).copy()
    data['Salary'] = data.pop('MonthlyIncome') * 12
    data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
    data = data.drop(columns=['EmployeeCount', 'Over18', 'StandardHours'])
    # Independently calculate expected values directly from the source columns.
    expected_active = int((source['Attrition'] == 'No').sum())
    expected_payroll = float(source.loc[source['Attrition'] == 'No', 'MonthlyIncome'].sum() * 12)
    summary = AnalyticsEngine(data).get_summary_statistics()
    actual_payroll = CompensationEngine(data).get_compensation_summary()['total_payroll']
    assert summary['headcount'] == expected_active
    assert np.isclose(actual_payroll, expected_payroll)
    artifact = train_attrition_model(data)
    assert not set(artifact.train_employee_ids) & set(artifact.test_employee_ids)
    assert len(artifact.train_employee_ids) + len(artifact.test_employee_ids) == len(data)
    # Exercise the exported artifact's production inference adapter as well.
    held_out = data[data['EmployeeID'].astype(str).isin(artifact.test_employee_ids)]
    predictions = artifact.engine.predict(held_out.drop(columns='Attrition'))
    checked = binary_metrics(held_out['Attrition'], [p['risk_score'] for p in predictions],
                             float(data.loc[data['EmployeeID'].astype(str).isin(artifact.train_employee_ids), 'Attrition'].mean()))
    assert np.isclose(checked['brier_score'], artifact.metrics['brier_score'])
    assert np.isclose(checked['roc_auc'], artifact.metrics['roc_auc'])
    # Fixed labels, model grids and seeds: no threshold tuning on these results.
    negative_controls = []
    for seed in (7, 19, 43):
        shuffled = data.copy()
        shuffled['Attrition'] = np.random.default_rng(seed).permutation(data['Attrition'].to_numpy())
        metrics = train_attrition_model(shuffled).metrics
        evaluation = ModelEvaluationPolicy().evaluate(metrics)
        negative_controls.append({'label_permutation_seed': seed, 'metrics': metrics, 'evaluation': evaluation})
    assert all(not r['evaluation']['passed'] for r in negative_controls), 'Random-label control unexpectedly passed: investigate; do not tune gates to this fixture'

    waltons = load_waltons()
    survival_data = pd.DataFrame({'EmployeeID': [f'W{i}' for i in range(len(waltons))],
                                  'Tenure': waltons['T'] / 12, 'Attrition': waltons['E'],
                                  'Dept': waltons['group']})
    actual = SurvivalEngine(survival_data).fit_kaplan_meier()['overall']
    # Independent product-limit recurrence, including tied events and censoring.
    times = sorted({0.0, *waltons['T'].astype(float)})
    survival, previous, area, expected = 1., 0., 0., []
    for time in times:
        area += survival * (time - previous)
        at_risk = int((waltons['T'] >= time).sum())
        deaths = int(((waltons['T'] == time) & (waltons['E'] == 1)).sum())
        if at_risk:
            survival *= 1 - deaths / at_risk
        expected.append(survival)
        previous = time
    probabilities = [p['survival_probability'] for p in actual['survival_function']]
    assert np.allclose(probabilities, expected, atol=0.000051, rtol=0)
    assert np.isclose(actual['mean_survival_months'], area)
    assert len(actual['confidence_intervals']['lower']) == len(times)
    return {
        'scope': 'synthetic_and_public_mechanics_validation_not_enterprise_certification',
        'versions': {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in
                     ('numpy', 'pandas', 'scikit-learn', 'scipy', 'lifelines', 'xgboost', 'lightgbm')}},
        'ibm': {'source_url': IBM_URL, 'source_commit': IBM_COMMIT, 'sha256': IBM_SHA256,
                'license': 'ODbL database / DbCL contents; see upstream README',
                'synthetic': True, 'rows': len(data), 'active_employees': expected_active,
                'annualized_payroll': actual_payroll,
                'mapping': 'EmployeeNumber→EmployeeID; Department→Dept; YearsAtCompany→Tenure; PerformanceRating→LastRating; JobRole→JobTitle; MonthlyIncome×12→Salary; Yes/No→1/0',
                'metrics': artifact.metrics, 'evaluation': ModelEvaluationPolicy().evaluate(artifact.metrics)},
        'random_label_controls': negative_controls,
        'waltons': {'source': 'lifelines.datasets.load_waltons', 'rows': len(waltons),
                    'fixture_sha256': hashlib.sha256(waltons.to_csv(index=False).encode()).hexdigest(),
                    'units': 'Original T units mapped mechanically to engine months; not employment durations',
                    'curve_points_checked': len(times), 'independent_restricted_mean': area,
                    'engine_restricted_mean': actual['mean_survival_months']},
        'checks_passed': ['source_hash', 'active_headcount', 'annualized_active_payroll',
                          'disjoint_employee_holdout', 'exported_artifact_inference_matches_evaluation', 'three_random_label_controls_rejected',
                          'independent_product_limit_curve', 'independent_step_rmst', 'aligned_ci_arrays'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ibm-data', help='Optional local copy; SHA256 must match the pinned fixture')
    parser.add_argument('--output', default='docs/validation/public-benchmark-results.json')
    args = parser.parse_args()
    result = run(args.ibm_data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'PASS: {len(result["checks_passed"])} benchmark checks; report: {output}')
