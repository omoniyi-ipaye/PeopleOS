#!/usr/bin/env python3
"""Exercise the unmodified ASGI app over TCP with independently specified dummy data.

Starts an isolated local server and preserves inputs, responses and expected/actual
checks. No engine mocks, dependency overrides, production data or remote services.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['EmployeeID', 'Dept', 'Salary', 'Tenure', 'LastRating', 'Age',
          'Gender', 'JobTitle', 'JobLevel', 'Location', 'HireDate', 'ManagerID',
          'Attrition', 'HireSource', 'SnapshotDate', 'PayPeriod', 'Currency']


def workforce():
    rows = []
    for date in ['2025-01-01', '2026-01-01']:
        for i in range(120):
            rows.append(dict(zip(FIELDS, [f'{i:04d}', ['001', '002', ''][i % 3],
                50000 if date == '2025-01-01' else (60000 if i < 40 else 90000),
                2, '' if i < 20 else 4, 30, 'Female' if i % 2 else 'Male',
                'Analyst', 'L03', 'Madrid', '2024-01-01', '0000',
                0 if date == '2025-01-01' or i < 80 else (1 if i < 100 else ''),
                'Referral' if i % 2 else 'Agency', date, 'annual', 'EUR'])))
    return rows


def csv_bytes(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'docs/validation/live-dummy-data')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks, responses = [], {}
    complete = False
    def check(name, actual, expected):
        passed = (math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8)
                  if isinstance(actual, (int, float)) and isinstance(expected, (int, float))
                  else actual == expected)
        checks.append(dict(name=name, actual=actual, expected=expected, passed=passed))
        print(('PASS' if passed else 'FAIL') + ': ' + name, flush=True)

    with tempfile.TemporaryDirectory(prefix='peopleos-acceptance-') as home:
        env = {**os.environ, 'PEOPLEOS_HOME': home,
               'PEOPLEOS_WORKSPACE_REGISTRY': str(Path(home)/'workspace.json'),
               'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        log = output.joinpath('server.log').open('w')
        def start():
            proc = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'api.main:app',
                '--host', '127.0.0.1', '--port', str(port)], cwd=ROOT,
                env=env, stdout=log, stderr=log)
            for _ in range(120):
                if proc.poll() is not None:
                    raise RuntimeError('Server exited; inspect server.log')
                try:
                    if client.get('/').status_code == 200:
                        return proc
                except httpx.ConnectError:
                    time.sleep(.25)
            proc.terminate()
            raise RuntimeError('Server did not start')
        client = httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=120, trust_env=False)
        proc = start()
        def request(name, path, *, method='GET', expected_status=200, **kwargs):
            response = client.request(method, path, **kwargs)
            try:
                body = response.json()
            except ValueError:
                body = {'non_json': response.text}
            responses[name] = {'path': path, 'status': response.status_code,
                'snapshot': response.headers.get('x-peopleos-snapshot'),
                'dataset': response.headers.get('x-peopleos-dataset'), 'body': body}
            check(name + ' HTTP status', response.status_code, expected_status)
            return body
        def upload(name, rows, expected_status=200):
            content = csv_bytes(rows)
            output.joinpath(name+'.csv').write_bytes(content)
            return request(name, '/api/upload', method='POST', expected_status=expected_status,
                files={'file': (name+'.csv', content, 'text/csv')})
        try:
            a = upload('workforce-a', workforce())
            aid = a['dataset_id']
            summary = request('summary-a', '/api/analytics/summary')
            for key, expected in {'record_count':120, 'active_count':80, 'attrition_count':20,
                    'observed_attrition_share':.2, 'salary_mean':75000, 'salary_median':75000,
                    'tenure_mean':2, 'lastrating_mean':4, 'department_count':3}.items():
                check('A summary '+key, summary[key], expected)
            check('A response dataset', responses['summary-a']['dataset'], aid)
            status = request('status-a', '/api/status')
            for key, expected in {'source_rows':240, 'current_rows':120, 'active_rows':80,
                    'unknown_status_rows':20}.items():
                check('A snapshot '+key, status['integrity']['snapshot'][key], expected)
            check('Untrained model is unavailable', status['integrity']['model_ready'], False)
            compensation = request('compensation-a', '/api/compensation/analysis')
            for key, expected in {'total_payroll':6000000, 'avg_salary':75000,
                    'median_salary':75000, 'headcount':80}.items():
                check('A compensation '+key, compensation['summary'][key], expected)
            departments = request('departments-a', '/api/analytics/departments')['departments']
            check('Department groups reconcile', sum(d['headcount'] for d in departments), 80)
            check('Leading-zero department labels survive', sorted(d['dept'] for d in departments), ['001','002','Unknown'])
            # Unknown has 13 salaries at 60k and 13 at 90k: sample variance uses n-1.
            check('Department salary dispersion remains measured',
                next(d['salary_std_dev'] for d in departments if d['dept']=='Unknown'),
                math.sqrt(26*15000**2/25))
            team = request('team-a', '/api/team/health')
            check('Team groups reconcile', sum(d['headcount'] for d in team), 80)
            check('Unknown team retained', next(d['headcount'] for d in team if d['dept']=='Unknown'), 26)

            survey = [{'EmployeeID':f'{i:04d}', 'SurveyDate':'2026-01-01',
                       'eNPSScore':10 if i<30 else (7 if i<40 else 0)} for i in range(60)]
            survey += [{'EmployeeID':'OUTSIDE', 'SurveyDate':'2026-01-01','eNPSScore':10},
                       {'EmployeeID':'', 'SurveyDate':'2026-01-01','eNPSScore':10},
                       {'EmployeeID':'0060', 'SurveyDate':'2026-01-01','eNPSScore':11}]
            content=csv_bytes(survey);output.joinpath('enps.csv').write_bytes(content)
            request('survey-upload', '/api/sentiment/upload/enps', method='POST',
                    files={'file':('enps.csv',content,'text/csv')})
            enps=request('enps-a', '/api/sentiment/enps')
            check('eNPS (30 promoters - 20 detractors) / 60', round(enps['overall_enps'],1), 16.7)
            check('eNPS valid response count', enps['total_responses'],60)
            sentiment=request('sentiment-a','/api/sentiment/analysis')
            coverage=sentiment['survey_coverage']['enps']
            for key, expected in {'input_rows':63,'matched_rows':61,'unmatched_rows':1,
                    'missing_employee_id_rows':1,'invalid_score_rows':1,'valid_score_rows':60}.items():
                check('Survey coverage '+key,coverage[key],expected)
            scenario=request('scenario-a','/api/scenario/simulate/compensation',method='POST',
                json={'target':{'scope':'all'},'adjustment_type':'percentage','adjustment_value':10,'time_horizon_months':6})
            check('Six-month 10% raise on annual 6m payroll',scenario['cost_impact']['salary_change'],300000)
            check('Scenario affected active population',scenario['affected_employees'],80)
            check('Scenario preserves Unknown department',sorted(scenario['affected_departments']),['001','002','Unknown'])
            for name,path in [('quality-a','/api/quality-of-hire/analysis'),
                    ('experience-a','/api/experience/analysis'),('fairness-a','/api/fairness/analysis'),
                    ('survival-a','/api/survival/kaplan-meier'),('structural-a','/api/structural/analysis'),
                    ('succession-a','/api/succession/summary'),('geo-a','/api/geo/summary'),
                    ('network-a','/api/network/summary'),('model-lab-a','/api/model-lab/validation'),
                    ('search-a','/api/search/status'),('forecast-a','/api/analytics/forecast')]:
                request(name,path)
            result=lambda name:responses[name]['body']
            check('Experience cannot infer a score without surveys',result('experience-a')['experience_index']['overall_exi'],None)
            for row in result('fairness-a')['four_fifths']:
                if row['attribute']=='Gender':
                    check(row['group']+' observed retained share',row['selection_rate'],.8)
                    check(row['group']+' favorable outcome ratio',row['ratio'],1)
            survival=result('survival-a')['overall']
            check('KM survival after 20 events among 100 known outcomes',survival['survival_at_24mo'],.8)
            check('KM restricted mean through 24 months',survival['mean_survival_months'],24)
            check('KM does not extrapolate unsupported 36-month survival',survival['survival_at_36mo'],None)
            check('Recorded manager has 79 active direct reports',result('structural-a')['span_of_control']['summary']['max_span'],79)
            succession=result('succession-a')
            check('Succession department totals reconcile',sum(x['Total'] for x in succession['bench_strength']),80)
            check('Missing readiness remains unassessed',sum(x['Unassessed'] for x in succession['bench_strength']),80)
            check('Missing readiness has no bench score',all(x['BenchStrength'] is None for x in succession['bench_strength']),True)
            check('City does not invent country mapping',result('geo-a')['unknown_country_count'],80)
            check('Reporting lines cannot invent collaboration',result('network-a')['available'],False)
            check('No prospective model metrics invented',result('model-lab-a')['metrics'],None)
            check('Two annual snapshots cannot invent monthly forecasts',result('forecast-a')['success'],False)
            check('No embeddings means unavailable search',result('search-a')['available'],False)
            # One sparse rating and no known outcomes cannot become a healthy workforce.
            b=[{**r,'EmployeeID':'B'+r['EmployeeID'],'Attrition':'','Salary':50000,
                'LastRating':5 if i==0 else ''} for i,r in enumerate(workforce()[120:])]
            upload('workforce-b',b)
            bs=request('summary-b','/api/analytics/summary')
            request('status-b','/api/status')
            request('departments-b','/api/analytics/departments')
            check('Unknown outcomes do not become active',bs['active_count'],0)
            check('Unknown attrition stays null',bs['observed_attrition_share'],None)
            check('No active salary stays null',bs['salary_mean'],None)
            quality=request('quality-b','/api/quality-of-hire/source-effectiveness')
            check('Sparse source ratings cannot create scores',all(x['quality_score'] is None for x in quality),True)
            request('stale-scenario','/api/scenario/'+scenario['scenario_id'],expected_status=404)
            sb=request('sentiment-b','/api/sentiment/analysis')
            check('Dataset switch clears survey evidence',sb['summary']['survey_flags_available'],False)
            request('activate-a',f'/api/platform/workspaces/local/datasets/{aid}/activate',method='POST')
            restored=request('summary-restored','/api/analytics/summary')
            check('A -> B -> A restores original outputs',restored,summary)
            duplicate=workforce()+[workforce()[0]]
            upload('invalid-duplicate',duplicate,expected_status=400)
            check('Rejected upload leaves current output intact',request('summary-after-invalid','/api/analytics/summary'),summary)
            proc.terminate();proc.wait(timeout=15)
            proc=start()
            request('restart-load','/api/upload/status')
            after=request('summary-restart','/api/analytics/summary')
            check('Fresh-process output matches pre-restart',after,summary)
            final=request('status-restart','/api/status')
            check('Fresh-process canonical history preserved',final['integrity']['snapshot']['source_rows'],240)
            complete = True
        finally:
            proc.terminate();proc.wait(timeout=15);client.close();log.close()
            output.joinpath('responses.json').write_text(json.dumps(responses,indent=2,allow_nan=False)+'\n')
            output.joinpath('checks.json').write_text(json.dumps({'commit':subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'checks':checks,
                'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
                    'src/analytics_engine.py','src/scenario_engine.py','src/succession_engine.py',
                    'api/routes/sentiment.py',
                    'scripts/live_dummy_acceptance.py']},
                'versions':{p:importlib.metadata.version(p) for p in ['pydantic','uvicorn','pandas','fastapi']},
                'complete':complete,'passed':sum(x['passed'] for x in checks),'failed':sum(not x['passed'] for x in checks)},indent=2)+'\n')
    print(f"RESULT: {sum(x['passed'] for x in checks)} passed; {sum(not x['passed'] for x in checks)} failed")
    return int(any(not x['passed'] for x in checks))


if __name__ == '__main__':
    raise SystemExit(main())
