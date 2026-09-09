"""Opt-in real loopback API + installed Ollama smoke acceptance, fictional data.

Runs normal production upload/runtime initialization without client replacement.
Selector adversarial cases remain in validate_local_llm.py. This is not browser
acceptance, offline installation validation, or predictive quality validation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(report):
    import httpx
    import uvicorn
    from src.utils import load_config
    config = load_config()['ollama']
    report['configuration'] = config
    if urlparse(config['host']).hostname not in {'localhost', '127.0.0.1', '::1'} or config['model'].endswith('-cloud'):
        raise RuntimeError('Only loopback installed local models are accepted')
    with urlopen(config['host'] + '/api/version', timeout=5) as response:
        report['runtime_version'] = json.load(response)
    from api.main import app
    from api.runtime_registry import get_local_state
    # Only the fictional CSV generator is shared; no cloud transport is created.
    from scripts.validate_cloud_llm import fictional_roster
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error', access_log=False))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while not server.started:
            if time.monotonic() >= deadline:
                raise RuntimeError('Local API startup timed out')
            time.sleep(.05)
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=60, trust_env=False) as api:
            upload = api.post('/api/upload', files={'file': ('synthetic-local.csv', fictional_roster(), 'text/csv')})
            upload.raise_for_status()
            client = get_local_state().llm_client
            report['production_client_type'] = type(client).__name__
            report['model_digest'] = getattr(client, 'model_digest', None)
            if client is None or not client.is_available or not report['model_digest']:
                raise RuntimeError('Production runtime did not initialize an installed local model with digest')
            baseline = api.get('/api/analytics/summary')
            baseline.raise_for_status()
            report['independent_fixture'] = {'source_rows': 120, 'active_count': 80, 'api_summary': baseline.json()}
            response = api.post('/api/intelligence/investigate', json={
                'question': 'What is current headcount?', 'dataset_version': upload.json()['dataset_id']})
            response.raise_for_status()
            result = response.json()
            items = [e for t in result['evidence']['tool_results'] for e in t['evidence']]
            cited_headcount = any(e['metric'] == 'headcount' and e['value'] == 80 and
                f"[{e['evidence_id']}; {e['source_tool']}]" in result['answer'] for e in items)
            passed = (baseline.json()['headcount'] == 80 and cited_headcount and
                result.get('model') == config['model'] and
                'Current active employee count: 80' in result['answer'] and
                not any('failed verification' in w for w in result.get('warnings', [])))
            report['cases'].append({'name': 'live_api_local_headcount', 'passed': passed, 'response': result})
            report['status'] = 'passed' if passed else 'failed'
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'incomplete', 'cases': [],
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
            ['scripts/validate_local_llm_api.py', 'scripts/validate_cloud_llm.py', 'config.yaml',
             'src/llm_client.py', 'src/safe_llm_client.py', 'src/agent/orchestrator.py']},
        'scope': 'Synthetic upload through actual HTTP API, normal production local model initialization, one headcount investigation; no provider injection'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    try:
        with tempfile.TemporaryDirectory(prefix='peopleos-local-api-') as home:
            os.environ['PEOPLEOS_HOME'] = home
            os.environ['PEOPLEOS_WORKSPACE_REGISTRY'] = str(Path(home) / 'workspaces.json')
            os.environ['PEOPLEOS_AGENT_AUDIT_PATH'] = str(Path(home) / 'audit.jsonl')
            run(report)
    except Exception as exc:
        report['status'] = 'failed'
        report['harness_error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'report': str(args.output)}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
