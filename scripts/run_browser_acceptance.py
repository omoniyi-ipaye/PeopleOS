#!/usr/bin/env python3
"""Serve the real isolated API and built frontend for repository browser tests.

Invoked by Playwright webServer, not a replacement browser-control mechanism.
Requires an ordinary Node production build in web/.next. The fixed API port is
part of the existing Next.js development proxy contract. Never reuse a runtime.
"""
from contextlib import ExitStack
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'web' / 'browser-artifacts'
    output.mkdir(parents=True, exist_ok=True)
    if not (ROOT / 'web' / '.next' / 'BUILD_ID').exists():
        raise RuntimeError('Run npm run build in web before browser acceptance.')
    # Refuse to interact with a user runtime or previous test server.
    for port in (8000, 3000):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    processes = []
    with ExitStack() as stack:
        home = stack.enter_context(tempfile.TemporaryDirectory(prefix='peopleos-browser-'))
        env = {**os.environ, 'PEOPLEOS_HOME': home,
               'PEOPLEOS_WORKSPACE_REGISTRY': str(Path(home) / 'workspace.json'),
               'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
               'NEXT_TELEMETRY_DISABLED': '1'}
        opener = build_opener(ProxyHandler({}))

        def launch(command, cwd, filename, url):
            log = stack.enter_context((output / filename).open('w'))
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=log)
            processes.append(proc)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and not stop.is_set():
                if proc.poll() is not None:
                    raise RuntimeError(f'{filename}: server exited with {proc.returncode}')
                try:
                    with opener.open(url, timeout=2) as response:
                        if response.status == 200:
                            return
                except (URLError, TimeoutError):
                    pass
                stop.wait(.2)
            raise RuntimeError(f'{filename}: server readiness timed out')

        try:
            launch([sys.executable, '-m', 'uvicorn', 'api.main:app', '--host',
                    '127.0.0.1', '--port', '8000'], ROOT, 'api.log', 'http://127.0.0.1:8000/')
            # Launch Next directly so shutdown owns the actual process, not an npm shell.
            launch([shutil.which('node') or 'node', 'node_modules/next/dist/bin/next',
                    'start', '--hostname', '127.0.0.1', '--port', '3000'],
                   ROOT / 'web', 'frontend.log', 'http://127.0.0.1:3000/')
            print('Isolated PeopleOS browser acceptance servers ready', flush=True)
            while not stop.wait(.25):
                if any(proc.poll() is not None for proc in processes):
                    raise RuntimeError('Acceptance server stopped unexpectedly')
        finally:
            for proc in reversed(processes):
                if proc.poll() is None:
                    proc.terminate()
            for proc in reversed(processes):
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == '__main__':
    main()
