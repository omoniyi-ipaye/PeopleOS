# PeopleOS Public Beta Guide

This guide covers local source runs and the archive format prepared for the public beta. Use the bundled fictional dataset for the first evaluation. This is a beta with a transitioning architecture, not a claim of production certification, hosted multi-user support or available signed installers.

## Packaged archive first use

These are the prepared archive names, **not a claim that downloads have been published**. Use these steps only when the corresponding archive and its separate `SHA256SUMS` file are available from the project's published release or build artifacts. Download the matching pair into one folder.

| System | Archive | Enclosed executable |
| --- | --- | --- |
| Windows x64 | `PeopleOS-windows-x64.zip` | `PeopleOS-windows-x64/PeopleOS.exe` |
| Linux x64 | `PeopleOS-linux-x64.tar.gz` | `PeopleOS-linux-x64/PeopleOS` |
| macOS Apple Silicon | `PeopleOS-macos-arm64.tar.gz` | `PeopleOS-macos-arm64/PeopleOS` |

Verify before extracting. Run from the download folder; a checksum mismatch means stop. Checksums detect changed bytes but are not a code signature or proof of publisher identity.

**Windows PowerShell:**

```powershell
$archiveName = 'PeopleOS-windows-x64.zip'
$checksumLine = Get-Content .\SHA256SUMS | Where-Object { $_ -match ('  ' + [regex]::Escape($archiveName) + '$') }
if (@($checksumLine).Count -ne 1) { throw 'Missing or ambiguous checksum' }
$expectedHash = ($checksumLine -split '\s+')[0]
if ((Get-FileHash $archiveName -Algorithm SHA256).Hash -ne $expectedHash) { throw 'Checksum mismatch' }
Expand-Archive $archiveName -DestinationPath .
.\PeopleOS-windows-x64\PeopleOS.exe
```

**Linux:**

```bash
sha256sum --check SHA256SUMS && tar -xzf PeopleOS-linux-x64.tar.gz && ./PeopleOS-linux-x64/PeopleOS
```

**macOS Apple Silicon:**

```bash
shasum -a 256 --check SHA256SUMS && tar -xzf PeopleOS-macos-arm64.tar.gz && ./PeopleOS-macos-arm64/PeopleOS
```

The launcher serves the included UI and API on `127.0.0.1` at an automatically selected port and opens your default browser after the service is ready. Keep the launcher running while using PeopleOS; closing a browser tab alone does not stop it. For terminal launches, stop with Ctrl+C. If a windowless launcher remains running, use your operating system’s process manager to stop the PeopleOS process before making a preservation copy; this beta has no tray shutdown control. Relaunch the same executable to restart. These archives include the runtime, so separate Python and Node installations are not required. Proceed to the sample walkthrough below.

Core analytics and deterministic investigation summaries can run offline after obtaining the archive. An Ollama server/model is optional and is not bundled; model downloads need separate setup. The slim desktop runtime also excludes optional predictive training packages. The packaging flow does not sign or notarize these executables. If your operating system blocks an unsigned app, stop and report the platform/message; this guide does not instruct you to disable or bypass operating-system protections.

## Install and run from source

Use Python 3.11 (used by repository CI), Node.js 22 and npm. The frontend declares Node.js >=20.9.0 in `web/package.json`. Run commands from the repository root unless shown otherwise. A full core install includes statistical and predictive packages; Ollama, Torch, sentence-transformers and FAISS are not required for the first run.

Create and activate an isolated Python environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the checked-in dependencies:

```bash
python -m pip install -r requirements.txt
cd web
npm ci
cd ..
```

Start the API, keeping this terminal open:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal, start the UI:

```bash
cd web
npm run dev -- --hostname 127.0.0.1
```

Open [PeopleOS](http://127.0.0.1:3000). The UI development proxy expects the API on port 8000. [API health](http://127.0.0.1:8000/api/health) and [API documentation](http://127.0.0.1:8000/docs) help diagnose startup problems. Stop each process with Ctrl+C when finished.

For optional embeddings/search, `python -m pip install -r requirements-advanced.txt` installs the advanced tier, including core. Optional local synthesis requires a separately installed and running Ollama server and a compatible model. Deterministic investigation summaries remain available without it; installing an LLM is not a prerequisite for this walkthrough.

## First run with fictional data

1. Open **Data & Sources** at `/upload`.
2. Choose **Load sample**. The bundled `sample_hr_data.csv` is fictional; its load route explicitly declares annual pay in a single illustrative currency, USD. Country/location fields do not imply currency conversion.
3. Confirm the active population, then explore workforce analytics and the People Intelligence page at `/advisor`. Try an aggregate question such as “How does attrition vary by department?” Inspect evidence, limitations and confidence alongside the answer.
4. Open **System Health** at `/platform` to inspect active dataset/version and runtime health.
5. Restart the backend and revisit Data & Sources to check that the expected dataset returns.

The sample button is disabled when a dataset is already active. Use a separate test environment to preserve existing work. **Reset** clears the active data selection while retaining dataset/model version history for auditability; it is not a secure erasure function. An upload activates a new dataset version and does not silently train or activate a predictive model.

## Import your own test data

Use synthetic or appropriately de-identified test data for beta feedback. Never upload workforce files, database copies or employee screenshots to public issues.

The upload route accepts **CSV or JSON**, not Excel files. Download the current Golden Schema template from Data & Sources or [the source-run template endpoint](http://127.0.0.1:8000/api/upload/template) (packaged users should use the Data & Sources button at the app’s actual port). The source template is [peopleos_template.csv](../data/templates/peopleos_template.csv). Follow the template and validation messages for required fields, types and optional capabilities. Historical analysis needs genuine dated observations; a single current snapshot cannot establish a historical trend.

### Declare salary units before pay analysis

PeopleOS does not annualize pay or convert currencies for you. Before import, ensure all monetary pay values—including `Salary`, `StartingSalary`, market references and band midpoints—are annual amounts in one shared reporting currency.

Declare those units in either supported way:

- Include `PayPeriod=annual` (or an accepted annual `PayFrequency`) and one shared three-letter `Currency` code on every row.
- In Data & Sources, select the annual-pay confirmation and enter the shared three-letter reporting currency before uploading.

The API equivalents are multipart form fields `salary_basis=annual` and `salary_currency=EUR` (replace EUR with the actual shared currency). A confirmation does not override conflicting source declarations. Mixed currencies, nonannual source pay periods and conflicting currency confirmations are rejected. Missing declarations allow population analysis but keep compensation disabled and pay amounts out of analytical results. Reimport with correct declarations to enable pay analysis.

## Supported beta scope and limits

- Local, single-owner evaluation of deterministic workforce analytics and governed aggregate investigations is the intended starting point. Evidence availability depends on the uploaded fields and sample size.
- Predictive model training and activation are explicit lifecycle actions. Insufficient outcomes or unusable inputs can block them. Model outputs are estimates and require evidence review.
- Small-group suppression and evidence gates can intentionally return limited or unavailable results. Do not interpret missing output as zero or as a favorable outcome.
- Succession, causal and individual-level legacy routes include retired or restricted surfaces. A navigation category is not a promise that every historical endpoint remains supported.
- The system must not execute or determine termination, discipline, demotion or pay reduction. Human review of the data and evidence remains necessary.
- Legacy process-global state remains transition debt. This beta guide does not establish multi-tenant isolation, organization-wide authentication, production capacity or regulatory compliance.
- Desktop packaging exists in the repository, but this guide does not assert that release binaries are published, signed or verified on every operating system.

## Storage, backup and recovery

Local persistence is split across several locations. Source runs do not automatically place every mutable file under a single directory.

| State | Default location / override |
| --- | --- |
| Dataset artifacts and lifecycle registry | OS-native PeopleOS home; `PEOPLEOS_HOME` overrides its root; `PEOPLEOS_WORKSPACE_REGISTRY` can separately override the registry |
| SQLite data | `data/peopleos.db` from `config.yaml`; setting `PEOPLEOS_HOME` redirects it to `<home>/data/peopleos.db` |
| Operational jobs | `.peopleos/jobs.json`; `PEOPLEOS_JOB_REGISTRY` override |
| Agent audit log | `data/agent_audit.jsonl`; `PEOPLEOS_AGENT_AUDIT_PATH` override |
| Legacy sessions and logs | `sessions/` and `logs/peopleos.log`, configurable in `config.yaml` |

The default PeopleOS home is `~/Library/Application Support/PeopleOS` on macOS; `%LOCALAPPDATA%\PeopleOS` on Windows (with an APPDATA/home fallback); and `$XDG_DATA_HOME/peopleos` or `~/.local/share/peopleos` on Linux. The path definitions are in [src/local_paths.py](https://github.com/omoniyi-ipaye/PeopleOS/blob/main/src/local_paths.py).

There is no full backup/restore endpoint in the current API. For a manual preservation copy, stop every PeopleOS process first, then copy the entire PeopleOS home, the configured database/job/audit/session/log locations, your configuration and original input files to protected storage. Record the source commit and environment overrides with the copy. Do not treat a CSV export or the registry alone as a complete backup. Copies contain sensitive data; Git exclusions are not encryption. Restore testing across versions and full recovery of in-memory predictive artifacts are not promised by this beta.

The governed `POST /api/platform/health/recover` action is **metadata recovery**, restricted to authorized owner/admin access. It can quarantine and reinitialize damaged registry metadata and mark interrupted jobs failed. It does not restore employee data, recover all historical state or activate a model. Preserve files before invoking it: metadata reinitialization can invalidate active runtime evidence while retaining source artifacts. If restart or recovery fails, retain the original files, reproduce with fictional data if possible, and report a sanitized error instead of editing registry JSON by hand.

## Troubleshooting and feedback

| Symptom | What to check |
| --- | --- |
| UI cannot reach API | Confirm both terminals are running and the API is on port 8000; check the local health endpoint. |
| Sample unavailable | Check for an existing active dataset and confirm the checkout includes `sample_hr_data.csv`. |
| Pay analysis unavailable | Check annual-pay and currency declarations, then reimport; no automatic FX conversion is performed. |
| AI summary unavailable | Use the deterministic evidence summary; check optional Ollama separately. |
| Predictive result unavailable | Check dataset outcomes, training/evaluation state and model activation in System Health. |
| Restart shows integrity/recovery error | Preserve state and inspect the error; metadata recovery is not a data backup restore. |

Use the [bug report](https://github.com/omoniyi-ipaye/PeopleOS/blob/main/.github/ISSUE_TEMPLATE/bug_report.md) or [feature request](https://github.com/omoniyi-ipaye/PeopleOS/blob/main/.github/ISSUE_TEMPLATE/feature_request.md) template. Include your source commit, operating system, Python/Node versions, installation route, expected behavior and minimal synthetic reproduction. Remove employee names/IDs, free-text reviews, salaries, tokens, local usernames and private paths from screenshots and logs. For suspected vulnerabilities, follow [SECURITY.md](../SECURITY.md).
