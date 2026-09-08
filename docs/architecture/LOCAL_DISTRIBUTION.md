# PeopleOS Local Distribution

Status: **TRANSITION — local-first distribution implementation**

## Product decision

PeopleOS is a **local-first product**. The primary experience is a downloadable application that runs workforce data and analytics on the user's computer. A hosted environment may exist as a synthetic-data demo, but real workforce analysis does not depend on a PeopleOS cloud service.

## Experience contract

A non-technical People/HR user should be able to:

1. Download PeopleOS for their operating system.
2. Open the application without installing Python, Node.js, Docker, a database, or an AI model.
3. Explore synthetic sample data or add a CSV/JSON workforce source.
4. Close and reopen PeopleOS without losing the active local workspace.
5. Upgrade PeopleOS without moving or overwriting workforce data.
6. Use deterministic People Intelligence without an LLM.
7. Enable local AI separately when desired.

The product must never make developer setup part of normal onboarding.

## Target architecture

```text
Installer / executable
        |
        v
PeopleOS local launcher
        |
        +--> selects free loopback port
        +--> resolves OS-native PeopleOS home
        +--> restores durable local workspace
        +--> starts bundled FastAPI runtime
        |
        v
http://127.0.0.1:<ephemeral-port>
        |
        +--> /api/*       governed PeopleOS API
        +--> /*           statically exported PeopleOS UI
        |
        v
Default browser / future thin native shell
```

The browser/native shell is presentation only. It does not own workforce state.

## Mutable local storage

All mutable data lives outside the application bundle.

```text
PeopleOS home/
├── data/
│   └── peopleos.db
├── control-plane/
│   └── workspaces.json
├── datasets/
├── models/
├── investigations/
├── exports/
├── backups/
├── logs/
└── config/
```

Default roots:

- Windows: `%LOCALAPPDATA%\\PeopleOS`
- macOS: `~/Library/Application Support/PeopleOS`
- Linux: `$XDG_DATA_HOME/peopleos` or `~/.local/share/peopleos`
- Override for portable/test use: `PEOPLEOS_HOME`

## Runtime boundaries

### Always included

- Current population resolution
- Deterministic workforce analytics
- Compensation and fairness screening
- Cohort survival/statistical analysis
- Quality-of-hire association analysis
- Scenario exploration
- Evidence-ledger People Intelligence fallback
- SQLite persistence and lifecycle registry

### Optional

- Predictive training/runtime
- Semantic/vector search
- Local LLM synthesis

Optional capabilities must degrade to an intentional product state. They must never make the local application fail to start.

## UI build strategy

The existing Next.js application remains the canonical product UI.

For normal web development it runs as a Next.js server and proxies `/api/*` to FastAPI.

For desktop packaging:

```bash
PEOPLEOS_DESKTOP_BUILD=1 npm run build
```

Next.js exports static assets into `web/out`. The packaged FastAPI process serves those files and the API on the same loopback origin, avoiding a second Node runtime in the installed application.

## Packaging

`desktop/peopleos.spec` creates a self-contained executable with PyInstaller. The package includes:

- Python runtime
- PeopleOS API/analytics code
- immutable `config.yaml`
- sample dataset/template assets
- exported web UI

The user's mutable workspace is never bundled into the executable.

Cross-platform builds are produced by `.github/workflows/local-desktop-build.yml` for Windows, macOS, and Linux. Every built executable must pass a smoke test that starts the packaged runtime and successfully retrieves both `/api/health` and the PeopleOS UI.

## Security boundary

The packaged runtime binds to `127.0.0.1` only and uses an ephemeral free port. It is not a LAN server. Existing PeopleOS local-first access controls remain authoritative for API behavior.

## Current transition gaps

1. The first packaging iteration opens the system browser. A thin native shell may replace this later without changing runtime/storage architecture.
2. The everyday desktop runtime uses `requirements-desktop.txt`, which excludes the heavier predictive training packages. `requirements-predictive.txt` provides the optional predictive tier; the source core dependency bundle still includes that heavier stack.
3. Model artifact persistence must use the `models/` directory before predictive activation is considered restart-durable.
4. Signed/notarized installers and automatic updates are a release-packaging layer after cross-platform executable builds are stable.
5. Backup/export UX still needs to be connected to the `backups/` and `exports/` directories.

## Build-readiness definition for local relaunch

Local distribution becomes **BUILD READY** when the same commit proves:

- architecture/output-integrity tests pass;
- normal frontend production build passes;
- static desktop frontend export passes;
- packaged executable smoke test passes on Windows, macOS and Linux;
- full browser E2E journey passes;
- closing/restarting restores a previously loaded local dataset;
- no optional model/LLM dependency is required for first-run analytics.
