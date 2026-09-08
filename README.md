# PeopleOS: Governed People Intelligence System

PeopleOS is a **local-first, privacy-preserving People Intelligence platform** for People Operations and HR leaders. It combines deterministic workforce analytics with a governed AI investigation layer so users can ask complex workforce questions, inspect the supporting evidence, understand limitations, and keep consequential employment decisions outside the agent autonomy boundary.

Sensitive workforce data is processed inside the PeopleOS environment. The default API binding is loopback-only, local AI synthesis can run through Ollama, and remote API access must be explicitly enabled and authenticated.

## Try the public beta

Start with the [Public Beta Guide](docs/PUBLIC_BETA_GUIDE.md) for a local source install, fictional sample data, salary declarations, recovery limits and troubleshooting. The beta scope is local evaluation; published installers and production readiness are not implied by the source build instructions.

- [First run and sample walkthrough](docs/PUBLIC_BETA_GUIDE.md#first-run-with-fictional-data)
- [Import and salary requirements](docs/PUBLIC_BETA_GUIDE.md#import-your-own-test-data)
- [Security and safe reporting](SECURITY.md)
- [Bug report](.github/ISSUE_TEMPLATE/bug_report.md) and [feature request](.github/ISSUE_TEMPLATE/feature_request.md) templates

---

## What PeopleOS Does

### Governed People Intelligence Agent
PeopleOS investigates workforce questions through a deterministic plan and an allowlisted set of aggregate analytics tools:

`question → plan → governed tools → evidence → sufficiency/confidence → policy gate → synthesis → verification/audit`

The agent:
- uses aggregate workforce evidence rather than unrestricted raw employee records;
- records workspace, dataset and model provenance;
- distinguishes **sufficient**, **limited** and **insufficient** evidence;
- refuses to infer a missing conclusion when evidence is insufficient;
- falls back to deterministic evidence summaries when Ollama is unavailable;
- blocks punitive or consequential employment recommendations at the model boundary;
- never has authority to terminate, demote, discipline, reduce pay, or otherwise execute consequential employment actions.

### Workforce Analytics
PeopleOS retains its deterministic analytics capabilities, including:
- retention and attrition analysis;
- survival forecasting;
- compensation and pay-equity analysis;
- workforce and department health;
- employee-experience and sentiment analysis;
- fairness analysis with small-group suppression;
- succession and quality-of-hire analytics;
- organizational structure and network analysis;
- scenario and causal analysis where supported by the dataset.

### Workspace and Lifecycle Control Plane
PeopleOS now maintains explicit control-plane identities for:
- **workspaces**;
- **dataset versions** and active dataset state;
- **model versions**;
- model lifecycle: `created → training → evaluating → candidate → active / rejected / failed`;
- **investigation sessions**;
- idempotent operational jobs;
- health, fitness and bounded recovery state.

The System Health page at `/platform` shows dataset/model history, active versions, fitness checks, access role, investigation state and the self-healing boundary.

---

## Safety, Privacy and Governance

PeopleOS uses deterministic controls around the probabilistic AI layer.

### Local-first access
The backend defaults to `127.0.0.1`. Loopback traffic is treated as the trusted local owner.

If you intentionally expose the API beyond loopback, configure both an API token and a server-side role:

```bash
export PEOPLEOS_API_HOST=0.0.0.0
export PEOPLEOS_API_TOKEN='replace-with-a-strong-secret'
export PEOPLEOS_API_ROLE='analyst'   # viewer | analyst | admin
export PEOPLEOS_API_ACTOR_ID='named-operator'
```

Remote clients cannot choose their own role in a request header. The role is assigned server-side after the bearer-token boundary succeeds.

### Role boundaries
- **owner**: trusted local owner; full control-plane access;
- **admin**: workspace/data/model administration, including governed model activation and recovery;
- **analyst**: investigations and read access, but no model activation or recovery;
- **viewer**: read-only system and lifecycle visibility.

### Bounded self-healing
PeopleOS currently operates at **L2 bounded auto-heal** for control-plane metadata only. Automatic recovery may repair registry metadata or mark interrupted jobs as safely retryable. It cannot:
- change employee data;
- activate a model;
- change policy thresholds;
- execute employment actions.

### Git safety
Runtime files span the OS-native PeopleOS data directory and, for source runs, repository-relative paths such as `.peopleos/`, `data/`, `sessions/` and `logs/`. See the [storage and recovery notes](docs/PUBLIC_BETA_GUIDE.md#storage-backup-and-recovery). Git exclusions do not encrypt data or make files safe to share.

---

## Architecture

The canonical machine-readable system definition is:

`model/system.json`

The architecture upgrade record is:

`docs/architecture/AGENT_SYSTEM_UPGRADE.md`

The canonical model captures the system boundary, AS-IS/TRANSITION/TARGET components, typed flows, lifecycle states, controls, risks, health loop, bounded autonomy and dependency-ordered build steps.

PeopleOS intentionally keeps the deterministic analytics engines responsible for calculations and state. The LLM is used for interpretation and synthesis, not as the source of truth and not as the enforcement layer.

---

## Setup

### Prerequisites
- Python 3.10+
- Node.js 22 recommended for the current Next.js 16 frontend
- Ollama is optional; core analytics and deterministic agent fallback do not require it

### Install the default core runtime

The default installation intentionally excludes the transformer/GPU/vector-search stack. It includes the API, deterministic analytics, predictive/statistical engines, governed agent runtime, local Ollama client and the transitional legacy Streamlit dependency required by the current ML module.

```bash
python -m pip install -r requirements.txt

cd web
npm ci
cd ..
```

### Optional advanced embeddings and semantic search

Install the advanced tier only when you want local sentence-transformer embeddings or FAISS semantic search:

```bash
pip install -r requirements-advanced.txt
```

The advanced tier includes the core runtime automatically. A normal PeopleOS install does not require Torch, sentence-transformers or FAISS.

### Optional local AI synthesis

Install Ollama and pull a compatible model, for example:

```bash
ollama serve
ollama pull gemma3
```

### Run PeopleOS

**Backend:**

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Alternatively, run `python -m api.main` from the repository root; it also defaults to loopback.

**Frontend:**

```bash
cd web
npm run dev -- --hostname 127.0.0.1
```

Open `http://localhost:3000`.

API documentation is available locally at `http://127.0.0.1:8000/docs`.

---

## Data Lifecycle

PeopleOS uses a Golden Schema. A template is available through the application/API.

Typical core fields include:
- `EmployeeID`
- `Dept`
- `Tenure`
- `Salary`
- `LastRating`
- `Age`

Additional fields unlock additional analytical capabilities, for example:
- `Attrition` for attrition/retention modelling;
- `PerformanceText` for NLP/sentiment capabilities;
- `HireSource` and interview scores for quality-of-hire analytics.

When data is uploaded, PeopleOS validates and preprocesses it, records a dataset version containing its content hash, row count, columns and basic quality metrics, and activates that dataset version. Predictive training and advanced vector indexing are not implicitly authorized by the upload action; they belong to their explicit lifecycle boundaries.

---

## Model Lifecycle

Predictive model lifecycle is explicit in the new control plane:

1. create a model version;
2. enter training state;
3. evaluate the resulting metrics;
4. accept as a candidate only if the deterministic evaluation policy passes;
5. activate only through a permitted governance action;
6. retire the previous active version;
7. monitor freshness and quality.

Agent investigations never implicitly authorize model activation.

> **Transition note:** Some legacy analytical routes still use the original process-global `AppState` runtime and its historical initialization behavior. The governed workspace/model control plane isolates the new lifecycle and agent surfaces while those legacy routes are migrated incrementally. This is tracked explicitly as transition debt rather than hidden as completed work.

---

## Verification and Build Readiness

Architecture checks can be run directly:

```bash
python scripts/validate_system_model.py
python scripts/check_build_readiness.py
```

The CI release gates cover:
- governed agent evidence/orchestration;
- model policy enforcement;
- privacy and access controls;
- workspace/data/model/session lifecycle invariants;
- RBAC, job idempotency and bounded recovery;
- canonical system-model validation;
- Next.js 16 / React 19 lint and production build;
- a browser-level user journey that boots the core runtime without Torch/sentence-transformers, activates sample data, opens System Health and runs a governed People Intelligence investigation.

The current architecture target is **BUILD READY WITH ASSUMPTIONS** while the legacy AppState routes remain in the transition layer. The repository must not be described as fully migrated until that debt is removed.

---

## Version

The current branch represents the **3.0 transition architecture**: PeopleOS evolving from a local HR analytics application into a governed People Intelligence system with explicit lifecycle, evidence, authorization and health boundaries.

---

## License

See the repository `LICENSE` file for the authoritative license terms. Any additional commercial-use conditions should be interpreted only from the repository’s actual license text, not from this README.
