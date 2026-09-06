# PeopleOS Agent System Upgrade

Status: **TRANSITION implementation complete; verification in progress**  
Branch: `upgrade/agent-system-foundation`

## Purpose

Evolve PeopleOS from a local People Analytics application with direct LLM prompting into a governed People Intelligence System that:

- preserves deterministic analytics as the source of truth;
- uses AI for interpretation and synthesis, not enforcement;
- traces conclusions to workspace, dataset, model and tool evidence;
- makes model/data lifecycle explicit;
- keeps consequential employment actions outside the autonomy boundary;
- detects degraded control-plane state and recovers only inside a bounded L2 envelope.

The canonical structural source of truth is `model/system.json`.

---

## AS-IS

```text
User
  |
  v
Next.js UI
  |
  v
FastAPI routes
  |
  +--> Analytics / ML / Compensation / Fairness / Survival / Scenario / Sentiment / Structural engines
  |
  +--> Strategic Advisor
           |
           +--> build analytics context
           +--> prompt Ollama directly
           +--> parse free text

Process-global AppState
  +--> active dataframe
  +--> initialized analytics engines
  +--> trained ML state
  +--> LLM client
```

Material AS-IS weaknesses observed during review:

1. advisor model output could bypass the intended validation path;
2. API defaulted to `0.0.0.0`;
3. PeopleOS had no explicit agent planner/tool/evidence model;
4. state and model lifecycle were implicit inside a process-global singleton;
5. data loading could initialize/train predictive models;
6. no durable investigation/workspace/model/dataset identity existed;
7. no system-level fitness/recovery model existed;
8. the frontend security/runtime baseline was on the Next.js 14 / React 18 generation.

---

## TARGET

```text
People / HR user
       |
       v
+---------------------------+
| Next.js product surface   |
| Advisor + System Health   |
+-------------+-------------+
              |
              v
+---------------------------+
| FastAPI trust boundary    |
| local owner / remote role |
+-------------+-------------+
              |
      +-------+--------+
      |                |
      v                v
+-------------+   +------------------+
| Workspace   |   | People           |
| control     |   | Intelligence     |
| plane       |   | Agent            |
+------+------+   +--------+---------+
       |                   |
       |                   v
       |            deterministic plan
       |                   |
       |                   v
       |            governed tool registry
       |                   |
       |                   v
       |              EvidenceBundle
       |       provenance/confidence/coverage
       |            sufficiency/unknowns
       |                   |
       |              sufficiency gate
       |                   |
       |          +--------+--------+
       |          |                 |
       |     insufficient       sufficient
       |          |                 |
       |   deterministic         policy-bound
       |     limitations        LLM synthesis
       |          |                 |
       |          +--------+--------+
       |                   |
       |                   v
       |             verified answer
       |                   |
       |                   v
       |               audit event
       |
       +--> dataset lifecycle
       |     validate -> version -> activate -> supersede
       |
       +--> model lifecycle
       |     create -> train -> evaluate -> candidate
       |                        -> reject
       |     candidate --governed activation--> active -> retired
       |
       +--> investigation sessions
       |
       +--> idempotent operation jobs
       |
       +--> health / fitness / bounded recovery
```

---

## System / process / build layers

### System layer

Implemented components:

- canonical `model/system.json`;
- local-first FastAPI trust boundary;
- deterministic RBAC policy;
- workspace control-plane store;
- dataset versions and activation state;
- model versions and lifecycle state;
- persistent investigation sessions;
- People Intelligence Agent;
- allowlisted aggregate tools;
- evidence sufficiency and provenance;
- policy-bound synthesis and deterministic fallback;
- privacy-preserving audit;
- health, freshness/fitness and bounded recovery;
- System Health product UI.

### Process layer

#### Investigation

```text
Request
 -> authenticate/establish trusted actor
 -> authorize investigate
 -> resolve workspace
 -> resolve/open investigation session
 -> bind active dataset/model provenance
 -> deterministic evidence plan
 -> execute allowlisted aggregate tools
 -> aggregate evidence
 -> detect unknowns/contradictions
 -> calculate confidence + coverage + sufficiency
 -> insufficient? deterministic limitation response
 -> otherwise optional local LLM synthesis
 -> employment-action policy enforcement
 -> audit
 -> return answer + evidence + warnings
```

#### Dataset lifecycle

```text
Upload
 -> legacy parser/runtime load [TRANSITION compatibility]
 -> hash source
 -> quality profile
 -> register dataset version
 -> validated
 -> activate
 -> previous active version becomes superseded
```

#### Model lifecycle

```text
Explicit train request
 -> permission gate
 -> idempotent job creation
 -> TRAINING
 -> legacy MLEngine invoked only across training boundary
 -> EVALUATING
 -> deterministic minimum-quality gate
 -> CANDIDATE or REJECTED
 -> separate model.activate permission gate
 -> ACTIVE
 -> previous active model RETIRED
```

#### Health/recovery

```text
Desired state
 -> sense registry / dataset / model / job state
 -> detect inconsistency, staleness or interruption
 -> diagnose
 -> authorize response
 -> metadata-only recovery if inside L2 envelope
 -> verify health
 -> expose degraded state where governed action is required
```

### Build layer

| Step | Build contract | State change / completion evidence | Status |
|---|---|---|---|
| STEP-001 | Establish workspace identity | durable workspace record | Done |
| STEP-002 | Register dataset version | hash, schema, rows, quality recorded | Done |
| STEP-003 | Activate dataset | one active dataset; prior active superseded | Done |
| STEP-004 | Train versioned model explicitly | model + idempotent job enter lifecycle | Done |
| STEP-005 | Evaluate model | deterministic pass/fail evidence | Done |
| STEP-006 | Governed activation | candidate → active; prior model retired | Done |
| STEP-007 | Open investigation session | session bound to workspace/dataset/model | Done |
| STEP-008 | Governed investigation | evidence, sufficiency, policy, audit | Done |
| STEP-009 | Health + recovery | fitness report / bounded actions | Done |
| STEP-010 | Release verification | architecture + agent + frontend gates | In verification |

---

## Controls and enforceable boundaries

### Model / agent controls

- all existing Ollama generation is wrapped by the guarded client transport;
- advisor/agent output is policy-validated outside prompt text;
- agent tools are explicitly allowlisted;
- first agent release is read-only and aggregate-oriented;
- fairness evidence suppresses small groups;
- client cannot choose its own trusted actor identity;
- insufficient evidence disables probabilistic synthesis;
- deterministic fallback remains available without Ollama.

### Access controls

Local loopback is the trusted owner. Wider exposure requires:

- `PEOPLEOS_API_TOKEN`;
- server-configured `PEOPLEOS_API_ROLE`;
- optional server-configured `PEOPLEOS_API_ACTOR_ID`.

Remote roles:

- viewer — read-only lifecycle/system state;
- analyst — read + investigations/sessions;
- admin — workspace/data/model administration and bounded recovery;
- owner — trusted local owner.

Clients cannot self-assert a role through a request header.

### Consequential-action boundary

The agent and self-healing loop have no authority to:

- terminate or discipline employees;
- demote employees;
- reduce compensation;
- change employee source data;
- activate a model automatically;
- change policy/evaluation thresholds automatically.

---

## Evidence and provenance

Every governed investigation now produces a canonical `EvidenceBundle` with:

- tool execution records;
- evidence items;
- confidence;
- execution coverage;
- sufficiency: `sufficient | limited | insufficient`;
- material unknowns;
- contradictions;
- verification notes;
- workspace ID;
- dataset version;
- model version.

Investigation persistence stores request IDs and SHA-256 question hashes rather than raw question text. Agent audit logging similarly avoids storing the raw answer/evidence payload by default.

---

## Reliability and adaptive operation

### Fitness policy

Current deterministic checks include:

- active dataset exists;
- active dataset freshness;
- active model state consistency;
- model freshness;
- minimum recorded model AUC where a model is active;
- workspace registry consistency;
- interrupted operation jobs.

### Autonomy level

**L2 — bounded auto-heal**.

Automatic response may:

- recreate an empty/missing local workspace metadata shell;
- ensure the local workspace exists;
- convert an interrupted RUNNING operation into FAILED with safe-retry evidence.

Maximum autonomous blast radius: **PeopleOS control-plane metadata**.

Structural changes, model activation, employee data mutation, policy changes and employment decisions require governed action.

---

## Frontend modernization

Target baseline:

- Next.js 16;
- React 19;
- Recharts 3;
- maintained current supporting dependencies.

The migration exposed stricter Recharts formatter typing and one numeric-format issue. These were fixed at the call sites rather than disabling TypeScript. The production build remains an objective release gate until CI is green and the regenerated lockfile is committed.

---

## Transition debt — explicit, not hidden

The upgrade deliberately uses a reversible transition rather than a high-risk big-bang rewrite.

The remaining legacy compatibility boundary is `AppState`:

```text
TARGET control plane / agent
        |
        +--> stable workspace/dataset/model/session identity
        |
        +--> explicit new lifecycle APIs
        |
        v
TRANSITION compatibility
        |
        +--> legacy AppState dataframe / analytics engines
        +--> legacy analytical routes
        +--> historical load_data initialization behavior
```

This means the architecture is not represented as fully migrated. New agent/control-plane behavior is governed and versioned, while legacy analytical routes are migrated incrementally behind the compatibility boundary.

This is the principal assumption behind the current `BUILD READY WITH ASSUMPTIONS` verdict.

---

## Verification

Automated verification includes:

- agent evidence contracts;
- deterministic planning/orchestration;
- malicious/prohibited model response enforcement;
- aggregate evidence selection;
- remote-access controls;
- fairness group suppression;
- audit privacy;
- workspace lifecycle invariants;
- dataset version/activation behavior;
- candidate-only model activation;
- deterministic model evaluation;
- question hashing;
- bounded recovery;
- RBAC;
- job idempotency and interruption handling;
- data/model fitness checks;
- canonical model validation;
- build-readiness validation;
- Next.js lint and production build.

Primary commands:

```bash
python scripts/validate_system_model.py
python scripts/check_build_readiness.py
pytest -q tests/test_agent_*.py tests/test_platform_*.py
cd web && npm run lint && npm run build
```

---

## Readiness verdict

Current structural verdict: **BUILD READY WITH ASSUMPTIONS**.

Required conditions before merge/release:

1. Agent Foundation CI passes at the final branch SHA.
2. Frontend Modernization CI passes at the final branch SHA.
3. generated `web/package-lock.json` reflects the Next.js 16 / React 19 dependency graph.
4. no new architecture validation or security-control failure exists.
5. merge into `main` is a governed owner decision.

We deliberately do **not** claim full TARGET completion while legacy AppState-backed analytical routes remain in the transition layer.
