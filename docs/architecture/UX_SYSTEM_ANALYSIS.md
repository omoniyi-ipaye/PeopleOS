# PeopleOS UX System Analysis

## Purpose

This document applies the PeopleOS System Design Architect method to the product experience. It treats UI/UX as a system of user goals, states, decisions, evidence, controls, handoffs, feedback and recovery—not as a collection of screens.

Lifecycle labels used below:
- **Observed** — confirmed from the current application, code or E2E screenshots.
- **Assumed** — reasonable product assumption requiring later validation.
- **Proposed** — target experience recommendation.

---

# 1. Current understanding

## Primary product promise

PeopleOS should help a People/HR leader move from workforce data to an evidence-backed decision without requiring them to understand the underlying analytics architecture.

The desired experience is:

`Connect workforce data → understand what matters → investigate why → decide what to explore or plan → verify evidence → act outside PeopleOS where human judgment is required`

The agent is not the product by itself. The product is the controlled decision-support loop around the agent and deterministic analytics.

## Core actors

| Actor | Primary job | Required experience |
| --- | --- | --- |
| People/HR Leader | Understand workforce health and decide what needs attention | Signal-first, low-jargon, executive-ready |
| People Analyst | Investigate patterns and validate evidence | Traceability, filters, evidence, methodology |
| People/Admin Operator | Maintain data/model/system readiness | Lifecycle visibility, recovery, permissions |
| Executive/Viewer | Consume trusted findings | Concise findings, provenance, limitations |

---

# 2. AS-IS experience map

```text
START
  |
  +--> No data
  |      |
  |      +--> Upload Data / Load Sample
  |              |
  |              +--> Data active
  |
  +--> Dashboard
  |      |
  |      +--> People Analytics
  |      |       +--> Workforce Health
  |      |       +--> Employee Experience
  |      |       +--> Flight Risk
  |      |
  |      +--> Strategic Planning
  |      |       +--> Scenario Planner
  |      |       +--> Retention Forecast
  |      |
  |      +--> Talent Management
  |      |       +--> Quality of Hire
  |      |
  |      +--> AI Tools
  |              +--> HR Advisor
  |              +--> PeopleOS Research
  |
  +--> Data & Settings
         +--> Upload Data
         +--> Sessions
         +--> System Health
         +--> Settings
```

### What this structure optimizes for
**Observed:** It mirrors application capabilities and internal functional categories.

### Where it creates friction
**Observed / inferred from the E2E journey:** A user must already know which PeopleOS capability contains the answer they need. The navigation is feature-oriented rather than decision-oriented.

---

# 3. AS-IS system experience findings

## F1 — Product entry does not adapt enough to system state

**Observed:** When data is active, the Upload page still gives the large upload drop zone high visual priority.

**Impact:** The user is not clearly moved from setup mode into analysis mode.

**Proposed rule:** Every primary screen should respond to the system lifecycle state.

```text
NO DATA       -> Setup state
DATA ACTIVE   -> Understand state
MODEL READY   -> Predictive capability available
MODEL ACTIVE  -> Governed predictive state
DEGRADED      -> Explain limitation + safe fallback
```

## F2 — Global status previously confused data readiness with model activation

**Observed:** The header displayed `ML Active` from the presence of predictive data while System Health showed no active model.

**Fix:** Replaced with `Predictive data ready`.

**Target:** Global context must independently represent:
- Dataset state
- Data freshness
- Predictive-data readiness
- Active model state
- AI synthesis availability

Never compress different lifecycle states into one green badge.

## F3 — Navigation reflects implementation domains

**Observed:** Sections such as People Analytics, Strategic Planning, Talent Management and AI Tools expose internal feature groupings.

**Impact:** Users must translate a business question into a product module before they can start.

**Proposed:** Organize the primary experience around the decision cycle:

```text
UNDERSTAND  -> What is happening?
INVESTIGATE -> Why is it happening?
PLAN        -> What could we do / what may happen next?
GOVERN      -> Can I trust the data/model/system?
```

## F4 — People Intelligence is visually strong but cognitively overloaded

**Observed:** The current Advisor page exposes agent architecture terminology such as allowlisted tools, deterministic synthesis and policy gating directly in the main surface.

**Positive:** This creates trust for expert users.

**Problem:** It competes with the business finding.

**Proposed:** Progressive disclosure:
1. Finding
2. Why it matters
3. Supporting evidence
4. Confidence / coverage
5. Advanced trace (tools, provenance, policy)

## F5 — Coverage and confidence need distinct semantics

**Observed:** A result could display `partial` while also showing `90% confidence`.

**Fix:** UI now labels these as `Partial coverage` and `90% confidence in available evidence`.

**Target model:**
- **Coverage** = how much of the requested question PeopleOS could verify.
- **Confidence** = how reliable the evidence PeopleOS did obtain appears to be.

These must never be merged into one score.

## F6 — Technical engine errors leaked into the People experience

**Observed:** The E2E screenshot exposed a NumPy dtype exception inside Limitations & Controls.

**Fix:** The UI now translates technical errors into business-readable capability gaps.

**Target:** Raw stack/library errors are observable to engineering logs only. Users see:
- what capability was unavailable;
- which part of the answer is affected;
- whether the rest of the answer is still trustworthy;
- what they can do next.

## F7 — System Health is useful but oriented toward builders

**Observed:** System Health exposes lifecycle IDs, L2 recovery vocabulary and implementation-level state.

**Proposed two-level view:**

**People-safe view**
- Workforce data: Ready / Needs attention
- Data freshness
- Predictive model: Active / Not trained / Needs review
- AI synthesis: Available / Deterministic fallback
- Overall trust state

**Advanced controls**
- Dataset IDs
- Model IDs
- jobs
- recovery class
- policy IDs
- detailed fitness checks

---

# 4. Target information architecture

```text
PeopleOS
|
+-- HOME / DECISION COCKPIT
|     +-- What needs attention
|     +-- What changed
|     +-- Data trust state
|     +-- Continue recent investigations
|
+-- UNDERSTAND
|     +-- Workforce Health
|     +-- Employee Experience
|     +-- Talent / Hiring Quality
|     +-- Retention Signals
|
+-- INVESTIGATE
|     +-- People Intelligence
|     +-- Investigation history
|     +-- Evidence workspace
|
+-- PLAN
|     +-- Scenario Planner
|     +-- Retention Forecast
|     +-- Workforce scenarios
|
+-- GOVERN
      +-- Data & Sources
      +-- Model Lifecycle
      +-- System Health
      +-- Access & Settings
```

### Why this is better
The information architecture matches the user’s decision loop rather than the codebase structure.

---

# 5. Target user journey

## Journey A — First-time user

```text
OPEN PEOPLEOS
   |
   v
No workforce data detected
   |
   v
SET UP DATA
   |-- Upload file
   |-- Load example
   |-- See schema template
   v
VALIDATE
   |-- Records
   |-- Coverage
   |-- Data quality issues
   v
ACTIVATE DATASET
   |
   v
WHAT PEOPLEOS CAN DO WITH THIS DATA
   |-- Available now
   |-- Requires model training
   |-- Requires additional fields
   v
DECISION COCKPIT
```

### Completion evidence
- Dataset ID/version created
- Active dataset state
- Quality summary visible
- User knows which capabilities are available

### Failure recovery
- Unsupported fields -> mapping guidance
- Insufficient records -> analytics-only path
- Missing predictive target -> no model claims
- Upload failure -> retained diagnostic and retry path

---

## Journey B — Returning People leader

```text
OPEN PEOPLEOS
   |
   v
DECISION COCKPIT
   |
   +--> What changed since last dataset?
   +--> What needs attention now?
   +--> Is the system/data trustworthy?
   |
   v
SELECT SIGNAL
   |
   v
INVESTIGATE
   |
   +--> Finding
   +--> Evidence
   +--> Coverage
   +--> Confidence
   +--> Unknowns
   |
   v
DECIDE NEXT STEP
   +--> Explore segment
   +--> Open relevant analysis
   +--> Run scenario
   +--> Save investigation
```

---

## Journey C — Ask People Intelligence directly

```text
QUESTION
   |
   v
Intent + evidence plan
   |
   v
Governed tools execute
   |
   v
Coverage gate
   |       
   +-- insufficient --> Explain what is missing + suggested next evidence
   |
   +-- sufficient/partial
           |
           v
        FINDING
           |
           +--> Evidence
           +--> Confidence
           +--> Gaps
           +--> Relevant next actions
           |
           v
        SAVE / CONTINUE / OPEN ANALYSIS
```

---

## Journey D — Predictive model lifecycle

```text
Predictive-ready dataset
   |
   v
TRAIN MODEL
   |
   v
Training job
   |
   +-- failed --> explain + retry
   |
   v
EVALUATION
   |
   +-- rejected --> show why; no activation option
   |
   +-- candidate
           |
           v
       REVIEW METRICS
           |
           v
       AUTHORIZED ACTIVATE
           |
           v
       ACTIVE MODEL
```

No page should imply that a model is active before this sequence completes.

---

# 6. Target page/state catalogue

## UX-01 Decision Cockpit

**Trigger:** authenticated/local user opens PeopleOS with an active dataset.

**Inputs:** dataset state, data freshness, high-level analytics, recent investigations, active model state.

**Primary output:** 3–5 prioritized workforce signals, not a wall of KPIs.

**Required blocks:**
1. `What needs attention`
2. `What changed`
3. `Ask People Intelligence`
4. `Trust state`
5. `Continue where you left off`

**Control:** No unsupported causal language; every signal links to supporting analysis/evidence.

**Success evidence:** user reaches a relevant investigation or analysis in <=2 interactions.

---

## UX-02 Data Setup & Readiness

**Trigger:** no active dataset or user explicitly changes data.

**States:** EMPTY -> UPLOADING -> VALIDATING -> READY -> ACTIVE -> SUPERSEDED / FAILED.

**Required blocks:**
- source selection
- schema/data-quality validation
- capability unlock summary
- activation confirmation
- dataset history behind secondary disclosure

**Success:** user knows what became available after activation.

---

## UX-03 People Intelligence Workspace

**Trigger:** direct question or click from a signal.

**State model:** EMPTY -> PLANNING -> RUNNING -> COMPLETE / PARTIAL / INSUFFICIENT / FAILED.

**Primary hierarchy:**
1. Finding
2. Business implication
3. Evidence
4. Coverage + confidence
5. Unknowns
6. Next investigation options
7. Advanced execution trace

**Never show:** raw exceptions, stack traces, library names as user-facing limitations.

---

## UX-04 Analysis Detail

Applies to Workforce Health, Employee Experience, Retention Signals, Quality of Hire.

**Pattern:**

`Headline finding -> trend/context -> segment breakdown -> evidence -> ask follow-up -> methodology`

Charts should answer a question stated above them rather than exist as independent dashboard widgets.

---

## UX-05 Plan / Scenario Workspace

**Trigger:** user wants to explore a potential intervention or future state.

**Required separation:**
- Observed baseline
- Assumption being changed
- Projected effect
- Uncertainty
- What the simulation does NOT prove

**Control:** Simulation output cannot be presented as causal truth unless causal requirements are met.

---

## UX-06 Trust Center

Replace the default System Health mental model with a user-facing Trust Center; retain advanced System Health controls within it.

**Primary states:**
- Data trust
- Model trust
- AI availability
- System availability
- Privacy / access state

**Advanced:** control-plane IDs, job states, recovery actions, detailed policy state.

---

# 7. Navigation redesign

## Proposed primary sidebar

```text
Home

Understand
  Workforce Health
  Employee Experience
  Retention Signals
  Quality of Hire

Investigate
  People Intelligence
  Saved Investigations

Plan
  Scenario Planner
  Retention Forecast

Govern
  Data & Sources
  Model Lifecycle
  Trust Center
  Settings
```

### Naming changes
- `HR Advisor` -> `People Intelligence`
- `Flight Risk` -> `Retention Signals`
- `Upload Data` -> `Data & Sources`
- `System Health` -> `Trust Center` with Advanced System Health inside
- `Sessions` -> `Saved Investigations` when sessions are user-facing investigation continuity

---

# 8. Cross-screen context bar

Every analysis screen should have one consistent context strip:

```text
Dataset: September workforce snapshot
Updated: 2h ago
800 people
Predictive model: Not active
AI synthesis: Local / fallback
```

Do not expose raw UUIDs by default. IDs belong behind `Details`.

---

# 9. UI design principles

1. **Signal before metric** — first explain what needs attention, then show the number.
2. **Business language before system language** — implementation detail is progressive disclosure.
3. **State is visible** — dataset/model/AI status must never contradict another screen.
4. **Evidence is one click away** — findings must be inspectable.
5. **Unknowns are first-class** — absence of evidence is shown, not hidden.
6. **One primary action per state** — avoid dashboards with many equally weighted buttons.
7. **Consequential decisions remain human** — UI should suggest investigation/systemic actions, not punitive employment actions.
8. **Continuity over page hopping** — a user can ask a follow-up from any analysis surface while retaining context.
9. **Progressive technical depth** — People leaders see plain language; analysts/admins can open technical detail.
10. **Visual hierarchy follows decision importance**, not component type.

---

# 10. UX health signals

Track the product experience as a system:

| Signal | Target meaning |
| --- | --- |
| Time to first trusted insight | Setup + first useful finding |
| Interactions to investigation | How quickly a user moves from signal to evidence |
| Investigation completion rate | Questions ending in useful complete/partial results |
| Insufficient-evidence recovery | Users successfully obtain missing evidence or choose a next path |
| Evidence-open rate | Whether users inspect provenance when it matters |
| Wrong-state incidents | UI claiming model/data/AI status inconsistent with backend lifecycle |
| Raw-error exposure | Must remain zero |
| Return-to-context rate | Saved/recent investigations successfully resumed |

---

# 11. Dependency-ordered UX backlog

## P0 — Correctness and trust
- [x] Replace misleading `ML Active` header state
- [x] Separate coverage from confidence wording
- [x] Hide raw technical tool errors from People-facing limitations
- [ ] Add global lifecycle context bar backed by one canonical status contract
- [ ] Format all evidence values consistently (currency, percentage, count, duration)

## P1 — Journey and information architecture
- [ ] Replace feature-first sidebar with Understand / Investigate / Plan / Govern
- [ ] Convert home page into Decision Cockpit
- [ ] Convert Upload into lifecycle-aware Data & Sources flow
- [ ] Rename Flight Risk to Retention Signals
- [ ] Rename HR Advisor to People Intelligence
- [ ] Convert Sessions into Saved Investigations

## P1 — Investigation experience
- [ ] Put finding and implication ahead of execution mechanics
- [ ] Add contextual follow-up prompts based on known gaps
- [ ] Add Save / Continue investigation workflow
- [ ] Move tool IDs/policy details into Advanced Trace
- [ ] Add direct links from evidence to relevant analysis surfaces

## P2 — Trust and governance UX
- [ ] Create Trust Center summary
- [ ] Move advanced System Health details behind progressive disclosure
- [ ] Add model lifecycle UI: predictive-ready -> train -> evaluate -> candidate -> activate
- [ ] Make recovery actions explain scope and consequence before execution

## P2 — Validation
- [ ] Browser-test no-data onboarding
- [ ] Browser-test active-data returning-user path
- [ ] Browser-test partial/insufficient investigation path
- [ ] Browser-test model lifecycle state transitions
- [ ] Browser-test degraded/no-Ollama path
- [ ] Browser-test permission-limited remote role path

---

# 12. UX readiness verdict

## Current
**UX BUILD READY WITH ASSUMPTIONS**

The system mechanics are strong enough to support the target experience, and the primary E2E journey is verified. The current UI is functional but still exposes the product architecture more than the user’s decision process.

## Target exit criteria
The UX becomes **BUILD READY** when:
1. primary navigation matches the decision cycle;
2. no-data, data-active, model-ready/active and degraded states have explicit UI behavior;
3. Decision Cockpit, People Intelligence and Trust Center use one canonical lifecycle context;
4. raw technical errors cannot appear in People-facing UI;
5. E2E tests cover first-time, returning, investigation, model lifecycle and degraded paths;
6. user-visible metrics are consistently formatted and provenance remains inspectable.
