# PeopleOS

### Your local People Analyst — deterministic workforce analytics with governed AI.

PeopleOS is a **local-first People Intelligence system for People Operations, People Analytics and HR teams**.

Bring in workforce data, understand what matters, ask questions in normal People language, drill into cohorts and inspect the evidence behind every answer — without making an LLM the source of truth.

> **PeopleOS analyses. People decide.**

PeopleOS is publicly available as **source-available software**. Non-commercial evaluation, learning, research and contribution are welcome under the repository license. **Commercial use requires the express written permission of Omoniyi Ipaye.** See [Commercial Use](COMMERCIAL_USE.md).

---

## Why PeopleOS

Most People teams have data, dashboards and increasingly AI tools — but still have to answer the hardest questions manually:

- Where are people leaving?
- Which teams deserve a closer look?
- How does pay differ across groups?
- What changes when we look at tenure, location or job level?
- Is an observed relationship meaningful or just noise?
- What evidence actually supports this conclusion?

PeopleOS is designed to make those investigations fast **without hiding the calculation behind AI-generated prose**.

The system separates calculation from interpretation:

```text
workforce data
    ↓
deterministic analytics
    ↓
governed cohort / downstream analysis
    ↓
typed evidence + provenance
    ↓
AI interpretation
    ↓
clear People-team answer
```

The AI can organise, explain and investigate verified evidence. It does **not** invent the underlying metrics.

---

## The experience

PeopleOS is intentionally simple by default and powerful when you drill deeper.

### 1. Add workforce data

Start with fictional sample data or import your own test dataset.

Supported import formats:

- `.xlsx`
- `.csv`
- `.json`

PeopleOS validates the dataset before analysis and fails closed when required meaning or units are unclear.

### 2. See what deserves attention

The Home and Insights experiences surface the most useful available signals based on what the current dataset actually supports.

PeopleOS does not pretend a capability exists when the required evidence is missing.

### 3. Ask PeopleOS

Ask questions in normal People language, for example:

- `What is our current headcount?`
- `Where are people leaving?`
- `How does pay look across departments?`
- `What should I look at first?`
- `Show average salary by job level in Engineering.`
- `Compare attrition by location.`

Simple questions get simple answers.

The supporting evidence, coverage, methodology, provenance and raw verified response remain available through progressive disclosure when you want to inspect them.

### 4. Explore deeper

PeopleOS supports governed aggregate drill-down across available workforce dimensions, including:

- department
- location
- job level
- job title
- gender
- tenure
- age
- salary
- rating

Categorical and numeric filters can be stacked to investigate an exact cohort, subject to privacy/support thresholds.

Examples:

```text
Engineering → Madrid → L3 → tenure < 2 years
```

or conversationally:

```text
Average salary by job level for Engineering employees in Madrid with under 4 years tenure.
```

Supported downstream operations include grouped summaries, rates, correlations and crosstabs.

PeopleOS enforces minimum cohort/group sizes and blocks identifier-like drill-down fields. The normal analytical experience remains **aggregate, not employee-ranking software**.

---

## What PeopleOS can analyse

Capabilities are dataset-dependent, but the deterministic analytical layer includes:

- workforce composition and department health
- headcount and workforce structure
- recorded attrition and retention
- tenure and cohort retention analysis
- compensation distribution and pay analysis
- employee-experience measures when genuinely present in the data
- quality-of-hire analysis
- fairness analysis with small-group suppression
- organizational structure and network analysis
- governed scenario planning
- predictive retention signals through an explicit model lifecycle
- aggregate derived analysis through the governed downstream-analysis runtime

PeopleOS distinguishes descriptive evidence, configured/assumed constructs and predictive evidence rather than presenting them as equally certain.

---

## Trust by design

PeopleOS is built around a simple rule:

> **No number should become more authoritative just because AI explained it.**

The deterministic evidence contract preserves the important context around a result, including:

- value
- population / denominator
- eligible and excluded observations where relevant
- semantic meaning
- validation state
- source tool
- dataset provenance

PeopleOS deliberately fails closed in important cases.

It will not:

- turn missing evidence into `0`;
- claim there is no hotspot when outcome evidence is unavailable;
- claim parity merely because a comparison could not be established;
- present observational correlation as causation;
- describe a model score as a validated future-event probability without the required validation;
- rank employees for termination, demotion or other consequential employment action;
- let an LLM directly execute arbitrary shell commands, network calls or unrestricted Python against employee data.

Causal questions abstain when the available evidence cannot establish causation.

---

## Local-first privacy boundary

PeopleOS is designed to run locally.

By default the backend binds to:

```text
127.0.0.1
```

Local AI synthesis can run through Ollama, and deterministic analytics continue to work without making a model the calculator or source of truth.

If you intentionally expose the API beyond loopback, configure authentication and a server-side role:

```bash
export PEOPLEOS_API_HOST=0.0.0.0
export PEOPLEOS_API_TOKEN='replace-with-a-strong-secret'
export PEOPLEOS_API_ROLE='analyst'   # viewer | analyst | admin
export PEOPLEOS_API_ACTOR_ID='named-operator'
```

Remote clients cannot choose their own authorization role in a request header.

### Role boundaries

- **owner** — trusted local owner with full control-plane access
- **admin** — workspace/data/model administration and governed recovery
- **analyst** — investigations and read access without model activation/recovery authority
- **viewer** — read-only lifecycle/system visibility

---

## Desktop experience

PeopleOS also supports a packaged local desktop experience.

The desktop lifecycle includes bounded in-app controls for:

- Open PeopleOS
- Restart app
- Quit PeopleOS

Users should not need a terminal, Task Manager or Activity Monitor to manage the running application.

The current build pipeline validates packaged Windows x64, macOS ARM64 and Linux x64 paths, including smoke/restart/archive checks.

Published, signed installers are a separate release step and are not implied by source availability.

---

## Getting started

For the full first-run walkthrough, use the [Public Beta Guide](docs/PUBLIC_BETA_GUIDE.md).

### Prerequisites

- Python 3.10+
- Node.js 22 recommended for the current Next.js frontend
- Ollama is optional

### Install the core runtime

```bash
python -m pip install -r requirements.txt

cd web
npm ci
cd ..
```

The default runtime intentionally excludes the heavy transformer/GPU/vector-search stack.

### Optional advanced embeddings

```bash
pip install -r requirements-advanced.txt
```

### Optional local AI

Install Ollama and pull a compatible model, for example:

```bash
ollama serve
ollama pull gemma3
```

### Run the backend

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

or:

```bash
python -m api.main
```

### Run the web app

```bash
cd web
npm run dev -- --hostname 127.0.0.1
```

Then open:

```text
http://localhost:3000
```

Local API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## Data and pay semantics

PeopleOS uses a normalized workforce schema and maps familiar HR fields into deterministic analytical contracts.

Typical fields include:

- `EmployeeID`
- `Dept`
- `Tenure`
- `Salary`
- `LastRating`
- `Age`
- `JobTitle`
- `JobLevel`
- `Location`
- `Attrition`

Additional fields unlock additional capabilities.

Monetary analytics require trustworthy pay-period and currency meaning. If those semantics cannot be established, PeopleOS keeps pay analysis unavailable rather than silently assuming annualization or currency equivalence.

---

## Predictive model lifecycle

Predictive functionality is deliberately separate from deterministic observed analytics.

A model follows an explicit lifecycle:

```text
created → training → evaluating → candidate → active / rejected / failed
```

Activation is a governed action. Asking PeopleOS a question never implicitly authorizes model activation.

Predictive scores are described as model-score bands unless future-event probability validity has actually been established for the intended use.

---

## Architecture

The canonical machine-readable system definition is:

[`model/system.json`](model/system.json)

The architecture upgrade record is:

[`docs/architecture/AGENT_SYSTEM_UPGRADE.md`](docs/architecture/AGENT_SYSTEM_UPGRADE.md)

The 10/10 product direction and drill-down principles are recorded in:

[`docs/product/PEOPLEOS_10_10_PRODUCT_PLAN.md`](docs/product/PEOPLEOS_10_10_PRODUCT_PLAN.md)

PeopleOS intentionally keeps deterministic analytics engines responsible for calculations and state. The probabilistic model sits **on top of evidence**, not underneath truth.

---

## Verification

The public-beta engineering candidate passed the complete validation matrix before merge to `main`:

- Agent Foundation
- Frontend Modernization
- E2E User Journey
- Analytics Validation
- People Team Browser Acceptance — desktop and mobile
- Local Desktop Build
- Release Security
- Local Ollama Acceptance

The browser suite exercises real People-team journeys including data import, pay-unit gating, malicious source labels, bad replacement data, concise AI answers with inspectable evidence, unsupported/causal abstention, dataset changes, mobile layout, scenario planning and Trust & Privacy.

CI is strong technical evidence. It is **not** a substitute for independent fresh-user usability testing or organization-specific prospective validation of predictive use cases.

---

## Pilot status and feedback

PeopleOS is engineering-ready for a **controlled public-beta pilot**. It is not a production certification.

The current intended scope is local, single-user People analytics and governed investigation with fictional/test data first.

If you try PeopleOS, please use the repository's **Pilot feedback** issue template for usability, trust, workflow or expectation feedback. Do not attach real employee or other sensitive personal data.

Before broader production use, teams should still perform their own security, privacy, legal, data-governance and intended-use review.

See:

- [Public Beta Guide](docs/PUBLIC_BETA_GUIDE.md)
- [Public Beta Release Checklist](docs/releases/PUBLIC_BETA_CHECKLIST.md)
- [Commercial Use](COMMERCIAL_USE.md)
- [Security Policy](SECURITY.md)
- [Bug report](.github/ISSUE_TEMPLATE/bug_report.md)
- [Feature request](.github/ISSUE_TEMPLATE/feature_request.md)
- [Pilot feedback](.github/ISSUE_TEMPLATE/pilot_feedback.md)
- [Commercial use request](.github/ISSUE_TEMPLATE/commercial_use_request.md)

---

## License and commercial use

The repository's [`LICENSE`](LICENSE) file is authoritative.

PeopleOS is **source-available, not OSI open source**, because the license includes an additional commercial-use restriction.

You must obtain **express written permission from Omoniyi Ipaye before using PeopleOS for a commercial purpose**. See [COMMERCIAL_USE.md](COMMERCIAL_USE.md) for the plain-language policy and request process.

No commercial permission is implied by cloning, forking, modifying, evaluating or contributing to this repository.
