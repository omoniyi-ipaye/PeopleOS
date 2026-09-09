# PeopleOS 10/10 Product Plan

## Product promise

PeopleOS should feel like an AI People Analyst whose calculations can be trusted.

The experience must remain simple by default and powerful on demand:

- non-technical People users can install, add data, understand the workforce and ask questions without documentation;
- deterministic engines own calculations, denominators, scope and validation state;
- AI can investigate, compare and derive further analysis only through governed analytical contracts;
- evidence and methodology stay available through progressive disclosure rather than dominating the main UI;
- PeopleOS analyses; people decide.

## 10/10 experience target

A first-time People Operations user should be able to complete this flow without a terminal or technical documentation:

1. Download and open PeopleOS.
2. Explore with fictional sample data or add an ordinary Excel/CSV HR export.
3. Let PeopleOS validate and explain what it understood.
4. Open Home and immediately understand what deserves attention.
5. Ask a workforce question in normal People language.
6. Receive a concise answer based on deterministic evidence.
7. Drill into the answer when more depth is needed.
8. Compare groups, break down metrics, run governed statistical analyses and visualise results.
9. Inspect population, exclusions, methodology and provenance when desired.
10. Quit/restart/reopen the local app without OS process tools.

## Progressive drill-down model

The normal product remains clean. Depth appears only after a user chooses to explore.

### Level 1 — Insight

Examples:

- Recorded attrition: 17.3%
- Engineering has elevated observed attrition
- Salary dispersion is higher in Sales
- Manager spans are high in Customer Success

Every material finding should expose an `Explore` or `Drill down` action.

### Level 2 — Breakdown

Supported dimensions should be selected from the actual dataset and privacy/measurement contracts, for example:

- Department
- Location
- Job level
- Tenure band
- Employment type
- Gender or other protected dimensions only when minimum-group controls permit it
- Organizational layer / manager structure when valid

PeopleOS must always carry the analytical context:

- current filters
- eligible population
- measured population
- excluded/missing population
- dataset version / provenance

### Level 3 — Nested drill-down

Allow progressive cohort exploration such as:

Workforce → Engineering → Madrid → L3 → tenure <2 years

The breadcrumb must make the current analytical population obvious.

Do not permit analytical drill-down to silently become employee ranking or individual employment decision support.

### Level 4 — Ask PeopleOS about this view

Every drill-down context should support `Ask about this view`.

Example conversation:

- What is different about Engineering employees with less than two years' tenure?
- Compare this with Product.
- Is the salary difference statistically distinguishable?
- Break the comparison down by tenure bands.
- Visualise the result.

The agent receives the active analytical context automatically and may request further governed deterministic calculations.

### Level 5 — Explore workspace

For advanced People Ops / People Analytics users, expose a lightweight analytical builder using People language rather than BI terminology.

Example controls:

- Measure: Recorded attrition share
- Break down by: Department
- Compare by: Location
- Filter: Tenure <2 years
- Statistical comparison: On/off

Minimum-group and measurement rules remain automatic and cannot be bypassed by the UI or AI.

## Governed downstream analysis runtime

AI may perform additional analysis over verified data/results, but it must not receive unrestricted shell, filesystem, network or arbitrary Python execution.

Preferred contract:

Verified workforce snapshot / deterministic result
→ typed analytical request
→ governed analysis runtime
→ deterministic derived result
→ evidence ledger + provenance
→ AI interpretation

Supported operations can grow over time, including:

- filters
- group-bys
- aggregate measures
- cross-tabs
- cohort comparisons
- correlations
- statistical tests
- sensitivity analysis
- derived tables
- safe visualisations

Every derived analysis must include its population, exclusions, method and source dataset version.

## Individual-data boundary

Normal analytical drill-down remains cohort/group level.

Acceptable:

Engineering → Madrid → L3 → tenure <2 years

Not acceptable as an analytical shortcut:

- rank employees by risk
- identify who is causing a workforce problem
- recommend who to fire, discipline, demote or exclude

If a legitimate data-quality workflow later requires record-level inspection, build it as a separate permissioned Record Review mode, clearly separated from employment-decision analytics.

## Primary UX principles

- Five-item primary navigation: Home / Ask PeopleOS / Insights / Plan / Data.
- Trust & Privacy and Settings are secondary.
- One page = one primary question.
- Methodology uses progressive disclosure (`About this number`, `Why you can trust this answer`, `How this was calculated`).
- Human language over system terminology.
- Capability-aware UI: do not foreground unsupported analyses.
- Every error must provide a useful next action.
- Clean visual hierarchy, generous whitespace, restrained accent use, very little simultaneous text.
- Desktop packaged app is the normal distribution path; source install is contributor/developer documentation.

## Pilot release gates

We should not call the product 10/10 solely because the design looks good. The target requires:

- deterministic exact-head validation green;
- governed-agent contracts green;
- desktop + mobile browser acceptance green;
- packaged desktop build/smoke/restart green;
- security gates green;
- concise human-facing answers with evidence drill-down;
- Excel/CSV onboarding and error recovery verified;
- fresh production screenshot review;
- native fresh-user installation/usability pilot;
- accurate release/licence wording;
- no misleading predictive or causal claims.

## Definition of 10/10

10/10 is reached when a non-technical People professional can independently install/open PeopleOS, add or explore data, understand the primary insights, ask follow-up questions, drill deeper, inspect trust evidence when needed, recover from mistakes, and exit/restart the app — while every displayed calculation remains traceable to a deterministic, population-aware analytical contract.
