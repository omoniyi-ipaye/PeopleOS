# PeopleOS specialist improvement cycle 001

This cycle improves a reviewable candidate for aggregate People-team investigation. It does not certify enterprise readiness, future employee predictions, or perfect trust. The baseline is `bd98b6af2687046c90148b5267896531d0326417`; the candidate and its CI evidence are linked from [PR #3](https://github.com/omoniyi-ipaye/PeopleOS/pull/3).

The durable operating specification is [model/quality_cycle.json](../../model/quality_cycle.json), connected to [the canonical system model](../../model/system.json). Each iteration pins the repository head, finishes outstanding verification, assigns separate specialist ownership, reproduces failures, implements bounded corrections, and runs independent acceptance gates. A failed gate returns to its owner. A passing gate qualifies that version and tested use case only.

## Findings and corrections

| Area | Reproduced problem | Candidate correction |
|---|---|---|
| Access | Remote viewer/analyst and foreign-origin local requests could mutate legacy routes; legacy sessions accepted filesystem paths | Central mutation authorization, local Host/Origin/proxy validation, retired legacy file sessions and non-cacheable API evidence |
| Recovery | Invalid registry bytes could be overwritten during constructor initialization or recovery | Fail closed at startup; preserve verified quarantine before explicit reconstruction; require dataset reconciliation; invalidate orphaned runtime evidence |
| Analytics | Salary department/tenure groups dropped unknown values; measured/excluded counts disappeared in API responses | Reconciled Unknown groups and explicit measured populations; consistent valid-measurement rules |
| ML interpretation | Overall accuracy could obscure poor positive-class recall | Fixed threshold, confusion counts, recall support and majority baseline exposed; no holdout tuning |
| Agent behavior | Missing tool evidence, unsupported scopes, omitted requested metrics and sparse measurements could appear complete | Required-metric and scope checks; warnings and denominators retained; model selects verified evidence IDs while server renders factual claims |
| Surveys | Missing required EmployeeID could be fabricated before validation and replace a valid survey | Validate original columns and prepare the entire candidate before committing survey state |
| Design | Stale answers, fabricated coverage fallback, hidden gaps, misleading source counts and crowded mobile navigation | Snapshot-bound answers; visible evidence limitations; record/employee distinction; accessible mobile drawer |

Independent review reproduced failures outside the authors' initial tests, then checked the corrections. It also identified the recovery/runtime integration defect after the initial recovery suite passed. This is why successful unit tests alone are insufficient.

## Acceptance evidence

The following gates run against the published candidate:

- Analytics Validation: arithmetic, population, measurement, model isolation, API integrity, agentic and recovery regression suites; nine public-benchmark checks; 96 live HTTP dummy-data checks.
- Agent Foundation: architecture, policy, tool, platform and output-integrity checks.
- Frontend Modernization: renderer assertions, design governance, lint and production build.
- People Team Browser Acceptance: four actual application journeys on desktop and mobile Chromium, using isolated synthetic data, real FastAPI and the production Next.js frontend. No API-response mocks or local LLM service are supplied.
- Local Desktop Build: existing Windows x64, macOS ARM64 and Linux x64 package, restart and smoke gates. Existing Windows launcher and packaging settings are preserved.

The browser gate stores screenshots, server logs, reports and failure traces/videos as `people-team-browser-evidence`. Passing browser assertions are not a claim of complete visual or accessibility review. The first browser candidate ran eight journeys: four passed and four stopped at an undersized second test fixture. The fixture was corrected to 60 records while preserving 30 active expectations and the existing 50-row minimum. Twelve actual desktop/mobile screenshots were inspected; no severe layout defect was observed, and the review identified an unknown-status explanation to add beside the active count. Exact final candidate results belong in the PR and cycle ledger; pending or blocked gates remain explicit.

## Operating boundaries and remaining work

Hourly recurring iterations are enabled and bounded, with one integrating cycle per candidate. They can inspect code, use synthetic/public fixtures, delegate specialists, make reversible corrections and update a draft PR. They cannot automatically merge, deploy, ingest real employee data, contact others, make employment decisions, weaken failing controls, or tune on a final holdout. If access is blocked or no justified correction is found, the iteration records that result rather than manufacturing progress.

The application agent remains a governed aggregate, read-only investigator with deterministic planning and an optional evidence selector. Controlled selector tests do not validate a real installed LLM. Representative future-outcome ML validation, actual LLM grounding/injection/latency evaluation, workload/recovery objectives, individual enterprise identity and retention requirements, broader accessibility testing, and a real People-team pilot remain required before stronger readiness claims.

Registry corruption present at startup requires restoration from a verified backup. A running instance can explicitly quarantine invalid metadata and rebuild a degraded shell; this does not reconstruct lost dataset/model history. Operators must reconcile the preserved metadata and explicitly reactivate a dataset.
