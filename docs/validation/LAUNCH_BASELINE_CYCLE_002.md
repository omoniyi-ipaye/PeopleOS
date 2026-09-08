# PeopleOS launch baseline — Cycle 002

Cycle 002 is an exhaustive pre-pilot hardening pass over the candidate that follows
`cc197912b107fdd4e67e81e163ab9e351cdaac27`. The final candidate commit and hosted CI
links are recorded only after publication. This document does not certify the product
for enterprise production or consequential employment decisions.

## Audited surface

- 20 analytics/ML engines and all 126 public methods have executable test references.
- 133 API operations across 129 paths were exercised through the actual middleware;
  the inventory includes 25 unsafe operations.
- All 13 routed product pages are covered by 18 desktop/mobile browser journeys.
- 30 renderer edge cases cover loading, missing, error, retry, stale-result and bounded
  input states.

The detailed contracts live in `model/analytics_launch_matrix.json`,
`model/agentic_launch_matrix.json`, `docs/validation/enterprise-controls-cycle-002.json`,
`docs/validation/RENDERER_VALIDATION_MATRIX.md`, and
`web/e2e/action-process-coverage.json`.

## Corrections supported by reproductions

1. Individual disclosure and workforce-ranking prompts now stop before tool execution;
   negation and unsupported scopes cannot borrow unrelated evidence.
2. Quality-of-hire individual risk output and ungrounded NLP employee summaries are
   retired. They return unavailable without an LLM call or risk count.
3. Workforce metrics require analyst access, remote individual routes are blocked,
   deprecated routes return 410, and governed grouped outputs suppress cells below 10.
4. Concurrent local registry writes are serialized and atomic. Failed registration or
   activation cleans up incomplete state, sessions are actor-bound, and security denials
   are non-cacheable.
5. Extreme finite compensation inputs cannot leak non-finite JSON; sentiment/eNPS inputs
   are finite and consistent, and eNPS grouping is allow-listed.
6. Product renderers distinguish unavailable data from zero and expose failed, loading,
   retry, stale and bounded-scenario states without leaving unsafe actions enabled.

## Executed local evidence

- Analytics/ML specialist matrix: 207 passing cases.
- Agentic governed matrix: 164 passing cases.
- Controls and recovery matrix: 100 passing cases.
- Independent cross-matrix challenge: 280 passing cases.
- Analytics Validation workflow selection: 547 passing cases.
- Agent Foundation workflow selection: 115 passing cases.
- Renderer suite: 30 passing cases; typecheck, design check and production build pass.
- Live API dummy-data acceptance: 96 passing checks.
- Public benchmark: nine integrity checks pass, including disjoint holdout, artifact
  inference parity, three rejected random-label controls, survival curve and RMST checks.

The repository-wide legacy pytest collection also completed with 635 passes and 29
failures. An independent reviewer executed the same six failing files at the pinned
baseline and observed the identical 29 failures and 44 passes: Cycle 002 introduced zero
of them. They remain explicit debt: 10 stale upload fixtures that omit the Golden Schema,
nine expectations for retired or replaced LLM/SHAP/metric contracts, six database
implementation/semantic gaps, and four vector tests run without the advanced FAISS
dependency. They block a claim that every legacy or optional capability is green;
semantic search remains unavailable until an advanced-dependency job passes.

## Candidate gates

The same published commit must pass analytics/public-benchmark/live-dummy validation,
Agent Foundation, frontend checks, actual desktop/mobile Playwright journeys, and
Windows x64, macOS ARM64 and Linux x64 packaged restart/smoke tests. Browser screenshots
and traces must be inspected when CI publishes them. Until then, Cycle 002 is not a
completed launch baseline.

## Readiness boundary

If every candidate gate passes, the recommendation is a controlled local People-team
pilot with synthetic or approved non-sensitive data. Enterprise production still needs
a configured real-LLM grounding/injection/timeout evaluation, prospective organization-
specific predictive validation, enterprise identity/audit/retention/key-management
controls, accessibility and localization review, realistic load/recovery objectives,
and observed outcomes from a real People-team pilot.

The recurring improvement automation remains paused until this baseline is complete and
explicitly accepted. No merge, deployment, real employee-data ingestion or employment
decision is part of this cycle.
