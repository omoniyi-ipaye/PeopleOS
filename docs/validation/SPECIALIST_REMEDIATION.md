# Specialist review remediation

Baseline: `d96ef3c22c9cf75666605ce10babafa41404f68a`. Scope: all eleven grouped findings in the [independent review](PEOPLE_ANALYTICS_ML_SPECIALIST_REVIEW.md), plus the concrete recovery defects found while implementing the fixes.

The original defect reproductions are now correctness regression tests. A passing result now requires the corrected behavior. The historical specialist report preserves the original observations; it is not a description of the repaired candidate.

| Finding | Implemented correction | Acceptance evidence |
|---|---|---|
| F01 — dataset and result provenance | Candidate snapshot preparation, artifact checksum validation, shared mutation lock, dataset-bound training/investigations, auxiliary-state reset and per-generation scenario caches. Conflicting sessions/requests are rejected. | HTTP A → B → A switch; exact restored categories/IDs; wrong-dataset requests blocked; failed upload/activation retains prior state; old scenario retrieval rejected. |
| F02 — sparse outcomes become active | Required/declared fields survive sparsity handling. One known departure and nineteen unknown statuses remain one departure, nineteen unknown and zero known active. Oversize imports fail instead of truncating populations. | Exact source-count, missingness and import-limit regressions. |
| F03 — misleading homepage | Loading, error, unavailable and measured zero have distinct output. Raw row count never substitutes for active workforce. | Real React server rendering of absent outcomes, loading/error summary and measured zero tenure. |
| F04 — advisor contracts | Shared probability, model and exact active-population validator; actual experience score/segment fields; nullable fairness ratios; empty experience evidence is partial. | Invalid/stale evidence rejected; measured ExI 75 preserved; all-retained undefined ratios remain null. |
| F05 — onboarding populations | Matched survey identities with exclusion coverage; latest response per person/stage consistently applied; totals calculated before display truncation. Unsupported warning evidence stays unavailable. | Fifteen affected respondents remain 15; scores 1 then 5 produce latest 5 throughout; excluded respondents and null flags survive API serialization. |
| F06 — source scores | One common supported component set/weights across sources; at least 10 measured observations per required component; missing support suppresses grades/rankings. Counts, coverage, weights and reasons retained in API and displayed with source cohorts. | One measured rating among 20 hires cannot produce grade A. Missing retention does not cause source-specific reweighting. |
| F07 — activation ordering | Transform, score and validate candidate before persisting activation and swapping runtime state. | Controlled scoring failure leaves candidate inactive; successful activation covers exactly 18 active rows of a 20-person fixture. |
| F08 — identity fidelity | CSV reads preserve lexemes before mapping. Artifact restore preserves identifiers and categories; only declared measurements receive numeric conversion. Artifact hashing uses the exact bytes parsed. | Leading-zero employee/manager IDs and department 001 survive import/restore; categorical 001 and 1 have different fingerprints; JobLevel L03 stays categorical. |
| F09 — payback | Expansion has no reported payback without a supported cash-flow schedule. Loss-making annual totals cannot imply positive payback. Reduction helper divides one-off severance by monthly savings. | Expansion payback null; existing direct reduction arithmetic has a 3-month known answer. Individual reduction API remains disabled. |
| F10 — team grouping | Null/blank departments are explicit Unknown groups. | Team headcounts reconcile to all 20 active people, including 10 Unknown. |
| F11 — legacy/dormant displays | Missing compa ratios show Unavailable; dispersion cards use neutral descriptive labeling; legacy tenure uses valid active rows; misleading quartile-equity explanation removed. Proxy succession and individual risk entry points retired. | Compensation known answers and renderer checks; explicit unavailable legacy paths, with no individual score output or fallback probability. |

## Added safeguards

- **Trust Center dataset integrity:** selected/runtime identity, model readiness, source/current/active counts and unknown statuses. This is an operational integrity check, separate from statistical validity.
- **Recovery that preserves evidence:** synchronized restore/activation/reset, rollback on failed preparation/persistence, same-dataset model reconciliation and real retraining when a persisted successful job has lost its in-memory artifact.
- **Responses bound to snapshots:** dataset/snapshot response headers; reset/switch races return HTTP 409; threaded NLP cannot repopulate stale caches. Scenario history is limited to 100 entries in the current generation and comparisons require compatible horizons.
- **Views follow dataset changes:** upload/reset clears cached query values; an observer checks the dataset generation on window focus and every 15 seconds. Scenario configuration changes clear the previous result and pending inputs are disabled.
- **Observed fairness without a model:** recorded-outcome disparity can be computed before predictive training; the existing minimum-group and interpretation limits still apply.

## Verification

CI now includes four specialist regression suites: the converted original findings, specialist measurement cases, route-level runtime acceptance and snapshot recovery. Frontend CI includes the original renderer suite and converted/extended specialist renderer cases. The existing pinned IBM synthetic benchmark, shuffled-label controls and Waltons survival checks remain required.

Local candidate verification: **319 analytics/integrity checks plus eight additional foundation checks**, **13 React renderer checks**, and **nine pinned public benchmark assertions** passed. Next.js builds all 16 static pages; lint reports zero errors and 19 warnings. Packaged CI verification remains required on the published head. The architecture model and source-review inventory are updated alongside this report. No launcher, frozen configuration-path behavior, desktop packaging dependencies or smoke assertions were weakened.

## Operating limits

The mutation boundary supports the existing single local workforce runtime in one application process. Non-local data workspaces and cross-process writers are not newly enabled. Canonical dataset artifacts persist; runtime model artifacts remain in memory, with explicit retraining after restart. Survey uploads and scenario sessions are cleared when activating another generation and must be uploaded/run again.

Complete source coverage and software regression checks do not establish prospective attrition accuracy, causal effects, survey construct validity, adjusted pay equity, LLM faithfulness or embedding relevance. Those stronger claims still require representative labeled data and intended-use validation. The public IBM dataset is synthetic; benchmark performance does not certify enterprise outcomes.
