# Live dummy-data validation

## Result

**96 live acceptance checks passed: 35 HTTP-response checks and 61 value/integrity checks across 25 distinct API routes.** The harness started the real FastAPI application with Uvicorn, uploaded CSV files over TCP, and used a separate temporary workspace. It did not mock engines, override dependencies, or use real employee data.

Four real React component rendering cases passed using the captured API responses. The existing 13 renderer checks, 69 targeted Python regressions, three audit-inventory checks, and nine pinned public benchmark assertions also passed. Lint has zero errors and 19 existing warnings.

## Independently specified answers

Dataset A contains 120 fictional employees observed at two dates: 240 source rows. The latest snapshot has 80 known active employees, 20 recorded departures and 20 unknown statuses. Of the active employees, 40 earn 60,000 annually and 40 earn 90,000. Department labels include `001`, `002`, and missing values. The earlier snapshot deliberately has different salaries and outcomes.

| Measurement | Expected | Actual |
|---|---:|---:|
| Current employees represented | 120 | 120 |
| Known active employees | 80 | 80 |
| Recorded departures | 20 | 20 |
| Unknown outcome statuses | 20 | 20 |
| Observed departure share among known outcomes | 20 / 100 = 20% | 20% |
| Annual active payroll | 40 × 60,000 + 40 × 90,000 = 6,000,000 | 6,000,000 |
| Average active salary | 75,000 | 75,000 |
| Average valid active rating | 4.0 | 4.0 |
| Active department groups, including Unknown | 3 | 3 |
| Active employees in Unknown department | 26 | 26 |
| Six-month cost of a 10% pay increase | 6,000,000 × 10% × 6/12 = 300,000 | 300,000 |
| eNPS: 30 promoters, 10 passives, 20 detractors | (30 − 20) / 60 × 100 = 16.7 | 16.7 |
| Excluded survey rows | 1 unmatched ID + 1 absent ID + 1 invalid score | 3 |
| Gender favorable-outcome ratios | Both 1.0 | Both 1.0 |
| Kaplan–Meier survival at 24 months | 1 − 20/100 = 0.8 | 0.8 |
| Restricted mean survival through 24 months | 24 months | 24 months |
| Direct reports, excluding the manager's self-reference | 79 | 79 |
| Unassessed employees in succession totals | 80 | 80 |

Amounts are unitless annual salary amounts supplied by the fixture; no currency conversion is assumed. Observed departure share is not an annual turnover rate. Scenario cost arithmetic is verified; the assumed retention effect is not a validated causal estimate. Department count represents displayed active groups, including Unknown, rather than only named departments.

Dataset B preserves 120 employee records but has no known outcomes and only one measured rating. It produces zero **known active** employees, unavailable observed attrition and active salary, and no unsupported source-quality scores. Unknown does not mean departed or healthy.

The A → B → A sequence restores A's exact summary. Switching datasets clears survey evidence and rejects old scenarios. A duplicate employee/snapshot upload is rejected without changing the selected data. A fresh server process restores the same 240-row history and exact current summary.

## Defects exposed and corrected

1. Missing departments caused compensation scenario serialization to fail with HTTP 500. Scenario groups now preserve an explicit Unknown group that can also be selected.
2. Succession bench summaries lost 26 active employees with missing departments because null equality could not select their rows. Unknown now retains their population and unassessed status.
3. Analytics summary group count excluded Unknown while the department table included it. Both now use normalized groups consistently, including blank/whitespace labels.
4. Department salary dispersion was always unavailable because the aggregate never populated the field consumed by the API. The sample standard deviation now uses valid active salaries and stays unavailable with fewer than two observations.
5. The first live CI run exposed nonfinite statistics in the combined sentiment response. Pydantic 2.5.3 preserved `NaN` in nested outputs, causing HTTP 500; the newer local serializer had hidden this defect. The route now explicitly converts undefined statistics to JSON null using the shared serializer. The failure was reproduced locally with Pydantic 2.5.3; all 96 checks pass after the correction.

These changes operate on engine copies and preserve source identifiers and categories in canonical dataset artifacts. Four focused regression tests cover the corrections; the live acceptance script is now a required Analytics Validation CI step. Desktop launcher, packaging and smoke-test assertions are unchanged.

## Actual ML benchmark results

The production training and inference code was also rerun on the pinned **1,470-row synthetic IBM HR benchmark**. Evaluation used an untouched 294-employee holdout. All three shuffled-label negative controls were rejected. The separate Waltons benchmark checked survival mechanics.

| Holdout metric | Result |
|---|---:|
| ROC AUC | 0.8037 |
| Average precision | 0.4886 versus prevalence baseline 0.1599 |
| Brier score, lower is better | 0.1077 versus baseline 0.1343 |
| Accuracy at the current classification threshold | 85.4% |
| Recall at the current classification threshold | **17.0%: 8 of 47 recorded departures detected** |
| Precision at that threshold | 66.7% |

Passing the retrospective candidate checks means the score improves on those baselines. It does **not** mean the threshold is suitable for operational detection. The low recall is a material limitation; threshold selection and acceptance criteria require a separate validation population and intended-use costs. This run does not establish future-departure accuracy or justify individual employment recommendations.

## Reproduce and inspect

With the repository validation dependencies installed:

```bash
python scripts/live_dummy_acceptance.py --output docs/validation/live-dummy-data
python scripts/benchmark_analytics.py --ibm-data /path/to/pinned/emp_attrition.csv --output docs/validation/live-dummy-data/benchmark-results.json
cd web
node tests/live-response-renderers.cjs
```

The IBM file must match the existing pinned checksum. The optional `--ibm-data` argument avoids fetching it again.

- [Expected versus actual checks](live-dummy-data/checks.json)
- [Full API response capture](live-dummy-data/responses.json)
- [Actual component output](live-dummy-data/rendered-output.html)
- [Renderer checks](live-dummy-data/renderer-checks.json)
- [ML and survival benchmark evidence](live-dummy-data/benchmark-results.json)
- [Main dummy workforce](live-dummy-data/workforce-a.csv)
- [Unknown-outcome workforce](live-dummy-data/workforce-b.csv)
- [Survey fixture](live-dummy-data/enps.csv)

The evidence records the base commit, exact hashes of the corrected sources, and runtime dependency versions. Final local acceptance used Python 3.12, Pydantic 2.5.3 and Uvicorn 0.34.3; CI uses its pinned validation environment, including Python 3.11, Pydantic 2.5.3 and Uvicorn 0.27.0.

## Coverage limits

The browser service could not open the local preview (`ERR_BLOCKED_BY_CLIENT`). Rendering checks therefore use actual React server-rendered output fed by live API responses; they do not verify browser hydration, interactions, responsive layout or cache timing visually. The static HTML capture has noninteractive controls.

Live route checks cover descriptive analytics, compensation, teams, survey sentiment, quality of hire, experience, fairness, survival, structural and succession summaries, geography, scenario arithmetic, lifecycle switching and restart. Network, search, forecasting and prospective Model Lab paths correctly report unavailable evidence for these inputs. No live LLM, embedding relevance, prospective HR prediction, multi-user load or enterprise security claim is established by this run.
