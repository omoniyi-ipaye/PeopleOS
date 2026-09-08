# Independent People Analytics and ML specialist review

**Historical review of d96ef3c. Subsequent corrections are tracked in [SPECIALIST_REMEDIATION.md](SPECIALIST_REMEDIATION.md). The reproduction files have been converted into correctness regression tests; the defect descriptions and original reproduction counts below describe the reviewed baseline.**

Reviewed source: `d96ef3c22c9cf75666605ce10babafa41404f68a`, branch `fix/analytics-validation`, 8 September 2026. This is an independent specialist-oriented code review by an AI agent, not a human professional certification. Existing audit manifests and validation reports were treated as coverage hypotheses, not proof of correctness.

**Conclusion:** the earlier repairs materially improve population handling, fail-closed behavior and claim labeling, but remaining live defects can associate results with the wrong dataset, turn unknown outcomes into reassuring values, and distort evidence passed to the advisor. The current implementation should not be represented as comprehensively validated for enterprise decisions. The most urgent work is shared data provenance and measurement contracts, followed by consumer fidelity.

No production code, existing tests, model artifacts, runtime packaging or Windows build path was changed by this review. The review adds reproducible synthetic cases and this report. Packaged desktop validation was outside this independent analytics review.

## Verification and interpretation

- `tests/test_specialist_review_findings.py`: **16 passing defect reproductions**. These intentionally assert current incorrect behavior. A green result means the defect was reproduced; these are not 16 correctness passes. Convert each assertion into the intended contract when repairing it.
- `web/tests/specialist-review-renderers.test.cjs`: **3 passing defect reproductions**, independently created and run by the parent reviewer against real React server rendering of the current homepage. These confirm rendered output, not just source-string patterns.
- The parent independently reran all 16 Python cases and all 3 SSR cases. Lifecycle tests use temporary registries and controlled model fitting/scoring to isolate provenance and atomicity; they do not measure predictive accuracy.
- Inputs are synthetic with explicit known answers. Public benchmarks from the prior audit were not rerun during this independent review. No prospective HR dataset, production deployment, live LLM output, browser interaction suite or packaged desktop was evaluated here.

Reproduction commands, from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/scratch/21af8284a212/test-venv/bin/python -m pytest -q tests/test_specialist_review_findings.py
cd web
node --test tests/specialist-review-renderers.test.cjs
```

## Prioritized findings

Priority reflects result integrity and reachability: **P1** can materially misstate current live evidence; **P2** is a concrete bounded analytic or lifecycle defect; **P3** is primarily a dormant or presentation issue. Findings distinguish direct reproduction from additional source traces of the same cause.

### F01 — P1: runtime data and displayed provenance can diverge

**Sources:** `api/routes/platform.py:89–112`, `api/routes/intelligence.py:60–82`, `src/platform/runtime_loader.py:98–146`, `api/routes/nlp.py:83–84`, and `api/routes/scenario.py` (`_scenario_cache`, `_convert_result`, retrieval/history/compare routes).

Dataset activation updates the workspace registry without loading the selected artifact into the runtime. Training then supplies the registry's active dataset ID while fitting `state.raw_df`. A controlled route-level reproduction registers dataset A, leaves B rows in the runtime, and obtains a model labeled A from a fit that received B. The intelligence route similarly accepts a payload/session/workspace dataset label while constructing its agent from the current application state; the requested historical artifact is not bound to that state.

Even real `activate_dataframe` resets leave auxiliary survey frames and NLP results intact. A reproduction activates 20 NEW employees while retaining an OLD employee's survey and receives eNPS 100 from that unrelated response, alongside the OLD NLP cache. Scenario results are stored in a process-global cache without dataset provenance: after changing to a three-person population, retrieving a previous scenario still returns 20 affected people with no dataset ID. Retaining historical scenarios is legitimate, but their source population must be explicit and comparisons must validate compatibility.

**Impact:** evidence and model records can appear to refer to the selected dataset while actually describing another population. This undermines every downstream correctness guarantee even when each formula is correct.

**Repair:** bind a runtime snapshot, auxiliary inputs, model and outputs to a validated dataset identity/content fingerprint. Load and validate the artifact on activation; reject incompatible training and investigation requests. Scope or invalidate derived caches and attach immutable provenance to historical results. Treat auxiliary survey populations and permissible joins as explicit contracts.

**Evidence:** three Python reproductions cover training, auxiliary-state activation and scenario retrieval. The intelligence and cached NLP endpoint implications are source-traced, not separately exercised through HTTP.

### F02 — P1: sparse outcome columns become fictional active employees

**Sources:** `src/data_loader.py:200–204`; `src/population.py:75–88`.

The ingestion cleanup drops any column with more than 90% missing values, including `Attrition`. Population resolution interprets absence of that column as an active-only input. With 20 records containing one known exit and 19 unknown statuses, ingestion removes the column and analytics reports 20 active employees. Neither the known departure nor unknown-status population survives.

**Known answer:** these records cannot establish 20 active people; the source contains one known departure, no known active employee and 19 unknown outcomes. An explicit active-only upload contract would be different, but was not supplied here.

**Repair:** protect identity and outcome contract fields from generic sparsity pruning; retain missing-status semantics through ingestion and population resolution. Validate both before and after normalization. Show unavailable measures when the required population cannot be established.

### F03 — P1: the live homepage renders missing/loading data as reassuring measurements

**Sources:** `web/app/page.tsx:37–47,72`; parent SSR reproduction file.

The page coalesces a missing observed attrition share to zero, then renders `0.0%` and says it is below the watch threshold. During summary loading, raw file `row_count` is labeled as the active workforce. A measured zero years of tenure is rendered as unavailable because its display uses a truthiness test.

**Evidence:** real React SSR demonstrates all three cases, including a loading raw row count of 42. The defect is in the current landing page, despite improved explicit retrospective labels elsewhere.

**Repair:** separate loading, error, unavailable and measured zero states. Only render active population metrics from a resolved summary; require an observed finite value before generating threshold narratives. Preserve genuine numeric zero.

### F04 — P1/P2: advisor adapters do not preserve engine contracts

**Sources:** `src/agent/adapters.py:72–110`; `src/agent/people_tools.py`, `FairnessOutcomeTool` and `EmployeeExperienceTool`.

Three reproduced mismatches affect the agent path:

1. **P1, retention evidence:** given synthetically constructed invalid runtime state, the retention tool accepts an OLD employee's score of 2.0, no usable model, and a NEW current population. It returns successful evidence, including mean risk 2, tagged with the requested new dataset version. This proves missing adapter validation; it does not show that normal trained-model inference produces scores above one. Validation on the direct predictions API does not protect this adapter. Validate finite probability bounds, employee identity/population, model readiness and snapshot provenance at a shared boundary.
2. **P2, experience evidence:** the engine returns a measured `overall_exi` of 75 and list-shaped segment output. The adapter looks for different score keys and dictionary-shaped segments, reporting success with no evidence. Align the typed contract and mark unsupported/empty evidence as unavailable.
3. **P2, fairness missingness:** an all-retained population correctly has an undefined departure parity ratio. The adapter attempts `float(None)` and fails instead of carrying the unavailable comparison and its reason. Preserve meaningful nulls and denominators.

The current advisor UI explicitly labels evidence quality/weights as heuristics rather than probabilities of truth. Fixed evidence weights alone are therefore not elevated as a separate statistical-certainty defect. Deterministic synthesis still uses the more ambiguous word “Confidence”; harmonizing that prose is a lower-priority consistency improvement.

### F05 — P2: onboarding totals and response populations disagree

**Sources:** `src/sentiment_engine.py:417,473–495,574` onward; sentiment onboarding/health/early-warning API routes.

A 15-person low-score cohort produces an onboarding trajectory total of 15 at risk, but early warnings report only 10 because they count the presentation-limited list. Separately, a person's superseded response contributes to onboarding health even though trajectory analysis selects the latest response per survey type. Scores 1 then 5 yield latest trajectory score 5 but health average 3.

**Impact:** the same underlying onboarding population yields conflicting totals and averages. These are reachable API results; a dedicated current web display was not established for each result.

**Repair:** compute totals before truncating display lists. Use a common eligible-response definition for snapshot measures, or explicitly label a historical-response average and report its unit/denominator. Test duplicates, survey types, dates and missing responses together.

### F06 — P2: source quality grades can imply more evidence and comparability than exists

**Sources:** `src/quality_of_hire_engine.py:217–253,291–318`; `api/routes/quality_of_hire.py` (`_safe_source`); `web/app/quality-of-hire/page.tsx` source table and highest-composite summary.

A hiring source with 20 hires but only one observed rating receives score 100 and grade A. The eligibility threshold uses total hires, not the measured denominator. The API **does retain** `performance_observations=1`; the current source table presents hires, average rating, retained share and composite without that observation count.

The composite also reweights components according to availability. Two sources with identical observed ratings of 4 score 85.7 and 75 when one has known retained outcomes and the other has unknown outcomes. These scores measure different component mixtures; they are not a like-for-like source ranking. Labeling the score heuristic is useful, but does not establish comparability.

**Repair:** require component-specific measured sample support, display coverage and effective weights/components, and compare a defined common construct. Suppress grades/rankings when insufficient or incomparable evidence would dominate. Validate on held-out organizational outcomes before interpreting a composite as useful for hiring decisions.

### F07 — P2: failed model scoring leaves the model marked active

**Sources:** `api/routes/platform.py:121–145`; `src/platform/model_lifecycle.py` activation; workspace model state transitions.

The route persists model activation before transforming/scoring the current population. A controlled transform failure returns HTTP 409 while the workspace's model remains `ACTIVE` and `active_model_id` points to it. Registry success and runtime readiness therefore disagree; replacing a previously active model also requires reliable rollback.

**Repair:** validate compatibility and calculate candidate outputs before committing activation, then atomically swap the model and derived state, or restore all prior state on failure. This reproduction proves transition ordering, not a failure in the model algorithm.

### F08 — P2: CSV loading irreversibly changes identifiers

**Source:** `src/data_loader.py:86` and downstream identity joins.

Default CSV type inference converts employee identifier `0001` to `1` before column mapping. This breaks exact identifier fidelity and can affect employee, manager, survey and longitudinal joins when related inputs preserve the original string.

**Repair:** preserve identifier lexemes at read time, including aliased ID columns, then apply one explicit normalization policy across linked tables. Reject collisions and verify round trips. The reproduction proves lexical loss; it does not claim every numeric ID dataset produces a failed join.

### F09 — P2: a loss-making scenario reports a finite positive payback

**Source:** `src/scenario_engine.py:709` and scenario API conversion.

Headcount expansion calculates payback using the absolute annual net impact. A synthetic expansion with negative modeled annual net impact consequently returns a positive finite payback. A continuing negative net does not repay the modeled cost.

**Repair:** return unavailable/no payback with an explanation when the benefit/net cash-flow definition cannot recover investment. Define which costs are one-off versus recurring and use a consistent cash-flow basis. The current scenario page shows costs/net/ROI but does not render this payback field; the defect is exposed through the API/helper contract.

### F10 — P2: missing departments disappear from team totals

**Source:** `src/team_dynamics_engine.py`, department equality filters in team health and composition.

Twenty active employees, ten with missing department, produce composition headcounts summing to ten. Iterating unique departments and testing equality against a null value does not select the null group.

**Repair:** retain an explicit Unknown department or report the excluded population and denominator; reconcile grouped totals to the eligible population. Team API outputs are affected. The current workforce-health page uses the separate analytics engine, whose Unknown grouping already addresses this case; it should not be described as affected by this team-specific reproduction.

### F11 — P3: dormant and legacy helpers still misrepresent evidence

These are not additional claims about the current homepage or governed individual-risk API:

- **Reproduced:** `src/compensation_engine.py:136` onward labels missing compa ratios “Near reference.” Missing is not evidence of reference alignment. The direct helper/legacy path requires an explicit unavailable category.
- **Source-reviewed:** legacy `ui/components.py` compensation rendering expects older Good/Fair status labels, whereas the engine emits dispersion descriptions; its fallback coloring and equity explanation no longer faithfully represent the metric.
- **Source-reviewed:** the legacy overview in `main.py`/`ui/dashboard_layout.py` uses an outer raw-population tenure mean; its salary-quartile explanation suggests the count shape establishes balanced pay/equity, which quartile counts cannot establish.
- **Source-reviewed:** the legacy succession path can describe rating/tenure proxies as “High-Potential Employees.” The current succession API instead has a separate recorded-potential contract. Preserve proxy limitations in every surviving consumer or remove the legacy surface.
- **Source-reviewed:** dormant employee-detail/risk components contain fallback risk values and thresholds that differ from the current model contract. The live individual employee risk route is governed/disabled; these are reactivation hazards, not demonstrated current individual-risk exposure.

## Complete engine coverage

All 20 `src/*_engine.py` modules were source-reviewed, including material calculation methods and their principal consumers. “No additional defect established” is not an exhaustive behavioral proof. Dynamic coverage below means the focused reproduction exercises that engine/path, not every branch. Related shared lifecycle findings can affect engines even where their own arithmetic is sound.

| Engine | Principal path reviewed | Independent assessment and verification limit |
|---|---|---|
| `analytics_engine.py` | Summary, departments, active population; current overview/workforce health | Known-outcome denominators and Unknown grouping improved. F02/F03 reproduce upstream/rendered failures; not every aggregation dynamically tested. |
| `causal_engine.py` | Estimator availability and causal API/governance | Fail-closed behavior avoids substituting correlation for causal estimates. No new concrete defect established; observational identification and intervention validity unproven. |
| `clustering_engine.py` | Active employee selection, finite features, ID/cluster mapping | Source-reviewed distinct-point and stale-result guards. No new defect established; cluster usefulness/stability on real organizations unvalidated. |
| `compensation_engine.py` | Positive pay, dispersion, descriptive gaps, compa ratio; legacy components | F11 reproduces missing ratio classification. Descriptive pay differences do not establish adjusted inequity or fairness. |
| `experience_engine.py` | Fixed-scale scoring, response coverage, segments; current page and advisor | F04 reproduces loss of valid evidence in adapter. Component weighting and organizational construct validity remain validation gaps. |
| `fairness_engine.py` | Recorded outcomes, group support, undefined comparisons; API/agent | F04 reproduces adapter failure on valid null. Protected-group parity descriptions do not validate fairness of untested employment decisions. |
| `forecasting_engine.py` | Monthly census construction, completeness, candidate/naive comparison | Source-reviewed period requirements and absence of invented intervals. Requires rolling-origin and prospective evidence across realistic history lengths; no new numerical defect established. |
| `merge_engine.py` | Snapshot/ID merge and change tracking; ingestion | Source-reviewed reconciliation and latest-record handling. F08 affects upstream identity fidelity; not a concurrent import/rollback stress test. |
| `ml_engine.py` | Inference, preprocessing, model evidence, direct risk and agent consumers | Shared training uses fold-local fitting, held-out baseline/calibration checks. F01/F04/F07 expose lifecycle/consumer bypasses. No prospective attrition prediction claim established. |
| `model_lab_engine.py` | Evaluation/reliability/backtest outputs | Backtesting explicitly unavailable where unsupported; reliability is heuristic. No independent predictive-performance benchmark rerun. |
| `network_engine.py` | Recorded relationship availability, graph metrics and API | Source-reviewed fail-closed data requirements. No fabricated network fallback found in reviewed path; no full graph ground-truth benchmark run. |
| `nlp_engine.py` | Text filtering, structured-output validation, batches, caches; NLP API | F01 traces stale cached output. Limited text samples for themes/skills and partial batch success require coverage accounting; no live LLM quality evaluation. |
| `quality_of_hire_engine.py` | Source/cohort scores, observed outcomes, API sanitizer, current page | F06 dynamically establishes weak measured support and variable-component comparisons. Retrospective observations are not prospective source effectiveness. |
| `scenario_engine.py` | Pay/headcount assumptions, costs, simulation, API/current page | F01/F09 reproduce cache scope and payback. Model assumptions and Monte Carlo draws are exploratory, not empirically calibrated outcomes. |
| `sentiment_engine.py` | eNPS response populations, onboarding, warnings and API | F01/F05 reproduce unmatched old survey use, truncated totals and differing response populations. No survey psychometric or nonresponse-bias validation. |
| `structural_engine.py` | Role tenure, promotion/span measures, organizational API | Source-reviewed duration and manager-link guards. Graph cycles, all invalid-date combinations and organizational interpretation not dynamically exhausted. |
| `succession_engine.py` | Recorded readiness/potential, bench measures; current API and legacy grid | Current and legacy contracts differ (F11). Bench counts are not validated critical-role coverage; proxy labels require careful retention/removal. |
| `survival_engine.py` | Kaplan–Meier steps, RMST/horizons, Cox path; current retention page | Source-reviewed censoring-related contracts and null unsupported horizons. Cohort selection/left truncation and in-sample Cox interpretation still need domain-specific validation; no new arithmetic defect established. |
| `team_dynamics_engine.py` | Team composition, health, span, API and legacy tab | F10 reproduces dropped Unknown department population. Composite interpretation and coverage are not an enterprise health validation. |
| `vector_engine.py` | Embedding/index readiness, vector validation, IDs and search consumers | Source-reviewed finite-vector, availability and stale-index guards. Real FAISS backend unavailable in this environment; no retrieval relevance/scale benchmark or concurrency proof. |

## End-to-end consumer coverage and reachability

| Layer | Reviewed scope | Evidence boundary |
|---|---|---|
| Ingestion and populations | Data loader, preprocessor, population resolver, merge, runtime loader | Synthetic ingestion/activation/identity cases; no exhaustive file-format or concurrent import testing. |
| Model lifecycle | Shared training, lifecycle service, workspace/job registries, platform routes | Real route/service transitions with controlled fitting/scoring. Does not establish accuracy, calibration transportability or prospective labeling. |
| API and schemas | Principal analytics, prediction, experience, quality-of-hire, sentiment, scenario, team, succession, platform and intelligence paths; schema conversion boundaries | Source traces and focused direct route calls. No complete HTTP authorization or every-schema-field integration suite. QoH observation count survives API serialization. |
| Agent and text | Adapter/tool contracts, evidence aggregation, deterministic synthesis, interpretation helpers | Three adapter reproductions. Parent independently confirmed current advisor heuristic labels. No live LLM hallucination/faithfulness evaluation. |
| Current React | Overview, workforce health, flight risk, employee experience, quality of hire, retention forecast, scenario planner; advisor trace independently checked by parent | Homepage has real SSR evidence. Other principal pages source-reviewed for metrics, missingness, claim labels and consumer shape; not every interaction or chart visually executed. |
| Dormant React | Compensation/NLP/succession tabs, nine-box and feature-importance/employee-detail helpers | Source-reviewed reactivation hazards. Presence in the tree is not proof of current reachability. |
| Legacy Streamlit | Main analytics/ML flow, overview, diagnostics, compensation, succession, team and detail components | Source-reviewed; not launched. Distinct from the current web application. |
| Export/report helpers | Export module and interpretation/report consumers | Source-reviewed evidence/claim flow; no exhaustive generated XLSX/PDF fidelity or visual-layout test. Unsafe source values can propagate without shared contracts. |

## Repair order and acceptance evidence

1. Establish immutable dataset/runtime/model provenance and atomic activation, including auxiliary inputs and caches. Exercise A → B → A switching, historical investigation, failed activation and multi-workspace separation through the real API.
2. Protect outcome/identity columns and carry unknown denominators to the UI. Turn F02/F03/F08 into correctness tests with exact expected populations, identifiers, nulls and zero display states.
3. Give API and agent consumers one validated typed evidence contract. Test invalid scores, incompatible IDs/models, empty evidence and undefined fairness comparisons at both boundaries.
4. Reconcile survey totals/populations, source score coverage, scenario cash-flow semantics and grouped team totals. Show measured sample counts and assumptions at the point of interpretation.
5. Remove or align dormant/legacy paths before reactivating them. Then evaluate predictive models, survival/forecast assumptions, NLP and retrieval quality on pinned public benchmarks and synthetic edge cases, followed by appropriate prospective organizational data. Report baselines, uncertainty, calibration, temporal/entity separation and limitations separately from software correctness.

The 19 reproductions provide concrete repair targets. Complete engine source coverage and prior green checks do not, by themselves, establish enterprise accuracy, causal validity, fairness, forecast reliability or deployment readiness.
