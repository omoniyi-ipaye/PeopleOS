# Analytics and ML validation — 2026-09-08

The synthetic known-answer tests and pinned public benchmarks validate specific calculations and failure boundaries. They do **not** certify PeopleOS for enterprise decisions or establish future-departure accuracy on a customer's workforce. This change builds on desktop PR #2, head `523727e`, whose Windows x64, macOS ARM64 and Linux x64 packaged smoke checks passed. The Windows launcher, packaging specification and desktop dependencies are preserved.

## Findings and changes

| Area | Observed defect | Corrected behavior and evidence |
|---|---|---|
| Current workforce and pay | Some legacy tests counted departed employees; salary infinities and invalid values entered aggregates | Canonical current/active population; finite positive salary statistics; synthetic headcount/payroll oracle and independent IBM source totals |
| Model fitting | Preprocessing was fitted before model-selection CV; ordinal category codes could be interpolated by SMOTE | Raw employee holdout; cloned preprocessing fitted inside each CV training fold; deterministic RF/XGBoost/LightGBM grid selection by training CV average precision; no synthetic category interpolation |
| Model evaluation | AUC and permissive absolute Brier cutoff could approve a worse-than-baseline model; reliability reflected sample size | Report training-prevalence Brier baseline, prevalence/AP baseline, weighted 10-bin ECE, test class counts, split scope and seed; candidate gate requires baseline improvements and minimum evidence |
| Model Lab | Retraining on current outcomes was labelled a historical backtest; feature-pruning lift was invented; optimize claimed an applied change | No backtest without timestamped pre-outcome predictions and mature follow-up; no numeric lift; review-only plan; sensitivity cannot refit the artifact's preprocessing |
| Survival | Trapezoids incorrectly integrated the KM step curve; intervals and curve lengths differed; unsupported horizons extrapolated | Step RMST with explicit restriction horizon, full arrays and segment curves, no horizons beyond support; Cox uses configured covariates and excludes post-outcome fields |
| Experience | Nonrespondents received HRIS proxy scores; scales varied by row; fractional score-band gaps; coverage missed signal types | Only in-range measured responses; fixed documented scales; actual respondent denominator; contiguous bands and coverage visible in UI |
| eNPS | Invalid scores became passives | Only numeric responses in 0–10 enter the response denominator |
| Fairness | Undefined all-zero favorable ratios appeared clean; age labels mismatched boundaries; duplicate predictions inflated samples | Undefined evidence remains unavailable, half-open age bands, unique prediction IDs and valid probabilities/outcomes required |
| Forecasting | Three dates became interpolated daily history; departed employees inflated counts; fixed confidence bands implied uncertainty evidence | At least 12 complete monthly censuses, active population, no missing-month filling; last-three-month model selection against naive MAE; no invented prediction intervals |
| Financial scenarios | Annual raise costs were multiplied by 12 again; requested horizon was ignored; correlation/ML was treated as causal pay response | Annual cost × months/12; missing/nonfinite salaries rejected; explicitly configured hypothetical baseline/pay response, default effect zero; supplied cohort cost in simulation; headcount financial summaries reconcile |
| Hiring source | Missing quality components contaminated the composite and could create a failing grade; correlation magnitude became percentage uplift | Weight only measured components; missing grade unavailable; constant correlations excluded; remove fabricated percentage uplift and automatic weight-changing recommendation |
| Promotion screening | Code claimed controlled regression while comparing raw group means | Explicit unadjusted years-since-last-promotion comparisons, Welch two-group test; no claim of time-to-promotion, confounder control or multiple-testing adjustment |
| Text analysis | Failed inference became neutral sentiment; unvalidated IDs/scores were accepted | Missing results remain missing; reject unknown/duplicate IDs, invalid scores and labels; absent average is null; model output labelled exploratory |
| Network analytics | All coworkers in a department were linked and interpreted as measured collaboration and individual influence | Product endpoints report unavailable until measured relationships and validated semantics exist; legacy graph engine is not production evidence |

## Public benchmark evidence

Run `python scripts/benchmark_analytics.py --output benchmark-results.json` after installing `requirements-validation.txt`. For offline use, provide `--ibm-data /path/to/emp_attrition.csv`; the same SHA256 is required. The script fails on changed data, arithmetic/curve mismatches, overlapping employee partitions, or a passing shuffled-label control. Raw public data is not vendored.

The checked-in [machine-readable run](public-benchmark-results.json) records dependency versions, source hashes, seed, model grids/selected parameters, class counts, calibration bins and evaluation checks. CI publishes its own report for the actual Python 3.11 dependency environment. Numerical results may vary slightly between these recorded environments; thresholds are not tuned to each run.

IBM's [employee-attrition-aif360 repository](https://github.com/IBM/employee-attrition-aif360/tree/13287d5f717dc978eda249aef4665e04c7cec8b0) describes its HR fixture as synthetic in its notebook. It contains 1,470 rows; the benchmark annualizes MonthlyIncome explicitly and excludes identifiers from model features. The upstream README specifies ODbL for the database and DbCL for its contents. This fixture has no validated feature-as-of timestamp or future outcome horizon.

| Dataset/control | Holdout AUC | Average precision | Brier error | Baseline Brier | Minimum retrospective gate |
|---|---:|---:|---:|---:|---|
| IBM synthetic HR, original labels | 0.8037 | 0.4886 | 0.1077 | 0.1343 | Pass |
| Shuffled labels, seed 7 | 0.4883 | 0.1865 | 0.1410 | 0.1343 | Reject |
| Shuffled labels, seed 19 | 0.4897 | 0.1814 | 0.1376 | 0.1343 | Reject |
| Shuffled labels, seed 43 | 0.4589 | 0.1652 | 0.1446 | 0.1343 | Reject |

The original-label run selected XGBoost, with a 1,176/294 employee train/test split, holdout class counts 247/47, AP prevalence baseline 0.1599 and weighted ECE 0.0372. These are classification diagnostics, not a “percentage of employees correctly predicted to leave.” The test partition is evaluated once after candidate selection. Three fixed negative controls are useful regression checks, not a general estimate of the false-approval rate.

Waltons, distributed with lifelines, supplies 163 survival observations. An independent product-limit recurrence matched all 33 returned curve points within display rounding. Independently integrated restricted mean was 50.133969323185624; engine result was 50.13396932318564. The fixture's original time units are mapped mechanically to the engine's month convention solely for verification; these are not employment records. See [lifelines survival examples and restricted-mean discussion](https://lifelines.readthedocs.io/en/latest/Examples.html).

Fold-local preprocessing follows the [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html). Calibration metrics describe probability behavior; a rank metric alone does not establish calibration. See [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).

## Capability inventory and remaining enterprise gates

This is a source/API inventory, not a claim that every legacy engine has received independent predictive validation.

| Capability / implementation | Current permitted interpretation | Remaining evidence |
|---|---|---|
| Analytics, compensation, population and merge | Descriptive observed records with explicit denominators | Source-system reconciliation, data completeness, currency/pay-period contracts and real export acceptance tests |
| Random Forest, XGBoost, LightGBM (`model_training`) | Retrospective employee classification only | Feature availability at scoring time, prospective prediction horizon, mature temporal holdout, uncertainty across cohorts/time, calibration stability, threshold costs and subgroup error bounds |
| Low-level `MLEngine.train_model` | Legacy preprocessed-matrix evaluation, explicitly unvalidated | Cannot attest upstream preprocessing isolation; governed lifecycle uses raw-data `train_attrition_model` |
| KM/Cox survival | Cohort curve / observational hazard association | Verify exit/censoring duration semantics; censoring assumptions, event-per-parameter adequacy, PH diagnostics, transportability and temporal validation |
| Forecasting | Monthly active census extrapolation; validation period used for model selection | Independent later-period test, rolling-origin stability, useful intervals and complete census attestation |
| Fairness | Aggregate disparity screening with small-group suppression | Label quality, selection-bias review, subgroup confidence intervals, intersectional stability; no fairness certification from a non-significant test |
| Experience / survey sentiment | Respondent composite and survey response summaries | Source-specific scale mapping, nonresponse bias, instrument validity, comparable component coverage and repeated survey-wave contracts |
| Quality of hire / structural | Exploratory cohort composites and unadjusted group differences | Comparable tenure exposure, valid component denominators, adjusted models and multiplicity correction; no causal hiring uplift |
| Scenario | Arithmetic under stated finance/response assumptions | Locally validated annual salary/cost definitions, causal intervention evidence; no empirical uncertainty claimed from configured draws |
| Team dynamics / succession | Team composites are labelled heuristics; succession summarizes recorded assessments, with unassessed coverage; individual rankings remain blocked | Instrument validity, assessment provenance/freshness, role-specific coverage and external outcomes; no future-readiness accuracy claim |
| Clustering / causal / network | Governed clustering and causal estimation remain blocked; network now unavailable | Validated aggregate use case, measured relationships, causal identification and diagnostics as applicable |
| NLP / vector retrieval / LLM agent | Exploratory text output, retrieval or evidence-grounded synthesis | Domain-labelled evaluation, retrieval recall/precision, citation faithfulness, abstention/error rates; no public HR NLP/embedding accuracy claim in this run |

The minimum activation thresholds (AUC ≥ .60, Brier ≤ .30 and better than training-prevalence baseline, AP above prevalence, weighted ECE ≤ .15, test ≥ 50 with ≥ 10 of each class, isolated holdout and fold-local preprocessing) are deliberately a **minimum retrospective gate**. They do not replace the outstanding enterprise acceptance criteria above. Public and synthetic data cannot close those organization-specific gates.

## Regression and release checks

`Analytics Validation` runs known-answer, model-isolation, existing engine and API-integrity tests plus the pinned benchmark. `Agent Foundation` retains its architecture/integrity gate. `Local Desktop Build` still packages and smoke-tests Windows x64, macOS ARM64 and Linux x64; none of its launcher settings or smoke assertions is weakened. Frontend production build and lint verify the new baseline metrics, retrospective notice, null text-analysis state and measured survey coverage.

Local verification for this change: **186 selected regression checks passed**, including **41 new synthetic/model-isolation cases**; nine public benchmark assertions passed. Next.js production build completed for 16 static pages. ESLint reported zero errors and 55 warnings. Statistical precision-loss warnings in constant-group negative controls are expected and the outputs are unavailable rather than false significance claims. Cross-platform release evidence must come from the PR's packaged jobs; a source test alone cannot establish Windows/macOS/Linux packaged behavior.


## Second integrity pass — teams, succession, surveys, retrieval and agent evidence

This follow-up adds **25 independent synthetic regression cases** in `tests/test_extended_integrity.py`. Together with the selected prior regressions, **211 selected tests pass locally**. The frontend production build passes for 16 pages; lint reports zero errors and 43 warnings. Analytics CI now includes these new cases and reruns the existing pinned public benchmarks. Public benchmarks validate the model/survival paths described above; they do not validate the additional talent, survey or embedding constructs.

| Area | Reproduced integrity defect | Correction |
|---|---|---|
| Team population and denominators | Departed/duplicate rows counted; unrated employees diluted high-performer percentages; missing components could become healthy/critical scores | Resolve current active employees; clean finite in-range measurements; use measured rating denominator and expose observation counts; unavailable health remains null |
| Team API filters and distributions | Missing filter columns were ignored; invalid job-level strings failed internally; missing gender became “other”; age groups used inaccurate generation labels; locations after the top ten disappeared | Reject unsupported/malformed filters; separate unknown gender; use explicit age intervals; retain all locations; expose measurement coverage and descriptive attrition-share semantics |
| Succession API and dashboard | Summary read nonexistent engine keys and reported zero; tenure/performance/risk proxies were presented as readiness; performance was reused as potential | Read actual engine keys; require recorded readiness and separately measured potential; retain “Unassessed”; expose assessed denominator and coverage; aggregate counts directly in UI instead of averaging rounded department scores; display errors as unavailable |
| Hiring cohorts | Missing ratings diluted performance percentages; tenure-free rows entered exposure-restricted comparisons | Report measured rating/outcome counts; compute percentages from measured observations; require tenure for the minimum-exposure cohort |
| Survey joins, dates and onboarding | Employee history multiplied survey rows; a survey's existing grouping column was overwritten by a suffixed join; timezone/end-date filtering excluded valid responses; invalid dates became trend periods; no measured dimensions could appear healthy | Preserve survey grouping or join unique employees many-to-one; normalize dates to UTC and include the complete selected end day; exclude invalid trend dates; require dates for requested filters; missing onboarding health stays unavailable; trajectory uses the latest dated recognized stage and excludes unknown stages |
| Vector index and search | Empty/failed rebuild retained old employee results; misaligned metadata could misattribute passages; caller mutations changed stored IDs; inference failures looked like no matches; reciprocal distance appeared as a percentage match | Clear previous index before rebuild; require aligned finite vectors; copy metadata; return an unavailable error on inference failure; retain squared distance and explicitly label the compatibility value a ranking score |
| Agent evidence | Failed results, absent numeric measurements and explicit dataset/model mismatches could count as support; replayed IDs could count twice | Normalize nonfinite/pandas-null values; exclude failed/blocked, missing-metric, wrong-source and explicitly out-of-scope evidence; deduplicate result/evidence IDs; keep units/currencies distinct during contradiction checks |

Succession input contract: `SuccessionReadiness` accepts `Ready Now`, `Ready 1-2 Years`, `Developing`, and `Early Career`; other/missing values are unassessed. Its descriptive index averages weights 1, .7, .3 and .1 over assessed active employees. `PotentialRating` must be a separate recorded 1–5 assessment; nine-box classification requires both it and a valid `LastRating`. These records are not independently verified predictions, and the index does not measure the percentage of critical roles with qualified successors.

The retrieval oracle supplies controlled vectors `[0,0]` and `[3,4]`: squared distance is 25 and the compatibility ranking score is 1/26. A lightweight index double isolates our indexing/metadata logic; this does not test a downloaded embedding model or establish recall/precision on workforce text. FAISS documents that [METRIC_L2 returns squared Euclidean distance](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances). Real embedding relevance, source-to-passage faithfulness and dataset-refresh concurrency still require separate evaluation.

Remaining limits: team composites and salary dispersion are not validated measures of team health or adjusted pay equity. Survey dimension correlations, individual early-warning heuristics, repeated-wave comparability and response bias still need dedicated validation; this pass does not establish causal drivers or forecast disengagement. Legacy succession candidate methods remain unvalidated and individual product endpoints stay blocked. Evidence filtering checks explicit version conflicts but does not authenticate unversioned evidence or certify LLM citation faithfulness. Geographic database/workspace reconciliation and other legacy engines have not received independent accuracy acceptance in this pass. The capability inventory above remains the enterprise acceptance backlog.
