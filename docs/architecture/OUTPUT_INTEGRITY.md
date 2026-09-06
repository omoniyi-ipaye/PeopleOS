# PeopleOS Output Integrity

Status: **TARGET WITH EXPLICIT ASSUMPTIONS — verification pending on final SHA**

This document is the canonical human-readable contract for how PeopleOS turns workforce data into user-visible results. The governing principle is: **a numerically correct calculation is not a trustworthy output if the population, denominator, statistical meaning, model quality, governance boundary, or UI label is wrong.**

## End-to-end integrity chain

`raw field → schema mapping → population resolution → metric/model definition → calculation → evidence → quality/sufficiency → policy boundary → UI wording → audit/reproducibility`

Every material output must preserve this chain.

## Population contracts

**Historical population** retains validated source rows, including repeated snapshots, only for analyses that explicitly require history.

**Current population** contains exactly one latest observation per `EmployeeID`. If `SnapshotDate` exists, the latest valid snapshot wins.

**Active population** is the current population with normalized `Attrition == 0`. Current headcount, payroll, salary distribution and active-workforce scoring use this population unless explicitly declared otherwise.

**Unknown outcome** stays missing. Unrecognized Attrition labels are never coerced to active, and predictive training is disabled when the target is ambiguous or single-class.

## Canonical metric contracts

| Metric | Meaning | Population | Limitation |
|---|---|---|---|
| Headcount | Current active employees | Active | Not dataframe row count |
| Record count | Current unique employee observations | Current | Can include departed/unknown outcomes |
| Observed attrition share | Share of known current outcomes marked departed | Current known outcomes | **Not period turnover rate** |
| Current payroll | Sum of valid positive salary | Active | Excludes departed/invalid salaries |
| Salary dispersion consistency | Within-department dispersion summary | Active | **Not adjusted pay equity** |
| Gender pay gap | Descriptive raw/stratified disparity | Active eligible groups | Not causal/legal determination |
| Four-fifths ratio | Favorable rate / highest favorable rate | Eligible groups | Attrition context uses retention as favorable outcome |
| Quality of Hire | Observed source/pre-hire associations + configurable descriptive composite | Qualified cohorts | Not causal source effectiveness |
| Experience composite | Configured weighted composite of explicit measured experience signals | Measured-signal population | No HRIS-proxy-derived engagement score |
| Survival | Cohort time-to-event survival from a defined origin | Valid survival cohort | Not individual next-period departure probability |
| Scenario output | Sensitivity result under configured assumptions | Aggregate scenario scope | Not causal forecast or empirical probability |
| Evidence quality | Heuristic support score | Investigation | **Not probability that conclusion is true** |

## Statistical integrity

- Correlation is association, not percent uplift or causation.
- Statistical significance is not proof of unfairness, causation, practical importance, or absence of an issue.
- Minimum group sizes are enforced for fairness/disparity screening.
- Undefined fairness rates remain unknown rather than being represented as zero.
- Learned preprocessing is fitted on training rows after the raw-row split.
- Holdout data does not influence imputation, scaling, outlier bounds, category encoding, SMOTE, model selection, or tuning.
- Brier score/calibration are reported separately from discrimination metrics such as F1/ROC AUC.
- Kaplan–Meier output is cumulative survival from its time origin; it is not automatically a conditional future probability.
- Cox hazard ratios are associations and are subject to proportional-hazards assumptions.
- Forecasting requires genuine repeated dated observations; PeopleOS does not synthesize history from HireDate or tenure.

## Consequential-action boundaries

PeopleOS does **not** expose or automate:

- individual retention-risk ranking or employee risk-detail endpoints;
- individual survival-risk ranking;
- individual experience/engagement scoring;
- manager experience ranking;
- individual succession/high-potential/promotion-readiness ranking;
- individual stagnation or named-manager span ranking;
- employee salary-outlier or compa-ratio lists;
- headcount-reduction selection by performance, tenure or cost;
- risk-targeted retention interventions;
- causal intervention recommendations without a validated identification design;
- employee cluster membership.

Aggregate screening remains available where it can support investigation without becoming an automated people decision.

## Predictive lifecycle

1. Dataset activation establishes current/active populations and initializes read-only analytics.
2. Training is an explicit governed operation.
3. Raw rows are split before learned preprocessing.
4. Candidate evaluation occurs on an untouched holdout.
5. Activation requires the exact evaluated runtime artifact.
6. Only the active workforce is scored.
7. Model fitness and calibration are displayed separately from aggregate score distribution.

A process restart without durable predictive artifact persistence fails closed. Durable artifact persistence is still a production assumption/debt item.

## Evidence and synthesis

`overall_confidence` and item `confidence` fields remain for compatibility, but their semantic contract is **heuristic evidence quality/reliability weight**. They are not statistical confidence and not probabilities of truth.

Evidence quality now accounts for:
- evidence kind (observed, derived, assumed, unknown),
- usable tool contribution,
- declared item reliability,
- known gaps,
- material cross-tool contradictions in the same scope.

Assumption-only evidence cannot become sufficient simply because a tool returned successfully.

## Output-boundary remediation completed

- snapshot-safe population resolution;
- active-first headcount and compensation;
- observed attrition-share terminology;
- conservative critical-field mapping;
- salary-dispersion vs pay-equity distinction;
- fairness favorable-outcome ratios + group suppression;
- leakage-safe predictive preprocessing/evaluation;
- explicit model activation binding;
- aggregate-only prediction/survival/experience/succession/structural/compensation boundaries;
- measured-signal-only experience outputs;
- observational Quality-of-Hire semantics;
- exploratory Scenario Planner semantics;
- unvalidated causal API disabled;
- synthetic historical forecasting removed;
- cluster-member exposure removed;
- heuristic evidence-quality semantics enforced;
- output-integrity tests and compile coverage wired into CI;
- user-facing UI language aligned to these contracts.

## Explicit assumptions / remaining non-blocking debt

- Observational HR data cannot identify causal intervention effects without an appropriate identification design.
- Scenario response/cost parameters require organization-specific validation.
- Configurable composite metrics require local validation before being treated as decision thresholds.
- Predictive runtime artifacts are process-local; durable artifact persistence is required for resilient multi-process production deployment.
- Several backward-compatible API field names (`turnover_rate`, `best_predictors`, `equity_scores`) remain until a future breaking API version.
- Legacy engine methods that are unreachable behind governed API boundaries can be deleted in a later cleanup.

Final readiness remains **BUILD READY WITH ASSUMPTIONS** because the explicit assumptions above are real product constraints, not unresolved semantic ambiguity.
