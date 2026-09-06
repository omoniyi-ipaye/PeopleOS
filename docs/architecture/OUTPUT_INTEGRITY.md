# PeopleOS Output Integrity

Status: **TRANSITION — integrity remediation in progress**

This document is the canonical human-readable contract for how PeopleOS turns workforce data into user-visible results. The governing principle is: **a numerically correct calculation is not a trustworthy output if the population, denominator, statistical meaning, model quality, or UI label is wrong.**

## End-to-end integrity chain

`raw field → schema mapping → population resolution → metric/model definition → calculation → evidence → quality/sufficiency → policy boundary → UI wording → audit/reproducibility`

Every material output must preserve this chain.

## Population contracts

### Historical population
All validated source rows, including repeated employee snapshots. Historical rows are retained only for analyses that explicitly require time/history.

### Current population
Exactly one latest observation per `EmployeeID`. If `SnapshotDate` exists, the latest valid snapshot wins. Current-state analytics must never count historical snapshots as multiple people.

### Active population
Current population rows with normalized `Attrition == 0`. Current headcount, payroll, salary distribution, current experience, and current structural metrics use this population unless a metric explicitly declares otherwise.

### Unknown outcome
Unrecognized Attrition labels remain unknown/missing. They are never silently coerced to active. Predictive training is disabled when the target is ambiguous or single-class.

## Metric contracts

| Metric | Canonical meaning | Population | Important limitation |
|---|---|---|---|
| Headcount | Current active employees | Active | Not row count |
| Record count | Current unique employee observations | Current | May include departed/unknown outcome rows |
| Observed attrition share | Share of known current rows with Attrition=1 | Current known outcomes | **Not a period turnover rate** |
| Current payroll | Sum of valid positive Salary | Active | Excludes departed and invalid salary rows |
| Salary dispersion consistency | Within-department salary dispersion summary | Active | **Not adjusted pay equity** |
| Gender pay gap | Descriptive raw and job-title-stratified disparity | Active eligible groups | Not causal/legal equity determination |
| Department median ratio | Salary / department median when no true band midpoint exists | Active | **Not formal compa-ratio** |
| Four-fifths ratio | Group favorable-outcome rate / highest favorable rate | Eligible groups | For Attrition, favorable outcome = retention |
| Prediction distribution | Aggregate distribution from activated evaluated model | Active scoring population | Model probabilities require calibration review |
| Evidence quality | Heuristic quality/coverage score for investigation evidence | Investigation | **Not probability that a conclusion is true** |

## Statistical integrity

- Correlation is association, not percent uplift and not causation.
- Group disparity is a screening signal, not proof of bias or discrimination.
- Minimum group sizes are enforced before fairness/disparity comparison.
- Undefined TPR/FPR values remain unknown rather than being written as zero.
- Model evaluation uses an untouched holdout.
- Learned preprocessing is fitted on training rows only.
- SMOTE belongs inside cross-validation folds and training data only.
- Brier score/calibration are evaluated separately from discrimination metrics such as ROC AUC/F1.

## Predictive governance

- Training is explicit and never occurs during upload.
- Candidate activation requires deterministic evaluation checks.
- The exact evaluated runtime artifact must be available before activation.
- A process restart without a durable model artifact fails closed.
- Predictive API output is aggregate-first.
- Individual risk-detail and high-risk employee ranking endpoints are disabled.
- Feature importance describes model influence, not causal attrition drivers.

## Evidence and synthesis

`confidence` fields retained for compatibility represent a **heuristic evidence-quality weight**, not statistical confidence. Model F1 must never be converted into confidence that an individual model output or evidence claim is true. Statistical reliability belongs in explicit metadata (sample size, p-value, Brier score, calibration error, held-out metrics).

## Integrity status labels

- **Observed** — directly verified in current code/runtime.
- **Assumed** — necessary assumption not yet proven by source/runtime evidence.
- **Unknown** — evidence not available.
- **Proposed** — target behavior not yet implemented.

## Current remediation status

### Implemented
- canonical current/latest-snapshot population resolver
- normalized Attrition outcome
- conservative mapping for critical fields
- snapshot-safe ingestion boundary
- active-first headcount and compensation populations
- observed attrition share semantic contract
- salary-dispersion vs pay-equity distinction
- retention-based four-fifths calculation and minimum group suppression
- leakage-safe raw-row split before predictive preprocessing
- model evaluation includes ROC AUC, Brier score and calibration signal
- individual predictive ranking endpoints disabled
- evidence `confidence` explicitly classified as heuristic quality
- output-integrity regression tests wired into CI

### Remaining review
- survival-analysis censoring/time-origin definitions
- scenario-planner causal/forecast assumptions
- employee-experience composite weighting
- quality-of-hire correlation language and recommendation thresholds
- agent sufficiency/contradiction rules
- all frontend labels for deprecated `turnover_rate` compatibility fields
- durable predictive model artifact persistence/recovery

The system must remain **BUILD READY WITH ASSUMPTIONS** until these remaining paths are reviewed and verified.