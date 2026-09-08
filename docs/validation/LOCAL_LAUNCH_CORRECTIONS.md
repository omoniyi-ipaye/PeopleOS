# Local-first launch correction pass

Baseline: `2ee3ff28d5312a07b34801dee138822fcd5cd99e` (local equivalent tree `8cf966d`).
Candidate identity and final CI outcomes are recorded in PR #3 and its immutable workflow runs. This document is the pre-publication checkpoint.

Six specialist tracks covered input accuracy, historical observations, ML feature validity, agent grounding, result presentation, and independent counterexample review. The user authorized integrating and merging verified corrections; no deployment or employee-data ingestion was authorized.

## Reproduced failures and corrections

| Failure | Correction and independent acceptance |
|---|---|
| One negative salary, tenure or age removed an employee from unrelated headcount | Keep the employee; mark invalid measurements missing and disclose excluded measurement counts. Independent composed fixture retains 60 people, 59 valid salaries and payroll 2,950,000. |
| An invalid snapshot date could cause a newer departure to disappear from current state | Reject unresolved observation dates before activating an upload. |
| Annual and monthly salaries were averaged together; inconsistent currency was accepted | Explicit pay-basis metadata must declare annual amounts; explicit currency must be complete and shared. No inferred FX or automatic conversion. |
| Salary observations 50k/60k/70k were stored as 50k/50k/60k against their incoming dates | Snapshot the successfully written state, use UTC ordering and transactional row rollback. Existing corrupt history cannot be reconstructed without source uploads. |
| Arbitrary exit-outcome copies produced perfect classifier evaluation | Closed canonical predictor contract, audited exclusions, frozen inference inputs. Constant legitimate predictors plus outcome copies now yield AUC 0.5 and fail model activation. |
| Legacy advisor returned invented 9000 headcount and causal claims against actual 80 | Compatibility endpoints use the same governed investigation, authorization, snapshot and evidence boundary. No model-created factual prose is displayed. |
| Named Finance/Madrid scopes could silently receive company-wide answers | Unsupported scopes and unrecognized question terms abstain without substituting unrelated global evidence. |
| High performance/tenure fabricated potential without an assessment | High-potential membership requires explicit valid PotentialRating; absent evidence produces no candidate. |
| Observed attrition share was classified against an unsupported 15% threshold | Descriptive labels and neutral presentation; missing fractions do not become zero or revive stale legacy values. |
| Absolute salary increases displayed as percent raises | Scenario names retain amount/percentage semantics, including market adjustments. |

## Validation contract

Required candidate workflows: Analytics Validation (including all new launch regressions, public benchmarks, 20,000-row stress and live dummy API checks), Agent Foundation, Frontend Modernization, People Team Browser Acceptance, and Windows x64/macOS ARM64/Linux x64 packaged restart/smoke checks. The browser suite adds a normal file-upload journey proving invalid salary preserves headcount and mixed pay rejection preserves the previous active dataset.

Local evidence before publication: 749 full-repository tests, 40 frontend renderer tests, TypeScript checking, 96 live API/restart checks, all 9 public-benchmark mechanics checks and 34 stress checks over 20,000 fictional employees passed. Independent specialist selection: 87 checks passed, plus a separately composed arithmetic fixture. Exact candidate CI and browser totals are recorded in the PR after execution, not assumed here.

## Limits that remain part of the supported scope

- No actual Ollama/gemma3, FAISS/sentence-transformers or SHAP execution was established in this environment. Deterministic evidence remains usable; controlled model responses validate the safety boundary, not real-model behavior or semantic relevance.
- The public classifier benchmark's model fails its activation gate after the stricter predictor contract. Nine passing benchmark checks mean mechanics and rejection controls behaved correctly, not that this model is useful for deployment. No holdout was tuned to restore a pass.
- Missing pay metadata retains legacy annual/shared-currency assumptions with visible warnings. Data without source-unit confirmation does not establish comparable pay.
- Unknown language or unsupported filters can be refused even when a human understands the question. The agent must not substitute a different scope.
- Recorded potential assessments are accepted as supplied; the software does not independently validate talent-review judgment.
- Production canonical dataset artifacts retain full validated history. Legacy direct SQLite upserts retain their documented arrival-order update semantics; prior shifted historical values require original uploads for reconstruction.
- A merge of these corrections is not an unrestricted launch certification or a guarantee of no remaining bugs. Predictive future-outcome validity and actual local-model evaluations remain separate acceptance gates.
