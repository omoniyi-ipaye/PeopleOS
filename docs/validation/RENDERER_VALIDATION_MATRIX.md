# Renderer Validation Matrix

Cycle 002 reviewed the production web routes at baseline `cc197912b107fdd4e67e81e163ab9e351cdaac27`. The review maps analytical values to their user-facing meaning and checks loading, failure, unavailable, zero, and stale-source behavior. It does not treat server rendering as browser interaction evidence.

## Routed product surfaces

| Route | Evidence rendered | Edge-state result |
|---|---|---|
| `/` | Active population, known outcomes, tenure, largest department | Existing tests preserve missing outcome and measured-zero distinctions; status and summary failures are explicit. |
| `/advisor` | Synthesized answer, evidence ledger, tool coverage, gaps and conflicts | Existing tests bind answers to source generation and hide stale answers. A live LLM is outside the current browser gate. |
| `/workforce-health` | Department population, attrition share, tenure, rating, correlation | Empty department responses now remain unavailable; secondary correlation and threshold failures are visible. |
| `/flight-risk` | Aggregate score bands and retrospective model diagnostics | Platform failure now blocks the view with an explicit unverified-state message rather than claiming the model is inactive. |
| `/employee-experience` | Explicit survey composite, respondents, coverage, segments and associations | Omitted respondent counts and coverage now remain unavailable rather than becoming measured zero. |
| `/quality-of-hire` | Historical population, source cohorts and pre-hire associations | Omitted population and measurement counts now remain unavailable. |
| `/retention-forecast` | Kaplan–Meier curve, cohort measures and Cox availability | Loading, error and empty-curve states exist. Dataset provenance and stale-source binding remain to be added. |
| `/scenario-planner` | Scoped sensitivity outputs, costs, assumptions and risks | Numeric inputs are finite and bounded, select labels are bound, unavailable results stay unavailable, and stale results are explained. |
| `/search` | Retrieved workforce text and similarity ranking | Capability loading/failure is explicit, index size cannot default to zero, malformed ranking values remain unavailable, and failed searches can be retried. |
| `/platform` | Dataset, model, actor, recovery and fitness state | Whole-page failure is explicit. Partial rendering when an ancillary request fails remains future work. |
| `/settings` | Runtime, dataset and capability registry | Status failure is explicit. Health-only failure and source provenance remain future work. |
| `/upload` | Dataset lifecycle and activation actions | Source state and mutation failures are explicit; mutating controls fail closed until status is verified. Reset confirmation remains future work. |
| `/sessions` | Investigation lifecycle and dataset identity | Loading, failure and empty states exist. Invalid timestamp fallback and retry remain future work. |
| `/design-system` | Internal component reference | This is a development reference and is not analytical evidence. |

## Executable evidence

`npm run test:analytics` executes 30 real-component server-render cases across dashboard, advisor, compensation, NLP, predictive explanation, experience, quality-of-hire, workforce health, research, upload, scenario and model-state boundaries. `npm run lint` completes with no errors. `npm run build` compiles and type-checks all 14 routes.

The browser suite remains the authority for hydration, keyboard behavior, mobile navigation, stale-query behavior and screenshots. Its Cycle 002 expansion should cover the new action and process cases in `web/e2e/people-team.spec.ts`.

## Coverage gaps before an exhaustive claim

Several existing analytical renderers have no routed product entry point: Compensation, Succession and NLP diagnostic tabs plus multiple chart components. API families for detailed compensation, succession, team dynamics, fairness, structural analysis, clustering, forecasting, group comparison and several sentiment operations likewise lack an end-user route. Those are product-coverage gaps, not passing renderer validations.

Remaining human checks include screen-reader flow, focus order, contrast under actual display profiles, text expansion/localisation, and comprehension testing with People-team users. Actual browser screenshots must be reviewed from the candidate CI artifacts before the design gate is closed.
