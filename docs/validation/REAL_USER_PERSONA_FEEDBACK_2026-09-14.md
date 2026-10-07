# PeopleOS real-user persona walkthrough

Date: 2026-09-14

Local app: `http://127.0.0.1:3001`

Data: fictional PeopleOS sample/current local dataset only; no real workforce data entered
State restored after test: app lock disabled; 800 source rows, 662 active employees; browser left on Home

This is an agent-led usability walkthrough, not independent human acceptance. The personas are deliberately task-oriented and the observations below are based on live browser interaction with the running local app.

## Personas and journeys

### 1. Cautious first-time People Ops owner

Goal: understand whether the system is safe and where to begin.

Path: Home -> Ask PeopleOS -> `Where are people leaving?` -> submit -> `By department` -> supported result -> Trust & Privacy.

What worked:

- Home immediately showed the active population, recorded-attrition boundary, source status, and a clear investigation CTA.
- The precise department drill-down returned department rates with sample sizes and a source/version label.
- Trust & Privacy clearly stated local ownership, deterministic mode, predictive insights off, and that PeopleOS does not make employment decisions.

Friction:

- The visible starter question `Where are people leaving?` produced `Not enough evidence` with `Unrecognized population scope or analysis terms...` even though the dataset supports a department attrition calculation. Choosing `By department` then worked.

Severity: P1 onboarding/usability issue. A new user can reasonably conclude the system cannot answer its own suggested question.

Recommendation: map each starter prompt directly to a known supported query, or make the first interaction ask for the missing scope before showing a failed answer.

### 2. HR analyst checking pay and evidence

Goal: inspect insight areas, follow a compensation lead, and verify provenance.

Path: Insights -> Workforce -> Attrition by department -> submit -> open `Why you can trust this answer` -> open `Evidence ledger` -> Ask PeopleOS -> `How does pay look?` -> submit -> `By department` -> direct `Headcount by department` query.

What worked:

- Workforce insight showed 662 active people, eight departments, recorded attrition, a department table, and related-pattern context.
- The answer provenance panel exposed evidence quality (92%), coverage (100%), tools with evidence (1/1), and known gaps (0).
- The evidence ledger exposed the derived-analysis identifier and source.
- The direct supported query returned department headcount with sample sizes.

Friction:

- Insight links pre-filled Ask PeopleOS but did not run the query; the user must infer that a second `Ask PeopleOS` click is required.
- The starter compensation prompt also returned `Not enough evidence` until the user selected `By department`.

Severity: P2 for the pre-filled-but-not-submitted transition; P1 for the starter-prompt mismatch noted above.

Recommendation: make the pre-filled state explicit (`Review and run`) or submit automatically when an insight link is selected. Keep the narrower follow-up when the natural-language intent is genuinely underspecified.

### 3. Executive or Finance partner pressure-testing a decision

Goal: explore a pay assumption without treating it as a recommendation.

Path: Plan -> default 5% pay change / whole workforce -> Explore scenario -> switch to one department -> Engineering -> Explore scenario.

What worked:

- The planner required an explicit scope and department before running a departmental scenario.
- Results showed people in scope (220), baseline and modeled outcome, modeled cost, ROI, the `Exploratory` label, and a pressure-test list.
- The copy explicitly warned the user to validate causal and financial assumptions before making a workforce decision.

Feedback: this is appropriately bounded for a public-beta demo. The negative modeled impact (`-1.2M`, ROI `-100.0%`) is visible, but a Finance user would benefit from a clearer currency/unit label adjacent to the result.

Severity: P2 clarity issue.

### 4. Security-conscious owner

Goal: set, use, and remove the local six-digit owner lock.

Path: Settings -> enter fictional PIN `123456` -> Set owner lock -> Lock app -> wrong PIN -> correct PIN -> Remove app lock -> confirm current PIN.

What worked:

- Setup changed the screen to `App lock is ready` and exposed lock/change/remove controls.
- Lock removed the navigation and presented a dedicated locked state with a clear owner-PIN prompt.
- Wrong PIN produced `That PIN did not unlock PeopleOS.` without exposing the protected UI.
- Correct PIN restored the app.
- Removal succeeded after confirming the current PIN and returned to the setup state.
- The temporary PIN was removed; `/api/app-lock/status` ended as `{"enabled":false,"locked":false}`.

Friction:

- A five-digit PIN left the submit button enabled and only showed `Use exactly six digits for the owner PIN.` after submission. This is safe, but an inline validity cue would be friendlier.
- During the first removal attempt, the running API process was stale and returned `Not Found` because it had not loaded the newer `/change` and `/disable` routes. After restarting the exact PeopleOS API process and refreshing the browser, removal succeeded. This is a local runtime/startup observation, not a reproduced current-route defect.

Severity: P2 validation affordance; P1 operational risk if a packaged/local launch can serve a stale API process.

Recommendation: ensure the supported launch path cannot leave an old backend process serving the UI, and add visible six-digit validation before submit.

### 5. Compact-screen user

Goal: see whether navigation can be reduced without losing access.

Path: Trust & Privacy -> Collapse navigation -> Expand navigation.

Result: the navigation collapsed to an `Expand navigation` control and restored successfully. No blocking issue found in this quick check.

## Launch disposition from this walkthrough

The core user journeys are usable with fictional data: inspect the workforce, ask a precise question, open evidence, pressure-test an aggregate scenario, review trust boundaries, and use the owner lock. The app was left on Home with verified local data and no lock configured.

The largest public-beta usability gap was the mismatch between starter prompts and the query vocabulary accepted by the evidence engine; it is addressed and re-verified below. The pre-filled Ask transition, scenario currency labeling, and PIN validation were also addressed. This walkthrough does not replace independent fresh-user acceptance, production deployment checks, or organization-specific validation.

## Remediation and second walkthrough

The following changes were made after the first round:

- Starter prompts now use supported canonical questions and run in one click: workforce structure, recorded attrition by department, salary by department, and headcount by role.
- Insight deep links now automatically run their pre-filled Ask PeopleOS question instead of stopping at an unsubmitted form.
- Scenario results now show the reporting currency in the source line, net-impact label, and cost/benefit values.
- Owner-lock setup now shows digit progress, matching guidance, and keeps `Set owner lock` disabled until both entries are valid and equal.
- Added regression coverage for the starter-question and insight-link paths.

### Second-round personas

1. First-time owner: clicked `Where is recorded attrition?`. It immediately produced a supported department answer with evidence; no failed starter state.
2. Analyst: opened Insights -> Workforce -> Attrition by department. The link landed on Ask PeopleOS and automatically produced the supported answer; no second submit was needed.
3. Finance partner: ran the default pay scenario. The result displayed `monetary amounts in USD`, `Modeled net impact (USD)`, and currency-prefixed cost/benefit values without the earlier ambiguity.
4. Security-conscious owner: entered five digits and saw `5/6 digits entered` with the action disabled; entering a mismatched six-digit confirmation showed `PIN entries do not match.` without saving anything.

Second-round disposition: all findings from the first live walkthrough were either fixed and re-verified or classified as an operational launch constraint rather than a current product defect. The browser remained on the live local app with fictional data; no real data or permanent PIN was used.

### Verification evidence

- Live browser sweep: all four revised starter paths returned supported answers; the workforce insight deep link returned a ready answer without a second submit; scenario currency and PIN feedback were visible.
- Web renderer/regression tests: 43 passed.
- New advisor-path regression tests: 2 passed.
- TypeScript check: passed.
- Production web build: passed.
- Owner app-lock tests: 8 passed.
- Lint: 0 errors; existing repository warnings remain.

### Third live round

The final browser sweep repeated the owner, analyst, and Finance paths against the running local app. The Finance path exposed one additional responsive presentation issue: the long net-impact value was ellipsized in the narrow fourth metric card even though the underlying result was available.

Severity: P2 readability issue.

Remediation: added a scoped readable-value treatment to the scenario net-impact card so the amount wraps instead of being truncated, while leaving the compact truncation behavior for ordinary metric cards unchanged. The browser check was repeated after the hot reload and the full amount remained visible.
