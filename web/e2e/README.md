# People-team browser acceptance

This gate runs the real built React application against the actual FastAPI app.
It does not intercept or replace API responses, load mock engines, or require an
external LLM. Every run creates a temporary PeopleOS data home and workspace
registry, and refuses to reuse a server already occupying ports 8000 or 3000.

From a normal development or CI environment with Python dependencies installed:

```sh
cd web
npm ci
npx playwright install --with-deps chromium
npm run build
npm run test:e2e
```

Set `PEOPLEOS_TEST_PYTHON` to the Python executable containing the backend
requirements if `python` is not the correct environment. Do not use a static
`PEOPLEOS_DESKTOP_BUILD=1` build for this gate; desktop packaging has its own CI.
The Playwright web-server hook starts both processes using
`scripts/run_browser_acceptance.py` and shuts them down when the suite finishes.

The four journeys run in Chromium desktop and emulated mobile viewports:

- Upload a known-answer workforce, check 80 active people, 20% observed departure
  share and 2.0-year tenure; inspect real agent evidence and Trust Center; activate
  a different dataset (60 records: 30 active and 30 unknown statuses) and verify
  30 active people, measured zero observed attrition share and
  measured zero tenure.
- Upload missing outcome values and confirm unavailable results do not become
  a measured zero or a reassuring statement.
- Confirm that the agent cannot investigate without a dataset, then confirm an
  unsupported question remains insufficient with a dataset loaded.
- Keep an answer open while another tab changes datasets; require the previous
  answer to disappear and a fresh investigation to use the new population.

Desktop and mobile checks also reject document-level horizontal overflow on the
cockpit, expanded evidence ledger and Trust Center. Screenshots at meaningful
checkpoints are attached even on passing runs. These are evidence for visual
review, not a substitute for a human assessment of visual quality.

`web/browser-artifacts/` contains JSON and HTML reports, frontend/backend logs,
checkpoint screenshots, and failure screenshots/videos/traces. CI must upload
this directory with `if: always()`, so failed and incomplete runs stay reviewable.
Retries are disabled to prevent an intermittent failure from being concealed.

Passing this gate demonstrates these particular workflows and fixture outputs.
It does not establish production security, prospective predictive accuracy,
all-browser compatibility, accessibility conformance, or universal correctness.

Test-runner setup follows the official [Playwright web-server documentation](https://playwright.dev/docs/test-webserver)
and [recording options](https://playwright.dev/docs/test-use-options#recording-options).
