# PeopleOS controlled pilot handoff — October 7, 2026

## Recommendation and scope

Proceed with a small, assisted pilot of 3–5 people using fictional sample data after all checks on the final integration pass. The source integration is [PR #18](https://github.com/omoniyi-ipaye/PeopleOS/pull/18); its live status and checks are the authoritative merge record. PR #11 history is included to retain the fairness review work.

This is a local, single-user People analytics pilot. Multi-user hosting, real employee-data deployment and organization-specific prediction claims remain outside the accepted scope. A merged source branch is not a public binary release or independent human acceptance.

## Repairs and validation

- Removed frontend production and development dependency advisories, including the unpatched braces dependency, rather than suppressing the dependency audit. Tailwind 4.3.3 and its PostCSS plugin replace Tailwind 3. The Next ESLint plugin alone uses tinyglobby through a scoped dependency override.
- Verified the actual Next root-discovery and internal-link lint rule against fixtures, including wildcard and array roots. Tailwind class-conflict handling was upgraded for version 4.
- Reviewed migrated styling and restored the outline component variant names changed incorrectly by the migration utility. The migration changes presentation, not analytics.
- Preserved citation selection checks while accepting valid grouped local-model citations. Added a numeric guard so digit-based invented measurements, sign changes and percent-unit changes fall back to deterministic evidence. Regression tests first reproduced the invented-headcount failure.
- Numeric validation is deliberately conservative and is not complete semantic verification of generated prose. Users should inspect evidence; generated interpretation is not an employment decision or a prediction guarantee.
- Enabled main protection: up-to-date passing required checks, pull requests, administrator enforcement, conversation resolution, and no force pushes or deletion. No independent human approval is claimed.
- Corrected shutdown instructions: packaged Settings provides Quit PeopleOS and Restart app; closing a browser tab does not stop the local backend.

Local validation passed: 1,240 Python tests plus 6 subtests (38 warnings), 54 frontend renderer/tooling tests, lint, design governance, production build, zero-advisory full dependency audit, and 30 desktop/mobile browser journeys including light/dark and keyboard-focus checks. Exact counts and final remote checks are recorded in PR #18. Remote checks additionally exercise actual local CPU AI, all engine forensic suites and Windows, macOS ARM64 and Linux packages.

The browser requirements are Safari 16.4+, Chrome 111+ or Firefox 128+ due to Tailwind 4. Existing source-available commercial-use restrictions remain in effect.

## First pilot session

1. Start from the final merged source or its reviewed build artifact. Follow the public beta guide; do not disable operating-system security protections.
2. Use the fictional sample, confirm active headcount and inspect its source evidence.
3. Import a fictional CSV, inspect its preflight findings, and activate it explicitly. Check that rejected replacement data leaves the active dataset intact.
4. Ask a supported workforce question, inspect its citations, and try an unsupported causal question. Confirm that limitations are clear.
5. Set an owner PIN, lock and unlock the app, then use Quit PeopleOS and restart it. Record unexpected reopening or lost state.
6. Capture installation friction, confusing terminology and incorrect answers using the existing pilot feedback template. Include platform and candidate version, but no real employee records.

Stop a session and report any exposure of records outside the intended local workspace, incorrect verified measurements, broken import recovery or ineffective owner lock. Prioritize these findings before inviting more testers.

## Remaining acceptance and distribution gates

A fresh user must still install and complete the journey on their own target machine. Automated checks cannot supply that independent result. Gather that evidence during the first assisted pilot.

Unsigned CI artifacts remain build evidence. A broadly distributed release still needs selected version/tag, archive/checksum and notices review, an explicit publication decision, and a decision on Windows signing and macOS notarization. No release or public binary has been published by this handoff.
