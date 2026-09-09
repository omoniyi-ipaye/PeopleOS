# Public beta release checklist

This is a preparation record, not a published release or a production certification.

Baseline before public-beta hardening: `ae986ef5be780790a798871d5a20fb6d2e95f343` (PR #5 merged).
Verified public-beta candidate: `90f1958708a78481b8931893076f14c076be92a1`.
Merged to `main`: `a300db48b2fe6d11076ed14fcdd891d170316764` via PR #6.

## Intended release scope

Local, single-user descriptive People analytics and evidence-backed investigations for People Operations and HR teams. Start with fictional sample data. PeopleOS shows source populations, exclusions, units, validation state and limitations. Predictive outputs remain behind explicit evaluation/activation controls and no prospective organization-specific predictive accuracy is claimed.

## Engineering gates — completed

- [x] Agent Foundation passed on the verified candidate.
- [x] Frontend Modernization passed on the verified candidate.
- [x] E2E User Journey passed on the verified candidate.
- [x] Analytics Validation passed on the verified candidate.
- [x] People Team Browser Acceptance passed its full desktop/mobile rerun on the verified candidate.
- [x] Local Desktop Build passed for Windows x64, macOS ARM64 and Linux x64 build/smoke/restart/archive paths.
- [x] Release Security passed across the validated frontend and Python dependency profiles.
- [x] Local Ollama Acceptance passed on the verified candidate.
- [x] README updated to the current People-team product positioning.
- [x] Commercial-use policy documented and a request template added.
- [x] Pilot feedback template added.

## Licensing status

PeopleOS remains **source-available**, not OSI open source.

The current `LICENSE` retains the Apache 2.0 text plus an additional commercial-use restriction. Commercial use is prohibited without the express written permission of Omoniyi Ipaye. See `COMMERCIAL_USE.md` for the plain-language explanation and the Commercial use request issue template for the permission workflow.

This repository must not describe the project as unrestricted open source while that additional restriction remains in force.

## Ready for pilot

The following are now engineering-ready for a controlled pilot:

- fictional sample-data onboarding;
- `.xlsx`, `.csv` and `.json` imports;
- deterministic workforce analytics;
- governed Ask PeopleOS investigations;
- privacy-bounded aggregate drill-down with typed cohort filters;
- evidence/provenance inspection;
- scenario planning with non-consequential boundaries;
- packaged desktop lifecycle controls;
- desktop/mobile browser acceptance coverage.

## Human / owner-controlled items still open

- [ ] Independent fresh-user installation and usability walkthrough on target machines.
- [ ] Capture pilot outcomes and recurring friction using the pilot feedback template.
- [ ] Decide whether Windows code signing and macOS notarization are required before broad public distribution.
- [ ] Choose a version/tag and final public release notes when publication is authorized.
- [ ] Review release archives, checksums and third-party/model notices before publishing binaries.
- [ ] Explicitly authorize creation of a GitHub Release / public binary distribution.
- [ ] Perform organization-specific prospective predictive validation before making stronger intended-use predictive claims.

## Distribution limitations

CI artifacts are build evidence until promoted to a versioned release. Checksums establish byte integrity, not publisher identity. No Windows signing, macOS notarization or publisher reputation claim is currently made. Do not tell users to disable system-wide security protections. Multi-user hosting, enterprise identity, organization-specific retention policies and real employee-data pilots remain outside the current public-beta engineering validation.

## Verification summary

PR #6 (`Prepare public beta distribution and close dependency security gaps`) merged the verified candidate after all eight required workflows passed. The final browser acceptance rerun passed the complete desktop/mobile journeys and its action/process coverage ledger. No privacy, evidence, security or analytical guardrail was weakened to obtain the passing result.

See also:

- `docs/product/PEOPLEOS_10_10_PRODUCT_PLAN.md`
- `COMMERCIAL_USE.md`
- `SECURITY.md`
- `docs/PUBLIC_BETA_GUIDE.md`
- PR #6 workflow evidence
