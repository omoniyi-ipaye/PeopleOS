# Public beta release checklist

This is a preparation record, not a published release or a production certification.

Baseline before public-beta hardening: `ae986ef5be780790a798871d5a20fb6d2e95f343` (PR #5 merged).
Historical verified public-beta candidate: `90f1958708a78481b8931893076f14c076be92a1`.
Historical merge to `main`: `a300db48b2fe6d11076ed14fcdd891d170316764` via PR #6.

## Current launch candidate

The current integration is [PR #18](https://github.com/omoniyi-ipaye/PeopleOS/pull/18), including the branch history and substantive work from PR #11. Its live checks and merge status are authoritative.

The October 7 security, compatibility and final acceptance handoff is recorded in [PUBLIC_PILOT_READINESS_2026-10-07.md](../validation/PUBLIC_PILOT_READINESS_2026-10-07.md). Earlier September evidence remains historical and must not be substituted for checks on the final candidate.

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
- [x] All engine-forensic review workflows passed on the current exact-head candidate.
- [x] Local owner app-lock setup, fail-closed lock screen, unlock, PIN change and PIN removal are covered by API/unit and browser acceptance tests.
- [x] Real vector benchmark executed with the pinned multilingual embedding model and FAISS; results remain synthetic acceptance evidence, not a public relevance claim.
- [x] README updated to the current People-team product positioning.
- [x] Commercial-use policy documented and a request template added.
- [x] Pilot feedback template added.
- [x] Current working-tree evidence and release boundaries recorded in [PUBLIC_PILOT_READINESS_2026-09-15.md](../validation/PUBLIC_PILOT_READINESS_2026-09-15.md).

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
- [ ] Verify the linked PR #18 is merged after all checks pass; its live GitHub state is the merge record.
- [x] Configure and read back the documented `main` branch protection, including required passing checks and administrator enforcement.
- [ ] Capture pilot outcomes and recurring friction using the pilot feedback template.
- [ ] Decide whether Windows code signing and macOS notarization are required before broad public distribution.
- [ ] Choose a version/tag and final public release notes when publication is authorized.
- [ ] Review release archives, checksums and third-party/model notices before publishing binaries.
- [ ] Explicitly authorize creation of a GitHub Release / public binary distribution.
- [ ] Perform organization-specific prospective predictive validation before making stronger intended-use predictive claims.
- [ ] Complete approved relevance/accuracy holdouts, load/concurrency, isolation, retention and independent review before making public semantic-search, NLP or predictive claims.

## Distribution limitations

CI artifacts are build evidence until promoted to a versioned release. Checksums establish byte integrity, not publisher identity. No Windows signing, macOS notarization or publisher reputation claim is currently made. Do not tell users to disable system-wide security protections. Multi-user hosting, enterprise identity, organization-specific retention policies and real employee-data pilots remain outside the current public-beta engineering validation.

## Verification summary

PR #6 (`Prepare public beta distribution and close dependency security gaps`) merged a historical verified candidate after all eight required workflows passed. The final browser acceptance rerun passed the complete desktop/mobile journeys and its action/process coverage ledger. No privacy, evidence, security or analytical guardrail was weakened to obtain the passing result.

For the current source snapshot, use [PUBLIC_PILOT_READINESS_2026-10-07.md](../validation/PUBLIC_PILOT_READINESS_2026-10-07.md). Remote CI, merge, release publication and independent pilot acceptance remain separate gates.

See also:

- `docs/product/PEOPLEOS_10_10_PRODUCT_PLAN.md`
- `COMMERCIAL_USE.md`
- `SECURITY.md`
- `docs/PUBLIC_BETA_GUIDE.md`
- PR #6 workflow evidence
