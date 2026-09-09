# Public beta release checklist

This is a preparation record, not a published release or a production certification.
Baseline: `ae986ef5be780790a798871d5a20fb6d2e95f343` (PR #5 merged).
Candidate branch: `release/public-beta-preparation`. Exact candidate and CI evidence
will be recorded in the preparation PR. Do not publish artifacts from mixed revisions.

## Intended release scope

Local, single-user descriptive People analytics and evidence-backed investigations.
Use fictional sample data first. Show source populations, exclusions, units and
limitations. Predictive outputs require their existing explicit evaluation/activation
controls; no prospective organizational predictive accuracy is claimed. Cloud acceptance
is test evidence only, not a shipped cloud-provider setting or offline validation.

## Required release gates

- [ ] Owner chooses distribution licence; current commercial restriction remains in force.
- [ ] All six existing candidate workflows pass, preserving all three desktop smoke/restart checks.
- [ ] New archive/checksum/provenance checks pass on Windows x64, macOS ARM64 and Linux x64.
- [ ] Native local Ollama acceptance passes with runtime/model identity and scope recorded,
      or local AI is explicitly an unvalidated optional preview in release notes.
- [ ] Dependency/security audit findings are triaged; no known release-blocking issue hidden.
- [ ] Published documentation matches actual UI, supported data and recovery capabilities.
- [ ] Fresh-user installation and dummy-data walkthrough completed on target machines;
      CI is supporting evidence, not a substitute for human usability feedback.
- [ ] Archive contents and third-party distribution notices reviewed. Downloaded model
      weights have their own terms and are not implicitly licensed by PeopleOS.
- [ ] Version/tag and release notes identify one verified commit and its checksums.
- [ ] Public publication authorized after the reviewable candidate and assets are ready.

## Baseline evidence

[PR #5](https://github.com/omoniyi-ipaye/PeopleOS/pull/5): six candidate workflows green;
24 production-browser journeys; 816 local Python tests and 42 renderer tests;
34 stress checks on 20,000 fictional employees, 96 live dummy checks and nine public
benchmark mechanics checks. These counts overlap and must not be added together.
Gemma 4 31B and GPT-OSS 120B each passed ten actual cloud selections, five pre-model
gates and one induced recovery case. Kimi K3 returned HTTP 402; native local inference
was not executed in that pass. Candidate-specific changes require fresh applicable checks.

## Licensing decision prepared for the owner

Recommended for the requested open-source launch: retain Apache License 2.0 terms and
copyright attribution, remove the appended commercial-use restriction, and state the
chosen terms consistently in README and release notes. Business use would be permitted.
The exact proposed edit is [PROPOSED_APACHE_LICENSE.patch](PROPOSED_APACHE_LICENSE.patch).
This document proposes that change; it does not grant new rights or change LICENSE.
Confirm ownership/contributor rights before relicensing contributions. If the restriction
is retained, describe the release as source-available rather than open source.

References: [current licence](../../LICENSE),
[Open Source Definition](https://opensource.org/osd),
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Distribution limitations

CI artifacts are temporary build evidence until promoted to a versioned release.
Checksums detect changed bytes; they do not establish publisher identity. No signing,
macOS notarization or Windows reputation claim is made. Do not tell users to disable
system-wide security protections. Local file copies are not a tested enterprise backup
service. Multi-user hosting, enterprise identity and real employee-data pilots are outside
this beta preparation pass.
