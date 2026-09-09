# PeopleOS Public Beta — Draft Release Notes

> Draft only. This document does not publish a GitHub Release or authorize binary distribution.

Proposed version: `v0.9.0-beta.1`

## Your local People Analyst

PeopleOS is a local-first People Intelligence system for People Operations, People Analytics and HR teams. It combines deterministic workforce analytics with governed AI so users can ask questions in normal People language, drill into cohorts, and inspect the evidence behind every answer.

**PeopleOS analyses. People decide.**

## What is included

- `.xlsx`, `.csv` and `.json` workforce import
- fictional sample-data onboarding
- workforce composition and department health
- recorded attrition and retention analysis
- tenure/cohort retention analysis
- compensation and pay analysis with explicit unit semantics
- employee-experience analysis when measured data is present
- quality-of-hire analysis
- fairness analysis with small-group suppression
- scenario planning with non-consequential boundaries
- governed Ask PeopleOS investigations
- privacy-bounded aggregate drill-down
- stacked cohort filters across supported dimensions
- grouped summaries, rates, correlations and crosstabs
- evidence, coverage, provenance and raw verified-response inspection
- packaged desktop lifecycle controls
- optional local Ollama synthesis

## Trust and safety model

PeopleOS keeps deterministic analytics responsible for calculations and state. The AI layer interprets governed evidence rather than inventing the underlying metrics.

The product fails closed when evidence is unavailable or insufficient. It does not convert missing evidence into zero, observational correlation into causation, or model score bands into validated future-event probabilities without the required validation.

Normal analytical workflows remain aggregate and privacy-bounded. PeopleOS does not provide employee-ranking workflows for termination, demotion or other consequential employment actions.

## Deep drill-down

People teams can progressively explore available data across dimensions such as department, location, job level, job title, gender, tenure, age, salary and rating, subject to privacy/support thresholds.

Example:

`Engineering → Madrid → L3 → tenure < 2 years`

Conversationally:

`Average salary by job level for Engineering employees in Madrid with under 4 years tenure.`

## Validation

The verified public-beta engineering candidate passed:

- Agent Foundation
- Frontend Modernization
- E2E User Journey
- Analytics Validation
- People Team Browser Acceptance (desktop and mobile)
- Local Desktop Build
- Release Security
- Local Ollama Acceptance

The candidate was merged to `main` through PR #6.

## Platforms

The build pipeline validates Windows x64, macOS ARM64 and Linux x64 package/smoke/restart/archive paths.

Code signing and macOS notarization are not currently claimed.

## License

PeopleOS is **source-available**, not OSI open source.

Non-commercial evaluation, learning, research and contribution are welcome subject to the repository license. **Commercial use requires express written permission from Omoniyi Ipaye.** See `COMMERCIAL_USE.md` and the repository Commercial use request issue template.

## Pilot expectations

This beta is intended for controlled evaluation with fictional/test data first. It is not a production certification. Independent security, privacy, legal, data-governance and intended-use review remains the responsibility of any adopting organization.

Pilot users should use the repository Pilot feedback issue template and must not attach real employee or other sensitive personal data.

## Not yet claimed

- signed/notarized public installers
- multi-user hosted enterprise deployment
- organization-specific identity/retention configuration
- prospective organization-specific predictive validity
- production certification
