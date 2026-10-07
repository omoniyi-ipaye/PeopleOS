# Protection for `main`

Protection was enabled and read back from GitHub on 2026-10-07. It applies to administrators as well as other contributors. Pull requests, current passing checks, and resolved conversations are required; force pushes and branch deletion are disabled. The repository remains single-maintainer: zero additional approving reviewers are required, which is not a claim of independent review.

## Required controls

- Require a pull request before merging.
- Require these controls for administrator changes as well; no bypass is used for the pilot integration.
- Require branches to be up to date before merge.
- Require conversation resolution before merge.
- Require the validated CI checks below before merge.
- Do not allow force pushes.
- Do not allow branch deletion.

## Required status checks

Required GitHub Actions job contexts (integration ID 15368):

- `agent-tests`
- `build`
- `user-journey`
- `known_answers_and_benchmarks`
- `real_people_team_journeys`
- `local_cpu`
- `frontend-dependencies`
- `package-linux-x64`, `package-windows-x64`, `package-macos-arm64`
- `python-dependencies (requirements-desktop.txt)`
- `python-dependencies (requirements-core.txt)`
- `python-dependencies (requirements-advanced.txt)`
- `python-dependencies (requirements-validation.txt)`

The integrating release review additionally verifies every engine-forensic job;
several older workflows share the same `forensic` job name, so those ambiguous
names are not used as substitutes for the unambiguous core/analytics gates.

## Maintainer policy

For public-beta development, prefer:

- one integrating pull request per product/release tranche;
- exact-head CI evidence before merge;
- no weakening of correctness, privacy or safety assertions to obtain a green build;
- no real employee data in issues, pull requests, fixtures, logs or screenshots;
- release/tag creation only after explicit owner authorization.

## Verification boundary

The protection API returned the configured rule with administrator enforcement,
strict status checks and conversation resolution enabled. This is live settings
evidence; no deliberate prohibited-push test was performed.
