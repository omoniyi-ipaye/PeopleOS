# Recommended protection for `main`

The repository `main` branch is currently unprotected. Before broad public contribution or release, configure a GitHub branch protection rule or repository ruleset for `main` with the following controls.

## Required controls

- Require a pull request before merging.
- Block direct pushes to `main` except for an explicitly designated emergency/maintainer bypass if desired.
- Require branches to be up to date before merge where practical.
- Require conversation resolution before merge.
- Require the validated CI checks below before merge.
- Do not allow force pushes.
- Do not allow branch deletion.

## Required status checks

Use the repository's current check names:

1. `Agent Foundation`
2. `Frontend Modernization`
3. `E2E User Journey`
4. `Analytics Validation`
5. `People Team Browser Acceptance`
6. `Local Desktop Build`
7. `Release Security`
8. `Local Ollama Acceptance`

If GitHub exposes job-level rather than workflow-level check names when configuring the rule, select the corresponding required jobs for each workflow and verify the rule against a test pull request before relying on it.

## Maintainer policy

For public-beta development, prefer:

- one integrating pull request per product/release tranche;
- exact-head CI evidence before merge;
- no weakening of correctness, privacy or safety assertions to obtain a green build;
- no real employee data in issues, pull requests, fixtures, logs or screenshots;
- release/tag creation only after explicit owner authorization.

## Current limitation

This file documents the intended repository rule. It does **not** itself enable GitHub branch protection. The repository owner must configure the rule in GitHub repository settings because the connected GitHub App used for the preparation pass does not have repository-administration permission.
