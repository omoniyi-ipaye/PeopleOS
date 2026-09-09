# Contributing to PeopleOS

Thanks for helping improve PeopleOS.

PeopleOS is publicly developed as **source-available software**. Contributions are welcome, especially around deterministic analytics, People-team usability, privacy, evidence quality, accessibility, documentation and safe agent behavior.

## Before you contribute

- Read the repository `LICENSE` and `COMMERCIAL_USE.md`.
- Commercial use of PeopleOS still requires express written permission from Omoniyi Ipaye. Contributing code does not grant commercial-use permission.
- Never include real employee data, personal data, credentials, API keys, private company documents or unredacted production screenshots/logs in issues, pull requests, fixtures or tests.
- Use fictional or synthetic workforce data for reproduction and validation.
- Security vulnerabilities should be reported according to `SECURITY.md`, not through a public issue.

## Product principles

Changes should preserve these core boundaries:

1. deterministic engines own calculations and analytical state;
2. AI interprets governed evidence rather than inventing metrics;
3. missing or insufficient evidence fails closed;
4. aggregate analysis stays privacy-bounded;
5. no employee ranking for termination, demotion or other consequential employment actions;
6. causal language requires causal evidence;
7. predictive score bands are not future-event probabilities unless prospectively validated for the intended use;
8. simple People-team UX comes before exposing technical implementation detail.

## Good contributions

Examples include:

- a reproducible analytics bug with synthetic known-answer data;
- a clearer People-language explanation without weakening the evidence contract;
- accessibility fixes;
- safer import validation;
- stronger privacy/support-threshold tests;
- new deterministic aggregate analysis operations;
- browser tests for real People-team workflows;
- documentation improvements;
- performance or packaging improvements that preserve local-first behavior.

## How to contribute

1. Fork the repository.
2. Create a focused branch.
3. Make the change and add meaningful tests.
4. Run the relevant validation locally where practical.
5. Open a pull request against `main` and explain the user impact, evidence contract, tests and any limitations.

For meaningful UI changes, include screenshots using fictional data.

Do not weaken an existing safety, privacy or correctness assertion simply to make a test pass. If a contract has genuinely changed, explain why the new behavior is safer or more correct and update the acceptance evidence accordingly.

## Pilot feedback

If you are evaluating PeopleOS rather than changing code, use the **Pilot feedback** issue template. Do not attach real employee or other sensitive personal data.
