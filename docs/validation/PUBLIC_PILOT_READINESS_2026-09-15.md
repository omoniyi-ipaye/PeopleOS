# PeopleOS broader public-pilot readiness

Review date: 2026-09-15. This is a current engineering handoff for the local
working tree. It is not a published release, a production certification or a
claim of organization-specific predictive validity.

## Current source and runtime

- Branch: `quality/cycle-017-ml-engine-forensic`
- Source base: `83bec3a68e857b784e984fe4b96e03584e75cb50`
- The pilot-hardening changes are committed on the PR branch. Keep the local
  `HEAD` and PR head aligned before relying on remote-CI evidence.
- The local UI is running at `http://127.0.0.1:3001` and the API at
  `http://127.0.0.1:8000`.
- The active fictional sample is loaded and integrity-verified: 800 source
  rows, 662 active employees, dataset version 6, annual USD pay semantics.
- Local Ollama is available with `qwen3.8:latest`. It is optional and remains
  governed by deterministic evidence, server-side citations and claim limits.
- Semantic search is intentionally not prepared in the current process. The
  optional embedding backend is available, but preparation remains an explicit
  user action; no relevance claim is implied by availability.

## What was completed for the controlled pilot

- All 20 engine modules were source-reviewed, with unavailable and exploratory
  contracts kept explicit instead of being presented as validated predictions,
  causal conclusions or employee rankings.
- The analytics layer completes governed deterministic calculations before the
  local model can explain them. Unsupported, insufficient or stale evidence
  fails closed.
- Ask PeopleOS uses a chat-first surface with a left-side answer drawer,
  plain People/HR language, grouped charts and tables, evidence inspection and
  aggregate drill-down links.
- Insight pages hand off to prefilled, runnable advisor questions rather than
  dead-end navigation.
- Scenario planning supports saved situations, comparison and AI-assisted
  explanation while preserving assumption-based, non-consequential framing.
- Onboarding is focused and sidebar-free. Import uses preview, column mapping,
  integrity review and explicit activation; optional local AI receives schema
  metadata only and never decides the final mapping.
- Owner app lock uses a six-digit PIN with fail-closed lock, unlock, change and
  removal flows.
- The shell now resets the main scroll position on route changes. This fixed a
  live visual defect where navigation could preserve the previous page’s scroll
  offset and hide the next page heading beneath the header.
- Lint warnings were removed without weakening the reviewed UI governance
  rules.

## Verification evidence

| Area | Current result | Evidence boundary |
| --- | --- | --- |
| Python core and API | `1,229 passed`, 38 warnings, 6 subtests | Local source checkout only; warnings are retained for review. |
| Targeted pilot/security gates | `65 passed`, 11 warnings | Focused checks; not a replacement for the full suite. |
| Frontend design governance | Pass; zero scoped exceptions | Contract/static governance, not visual certification by itself. |
| Frontend lint | Pass; zero warnings and zero errors | Source lint only. |
| Analytics renderer tests | `52 passed` | Component/renderer contracts with controlled fixtures. |
| Production web build | Pass; 17 routes generated | Build evidence, not release publication. |
| Static desktop build | Pass; 17 routes generated | Static bundle evidence, not cross-platform release certification. |
| Browser acceptance | `28 passed`: 14 desktop and 14 mobile | Real Playwright journeys cover setup, import, advisor, drill-down, scenarios, lock and recovery boundaries. |
| Live browser observation | Insights and specialist pages exercised with no console errors or warnings observed | Manual local observation; not independent fresh-user acceptance. |
| Local LLM | Ollama transport and qwen runtime ready | Local synthesis only; no claim that generated language is organizational truth. |
| Vector benchmark | Sentence Transformers + FAISS path exercised on synthetic checks | Geometry/order evidence only; no real-corpus Recall@k, nDCG or multilingual relevance claim. |
| macOS packaging | Apple Silicon package built and smoke-probed successfully | Local packaging evidence; no signed/notarized archive was published. |

## What this supports

PeopleOS is ready for a controlled, local, single-owner pilot using fictional or
appropriately de-identified data. The pilot can evaluate descriptive workforce
analytics, governed aggregate investigations, evidence/provenance UX, scenario
assumptions and optional local-AI explanations.

The evidence does not support a hosted multi-user launch, production capacity
claim, organization-specific predictive accuracy claim, causal decision support,
employee risk ranking or unrestricted commercial use. See the [engine audit](ENGINE_RENDERER_AUDIT.md)
and [public beta guide](../PUBLIC_BETA_GUIDE.md) for the detailed limits.

## Gates still requiring owner or external action

These are release controls, not hidden application defects:

1. Review the complete pushed PR diff and confirm the final exact-head checks.
2. Retain the remote-CI artifacts for the exact commit intended for the pilot.
3. Review/merge the pilot PR and configure the documented `main` branch rules.
4. Conduct an independent fresh-user installation and usability walkthrough on
   each target operating system.
5. Decide on Windows signing, macOS notarization, version/tag and GitHub
   Release/binary publication; review checksums, third-party notices and model
   notices before distribution.
6. Capture pilot feedback with fictional/de-identified data and complete the
   adopting organization’s security, privacy, legal and data-governance review.
7. Before stronger predictive, semantic-search or NLP claims, run approved
   organization-specific holdouts, calibration/relevance tests, subgroup
   review, load/concurrency, isolation and retention validation.

Until those gates are completed, describe this as a controlled public-beta
engineering candidate, not as a generally available production system.
