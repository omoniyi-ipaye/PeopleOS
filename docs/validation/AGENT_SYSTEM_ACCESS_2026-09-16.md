# PeopleOS agent system-access verification

Review date: 2026-09-16. This note records the implemented access boundary; it
does not claim unrestricted data disclosure or production certification.

## Implemented boundary

- The People Intelligence Agent registry now exposes the seven existing
  governed tools plus runtime/system status, complete active-snapshot profile,
  passive aggregate summaries for initialized engine families, scenario-library
  reads and explicit unavailable descriptors for predictive, semantic-search,
  network, causal, clustering and forecasting surfaces.
- `/api/intelligence/capabilities` returns the executable tool catalog, mapped
  read API routes, runtime availability and the derived-analysis contract.
- The deterministic analytical plan still runs first. For broad questions, the
  local model can select at most four additional available read-only tool IDs.
  The server validates exact IDs against the registry and executes adapters
  in-process against the same workspace and verified dataset snapshot.
- The second selection pass receives only aggregate-safe signals from the first
  pass. It cannot emit code, URLs, shell commands, writes or arbitrary
  parameters.
- Large engine payloads such as survival curves are compacted only for the
  narrative prompt; the complete redacted result stays in the server response
  and evidence ledger. The model uses short request-local citation keys, which
  PeopleOS expands back to canonical evidence IDs after verification.

## Data boundary

PeopleOS scans the complete active snapshot in-process for schema, coverage and
engine calculations. The model context is limited to schema and redacted
aggregate evidence. Employee identifiers, row-level employee records and free
text are withheld. Predictive model activation, semantic index preparation,
causal analysis and employment actions remain explicit human-controlled flows.

## Verification

- The focused agent/system-access, launch-matrix, acceptance, derived-analysis
  and local-LLM transport tests pass.
- The access tests assert full-frame scanning, row redaction, runtime
  availability reporting, unavailable-engine exclusion and a real second-pass
  tool execution after deterministic analysis. They also assert bounded model
  context and canonical citation restoration.
- A live local run against the persisted sample snapshot completed the
  multi-engine broad question with Qwen (`grounded_llm`), selecting additional
  sentiment, succession, team-dynamics and survival reads; the browser showed
  the grounded explanation and 45 supporting checks with no console errors.
- Remaining readiness evidence must still distinguish local tests and live
  browser observation from remote CI, independent fresh-user acceptance and
  organization-specific privacy/security review.
