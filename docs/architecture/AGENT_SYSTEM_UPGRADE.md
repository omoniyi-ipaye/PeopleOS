# PeopleOS Agent System Upgrade

Status: In progress
Branch: `upgrade/agent-system-foundation`

## Current architecture assessment

PeopleOS is currently a strong local-first People Analytics platform with deterministic analytical engines plus an LLM-backed Strategic Advisor. It is not yet an agentic system in the architectural sense because the current advisor follows a direct pattern: analytics context -> prompt -> Ollama -> parsed response.

The target is to evolve PeopleOS into a governed People Intelligence Agent System without replacing the deterministic analytics foundation.

## AS-IS

```text
User
  |
  v
Next.js UI
  |
  v
FastAPI routes
  |
  +--> Analytics / ML / Compensation / Fairness / Survival / Scenario / Sentiment / Structural engines
  |
  +--> Strategic Advisor
           |
           +--> Build analytics context
           +--> Prompt Ollama
           +--> Parse free text
```

## Key findings

### P0 - LLM safety enforcement can be bypassed
`LLMClient` contains prohibited-content validation, but the advisor route calls `llm_client.generate()` directly. That means the safety policy is not an enforceable system boundary.

Target: all externally returned AI advice must pass a single centrally enforced safety boundary.

### P0 - Local-first network boundary is too permissive by default
The documented backend command and `api.main` launch path bind to `0.0.0.0`. For a privacy-sensitive local-first People system, the default should be loopback-only (`127.0.0.1`) with explicit opt-in for LAN/server deployment.

### P0 - Frontend dependency security baseline needs modernization
Next.js is on the 14.x line. Upgrade should be handled as a security and compatibility workstream rather than a cosmetic dependency refresh.

### P1 - Global AppState will constrain multi-workspace and agent evolution
Current state is held in one process-wide singleton containing datasets, engines, trained models, risk scores, surveys and LLM state.

Target model:

```text
Workspace
  +-- Dataset version
  +-- Analysis run
  +-- Model version
  +-- Agent session
  +-- Evidence bundle
```

### P1 - Model lifecycle is coupled to data loading
Loading a dataset can initialize and train predictive models immediately. Training should become an explicit lifecycle with dataset versioning, evaluation gates and model activation.

### P1 - Missing governed agent orchestration
Target agent architecture:

```text
User question
  |
  v
Intent / task understanding
  |
  v
Agent orchestrator
  |
  +--> governed analytics tools
  +--> governed prediction tools
  +--> governed scenario tools
  +--> governed policy checks
  |
  v
Evidence bundle
  |
  v
Policy / risk gate
  |
  v
LLM synthesis
  |
  v
Verification
  |
  v
Answer + evidence + confidence
```

## Architectural principles

1. Keep deterministic rules, calculations, permissions and state transitions outside the LLM.
2. Use the LLM for interpretation, planning and synthesis where probabilistic reasoning adds value.
3. Do not give the LLM direct write authority over sensitive HR state.
4. Every material conclusion should be traceable to evidence, confidence and model/tool provenance.
5. Keep policy and safety controls enforceable outside prompt text.
6. Add agent complexity only when justified; start with one orchestrator and typed tools rather than many agents.
7. Keep local-first as the default deployment posture.

## Target first agent

### People Intelligence Agent

Initial scope: investigate workforce questions using existing engines and produce evidence-backed answers.

```text
Question
  -> intent classification
  -> evidence requirements
  -> tool selection
  -> analytics execution
  -> evidence aggregation
  -> contradiction/confidence check
  -> policy check
  -> synthesis
  -> verification
  -> answer
```

Initial tool families:
- workforce analytics
- retention / survival
- compensation
- fairness
- structural / span of control
- sentiment / employee experience
- scenario analysis

## Transition backlog

| Priority | Change | Status |
|---|---|---|
| P0 | Centralize LLM output safety enforcement | In progress |
| P0 | Default backend bind to loopback | In progress |
| P0 | Upgrade Next.js security baseline | Planned |
| P0 | Introduce explicit authentication / authorization design for non-local deployments | Planned |
| P1 | Introduce workspace/session architecture | Planned |
| P1 | Define typed PeopleOS tool contracts | Planned |
| P1 | Create canonical evidence/result schema | Planned |
| P1 | Separate model training from request lifecycle | Planned |
| P1 | Introduce agent orchestrator | Planned |
| P1 | Add policy/authorization gate | Planned |
| P2 | Add persistent agent/session state | Planned |
| P2 | Add verification/evaluation layer | Planned |
| P2 | Add provenance/confidence to generated conclusions | Planned |
| P2 | Add observability and audit events | Planned |
| P3 | Add bounded autonomous workflows | Planned |
| P3 | Add MCP interface if external agents should consume PeopleOS | Planned |
| P3 | Consider LangGraph only when workflow complexity warrants it | Planned |

## Build readiness

Current verdict: **NOT BUILD READY as an agentic system**.

PeopleOS is, however, a strong deterministic foundation for the target architecture. The analytics engines should largely be preserved and exposed as governed tools behind a new orchestration, evidence, policy and verification layer.

## Immediate implementation slice

The first slice is intentionally small and low-risk:

1. enforce LLM safety at the shared client boundary;
2. update advisor paths so they cannot bypass that boundary;
3. change local development/server defaults to loopback-only;
4. add regression tests for the safety boundary;
5. then start the canonical evidence/tool-contract layer.
