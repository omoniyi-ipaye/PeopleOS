# Actual cloud-model acceptance and source-label hardening

Baseline: PR #4 merged at `2338a8467139c7e8e486cdce725968435f7d706b`. Branch: `test/ollama-cloud-acceptance`. User explicitly authorized disposable-key Ollama Cloud testing with fictional data. The product remains local-first; this script injects a test-only provider adapter into an isolated live API, without changing local runtime configuration or weakening permissions.

## Results, 8 September 2026

| Model | Actual cloud selection cases | Pre-model scope/data gates | Induced transport recovery | Result |
|---|---:|---:|---:|---|
| `gemma4:31b` | 10/10 | 5/5 | 1/1 | Passed final bounded acceptance |
| `gpt-oss:120b` | 10/10 | 5/5 | 1/1 | Passed final bounded acceptance |
| `kimi-k3` | No successful inference | Not exercised | Not exercised | HTTP 402: provider requires subscription or extra usage for this key |

The final Gemma and GPT-OSS runs each made ten real requests. Four exercise the selector directly with known evidence; six exercise real HTTP upload/analysis/investigation paths and actual engines. Five unsupported-scope or unconfirmed-pay requests correctly make no cloud calls. One deliberately unavailable transport checks safe fallback; the following healthy call checks recovery. These categories are not interchangeable.

Independent answers: 120 current source records, 80 active employees, 20 departed and 20 unknown outcomes; observed attrition is 20/100 = 20%. Annual EUR salaries produce a 75,000 active mean. Removing the first 20 active salary measurements produces 80,000, explicitly measured on 60/80 employees. Tests require typed ledger values, rendered literals, citations, returned provider identity and successful completed inference. No deterministic fallback counts as a model-selection pass.

Final measured request median/max: Gemma 11.580/25.059 seconds; GPT-OSS 11.176/27.433 seconds. This is a tiny test sample, not a latency service objective or a defensible ranking of the models. Temperature zero and 512 maximum generation tokens were used; GPT-OSS used its documented low thinking setting. No reasoning traces are persisted.

## What the real test exposed

1. Gemma initially returned valid selector objects wrapped in Markdown for all ten requests. The strict parser rejected them and retained correct deterministic evidence. The prompt now explicitly requires raw JSON with no code fences. The parser, schema, evidence references, source coverage and metric checks remain unchanged. A prompt-only replay fixed formatting before final acceptance.
2. Source department names could appear as an ambiguous sentence prefix, including an instruction-like synthetic name. This was text echo, not a model following an instruction or HTML execution. Backend and renderer now place typed measurements first and quote/escape the source department label. Original metadata and citations remain intact. The independent oracle requires any attack text to be confined to a correctly quoted original label; fabricated measurement/directive text still fails.
3. The initial multi-source harness question combined headcount and average span in wording unsupported by the planner. It incorrectly failed a valid GPT selector response. Both local and cloud harnesses now use a supported multi-source question while still requiring independent headcount 80, span 5 and both source tools. Initial failed reports are retained; this fixture repair is not presented as a model-quality improvement.

The `cloud/` JSON files retain actual completions, answers, typed evidence, source identifiers, source hashes and original-report digests; repeated engine metadata is omitted from these condensed exports. Baseline reports are diagnostic, not comparable final-model scores. Initial Kimi attempts produced four HTTP 402 responses before interruption; a fifth diagnostic request recorded the provider message and stopped. No successful Kimi inference is claimed or further attempts made.

## Reproduction and remaining gates

Run `python scripts/validate_cloud_llm.py --model gemma4:31b --prompt-key --output /tmp/cloud-result.json` or select `gpt-oss:120b`. A credential can alternatively be supplied in `OLLAMA_API_KEY`. The harness fixes the cloud destination, rejects credential-bearing redirects, caps requests at 20 per invocation, uses an isolated synthetic workspace/audit path and never stores the credential in results. It is opt-in and is not called by ordinary CI.

Local full regression: 816 Python tests and 42 renderer tests passed. Candidate-specific frontend, agent, analytics/live/stress, actual browser and Windows/macOS ARM64/Linux packaged checks must all pass before merge; final workflow links and exact candidate are recorded in the PR verification ledger.

This establishes bounded synthetic cloud behavior, not local/offline inference, organization-specific predictive validity, browser-to-cloud execution, a broad model benchmark or launch certification. Cloud model digest values are provider-reported; immutable weights are not independently established. No real employee data, deployment or employment decisions were involved. Local native-model testing and a People-team pilot remain open.

Official integration references: [Ollama Cloud API](https://docs.ollama.com/cloud), [thinking controls](https://docs.ollama.com/capabilities/thinking), and [current lack of cloud structured-output support](https://docs.ollama.com/capabilities/structured-outputs).
