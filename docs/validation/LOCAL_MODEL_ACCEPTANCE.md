# Local model and pay integrity correction pass

Baseline: merged PR #3, `e09a6c12f2270256880c6a08eed0081fb1d6a8ba`. Successor branch: `fix/local-model-acceptance`. This is an integration checkpoint; exact published candidate and workflow results belong in its PR evidence. No deployment is included.

## Reproduced problems and corrections

- A candidate model could activate without rechecking its persisted evaluation. Activation now requires matching registered/runtime metrics, a passing current evaluation policy and compatible training snapshot. Rejections preserve the existing active model.
- Missing pay units were treated as annual comparable amounts. Normal upload now supports explicit annual/shared-currency declarations, preserves these in canonical artifacts and rejects conflicts. Undeclared amounts remain in source history but are masked from active analysis and model inputs; salary forecasts also require declared historical units. Workforce counts remain available.
- Legitimate whole-workforce paraphrases were rejected. Complete-question recognition now admits supported mean/pay/headcount questions while retaining named, temporal, negated and unsupported-statistic restrictions.
- Ollama server reachability was mistaken for configured-model availability. Exact installed model selection and configured transport timeout are checked. The acceptance harness uses independent numeric/source assertions and never counts deterministic fallback as actual inference.
- The original embedding model failed both cross-language synthetic retrieval cases. A pinned multilingual model passed the unchanged cases. Returned nested search metadata could also mutate later provenance; results now isolate that metadata.
- The older core-only E2E workflow failed after merge on a stale selector. It now uses accessible controls and checks actual API counts, rendered evidence and dataset identity. It also runs before merge on pull requests.

## Actual model evidence

`real-embeddings-before.json`: original model passed 10 of 13 checks; two cross-language cases and metadata isolation failed. `real-embeddings-after.json`: actual sentence-transformers/FAISS inference passed 13 of 13, repeated offline with repository NumPy/scikit-learn/PyYAML pins. Default model revision is `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. This small synthetic acceptance corpus is not a held-out population benchmark; no final holdout was tuned. Scores are ranking distances, not truth probabilities or validated no-match decisions. The multilingual model is a larger download.

`local-llm-blocked.json`: actual LLM inference was not executed. The Ollama package/runtime was unavailable and the official runtime download returned a cancelled network approval. The four-case executable harness is prepared, but its transport/oracle unit tests do not establish live LLM correctness.

## Verification checkpoint

Initial full Python integration: 805 passed; one review-hash freshness failure subsequently reconciled after source review (all three audit checks then passed). New pay API/restore/forecast cases and independent agent-language known answers passed. Renderer checks: 41 passed. Live API/restart checks: 96 passed. Stress: 34 checks passed on 20,000 fictional employees. Local production build and TypeScript passed. Local browser execution was blocked before any journey by missing Chromium and official CDN download timeouts/502; no local browser screenshots or pass are claimed. Candidate-wide browser, frontend, analytics/stress and all three desktop platforms must be verified on the published head; pending checks are not passes.

## Remaining intended-use limits

These corrections support trustworthy descriptive analysis under explicit input contracts. They do not establish prospective attrition validity, real-LLM grounding, organization-specific access/retention requirements or real People-team pilot outcomes. Predictive activation remains gated; failed public predictive evaluation is not tuned away. No production launch or promise of error-free outputs follows from this pass.
