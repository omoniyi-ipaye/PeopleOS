# Synthetic workforce stress validation

This gate creates a reproducible fictional organization and calculates its expected
answers independently of PeopleOS. It is intended to expose arithmetic, denominator,
privacy, persistence and load failures. It does not prove that synthetic behavior
represents any employer or validate future employee-level predictions.

## Current run

- Seed: `20260908`
- Fictional employee records: 20,000
- Known-answer and stress assertions: 34 passed, 0 failed
- Generation: 0.20 seconds
- Direct analytics: 0.44 seconds
- Real API upload and activation: 6.45 seconds
- Concurrent summary reads: 32/32 successful with identical bodies and dataset IDs
- Concurrent-read p95: 4.86 seconds on the local validation runner
- Total harness duration: 30.64 seconds

The machine-readable evidence is `docs/validation/synthetic-stress-results.json`.
Timings describe this one runner and are not a production capacity guarantee.

## Populations and edge cases

The generator includes five departments, four locations, four gender values, multiple
levels and hire sources, ages 18–69, tenure from zero to 25 years, manager hierarchies,
salary and rating distributions, promotions, interview scores and assessment scores.

Deliberate challenges include:

- 1,177 missing salaries, 1,053 missing ratings and 870 unknown employment outcomes;
- a seven-person department below the reporting minimum of 10;
- independently calculated headcount, payroll, attrition share and eNPS;
- row permutation and an unused HRIS column;
- a duplicate employee identity;
- finite inputs whose aggregate payroll cannot be represented;
- a shifted distribution whose observed attrition share changes from 33.51% to 56.59%;
- artifact write/read identity preservation;
- 64 simultaneous workspace creations across eight store instances; and
- a 20,000-row upload followed by 32 concurrent API reads.

## Acceptance contract

The gate fails if populations, denominators, payroll or eNPS disagree with the independent
calculation; any analytic result becomes non-finite; reordered rows change an answer;
extreme payroll does not fail closed; persistence changes identities; concurrent registry
writes are lost; API responses disagree; dataset provenance changes between reads; upload
exceeds 120 seconds; or concurrent-read p95 exceeds 10 seconds.

The generator is seeded and the fixture hash is recorded. It is not a fixed final holdout
and must not be used to tune operational ML thresholds.

## What remains outside this evidence

Synthetic data cannot establish survey construct validity, causal effects, organizational
fairness, real text/LLM groundedness, future prediction quality, multi-process/database
capacity, enterprise identity behavior or real-user usability. Those require separate
representative data, infrastructure and human validation.
