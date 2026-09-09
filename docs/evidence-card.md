# Evidence card

This is the experimentation equivalent of a model card: it states what evidence the repository can support, where that evidence came from, and where it must not be used.

## Intended use

- review a public implementation of lifecycle experimentation methods;
- demonstrate product-oriented interpretation of mostly null results;
- test data contracts, multiplicity corrections, temporal funnels, and guardrail denominators on synthetic data;
- provide a design brief for a future user-level randomized experiment.

## Out-of-scope use

- selecting a production email cadence from the aggregate benchmark;
- estimating general lifecycle-email return on investment;
- targeting individual users;
- treating recorded opens as a causal mechanism or as a reliable proxy for funding;
- using synthetic results as empirical evidence about a real campaign.

## Evidence tiers

| Tier | Artifact | Supports | Does not support |
| --- | --- | --- | --- |
| Public executable | Synthetic-data workflow and tests | Code correctness, estimator behavior, invariant checks | Real-world effect magnitude |
| Aggregate reproduction | `reports/source-benchmark.json` | Non-identifying descriptive rates and audit counts | Row-level verification by repository users |
| Prospective executable | `docs/prospective-synthetic-study.md` and `reports/prospective-synthetic-benchmark.json` | Reproducible randomization, fixed-window ITT inference, quality gates, and decision logic | A completed real-world experiment or campaign effect |

## Aggregate benchmark summary

- best recorded-open candidate, Template D: about 26.0%, versus about 18.3% for the runner-up;
- retrospective aggregate-control funding family: 3/24 raw positives and 1/24 after Holm/BH;
- surviving local snapshot difference: +0.311 pp (95% CI +0.122 to +0.500 pp; adjusted p ≈ 0.011);
- direct cadence family: no corrected advantage;
- preferred unsubscribe guardrail: about 2.28% unique-user risk among actual recipients;
- conservative new-link funnel: about 226K → 119K → 2.6K → 2.2K.

## Prospective synthetic benchmark summary

- seven concurrent cells with 8,400 fictional participants and complete 14-day ITT
  follow-up;
- every assignment, SRM, timing, latency, contamination, negative-control, and population
  gate passes;
- pre-specified candidate funding difference: +1.50 pp, nominal 95% CI -0.11 to
  +3.14 pp, Holm-adjusted p = 0.396;
- conservative 80% planning MDE: 2.78 pp from the declared 4% baseline, not the observed
  holdout result;
- candidate unsubscribe and complaint non-inferiority both remain inconclusive;
- decision: `continue_testing`, with no real-campaign effect claim.

## Retrospective source validity constraints

1. Controls are segment-level aggregates, shared across cadence comparisons, without aligned user-level timestamps.
2. Follow-up is incomplete for the longer schedule.
3. Pre-send outcomes exist and must be removed from post-treatment counts.
4. A reused assignment-order pattern was detected across groups, so the groups cannot be treated as independent production randomization draws.
5. Date-only open timestamps make same-day ordering ambiguous.
6. Recorded-open measurement is affected by mail-client behavior and should not be interpreted as attention with certainty.

## Privacy and provenance

The source event extract is private and not redistributed. It contained user-level behavioral data;
this repository exports no stable identifiers, addresses, message bodies, reason strings, or
row-level samples. Public code is independently written. The benchmark is limited to rounded or
resume-aligned aggregates and qualitative validity flags; the full audit remains private.

## Reliability controls

- strict one-to-one subject joins;
- pre-exposure outcome detection;
- fixed-window censoring checks;
- family-wise multiplicity correction;
- placebo and pre-treatment balance checks;
- unique-user guardrail denominators;
- temporal funnel ordering;
- deterministic synthetic tests and CI.

## Maintenance

Any change to an estimand, denominator, time window, or multiplicity family requires:

1. an update to `docs/analysis-contract.md`;
2. a new machine-readable report version;
3. tests covering the changed invariant;
4. a README review so the recommendation remains proportional to the evidence.
