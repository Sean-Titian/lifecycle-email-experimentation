# Reports

This directory separates executable synthetic output from non-identifying source benchmarks.

## `source-benchmark.json`

`source-benchmark.json` is a deliberately minimized handoff artifact from a private runtime audit. It
contains only the rounded or resume-aligned aggregates needed to support the public case study. Exact
operational counts, dates, group structures, and the full audit remain private. The artifact contains
no rows, identifiers, text fields, addresses, or source file paths.

The benchmark is included so public claims are machine-readable and auditable against the README. It is **not** an input dataset and cannot reproduce the private campaign from scratch.

## Generated reports

Reports produced from the public synthetic workflow may be regenerated locally. Their numbers demonstrate code behavior only. They should be labeled `synthetic` and must not overwrite the source benchmark.

## `prospective-synthetic-benchmark.json`

This tracked, deterministic aggregate is the executable acceptance test for the seven-cell
prospective design. Schema version 1.2.0 reports block-standardized primary and
pre-period negative-control risk differences with conservative Neyman variance, the six
pre-specified funding comparisons, pooled Newcombe-style intention-to-treat unsubscribe and
complaint bounds that are explicitly not block-adjusted, exact block allocation,
sample-ratio and contamination checks, active in-window delivery coverage, follow-up and
known-event latency completeness, and the resulting decision. It includes no synthetic rows
or identifiers and makes no empirical claim about a real campaign or the completeness of an
external sparse event feed.

CI rebuilds the report from packaged code and requires byte-for-byte equality, so changing
the estimand, seed, configuration, or serialization cannot silently leave stale numbers in
the repository.

## `prospective-synthetic-operating-characteristics.json`

This tracked aggregate-only artifact runs 20,000 synthetic replications under each of five
frozen scenarios. It reports primary family-wise behavior, nominal 95% effect-interval
coverage, candidate Holm superiority, negative-control behavior, guardrail
non-inferiority decisions, and the complete policy frequency. Every probability is
reported with a two-sided 99% Wilson Monte Carlo interval.

The benchmark preserves a shared holdout within each replication and regression-tests the
vectorized count kernel against the row-level production APIs. It is conditional on the
declared Bernoulli data-generating process, exact allocation, and operational quality
gates; it does not estimate a real campaign effect or validate an external event feed.

All frozen calibration gates pass, but `design_readiness` remains `continue_testing`.
Calibration verifies implementation behavior; it does not authorize rollout. CI rebuilds
this report from both source and the packaged wheel and requires byte-for-byte equality.

## Publication checks

Before committing any new report:

1. confirm its provenance is `synthetic` or `aggregate_only`;
2. scan keys and values for identifiers, text, paths, and secrets;
3. reconcile headline metrics with the analysis contract;
4. retain null comparisons and corrected p-values;
5. verify that timing and censoring limitations remain visible;
6. keep operating-characteristics outputs aggregate-only, with no replication rows,
   per-replication seeds, identifiers, or paths;
7. distinguish effect confidence intervals from Monte Carlo intervals around estimated
   operating-characteristic rates.
