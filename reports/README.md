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
prospective design. It reports the six pre-specified funding comparisons, simultaneous
intention-to-treat unsubscribe and complaint guardrails, sample-ratio and contamination
checks, pre-period negative controls, follow-up and event-latency completeness, and the
resulting decision. It includes no synthetic rows or identifiers and makes no empirical
claim about a real campaign.

CI rebuilds the report from packaged code and requires byte-for-byte equality, so changing
the estimand, seed, configuration, or serialization cannot silently leave stale numbers in
the repository.

## Publication checks

Before committing any new report:

1. confirm its provenance is `synthetic` or `aggregate_only`;
2. scan keys and values for identifiers, text, paths, and secrets;
3. reconcile headline metrics with the analysis contract;
4. retain null comparisons and corrected p-values;
5. verify that timing and censoring limitations remain visible.
