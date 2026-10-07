# Lifecycle Email Experimentation

[![CI](https://github.com/Sean-Titian/lifecycle-email-experimentation/actions/workflows/ci.yml/badge.svg)](https://github.com/Sean-Titian/lifecycle-email-experimentation/actions/workflows/ci.yml)

A product experimentation case study for choosing lifecycle-email content and cadence without mistaking opens for business impact.

The decision is deliberately practical: **which message should a lifecycle team test next, is there evidence to prefer daily or twice-weekly delivery, and what customer-risk guardrails must ship with that decision?** The repository turns that question into an analysis contract, a public-safe synthetic pipeline, multiplicity-aware inference, and a temporally valid funnel.

## Executive readout

| Decision | Evidence | Recommendation |
| --- | --- | --- |
| Which content deserves the next test? | **Template D** recorded an open rate of about **26.0%**, versus **18.3%** for the runner-up. | Advance Template D to a clean confirmatory experiment; do not infer funding impact from opens alone. |
| What does the funding comparison show? | Three of 24 retrospective aggregate-control comparisons had raw p < 0.05; only **one local difference** survived correction: **+0.311 percentage points** (95% CI +0.122 to +0.500 pp; adjusted p ≈ 0.011). | Treat this association as a hypothesis, not an experimental estimate, because controls lack aligned user-level timing. |
| Is daily better than twice weekly? | No direct daily-versus-weekly funding comparison was significant even before correction; none survived correction for either endpoint. | Do not select cadence from this dataset. Power a prospective cadence test instead. |
| What is the customer-risk guardrail? | Unique-user unsubscribe risk was about **2.28%** among actual recipients. | Make unique-recipient unsubscribe risk a primary guardrail, alongside complaints and delivery failures. |

The business takeaway is deliberately narrow: content engagement has a credible frontrunner, the downstream funding associations are mostly null or uncertain, and cadence remains an open product question.

### Prospective design, now executable

The next experiment is also implemented as a deterministic, public-safe rehearsal: three
content levels crossed with two cadences, plus a concurrent holdout; stratified block
randomization; 14-day intention-to-treat funding; SRM and contamination gates; and
simultaneous unsubscribe/complaint non-inferiority bounds. The generated benchmark is a
test of the decision system, not evidence that lifecycle email works. Its launch rule
defaults to `continue_testing` unless the pre-specified candidate has corrected primary
evidence, both candidate guardrails pass, and every integrity gate passes.

| Canonical synthetic check | Result | Decision meaning |
| --- | --- | --- |
| Design and population | 8,400 randomized and analyzed by ITT; 1,200 per cell | All 10 assignment, exact-block, SRM, concurrent-holdout, timing, known-event latency, delivery-coverage, contamination, negative-control, and population-integrity gates pass. |
| Pre-specified candidate funding | +1.50 pp versus holdout; block-adjusted nominal 95% CI -0.10 to +3.10 pp; Holm-adjusted p = 0.395 | Superiority is not established. |
| Analytic planning reference | 2.7839 pp at a declared 4% baseline, using baseline variance and Bonferroni alpha/6 for a target 80% power | This is a first-order approximation, not an empirically attained 80%-power MDE. |
| Repeated-simulation readiness | Candidate Holm superiority is 65.8% (99% Monte Carlo CI 64.9% to 66.6%) across 20,000 planning-reference replications. Even with an eight-point funding effect and safe true guardrail risks, the complete decision rule passes 0.10% (99% MC CI 0.057% to 0.176%). | Method calibration passes, but rare-event guardrail precision keeps the design at `continue_testing`. |
| Customer-risk precision | Unsubscribe UCB 1.21 pp vs 0.50 pp margin; complaint UCB 0.65 pp vs 0.30 pp margin | Non-inferiority is inconclusive, so the candidate stays in testing. |

These are synthetic outputs from a
[single deterministic rehearsal](reports/prospective-synthetic-benchmark.json) and a
[five-scenario operating-characteristics benchmark](reports/prospective-synthetic-operating-characteristics.json).
The latter runs 20,000 replications per scenario and gives every reported rate a two-sided
99% Wilson Monte Carlo interval. Neither artifact is a source-campaign estimate, and the
largest observed cell is not promoted after the fact.

### Resume metric crosswalk

An earlier resume version compressed this case to roughly 480K users, 24 cohorts,
10 message templates, and one
funding-rate difference that remained after correction (+0.311 pp; adjusted p = 0.011). Those values
reconcile to the private audit. This repository supplies the qualification that cannot fit in one
resume bullet: it is a retrospective aggregate-control snapshot difference, not causal uplift, and
the unmatched follow-up windows prevent a rollout claim.

## What this project demonstrates

- **Experimentation:** absolute and relative effects, 95% confidence intervals, two-sided tests, Holm/BH multiplicity control, analytic planning approximations, Monte Carlo operating characteristics with uncertainty, and placebo checks.
- **Prospective design:** stratified factorial randomization, concurrent holdout, sample-ratio
  checks, aligned data freeze, and intention-to-treat non-inferiority guardrails.
- **Product analytics:** an explicit decision memo, outcome hierarchy, capacity for null results, and customer-harm guardrails.
- **Data quality:** one-to-one join validation, user-level deduplication, exposure-window checks, and event-table aggregation before joins.
- **Behavioral funnels:** event-time ordering instead of counting users who completed steps at any point in their history.
- **Reproducibility:** standalone Python code, synthetic data, automated tests, linting, and machine-readable aggregate benchmarks.

## A funnel that respects time

The original population-state funnel counted historical link and funding states, including activity that occurred before treatment. This implementation defines eligibility first and then requires each event to occur after the preceding step.

```text
~226K eligible actual recipients
    └── ~119K opened on/after campaign start
            └── ~2.6K linked at least one day after first open
                    └── ~2.2K funded after that link
```

Open timestamps have date-level precision, so the one-day rule is intentionally conservative. The repository reports this strict sequence rather than presenting a non-temporal funnel as user behavior.

## Repository map

```text
src/email_experiment/                 analysis and synthetic-data package
tests/                                unit and invariant tests
data/                                 public data contract; raw sources excluded
docs/analysis-contract.md             estimands, units, windows, and claim rules
docs/experiment-design.md             prospective decision-grade redesign
docs/evidence-card.md                 intended use, evidence tiers, and limitations
reports/source-benchmark.json         aggregate-only source benchmark
reports/prospective-synthetic-benchmark.json
                                      reproducible aggregate design rehearsal
reports/prospective-synthetic-operating-characteristics.json
                                      five-scenario aggregate design diagnostic
```

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-benchmark.txt
python -m pip install -e ".[dev]"
python -m email_experiment generate-synthetic --output-dir data/synthetic
python -m email_experiment run-synthetic --output-dir reports/synthetic
python -m email_experiment run-prospective-synthetic --output reports/prospective-synthetic-benchmark.json
python -m email_experiment run-prospective-operating-characteristics --output reports/prospective-synthetic-operating-characteristics.json
pytest
ruff check .
```

The first command writes inspectable synthetic rows; the second runs the full analysis in memory and
writes only `reports/synthetic/aggregate.json`. Synthetic outputs validate code paths and statistical
behavior; they are not presented as evidence about the source campaign. See
[data/README.md](data/README.md) for the privacy boundary and schema expectations.

As a power sanity check, the default deterministic run finds no Holm-significant primary or
pre-treatment negative-control comparison. Its simple per-experiment normal-approximation
reference for a target 80% power is about 2.34–2.41 percentage points, far above the
simulated 0.42-point effect. This is not a calibrated family-wise MDE. A large-looking
synthetic win is not engineered into the demo.

The two prospective commands write only canonical aggregate reports; participant, event,
and replication rows are never written. CI regenerates both reports from the built wheel
and requires exact byte equality. See
[the prospective synthetic study](docs/prospective-synthetic-study.md) for the estimands,
multiplicity families, calibration gates, decision gates, and limitations.

Package 0.5.0 adds repeated-simulation calibration without changing the single-run
estimand or its `continue_testing` result. The new count-level benchmark preserves the
shared holdout and declared blocks, uses deterministic SHA-256-keyed PCG64 streams, retains
every requested replication, and matches 12 fixed row-level production-API checks to within
`1e-12` with exact Boolean and decision parity. That parity set is a regression check, not
proof over every possible count state. The release separates method calibration from
design readiness: all frozen calibration gates pass, while rare-event non-inferiority
precision makes the current complete launch rule impractical. A real deployment still
needs independently verified source watermarks because a sparse event table cannot prove
that an entirely missing endpoint feed is complete.

## Evidence boundary

Two evidence layers are kept separate:

1. **Public synthetic workflows** — safe to run, inspect, and modify. They demonstrate the retrospective analysis mechanics and prospective decision invariants.
2. **Private-source aggregate benchmark** — reproduced by running the same reasoning against a restricted event extract, then exporting only a minimized set of rounded or resume-aligned values to [`reports/source-benchmark.json`](reports/source-benchmark.json).

No raw event data, user identifiers, email addresses, message bodies, proprietary materials, or source-derived row-level samples are included. The implementation is an independent rewrite rather than a publication of source notebooks or templates.

## Why the source comparison stops here

The aggregate source benchmark contains useful signals, but it cannot support an experimental funding conclusion:

- control outcomes are aggregate snapshots without user-level timestamps, so treatment and control follow-up windows cannot be aligned;
- funding outcomes occurred after eligibility but before the first send and must be excluded from any post-exposure comparison;
- outcome collection ends before the full twice-weekly schedule and its follow-up window complete;
- all groups reuse the same pre-generated assignment-order block rather than independent production randomizations;
- raw opens have day rather than intra-day precision, preventing reliable same-day event ordering.

These limitations change the recommendation, not just the footnotes. The prospective design
in [docs/experiment-design.md](docs/experiment-design.md) is now executable on independent
synthetic data: user-level randomization, timestamped controls, a fixed outcome window,
pre-registered multiplicity families, and explicit unsubscribe/complaint guardrails. A
real authorized randomized experiment is still required before any rollout claim.

## Reproducibility and public safety

- Python 3.11+
- deterministic synthetic fixtures
- pinned canonical-report dependencies in `requirements-benchmark.txt`, while the test
  matrix still exercises the supported package range
- tests for joins, multiplicity, strict timezone handling, exact block allocation,
  block-standardized inference, SRM, delivery coverage, contamination, aligned freeze,
  report determinism, and guardrail denominators
- 20,000 replications across each of five frozen synthetic scenarios, with two-sided 99%
  Wilson Monte Carlo intervals and vectorized-to-production API parity checks
- Ruff linting and GitHub Actions CI
- MIT-licensed code; source data are not redistributed

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the data and dependency boundary.

## Status

This repository is a portfolio implementation of Product Analytics + Experimentation practice. It is suitable for learning, review, and method development; it is not a campaign recommendation or evidence of a general funding increase.
