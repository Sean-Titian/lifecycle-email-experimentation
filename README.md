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

### Resume metric crosswalk

My resume compresses this case to roughly 480K users, 24 cohorts, 10 message templates, and one
funding-rate difference that remained after correction (+0.311 pp; adjusted p = 0.011). Those values
reconcile to the private audit. This repository supplies the qualification that cannot fit in one
resume bullet: it is a retrospective aggregate-control snapshot difference, not causal uplift, and
the unmatched follow-up windows prevent a rollout claim.

## What this project demonstrates

- **Experimentation:** absolute and relative effects, 95% confidence intervals, two-sided tests, Holm/BH multiplicity control, minimum detectable effects, and placebo checks.
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
```

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m email_experiment generate-synthetic --output-dir data/synthetic
python -m email_experiment run-synthetic --output-dir reports/synthetic
pytest
ruff check .
```

The first command writes inspectable synthetic rows; the second runs the full analysis in memory and
writes only `reports/synthetic/aggregate.json`. Synthetic outputs validate code paths and statistical
behavior; they are not presented as evidence about the source campaign. See
[data/README.md](data/README.md) for the privacy boundary and schema expectations.

As a power sanity check, the default deterministic run finds no Holm-significant primary or
pre-treatment negative-control comparison. Its per-experiment 80% MDE is about 2.34–2.41 percentage
points, far above the simulated 0.42-point effect. A large-looking synthetic win is not engineered
into the demo.

## Evidence boundary

Two evidence layers are kept separate:

1. **Public synthetic workflow** — safe to run, inspect, and modify. It demonstrates the analysis design and its invariants.
2. **Private-source aggregate benchmark** — reproduced by running the same reasoning against a restricted event extract, then exporting only a minimized set of rounded or resume-aligned values to [`reports/source-benchmark.json`](reports/source-benchmark.json).

No raw event data, user identifiers, email addresses, message bodies, proprietary materials, or source-derived row-level samples are included. The implementation is an independent rewrite rather than a publication of source notebooks or templates.

## Why the source comparison stops here

The aggregate source benchmark contains useful signals, but it cannot support an experimental funding conclusion:

- control outcomes are aggregate snapshots without user-level timestamps, so treatment and control follow-up windows cannot be aligned;
- funding outcomes occurred after eligibility but before the first send and must be excluded from any post-exposure comparison;
- outcome collection ends before the full twice-weekly schedule and its follow-up window complete;
- all groups reuse the same pre-generated assignment-order block rather than independent production randomizations;
- raw opens have day rather than intra-day precision, preventing reliable same-day event ordering.

These limitations change the recommendation, not just the footnotes. The next step is the prospective design in [docs/experiment-design.md](docs/experiment-design.md): user-level randomization, timestamped controls, a fixed outcome window, pre-registered multiplicity families, and explicit unsubscribe/complaint guardrails.

## Reproducibility and public safety

- Python 3.11+
- deterministic synthetic fixtures
- tests for joins, multiplicity, time ordering, and guardrail denominators
- Ruff linting and GitHub Actions CI
- MIT-licensed code; source data are not redistributed

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the data and dependency boundary.

## Status

This repository is a portfolio implementation of Product Analytics + Experimentation practice. It is suitable for learning, review, and method development; it is not a campaign recommendation or evidence of a general funding increase.
