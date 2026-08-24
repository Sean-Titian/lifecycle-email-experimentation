# Data boundary and public schema

This repository does not redistribute the source campaign data. The original files contain user-level behavioral records and are not established as publicly licensed.

## What is public

- a deterministic synthetic-data generator and its generated fixtures;
- a schema that mirrors the analytical roles needed by the code without preserving source values;
- aggregate, non-identifying source benchmarks in [`../reports/source-benchmark.json`](../reports/source-benchmark.json).

Synthetic records exist to exercise joins, estimators, multiplicity corrections, timing rules, and funnel logic. Generate them with:

```bash
python -m email_experiment generate-synthetic --output-dir data/synthetic
python -m email_experiment run-synthetic --output-dir reports/synthetic
```

Their results must never be presented as source-campaign performance. Synthetic analysis artifacts belong under `reports/synthetic/`; do not commit subject-level CSV files.

## What is never committed

- raw or transformed source rows;
- stable user or account identifiers;
- email addresses, message bodies, reason strings, or free text;
- source-derived samples, even if they contain only a few rows;
- private notebooks, reports, templates, or trained artifacts;
- local filesystem paths or access credentials.

The `data/raw/` directory is intentionally ignored. A placeholder does not authorize placing private data there for commit.

## Public synthetic schema

| File / frame | Columns | Grain and validation |
| --- | --- | --- |
| `assignments.csv` | `participant_id`, `experiment`, `arm`, `assigned_at`, `messages_delivered`, `linked_before_entry`, `converted_before_entry` | one fictional participant per row; complete unique key; both arms present in every experiment |
| `events.csv` | `participant_id`, `event_type`, `event_at` | one event per row; known participant; valid timestamp; event types include delivery, open, new link, conversion, and unsubscribe |
| `manifest.json` | classification, derivation, observation end, row counts, generator configuration | labels every generated row as synthetic and records deterministic parameters |
| `reports/synthetic/aggregate.json` | population flow, effects, CIs, corrected p-values, MDEs, censoring, negative control, ordered funnel, guardrail | aggregate-only output; contains no participant key or event row |

The private aggregate control is not an input to the public workflow. Its minimum non-identifying
handoff lives separately in `reports/source-benchmark.json`; the synthetic schema does not pretend to
reconstruct those source records.

## Grain and join rules

1. Validate one assignment row per synthetic subject before joining.
2. Aggregate repeated delivery events to the declared subject-message or subject grain.
3. Never join a many-row event table directly to a subject summary and then count rows as users.
4. Fail closed on duplicated or ambiguous assignment keys.
5. Preserve timestamps until exposure windows and funnel order have been resolved.
6. Report the number of excluded pre-exposure and right-censored outcomes.

## Replacing synthetic data

Anyone adapting the code to authorized internal data should create a separate, untracked adapter. Before running inference, document:

- the legal basis for use;
- analysis unit and randomization unit;
- eligibility snapshot and treatment start;
- exposure and outcome windows;
- control construction;
- late-arriving-event policy;
- deletion, retention, and access controls.

The public repository neither requires nor expects access to the private source data.
