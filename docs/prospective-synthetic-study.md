# Prospective synthetic study

This study is an executable rehearsal of the prospective design. It uses only
parameterized synthetic records and therefore tests the analysis contract, not the
effectiveness of any real campaign.

## Decision before outcomes

The product decision is whether a pre-specified content-and-cadence candidate has
enough 14-day funding evidence to justify rollout while remaining within customer-harm
guardrails. The design has seven concurrent cells:

| Cell type | Content levels | Cadence levels |
| --- | --- | --- |
| Active | candidate concept, current message, challenger | daily, twice weekly |
| Holdout | no campaign | none |

The six active-cell funding comparisons against the same concurrent holdout form the
primary family. Holm controls the family-wise error rate at 0.05. Relative risks are
reported as supporting context; the absolute intention-to-treat risk difference is the
decision estimand. A large observed rate alone never selects a winner. Pooled factor
contrasts and content-by-cadence interactions are intentionally not claimed by this
computational rehearsal; they require a separately powered, pre-specified extension.

The canonical primary and pre-period negative-control summaries use a block-standardized
intention-to-treat risk difference across the declared lifecycle-segment, tenure-band, and
assignment-wave blocks. The variance is the conservative Neyman estimator: it does not
assume one common effect across blocks and omits the unidentified finite-population
treatment-effect variance term. Every block must retain exact seven-arm allocation with at least two units
per arm, and a zero or non-finite variance fails closed. The resulting normal intervals are
nominal, not simultaneous; Holm still controls the six-comparison primary family.

Exact common allocation makes the adjusted point estimate equal to the pooled difference
in this fixture. Relative-risk intervals remain pooled, supplementary, and not
block-adjusted. That separation prevents a design-alignment change from being presented as
a larger treatment effect.

## Randomization and analysis population

- eligibility is fixed before assignment;
- the unit of randomization and analysis is one synthetic participant;
- assignment is permuted within lifecycle-segment, tenure-band, and enrollment-wave
  blocks, with one slot for each of the seven cells per complete block;
- assignment is deterministic for a declared seed and invariant to eligible-row order;
- validation independently requires equal counts in all seven cells inside every block;
- every eligible randomized participant remains in the intention-to-treat denominator;
- the canonical assignment table is hashed so an accidental change is visible without
  publishing rows.

Exact allocation is a deterministic count contract, not a statistical SRM conclusion or
proof that an external ledger is complete. A production run must also reconcile the ledger
to its frozen eligibility snapshot and pre-registered randomization record. The harness also
reports sample-ratio diagnostics globally and within blocks. It compares a
pre-period activity indicator that was available before assignment across the primary
family as a negative control. A detected assignment-related difference can block
interpretation; a non-significant result is a diagnostic, not proof that balance is
perfect.

## Timestamps and data freeze

All timestamps must include an explicit timezone and are normalized to UTC; timezone-naive
values are rejected rather than assumed to be UTC. Duplicate column names are also rejected
before schema selection. The primary and guardrail interval is the
half-open window `[assigned_at, assigned_at + 14 days)`, so an event exactly on the
14-day boundary is not counted. Each event has both an occurrence time and an
availability time.

The analysis waits until every assignment has 14 full days of follow-up and the declared
event-latency buffer has elapsed. Events known to exist but unavailable at the data freeze
are reported as late arrivals and block the decision instead of being silently counted as
failures. Duplicate event keys, orphan events, occurrence after availability, or join
amplification also fail closed. This validates known synthetic events and their latency; a
sparse positive-event table cannot prove that an entire funding, unsubscribe, complaint, or
baseline feed is present. Production use therefore requires source-level completeness
watermarks or complete participant-level outcome snapshots as an additional gate.

## Contamination and guardrails

Holdout campaign deliveries, active-cell deliveries with the wrong content or cadence,
and any campaign exposure before assignment are contamination. The public report emits
only aggregate counts, never participant or event examples.

The canonical rehearsal additionally requires every active assignment to have at least one
correct delivery during `[assigned_at, assigned_at + 14 days)`. Missing or only out-of-window
delivery blocks the decision but does not remove the participant from ITT. This is a minimum
synthetic fixture-fidelity check, not a production adherence estimate or proof that every
scheduled send succeeded.

Unsubscribe and complaint are rollout-blocking guardrails. Each has six active-versus-
holdout risk-difference comparisons using all randomized users, not only delivered users.
Their one-sided upper confidence bounds use Bonferroni allocation across all 12 planned
guardrail comparisons. Every comparison is reported; the two comparisons for the
pre-specified candidate are rollout gates. A comparison passes only when its upper bound
is strictly below the pre-specified absolute harm margin; a non-significant harm test is
not evidence of safety. Because these events are rare and several block cells contain no
events, the bounds deliberately retain the conservative pooled Newcombe-style construction
and are explicitly not block-adjusted. A future stratified guardrail method must validate
one-sided coverage with zero cells before replacing it.

## Launch rule

The decision remains `continue_testing` unless all of the following hold:

1. the pre-specified candidate has a positive funding risk difference and a
   Holm-adjusted two-sided p-value below 0.05;
2. that candidate demonstrates non-inferiority for both unsubscribe and complaint, with
   all 12 planned comparisons still disclosed;
3. follow-up and event-latency windows are complete and aligned;
4. assignment integrity, exact within-block allocation, sample ratio, contamination,
   in-window active-delivery coverage, schema, foreign-key, uniqueness, and pre-treatment
   negative-control gates all pass.

The decision API requires these named gates: `assignment_contract`,
`exact_block_allocation`, `sample_ratio`, `concurrent_holdout`,
`aligned_14_day_followup`,
`complete_followup_and_latency_buffer`, `no_contamination`,
`active_delivery_coverage`, `pre_period_negative_control`, and
`itt_population_preserved`. Callers may add stricter gates, but omitting any required gate
is a contract error. The canonical report permits exactly this versioned set so a typo or
silent schema change cannot produce a launch-eligible result.

The canonical fixture is deliberately configured around small, uncertain effects. It is
not tuned to manufacture a launch recommendation. Its aggregate output is regenerated by
the command documented in the README and stored in
`reports/prospective-synthetic-benchmark.json`.

In the canonical run, 8,400 fictional participants are split equally across seven cells.
All quality gates pass, but the pre-specified candidate's observed funding difference is
+1.50 percentage points (block-adjusted nominal 95% CI -0.10 to +3.10 pp;
Holm-adjusted p = 0.395).
The declared-baseline 80% planning MDE is 2.78 pp, and the simultaneous unsubscribe and
complaint upper bounds both exceed their respective margins. The resulting
`continue_testing` decision reflects insufficient evidence and precision, not a failed
attempt to find the largest rate.

## What this can and cannot show

The study can show that a decision policy, denominator, time window, multiplicity family,
and declared synthetic failure gates are encoded consistently and reproducibly. It cannot
estimate a real campaign effect, prove an external event feed is complete, validate the
retrospective source benchmark, establish a production non-inferiority margin, or replace a
real randomized experiment.
