# Prospective experiment design

The retrospective benchmark is best used to design a cleaner next test. This document specifies the minimum design required before a lifecycle team makes a rollout decision.

## Decision and hypothesis

**Decision:** choose a message concept and delivery cadence that improve 14-day funding without an unacceptable increase in unsubscribe, complaint, or delivery-failure risk.

**Primary hypothesis:** at least one pre-specified lifecycle strategy changes 14-day funding relative to a concurrent no-campaign or business-as-usual holdout.

Recorded opens and links are secondary mechanisms, not substitutes for the business endpoint.

## Recommended design

Use a user-level, stratified, factorial randomization when traffic permits:

| Factor | Example arms |
| --- | --- |
| Content | Template D concept, current message, one additional challenger |
| Cadence | daily, twice weekly |
| Holdout | business-as-usual/no campaign concurrent control |

Randomize once at the user level before exposure and persist the assignment. Stratify on the small set of pre-treatment variables used operationally, such as lifecycle segment and tenure band. If a full factorial test is too costly, run two stages: confirm content against holdout first, then compare cadence using the winning content.

## Eligibility

- define the eligible lifecycle state from a timestamped snapshot;
- exclude users already funded before randomization;
- decide before analysis whether already-linked users are eligible for the funding endpoint;
- enforce communication-consent and contact-policy rules before assignment;
- retain all randomized eligible users in the intention-to-treat denominator.

An exclusion discovered after outcome observation requires a documented sensitivity analysis.

## Exposure and observation windows

- treatment clock: randomized assignment, with first successful delivery recorded separately;
- primary endpoint: funding within 14 full days of assignment;
- mechanism endpoints: delivered, recorded open, and new link inside fixed windows;
- guardrails: unsubscribe, complaint, and delivery failure through at least the primary window;
- data freeze: no analysis until every randomized arm has complete follow-up plus the agreed event-latency buffer.

Control users need the same synthetic assignment timestamp and observation window as treated users. A timeless aggregate control is insufficient.

## Outcomes

### Primary

Unique users funding within 14 days, analyzed by intention to treat.

### Secondary

- new-link conversion among users not linked at baseline;
- recorded open among delivered messages;
- ordered open → new link → fund progression;
- time to funding, if event timestamps are reliable enough for survival analysis.

### Guardrails

- unique-recipient unsubscribe risk;
- complaint/spam-report risk;
- bounce/drop risk;
- notification fatigue or downstream retention, if available.

Pre-specify a non-inferiority margin for each rollout-blocking guardrail rather than relying only on p > 0.05.

## Power and multiplicity

Calculate sample size from the control funding rate, minimum business-relevant absolute effect, desired power, and number of primary comparisons. Do not use the observed winning effect as the planning target without shrinkage.

Recommended hierarchy:

1. test the primary funding family with Holm control at α = 0.05;
2. evaluate guardrails against pre-specified non-inferiority margins;
3. interpret content/cadence interactions only if powered;
4. label all other segment analyses exploratory and report the full family.

Open-rate differences can guide iteration but cannot rescue a null funding result.

## Instrumentation and quality checks

Before reading outcomes:

- verify assignment uniqueness and immutability;
- test sample-ratio mismatch overall and within strata;
- confirm exposure timestamps do not precede assignment;
- check that holdout users are not contaminated by campaign sends;
- aggregate delivery-provider events to the subject-message grain;
- compare treatment arms on pre-treatment balance;
- validate outcome latency and freeze completeness;
- hash assignment blocks to catch accidental reuse or copying.

## Analysis plan

1. publish a CONSORT-style flow from eligibility through analysis;
2. report counts, rates, absolute effects, relative risks, 95% intervals, and adjusted p-values;
3. use intention to treat as the decision estimate;
4. use treatment-on-delivered or per-protocol views only as clearly labeled sensitivities;
5. report all planned arms, including null and harmful estimates;
6. repeat the analysis with late-event buffers and pre-defined missingness assumptions;
7. monitor heterogeneous effects only after the global decision and with shrinkage or held-out confirmation.

## Launch rule

A launch recommendation requires all of the following:

- a corrected primary funding result or a pre-specified decision rule met with sufficient precision;
- no unacceptable guardrail degradation;
- complete, aligned follow-up across arms;
- no unresolved randomization, contamination, or join-integrity issue;
- a monitoring plan for calibration of expected volume, event latency, and customer harm.

If these conditions are not met, the correct product decision is to keep testing—not to select the largest observed percentage.
