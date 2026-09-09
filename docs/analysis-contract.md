# Analysis contract

This contract defines what the project estimates before results are interpreted. Its purpose is to prevent denominator drift, post-hoc outcome selection, and causal language that the available design cannot support.

## 1. Product decision

The lifecycle team needs to decide:

1. which message concept should enter a confirmatory content experiment;
2. whether daily or twice-weekly delivery should be tested or adopted;
3. whether any engagement gain is accompanied by acceptable customer-risk guardrails.

The analysis may recommend a next experiment even when it cannot recommend a production rollout.

## 2. Units and timestamps

| Concept | Contract |
| --- | --- |
| Randomization unit | Subject/account. Cluster at this unit if repeated messages are analyzed. |
| Analysis unit | Unique eligible subject for funding and guardrails; subject-template pair for recorded-open comparisons. |
| Eligibility time | Fixed before any exposure. |
| Randomization / analysis time | Persistent user-level assignment, immediately before the intervention period; this starts the prospective ITT clock. |
| Exposure time | Every successful delivery is retained separately so assignment-to-delivery gaps and non-delivery remain auditable. |
| Primary follow-up | Pre-specified from the analysis time and identical for every randomized arm. |
| Event order | assignment ≤ delivery ≤ open < link < funding; date-only ties require a conservative rule. |

Outcomes before assignment are not post-treatment successes. If a historical extract has a gap
between its eligibility snapshot and first possible exposure, gap outcomes are disclosed and removed
from any post-exposure sensitivity analysis. A prospective design should avoid that gap by assigning
immediately before treatment begins. Subjects without enough observation time are right-censored
rather than silently labeled as failures.

## 3. Outcome hierarchy

### Primary business endpoint

Funding within the pre-specified post-exposure window, analyzed by intention to treat.

Report for every planned comparison:

- treatment and control event counts and denominators;
- absolute effect in percentage points;
- 95% confidence interval;
- relative risk with 95% confidence interval;
- two-sided p-value;
- adjusted p-value within its pre-registered family;
- approximate minimum detectable effect at 80% power.

### Secondary behavioral endpoints

- new linking inside the same fixed window;
- recorded open status for content exploration;
- temporally ordered open → new link → fund progression.

Recorded opens are an imperfect engagement proxy and are never substituted for the primary funding endpoint.

### Guardrails

- unique-recipient unsubscribe risk;
- complaint/spam-report risk;
- bounce and delivery-failure rates;
- any predefined downstream harm metric available in a prospective deployment.

Operational event counts may be reported alongside user risk but must not use the same label or denominator.

## 4. Estimands and comparison families

### Content exploration

Estimate template-level recorded-open proportions with Wilson 95% confidence intervals. When the same subject can receive multiple templates, paired comparisons must preserve the within-subject dependence.

This endpoint ranks content candidates; it does not estimate a funding difference.

### Campaign group versus aggregate control

The private benchmark contains 24 retrospective segment-by-cadence comparisons against aggregate control snapshots. Holm family-wise error control is primary because it remains valid under arbitrary dependence; Benjamini–Hochberg is a sensitivity analysis. Shared aggregate controls create dependent comparisons and must be disclosed. These are snapshot differences/associations, not user-level randomized estimates.

### Direct cadence comparison

Compare daily with twice-weekly assignment within matched lifecycle strata. Funding and valid
new-linking families are adjusted separately. A raw p-value is not a cadence decision.

### Prospective factorial family

The executable prospective rehearsal has three content levels crossed with two cadence
levels and a concurrent holdout. Its six active-cell-versus-holdout 14-day funding risk
differences are one pre-specified primary family under Holm family-wise error control at
α = 0.05. Every randomized eligible participant is analyzed in the assigned cell.

Unsubscribe and complaint are separate outcomes but one simultaneously covered family of 12
active-cell-versus-holdout comparisons. Each adverse risk difference receives a
simultaneous one-sided upper confidence bound using Bonferroni allocation. Non-inferiority
requires that bound to be strictly below the declared absolute margin; `p > 0.05` is not a
safety result. All 12 are disclosed, while the two bounds for the candidate named before
outcomes are the rollout gates. Pooled factor contrasts and interactions are not used to
rescue a null primary cell family.

## 5. Data-quality gates

Inference stops if any of these gates fail without a documented resolution:

1. assignment and subject tables cannot be made one-to-one;
2. an event-table join increases the unique-subject denominator;
3. assignment occurs after exposure or outcomes occur before eligibility;
4. arm sizes or assignment patterns indicate an unexplained randomization failure;
5. control eligibility, timing, or outcome definitions differ from treatment;
6. complete follow-up or the declared event-latency buffer is unavailable and censoring
   cannot be handled consistently;
7. holdout receives a campaign exposure or an active cell receives the wrong content or
   cadence;
8. the pre-treatment negative control shows multiplicity-adjusted evidence of an
   assignment-related difference.

The source benchmark detected ambiguous assignment keys; strict analysis excludes every affected
record rather than keeping an arbitrary row.

## 6. Negative controls and diagnostics

- test pre-exposure funding against assigned first-template position within segment;
- examine standardized differences in pre-treatment tenure/approval age;
- test sample-ratio mismatch and assignment-position balance;
- hash repeated assignment blocks to identify reused randomization structures;
- compare event-row counts with unique-subject counts to detect join amplification.

A negative control that shows a treatment effect triggers investigation; it is not explained away after viewing the primary endpoint.

## 7. Funnel contract

The strict new-link funnel includes actual recipients who were neither linked nor funded before campaign start. A subject advances only if:

1. an open occurs on or after campaign start;
2. a new link occurs after that open;
3. funding occurs after that link.

With date-only open timestamps, the canonical public benchmark requires linking at least the next calendar day. A same-day version may appear only as a labeled sensitivity analysis.

## 8. Missingness and censoring

- show observed and missing status denominators for template metrics;
- do not treat undelivered messages as observed non-opens without a declared estimand;
- exclude known pre-send outcomes from post-treatment effects;
- align treatment and control follow-up clocks;
- report the last observable timestamp for every endpoint;
- do not compare completed daily follow-up with incomplete twice-weekly follow-up.

## 9. Claim rules

The public write-up may say:

- Template D had the highest recorded-open rate in the analyzed schedule;
- one of 24 local funding comparisons remained non-null after multiplicity correction;
- no direct cadence advantage was supported;
- the strict temporal funnel is much smaller than the population-state funnel.

It may not say:

- lifecycle email generally caused higher funding;
- daily sending beats twice-weekly sending;
- opens caused links or funding;
- the aggregate control is equivalent to a randomized user-level holdout;
- synthetic output validates the source campaign.

## 10. Publication checklist

- [x] Code has been executed end to end.
- [x] Tests and lint pass.
- [x] All numbers reconcile to machine-readable report artifacts.
- [x] Multiplicity families and null results remain visible.
- [x] Outcome windows and censoring are disclosed.
- [x] Retrospective descriptive user-risk denominators use unique actual recipients;
  prospective rollout guardrails use all randomized users by intention to treat.
- [x] No raw rows, identifiers, private text, or local paths are present.
- [x] README recommendation matches the strength of the design.
