"""Decision rules for the synthetic prospective factorial harness.

The functions in this module operate on one row per randomized participant.
They deliberately require a seven-arm design: six active content-by-cadence
cells and one concurrent holdout.  Returned decisions describe whether a
pre-specified *synthetic* candidate met the declared statistical criteria;
they are not production rollout recommendations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from math import erfc, isfinite, sqrt
from statistics import NormalDist
from typing import Literal

import numpy as np
import pandas as pd

from .assignment import BLOCK_COLUMNS
from .contracts import (
    ContractError,
    require_columns,
    validate_binary,
    validate_unique_key,
)
from .statistics import (
    adjust_pvalues,
    compare_binary_proportions,
    compare_block_standardized_binary,
)

DEFAULT_GUARDRAILS = ("unsubscribe_14d", "complaint_14d")
EXPECTED_ACTIVE_CELLS = 6
BLOCK_STANDARDIZED_ESTIMATOR = "block_standardized_difference_in_means"
STRATIFIED_NEYMAN_VARIANCE = "stratified_neyman_conservative"
NOMINAL_NORMAL_INTERVAL = "nominal_normal_not_simultaneous"
POOLED_PRIMARY_ESTIMATOR = "pooled_difference_in_proportions"
POOLED_PRIMARY_UNCERTAINTY = "pooled_score_and_newcombe_not_block_adjusted"
POOLED_GUARDRAIL_POINT = "pooled_difference_in_proportions"
POOLED_GUARDRAIL_BOUND = "pooled_newcombe_style_bonferroni_not_block_adjusted"
CANONICAL_BLOCK_DEFINITION = "|".join(BLOCK_COLUMNS)
REQUIRED_QUALITY_GATES = (
    "assignment_contract",
    "exact_block_allocation",
    "sample_ratio",
    "concurrent_holdout",
    "aligned_14_day_followup",
    "complete_followup_and_latency_buffer",
    "no_contamination",
    "active_delivery_coverage",
    "pre_period_negative_control",
    "itt_population_preserved",
)


@dataclass(frozen=True)
class DecisionResult:
    """Machine-readable result for one pre-specified synthetic candidate."""

    status: Literal["criteria_met", "continue_testing"]
    candidate_arm: str | None
    scope: str
    reasons: tuple[str, ...]
    failed_quality_gates: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-native values without changing the immutable result."""

        result = asdict(self)
        result["reasons"] = list(self.reasons)
        result["failed_quality_gates"] = list(self.failed_quality_gates)
        return result


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool):
        raise ContractError("alpha must be a finite number between 0 and 1")
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ContractError("alpha must be a finite number between 0 and 1") from exc
    if not isfinite(value):
        raise ContractError("alpha must be a finite number between 0 and 1")
    if not 0.0 < value < 1.0:
        raise ContractError("alpha must be between 0 and 1")
    return value


def _validate_factorial_frame(
    frame: pd.DataFrame,
    *,
    id_col: str,
    arm_col: str,
    content_col: str,
    cadence_col: str,
    holdout_arm: str,
    outcome_cols: Sequence[str],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Validate the one-row-per-randomized-unit seven-arm contract."""

    required = [id_col, arm_col, content_col, cadence_col, *outcome_cols]
    require_columns(frame, required, frame_name="factorial analysis data")
    validate_unique_key(frame, id_col, frame_name="factorial analysis data")
    if frame.empty:
        raise ContractError("factorial analysis data must contain randomized units")
    if frame[arm_col].isna().any():
        raise ContractError("arm assignments must be complete")
    if not all(isinstance(value, str) and value.strip() for value in frame[arm_col]):
        raise ContractError("arm assignments must be non-empty strings")

    observed_arms = set(frame[arm_col])
    if holdout_arm not in observed_arms:
        raise ContractError("the declared holdout arm is missing")
    active_arms = tuple(sorted(observed_arms - {holdout_arm}))
    if len(active_arms) != EXPECTED_ACTIVE_CELLS:
        raise ContractError("the design must contain six active arms and one holdout")

    holdout = frame.loc[frame[arm_col] == holdout_arm]
    if holdout[[content_col, cadence_col]].notna().any().any():
        raise ContractError("holdout rows must not be assigned content or cadence")

    active = frame.loc[frame[arm_col] != holdout_arm]
    if active[[content_col, cadence_col]].isna().any().any():
        raise ContractError("active rows require complete content and cadence factors")
    for column in (content_col, cadence_col):
        if not all(isinstance(value, str) and value.strip() for value in active[column]):
            raise ContractError(f"{column} values must be non-empty strings")

    contents = tuple(sorted(active[content_col].unique()))
    cadences = tuple(sorted(active[cadence_col].unique()))
    if len(contents) != 3 or len(cadences) != 2:
        raise ContractError("active arms must form three content by two cadence levels")

    mappings = active[[arm_col, content_col, cadence_col]].drop_duplicates()
    observed_pairs = set(zip(mappings[content_col], mappings[cadence_col], strict=True))
    expected_pairs = set(product(contents, cadences))
    if (
        len(mappings) != EXPECTED_ACTIVE_CELLS
        or mappings[arm_col].nunique() != EXPECTED_ACTIVE_CELLS
        or observed_pairs != expected_pairs
    ):
        raise ContractError("each active arm must map to exactly one factorial cell")

    working = frame.copy()
    for outcome_col in outcome_cols:
        working[outcome_col] = validate_binary(
            working[outcome_col], name=outcome_col, allow_missing=False
        )
    return working, active_arms


def evaluate_primary_family(
    frame: pd.DataFrame,
    *,
    id_col: str = "participant_id",
    arm_col: str = "arm",
    content_col: str = "content",
    cadence_col: str = "cadence",
    outcome_col: str = "funded_14d",
    holdout_arm: str = "holdout",
    alpha: float = 0.05,
    block_cols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Evaluate six active-cell versus holdout funding ITT contrasts.

    Risk difference is the decision-scale estimand and relative risk is
    supplementary.  When ``block_cols`` is explicit, risk differences use the
    design-aligned block-standardized estimator with conservative Neyman
    variance; otherwise the legacy pooled comparison is retained for backwards-
    compatible descriptive use.  The six two-sided p-values form one Holm FWER
    family.  Nominal confidence intervals are not simultaneous.
    """

    family_alpha = _validate_alpha(alpha)
    if block_cols is not None:
        require_columns(
            frame,
            block_cols,
            frame_name="factorial analysis data",
        )
    working, active_arms = _validate_factorial_frame(
        frame,
        id_col=id_col,
        arm_col=arm_col,
        content_col=content_col,
        cadence_col=cadence_col,
        holdout_arm=holdout_arm,
        outcome_cols=(outcome_col,),
    )
    holdout = working.loc[working[arm_col] == holdout_arm, outcome_col]
    rows: list[dict[str, object]] = []
    for active_arm in active_arms:
        active = working.loc[working[arm_col] == active_arm, outcome_col]
        marginal_effect = compare_binary_proportions(
            int(active.sum()),
            int(active.size),
            int(holdout.sum()),
            int(holdout.size),
        )
        if block_cols is None:
            treatment_rate = marginal_effect.treatment_rate
            control_rate = marginal_effect.control_rate
            risk_difference = marginal_effect.absolute_effect
            standard_error: float | None = None
            ci_low = marginal_effect.absolute_ci_low
            ci_high = marginal_effect.absolute_ci_high
            p_value = marginal_effect.p_value_two_sided
            estimator = POOLED_PRIMARY_ESTIMATOR
            variance_estimator = POOLED_PRIMARY_UNCERTAINTY
            interval_method = "nominal_newcombe_not_simultaneous"
            block_definition: str | None = None
            block_count = 0
        else:
            pair = working.loc[
                working[arm_col].isin([active_arm, holdout_arm])
            ].copy()
            block_effect = compare_block_standardized_binary(
                pair,
                group_col=arm_col,
                treatment_label=active_arm,
                control_label=holdout_arm,
                outcome_col=outcome_col,
                block_cols=block_cols,
            )
            treatment_rate = block_effect.treatment_rate
            control_rate = block_effect.control_rate
            risk_difference = block_effect.absolute_effect
            standard_error = block_effect.absolute_standard_error
            ci_low = block_effect.absolute_ci_low
            ci_high = block_effect.absolute_ci_high
            p_value = block_effect.p_value_two_sided
            estimator = BLOCK_STANDARDIZED_ESTIMATOR
            variance_estimator = STRATIFIED_NEYMAN_VARIANCE
            interval_method = NOMINAL_NORMAL_INTERVAL
            block_definition = "|".join(block_cols)
            block_count = block_effect.block_count
        rows.append(
            {
                "family": "primary_funding_itt",
                "comparison_id": f"{active_arm}_vs_{holdout_arm}",
                "outcome": outcome_col,
                "active_arm": active_arm,
                "comparator_arm": holdout_arm,
                "active_successes": marginal_effect.treatment_successes,
                "active_total": marginal_effect.treatment_total,
                "holdout_successes": marginal_effect.control_successes,
                "holdout_total": marginal_effect.control_total,
                "active_rate": treatment_rate,
                "holdout_rate": control_rate,
                "risk_difference": risk_difference,
                "risk_difference_standard_error": standard_error,
                "risk_difference_ci_low_nominal": ci_low,
                "risk_difference_ci_high_nominal": ci_high,
                "relative_risk": marginal_effect.relative_risk,
                "relative_risk_ci_low_nominal": marginal_effect.relative_risk_ci_low,
                "relative_risk_ci_high_nominal": marginal_effect.relative_risk_ci_high,
                "relative_risk_interval_method": (
                    "pooled_log_wald_supplementary_not_block_adjusted"
                ),
                "p_value_two_sided": p_value,
                "estimator": estimator,
                "variance_estimator": variance_estimator,
                "interval_method": interval_method,
                "block_columns": block_definition,
                "block_count": block_count,
            }
        )

    results = pd.DataFrame(rows)
    results["p_value_holm"] = adjust_pvalues(
        results["p_value_two_sided"], "holm"
    )
    results["family_size"] = len(results)
    results["family_alpha"] = family_alpha
    results["holm_significant"] = results["p_value_holm"] < family_alpha
    results["superiority_pass"] = (
        results["holm_significant"] & (results["risk_difference"] > 0.0)
    )
    return results


def _wilson_interval_one_sided(
    successes: int,
    total: int,
    *,
    tail_alpha: float,
) -> tuple[float, float]:
    """Wilson limits using the one-sided critical value."""

    rate = successes / total
    z = NormalDist().inv_cdf(1.0 - tail_alpha)
    denominator = 1.0 + z**2 / total
    center = (rate + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * sqrt(rate * (1.0 - rate) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _risk_difference_upper_bound(
    active_successes: int,
    active_total: int,
    holdout_successes: int,
    holdout_total: int,
    *,
    tail_alpha: float,
) -> float:
    """Conservative Newcombe-style upper bound for active minus holdout risk."""

    active_rate = active_successes / active_total
    holdout_rate = holdout_successes / holdout_total
    _, active_upper = _wilson_interval_one_sided(
        active_successes, active_total, tail_alpha=tail_alpha
    )
    holdout_lower, _ = _wilson_interval_one_sided(
        holdout_successes, holdout_total, tail_alpha=tail_alpha
    )
    difference = active_rate - holdout_rate
    upper = difference + sqrt(
        (active_upper - active_rate) ** 2
        + (holdout_rate - holdout_lower) ** 2
    )
    return min(1.0, upper)


def evaluate_guardrail_family(
    frame: pd.DataFrame,
    *,
    margins: Mapping[str, float],
    outcome_cols: Sequence[str] = DEFAULT_GUARDRAILS,
    id_col: str = "participant_id",
    arm_col: str = "arm",
    content_col: str = "content",
    cadence_col: str = "cadence",
    holdout_arm: str = "holdout",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Evaluate 12 active-cell guardrail non-inferiority contrasts.

    Positive risk differences are harmful.  Each caller-supplied margin is on
    the absolute unique-randomized-user risk scale.  Bonferroni allocates
    ``alpha / 12`` to conservative one-sided Newcombe-style upper bounds, so a
    strategy passes only when its upper bound is *strictly below* the margin.
    """

    family_alpha = _validate_alpha(alpha)
    outcomes = tuple(outcome_cols)
    if len(outcomes) != 2 or len(set(outcomes)) != 2:
        raise ContractError("the guardrail family must contain two unique outcomes")
    if not isinstance(margins, Mapping):
        raise ContractError("guardrail margins must be an explicit mapping")
    if set(margins) != set(outcomes):
        raise ContractError("one explicit margin is required for every guardrail")
    validated_margins: dict[str, float] = {}
    for outcome in outcomes:
        margin = margins[outcome]
        if isinstance(margin, bool):
            raise ContractError(f"the {outcome} margin must be finite")
        try:
            value = float(margin)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"the {outcome} margin must be finite") from exc
        if not isfinite(value):
            raise ContractError(f"the {outcome} margin must be finite")
        if not 0.0 <= value < 1.0:
            raise ContractError(f"the {outcome} margin must be in [0, 1)")
        validated_margins[outcome] = value

    working, active_arms = _validate_factorial_frame(
        frame,
        id_col=id_col,
        arm_col=arm_col,
        content_col=content_col,
        cadence_col=cadence_col,
        holdout_arm=holdout_arm,
        outcome_cols=outcomes,
    )
    family_size = len(active_arms) * len(outcomes)
    if family_size != 12:
        raise ContractError("the guardrail family must contain 12 comparisons")
    per_comparison_alpha = family_alpha / family_size

    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        holdout = working.loc[working[arm_col] == holdout_arm, outcome]
        holdout_successes = int(holdout.sum())
        holdout_total = int(holdout.size)
        for active_arm in active_arms:
            active = working.loc[working[arm_col] == active_arm, outcome]
            active_successes = int(active.sum())
            active_total = int(active.size)
            effect = compare_binary_proportions(
                active_successes,
                active_total,
                holdout_successes,
                holdout_total,
            )
            margin = validated_margins[outcome]
            upper_bound = _risk_difference_upper_bound(
                active_successes,
                active_total,
                holdout_successes,
                holdout_total,
                tail_alpha=per_comparison_alpha,
            )
            passes = upper_bound < margin
            if passes:
                conclusion = "noninferior"
            elif effect.absolute_effect >= margin:
                conclusion = "harmful_or_over_margin"
            else:
                conclusion = "inconclusive"
            rows.append(
                {
                    "family": "rollout_blocking_guardrails_itt",
                    "comparison_id": f"{outcome}:{active_arm}_vs_{holdout_arm}",
                    "outcome": outcome,
                    "active_arm": active_arm,
                    "comparator_arm": holdout_arm,
                    "estimand": "intention_to_treat_unique_randomized_user_risk",
                    "harm_direction": "positive_risk_difference",
                    "active_successes": active_successes,
                    "active_total": active_total,
                    "holdout_successes": holdout_successes,
                    "holdout_total": holdout_total,
                    "active_rate": effect.treatment_rate,
                    "holdout_rate": effect.control_rate,
                    "risk_difference": effect.absolute_effect,
                    "relative_risk_supplementary": effect.relative_risk,
                    "point_estimator": POOLED_GUARDRAIL_POINT,
                    "uncertainty_method": POOLED_GUARDRAIL_BOUND,
                    "block_adjusted_uncertainty": False,
                    "noninferiority_margin": margin,
                    "margin_source": "caller_supplied_pre_specified",
                    "simultaneous_one_sided_upper_bound": upper_bound,
                    "noninferiority_pass": passes,
                    "conclusion": conclusion,
                    "family_size": family_size,
                    "family_alpha": family_alpha,
                    "bonferroni_alpha_each": per_comparison_alpha,
                }
            )
    return pd.DataFrame(rows)


def make_launch_decision(
    primary_results: pd.DataFrame,
    guardrail_results: pd.DataFrame,
    quality_gates: Mapping[str, bool],
    *,
    candidate_arm: str | None,
) -> DecisionResult:
    """Apply gates to a pre-specified arm without ranking observed effects.

    ``criteria_met`` means only that the declared synthetic decision rule was
    met.  Every named gate in ``REQUIRED_QUALITY_GATES`` must be present; callers
    may add stricter gates and those also participate in the decision.  A missing
    candidate, any failed gate, a non-positive Holm result, or either
    inconclusive guardrail returns ``continue_testing``.
    """

    require_columns(
        primary_results,
        [
            "active_arm",
            "risk_difference",
            "risk_difference_standard_error",
            "risk_difference_ci_low_nominal",
            "risk_difference_ci_high_nominal",
            "p_value_two_sided",
            "p_value_holm",
            "family_size",
            "family_alpha",
            "superiority_pass",
            "estimator",
            "variance_estimator",
            "interval_method",
            "block_columns",
            "block_count",
        ],
        frame_name="primary results",
    )
    require_columns(
        guardrail_results,
        [
            "active_arm",
            "outcome",
            "noninferiority_margin",
            "simultaneous_one_sided_upper_bound",
            "family_size",
            "family_alpha",
            "bonferroni_alpha_each",
            "noninferiority_pass",
            "point_estimator",
            "uncertainty_method",
            "block_adjusted_uncertainty",
        ],
        frame_name="guardrail results",
    )
    if not isinstance(quality_gates, Mapping):
        raise ContractError("quality gates must be an explicit mapping")
    invalid_gates = [
        name
        for name, passed in quality_gates.items()
        if not isinstance(name, str) or not name or type(passed) is not bool
    ]
    if invalid_gates:
        raise ContractError("quality gates require non-empty names and boolean values")
    missing_gates = set(REQUIRED_QUALITY_GATES) - set(quality_gates)
    if missing_gates:
        raise ContractError(
            "quality gates are missing "
            f"{len(missing_gates)} required decision-contract entries"
        )

    primary_arms = tuple(primary_results["active_arm"])
    if len(primary_arms) != EXPECTED_ACTIVE_CELLS or len(set(primary_arms)) != len(
        primary_arms
    ):
        raise ContractError("primary results require one row for each active arm")
    if primary_results["superiority_pass"].isna().any():
        raise ContractError("primary pass indicators must be complete")
    if not pd.api.types.is_bool_dtype(primary_results["superiority_pass"]):
        raise ContractError("primary pass indicators must be boolean")
    if primary_results["estimator"].isna().any():
        raise ContractError("primary method metadata must be complete")
    if not primary_results["estimator"].eq(BLOCK_STANDARDIZED_ESTIMATOR).all():
        raise ContractError("launch decisions require the block-standardized primary estimator")
    primary_method_columns = [
        "variance_estimator",
        "interval_method",
        "block_columns",
    ]
    if primary_results[primary_method_columns].isna().any().any():
        raise ContractError("primary method metadata must be complete")
    if not primary_results["variance_estimator"].eq(STRATIFIED_NEYMAN_VARIANCE).all():
        raise ContractError("launch decisions require conservative stratified Neyman variance")
    if not primary_results["interval_method"].eq(NOMINAL_NORMAL_INTERVAL).all():
        raise ContractError("primary interval metadata is inconsistent")
    if not primary_results["block_columns"].eq(CANONICAL_BLOCK_DEFINITION).all():
        raise ContractError("primary results use the wrong randomization-block definition")
    primary_numeric = primary_results[
        [
            "risk_difference",
            "risk_difference_standard_error",
            "risk_difference_ci_low_nominal",
            "risk_difference_ci_high_nominal",
            "p_value_two_sided",
            "p_value_holm",
            "family_size",
            "family_alpha",
            "block_count",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if primary_numeric.isna().any().any() or not np.isfinite(primary_numeric).all().all():
        raise ContractError("primary decision fields must be finite numbers")
    if not primary_numeric["family_size"].eq(EXPECTED_ACTIVE_CELLS).all():
        raise ContractError("primary results must declare one six-comparison family")
    if not np.isclose(
        primary_numeric["family_alpha"], 0.05, rtol=0.0, atol=1e-15
    ).all():
        raise ContractError("primary results must use Holm FWER alpha 0.05")
    if not primary_numeric["p_value_holm"].between(0.0, 1.0).all():
        raise ContractError("Holm p-values must be between zero and one")
    if not primary_numeric["p_value_two_sided"].between(0.0, 1.0).all():
        raise ContractError("primary raw p-values must be between zero and one")
    if not primary_numeric["risk_difference_standard_error"].gt(0.0).all():
        raise ContractError("primary standard errors must be positive")
    if not primary_numeric["block_count"].gt(0).all() or not np.isclose(
        primary_numeric["block_count"],
        np.round(primary_numeric["block_count"]),
        rtol=0.0,
        atol=0.0,
    ).all():
        raise ContractError("primary block counts must be positive integers")
    if primary_numeric["block_count"].nunique() != 1:
        raise ContractError("primary comparisons must use the same randomization blocks")
    expected_raw_p = primary_numeric.apply(
        lambda row: erfc(
            abs(row["risk_difference"] / row["risk_difference_standard_error"])
            / sqrt(2.0)
        ),
        axis=1,
    )
    if not np.isclose(
        primary_numeric["p_value_two_sided"],
        expected_raw_p,
        rtol=1e-12,
        atol=1e-15,
    ).all():
        raise ContractError("primary raw p-values are inconsistent with the estimator")
    expected_holm = adjust_pvalues(primary_numeric["p_value_two_sided"], "holm")
    if not np.isclose(
        primary_numeric["p_value_holm"],
        expected_holm,
        rtol=1e-12,
        atol=1e-15,
    ).all():
        raise ContractError("primary Holm p-values are inconsistent with the family")
    z_critical = NormalDist().inv_cdf(0.975)
    expected_ci_low = np.maximum(
        -1.0,
        primary_numeric["risk_difference"]
        - z_critical * primary_numeric["risk_difference_standard_error"],
    )
    expected_ci_high = np.minimum(
        1.0,
        primary_numeric["risk_difference"]
        + z_critical * primary_numeric["risk_difference_standard_error"],
    )
    if not np.isclose(
        primary_numeric["risk_difference_ci_low_nominal"],
        expected_ci_low,
        rtol=1e-12,
        atol=1e-15,
    ).all() or not np.isclose(
        primary_numeric["risk_difference_ci_high_nominal"],
        expected_ci_high,
        rtol=1e-12,
        atol=1e-15,
    ).all():
        raise ContractError("primary confidence intervals are inconsistent with the estimator")
    expected_primary_pass = (
        primary_numeric["p_value_holm"] < primary_numeric["family_alpha"]
    ) & (primary_numeric["risk_difference"] > 0.0)
    if not primary_results["superiority_pass"].eq(expected_primary_pass).all():
        raise ContractError("primary pass indicators are inconsistent with the family rule")

    expected_guardrail_rows = EXPECTED_ACTIVE_CELLS * len(DEFAULT_GUARDRAILS)
    if len(guardrail_results) != expected_guardrail_rows:
        raise ContractError("guardrail results require two rows for each active arm")
    duplicates = guardrail_results.duplicated(["active_arm", "outcome"], keep=False)
    if duplicates.any():
        raise ContractError("guardrail results contain duplicate arm-outcome rows")
    if set(guardrail_results["active_arm"]) != set(primary_arms):
        raise ContractError("primary and guardrail arms must match")
    if set(guardrail_results["outcome"]) != set(DEFAULT_GUARDRAILS):
        raise ContractError("guardrail results must contain the declared two outcomes")
    if guardrail_results["noninferiority_pass"].isna().any():
        raise ContractError("guardrail pass indicators must be complete")
    if not pd.api.types.is_bool_dtype(guardrail_results["noninferiority_pass"]):
        raise ContractError("guardrail pass indicators must be boolean")
    guardrail_method_columns = [
        "point_estimator",
        "uncertainty_method",
        "block_adjusted_uncertainty",
    ]
    if guardrail_results[guardrail_method_columns].isna().any().any():
        raise ContractError("guardrail method metadata must be complete")
    if not guardrail_results["point_estimator"].eq(POOLED_GUARDRAIL_POINT).all():
        raise ContractError("guardrail point-estimator metadata is inconsistent")
    if not guardrail_results["uncertainty_method"].eq(POOLED_GUARDRAIL_BOUND).all():
        raise ContractError("guardrail uncertainty metadata is inconsistent")
    if not pd.api.types.is_bool_dtype(guardrail_results["block_adjusted_uncertainty"]):
        raise ContractError("guardrail block-adjustment indicators must be boolean")
    if guardrail_results["block_adjusted_uncertainty"].any():
        raise ContractError("guardrail uncertainty must remain explicitly unadjusted")
    guardrail_numeric = guardrail_results[
        [
            "noninferiority_margin",
            "simultaneous_one_sided_upper_bound",
            "family_size",
            "family_alpha",
            "bonferroni_alpha_each",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if guardrail_numeric.isna().any().any() or not np.isfinite(
        guardrail_numeric
    ).all().all():
        raise ContractError("guardrail decision fields must be finite numbers")
    if not guardrail_numeric["family_size"].eq(expected_guardrail_rows).all():
        raise ContractError("guardrails must declare one 12-comparison family")
    if not np.isclose(
        guardrail_numeric["family_alpha"], 0.05, rtol=0.0, atol=1e-15
    ).all():
        raise ContractError("guardrails must use family alpha 0.05")
    expected_tail_alpha = guardrail_numeric["family_alpha"] / expected_guardrail_rows
    if not np.isclose(
        guardrail_numeric["bonferroni_alpha_each"],
        expected_tail_alpha,
        rtol=0.0,
        atol=1e-15,
    ).all():
        raise ContractError("guardrail Bonferroni allocation is inconsistent")
    if not guardrail_numeric["noninferiority_margin"].between(
        0.0, 1.0, inclusive="left"
    ).all():
        raise ContractError("guardrail margins must be in [0, 1)")
    expected_guardrail_pass = guardrail_numeric[
        "simultaneous_one_sided_upper_bound"
    ] < guardrail_numeric["noninferiority_margin"]
    if not guardrail_results["noninferiority_pass"].eq(
        expected_guardrail_pass
    ).all():
        raise ContractError("guardrail pass indicators are inconsistent with the bounds")

    failed_quality_gates = tuple(
        sorted(name for name, passed in quality_gates.items() if not passed)
    )
    reasons = [f"quality_gate_failed:{name}" for name in failed_quality_gates]
    if candidate_arm is None:
        reasons.append("pre_specified_candidate_required")
    elif candidate_arm not in set(primary_arms):
        raise ContractError("candidate_arm is not one of the active arms")
    else:
        primary_row = primary_results.loc[
            primary_results["active_arm"] == candidate_arm
        ].iloc[0]
        if not bool(primary_row["superiority_pass"]):
            reasons.append("primary_superiority_not_established")
        candidate_guardrails = guardrail_results.loc[
            guardrail_results["active_arm"] == candidate_arm
        ]
        failed_guardrails = candidate_guardrails.loc[
            ~candidate_guardrails["noninferiority_pass"].astype(bool), "outcome"
        ]
        reasons.extend(
            f"guardrail_noninferiority_not_established:{outcome}"
            for outcome in sorted(failed_guardrails)
        )

    status: Literal["criteria_met", "continue_testing"]
    status = "criteria_met" if not reasons else "continue_testing"
    return DecisionResult(
        status=status,
        candidate_arm=candidate_arm,
        scope="synthetic_method_check_not_a_rollout_recommendation",
        reasons=tuple(reasons),
        failed_quality_gates=failed_quality_gates,
    )


__all__ = [
    "DEFAULT_GUARDRAILS",
    "DecisionResult",
    "REQUIRED_QUALITY_GATES",
    "evaluate_guardrail_family",
    "evaluate_primary_family",
    "make_launch_decision",
]
