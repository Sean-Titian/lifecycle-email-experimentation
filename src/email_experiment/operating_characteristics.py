"""Vectorized operating characteristics for the prospective synthetic design.

The benchmark in this module is deliberately aggregate-only.  It draws block by
arm event counts directly from the declared Bernoulli data-generating process,
reuses one concurrent holdout in all six active-arm contrasts, and applies the
same statistical rules as :mod:`email_experiment.decision`.  It is a design
diagnostic, not evidence about a real campaign and not a rollout authorization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import erfc, isfinite, sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from .assignment import ACTIVE_ARMS, ALL_ARMS, BLOCK_COLUMNS, HOLDOUT_ARM
from .contracts import ContractError
from .decision import (
    DEFAULT_GUARDRAILS,
    REQUIRED_QUALITY_GATES,
    evaluate_guardrail_family,
    evaluate_primary_family,
    make_launch_decision,
)

SCHEMA_VERSION = "1.0.0"
SIMULATION_CONTRACT_VERSION = "lifecycle-operating-characteristics-v1"
SCENARIO_IDS = (
    "global_null",
    "configured_small_effects",
    "planning_reference",
    "strong_effect_safe_guardrails",
    "strong_effect_guardrails_at_margins",
)
LIFECYCLE_SEGMENTS = ("new", "established", "reactivation")
TENURE_BANDS = ("0_30d", "31_90d")
CANDIDATE_ARM = "challenger_a__twice_weekly"
PLANNING_REFERENCE_EFFECT = 0.027839028056397323
STRONG_EFFECT = 0.08

_SEGMENT_FUNDING = {"new": -0.006, "established": 0.0, "reactivation": 0.008}
_TENURE_FUNDING = {"0_30d": -0.002, "31_90d": 0.002}
_SEGMENT_PREPERIOD = {"new": -0.08, "established": 0.0, "reactivation": 0.08}
_TENURE_PREPERIOD = {"0_30d": -0.02, "31_90d": 0.02}


@dataclass(frozen=True)
class OperatingCharacteristicsConfig:
    """Frozen public parameters for the aggregate simulation benchmark."""

    seed: int = 20261006
    replications: int = 20_000
    units_per_arm_per_block: int = 100
    assignment_waves: int = 2
    family_alpha: float = 0.05
    monte_carlo_confidence: float = 0.99
    unsubscribe_noninferiority_margin: float = 0.005
    complaint_noninferiority_margin: float = 0.003
    pre_specified_candidate_arm: str = CANDIDATE_ARM


@dataclass(frozen=True)
class _Scenario:
    scenario_id: str
    description: str
    candidate_funding_effect: float
    challenger_a_funding_effect: float
    challenger_b_funding_effect: float
    daily_funding_effect: float
    active_unsubscribe_effect: float
    daily_unsubscribe_effect: float
    active_complaint_effect: float
    daily_complaint_effect: float
    centered_funding_heterogeneity: bool = False
    guardrails_at_margins: bool = False


@dataclass(frozen=True)
class _FamilyArrays:
    risk_difference: np.ndarray
    standard_error: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    raw_p: np.ndarray
    holm_p: np.ndarray
    holm_reject: np.ndarray
    superiority_pass: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class _GuardrailArrays:
    risk_difference: np.ndarray
    upper_bound: np.ndarray
    noninferiority_pass: np.ndarray


@dataclass(frozen=True)
class _SimulationArrays:
    primary: _FamilyArrays
    negative_control: _FamilyArrays
    guardrails: dict[str, _GuardrailArrays]
    invalid: np.ndarray
    criteria_met: np.ndarray
    primary_counts: np.ndarray | None = None
    negative_control_counts: np.ndarray | None = None
    guardrail_counts: dict[str, np.ndarray] | None = None


def _validate_config(config: OperatingCharacteristicsConfig) -> None:
    for name, value in {
        "seed": config.seed,
        "replications": config.replications,
        "units_per_arm_per_block": config.units_per_arm_per_block,
        "assignment_waves": config.assignment_waves,
    }.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ContractError(f"{name} must be an integer")
    if config.seed < 0:
        raise ContractError("seed must be non-negative")
    if config.replications < 2:
        raise ContractError("replications must be at least two")
    if config.units_per_arm_per_block < 2:
        raise ContractError("units_per_arm_per_block must be at least two")
    if config.assignment_waves < 1:
        raise ContractError("assignment_waves must be positive")
    if config.family_alpha != 0.05:
        raise ContractError("the frozen family alpha must be 0.05")
    if config.monte_carlo_confidence != 0.99:
        raise ContractError("the frozen Monte Carlo confidence level must be 0.99")
    if config.pre_specified_candidate_arm != CANDIDATE_ARM:
        raise ContractError("the frozen candidate must be challenger_a__twice_weekly")
    for name, value in {
        "unsubscribe_noninferiority_margin": (
            config.unsubscribe_noninferiority_margin
        ),
        "complaint_noninferiority_margin": config.complaint_noninferiority_margin,
    }.items():
        if isinstance(value, bool) or not isfinite(value) or not 0.0 <= value < 1.0:
            raise ContractError(f"{name} must be finite and in [0, 1)")


def _block_values(config: OperatingCharacteristicsConfig) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (segment, tenure, f"wave_{wave + 1:02d}")
        for segment in LIFECYCLE_SEGMENTS
        for tenure in TENURE_BANDS
        for wave in range(config.assignment_waves)
    )


def _scenario_definitions(config: OperatingCharacteristicsConfig) -> tuple[_Scenario, ...]:
    return (
        _Scenario(
            scenario_id="global_null",
            description=(
                "All arms share each block's outcome probabilities; this diagnoses "
                "primary and negative-control family-wise false-positive behavior."
            ),
            candidate_funding_effect=0.0,
            challenger_a_funding_effect=0.0,
            challenger_b_funding_effect=0.0,
            daily_funding_effect=0.0,
            active_unsubscribe_effect=0.0,
            daily_unsubscribe_effect=0.0,
            active_complaint_effect=0.0,
            daily_complaint_effect=0.0,
        ),
        _Scenario(
            scenario_id="configured_small_effects",
            description="The exact effect configuration used by the prospective fixture.",
            candidate_funding_effect=0.002,
            challenger_a_funding_effect=0.002,
            challenger_b_funding_effect=-0.001,
            daily_funding_effect=0.0005,
            active_unsubscribe_effect=0.0002,
            daily_unsubscribe_effect=0.0003,
            active_complaint_effect=0.0001,
            daily_complaint_effect=0.0001,
        ),
        _Scenario(
            scenario_id="planning_reference",
            description=(
                "The candidate receives the frozen 2.7839028 percentage-point planning "
                "effect; funding block heterogeneity is centered and guardrails are safe."
            ),
            candidate_funding_effect=PLANNING_REFERENCE_EFFECT,
            challenger_a_funding_effect=0.0,
            challenger_b_funding_effect=0.0,
            daily_funding_effect=0.0,
            active_unsubscribe_effect=0.0,
            daily_unsubscribe_effect=0.0,
            active_complaint_effect=0.0,
            daily_complaint_effect=0.0,
            centered_funding_heterogeneity=True,
        ),
        _Scenario(
            scenario_id="strong_effect_safe_guardrails",
            description=(
                "The candidate receives an eight percentage-point funding effect while "
                "all active-arm guardrails equal holdout risk."
            ),
            candidate_funding_effect=STRONG_EFFECT,
            challenger_a_funding_effect=0.0,
            challenger_b_funding_effect=0.0,
            daily_funding_effect=0.0,
            active_unsubscribe_effect=0.0,
            daily_unsubscribe_effect=0.0,
            active_complaint_effect=0.0,
            daily_complaint_effect=0.0,
        ),
        _Scenario(
            scenario_id="strong_effect_guardrails_at_margins",
            description=(
                "The candidate receives an eight percentage-point funding effect and all "
                "active-arm guardrail risks sit exactly at their non-inferiority margins."
            ),
            candidate_funding_effect=STRONG_EFFECT,
            challenger_a_funding_effect=0.0,
            challenger_b_funding_effect=0.0,
            daily_funding_effect=0.0,
            active_unsubscribe_effect=config.unsubscribe_noninferiority_margin,
            daily_unsubscribe_effect=0.0,
            active_complaint_effect=config.complaint_noninferiority_margin,
            daily_complaint_effect=0.0,
            guardrails_at_margins=True,
        ),
    )


def _stream_seed(base_seed: int, scenario_id: str, outcome: str) -> int:
    payload = "|".join(
        (SIMULATION_CONTRACT_VERSION, str(base_seed), scenario_id, outcome)
    ).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:16], byteorder="big")


def _rng(base_seed: int, scenario_id: str, outcome: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(_stream_seed(base_seed, scenario_id, outcome)))


def _probability_matrices(
    config: OperatingCharacteristicsConfig,
    scenario: _Scenario,
) -> dict[str, np.ndarray]:
    blocks = _block_values(config)
    block_count = len(blocks)
    arm_count = len(ALL_ARMS)
    funding = np.empty((block_count, arm_count), dtype=float)
    preperiod = np.empty_like(funding)
    unsubscribe = np.empty_like(funding)
    complaint = np.empty_like(funding)

    segment_center = float(np.mean(tuple(_SEGMENT_FUNDING.values())))
    for block_index, (segment, tenure, _) in enumerate(blocks):
        funding_block_effect = _SEGMENT_FUNDING[segment] + _TENURE_FUNDING[tenure]
        if scenario.centered_funding_heterogeneity:
            funding_block_effect -= segment_center
        preperiod_probability = (
            0.25 + _SEGMENT_PREPERIOD[segment] + _TENURE_PREPERIOD[tenure]
        )
        for arm_index, arm in enumerate(ALL_ARMS):
            active = arm != HOLDOUT_ARM
            if active:
                content, cadence = arm.split("__", maxsplit=1)
                content_effect = {
                    "current": 0.0,
                    "challenger_a": scenario.challenger_a_funding_effect,
                    "challenger_b": scenario.challenger_b_funding_effect,
                }[content]
                if arm == CANDIDATE_ARM:
                    content_effect = scenario.candidate_funding_effect
                cadence_effect = scenario.daily_funding_effect if cadence == "daily" else 0.0
                unsubscribe_effect = scenario.active_unsubscribe_effect + (
                    scenario.daily_unsubscribe_effect if cadence == "daily" else 0.0
                )
                complaint_effect = scenario.active_complaint_effect + (
                    scenario.daily_complaint_effect if cadence == "daily" else 0.0
                )
            else:
                content_effect = 0.0
                cadence_effect = 0.0
                unsubscribe_effect = 0.0
                complaint_effect = 0.0
            funding[block_index, arm_index] = (
                0.04 + funding_block_effect + content_effect + cadence_effect
            )
            preperiod[block_index, arm_index] = preperiod_probability
            unsubscribe[block_index, arm_index] = 0.004 + unsubscribe_effect
            complaint[block_index, arm_index] = 0.001 + complaint_effect

    matrices = {
        "funded_14d": funding,
        "pre_period_engaged": preperiod,
        "unsubscribe_14d": unsubscribe,
        "complaint_14d": complaint,
    }
    for outcome, probabilities in matrices.items():
        if not np.isfinite(probabilities).all() or (
            (probabilities < 0.0) | (probabilities > 1.0)
        ).any():
            raise ContractError(f"{scenario.scenario_id} has invalid {outcome} probabilities")
    return matrices


def _draw_counts(
    config: OperatingCharacteristicsConfig,
    scenario_id: str,
    outcome: str,
    probabilities: np.ndarray,
) -> np.ndarray:
    return _rng(config.seed, scenario_id, outcome).binomial(
        config.units_per_arm_per_block,
        probabilities,
        size=(config.replications, *probabilities.shape),
    )


def _two_sided_normal_p(z_statistics: np.ndarray) -> np.ndarray:
    flat = np.abs(np.asarray(z_statistics, dtype=float)).ravel() / sqrt(2.0)
    values = np.fromiter((erfc(float(value)) for value in flat), dtype=float, count=flat.size)
    return values.reshape(z_statistics.shape)


def _holm_rows(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(p_values, order, axis=1)
    multipliers = np.arange(p_values.shape[1], 0, -1, dtype=float)
    adjusted_sorted = np.minimum(
        1.0,
        np.maximum.accumulate(sorted_values * multipliers, axis=1),
    )
    adjusted = np.empty_like(adjusted_sorted)
    np.put_along_axis(adjusted, order, adjusted_sorted, axis=1)
    return adjusted


def _evaluate_block_family(
    counts: np.ndarray,
    *,
    units_per_arm_per_block: int,
    family_alpha: float,
) -> _FamilyArrays:
    n = float(units_per_arm_per_block)
    active = counts[:, :, 1:].astype(float)
    control = counts[:, :, [0]].astype(float)
    active_rates = active / n
    control_rates = control / n
    risk_difference = (active_rates - control_rates).mean(axis=1)

    active_variance = active_rates * (1.0 - active_rates) * n / (n - 1.0)
    control_variance = control_rates * (1.0 - control_rates) * n / (n - 1.0)
    block_count = counts.shape[1]
    variance = (
        (active_variance / n + control_variance / n).sum(axis=1) / block_count**2
    )
    valid = np.isfinite(variance) & (variance > 0.0)
    standard_error = np.sqrt(np.where(valid, variance, np.nan))
    z_statistic = np.divide(
        risk_difference,
        standard_error,
        out=np.zeros_like(risk_difference),
        where=valid,
    )
    raw_p = _two_sided_normal_p(z_statistic)
    raw_p = np.where(valid, raw_p, 1.0)
    holm_p = _holm_rows(raw_p)
    z_critical = NormalDist().inv_cdf(0.975)
    ci_low = np.maximum(-1.0, risk_difference - z_critical * standard_error)
    ci_high = np.minimum(1.0, risk_difference + z_critical * standard_error)
    holm_reject = valid & (holm_p < family_alpha)
    superiority_pass = holm_reject & (risk_difference > 0.0)
    return _FamilyArrays(
        risk_difference=risk_difference,
        standard_error=standard_error,
        ci_low=ci_low,
        ci_high=ci_high,
        raw_p=raw_p,
        holm_p=holm_p,
        holm_reject=holm_reject,
        superiority_pass=superiority_pass,
        valid=valid,
    )


def _wilson_limits(
    successes: np.ndarray,
    total: int,
    *,
    tail_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    rate = successes / total
    z = NormalDist().inv_cdf(1.0 - tail_alpha)
    denominator = 1.0 + z**2 / total
    center = (rate + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(rate * (1.0 - rate) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return np.maximum(0.0, center - half_width), np.minimum(1.0, center + half_width)


def _evaluate_guardrail_counts(
    counts: np.ndarray,
    *,
    units_per_arm_per_block: int,
    family_alpha: float,
    margin: float,
) -> _GuardrailArrays:
    totals = counts.sum(axis=1)
    total_per_arm = counts.shape[1] * units_per_arm_per_block
    active_successes = totals[:, 1:].astype(float)
    control_successes = totals[:, [0]].astype(float)
    active_rate = active_successes / total_per_arm
    control_rate = control_successes / total_per_arm
    risk_difference = active_rate - control_rate
    tail_alpha = family_alpha / (len(ACTIVE_ARMS) * len(DEFAULT_GUARDRAILS))
    _, active_upper = _wilson_limits(
        active_successes,
        total_per_arm,
        tail_alpha=tail_alpha,
    )
    control_lower, _ = _wilson_limits(
        control_successes,
        total_per_arm,
        tail_alpha=tail_alpha,
    )
    upper_bound = np.minimum(
        1.0,
        risk_difference
        + np.sqrt((active_upper - active_rate) ** 2 + (control_rate - control_lower) ** 2),
    )
    return _GuardrailArrays(
        risk_difference=risk_difference,
        upper_bound=upper_bound,
        noninferiority_pass=upper_bound < margin,
    )


def _simulate_scenario(
    config: OperatingCharacteristicsConfig,
    scenario: _Scenario,
    *,
    retain_counts: bool = False,
) -> _SimulationArrays:
    probabilities = _probability_matrices(config, scenario)
    primary_counts = _draw_counts(
        config, scenario.scenario_id, "funded_14d", probabilities["funded_14d"]
    )
    negative_counts = _draw_counts(
        config,
        scenario.scenario_id,
        "pre_period_engaged",
        probabilities["pre_period_engaged"],
    )
    primary = _evaluate_block_family(
        primary_counts,
        units_per_arm_per_block=config.units_per_arm_per_block,
        family_alpha=config.family_alpha,
    )
    negative = _evaluate_block_family(
        negative_counts,
        units_per_arm_per_block=config.units_per_arm_per_block,
        family_alpha=config.family_alpha,
    )

    guardrail_counts: dict[str, np.ndarray] = {}
    guardrails: dict[str, _GuardrailArrays] = {}
    margins = {
        "unsubscribe_14d": config.unsubscribe_noninferiority_margin,
        "complaint_14d": config.complaint_noninferiority_margin,
    }
    for outcome in DEFAULT_GUARDRAILS:
        outcome_counts = _draw_counts(
            config, scenario.scenario_id, outcome, probabilities[outcome]
        )
        guardrail_counts[outcome] = outcome_counts
        guardrails[outcome] = _evaluate_guardrail_counts(
            outcome_counts,
            units_per_arm_per_block=config.units_per_arm_per_block,
            family_alpha=config.family_alpha,
            margin=margins[outcome],
        )

    invalid = (~primary.valid.all(axis=1)) | (~negative.valid.all(axis=1))
    candidate_index = ACTIVE_ARMS.index(CANDIDATE_ARM)
    negative_control_pass = ~negative.holm_reject.any(axis=1)
    candidate_guardrails_pass = np.logical_and.reduce(
        tuple(
            guardrails[outcome].noninferiority_pass[:, candidate_index]
            for outcome in DEFAULT_GUARDRAILS
        )
    )
    criteria_met = (
        ~invalid
        & negative_control_pass
        & primary.superiority_pass[:, candidate_index]
        & candidate_guardrails_pass
    )
    return _SimulationArrays(
        primary=primary,
        negative_control=negative,
        guardrails=guardrails,
        invalid=invalid,
        criteria_met=criteria_met,
        primary_counts=primary_counts if retain_counts else None,
        negative_control_counts=negative_counts if retain_counts else None,
        guardrail_counts=guardrail_counts if retain_counts else None,
    )


def _wilson_interval(successes: int, total: int, confidence: float) -> tuple[float, float]:
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    rate = successes / total
    denominator = 1.0 + z**2 / total
    center = (rate + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * sqrt(rate * (1.0 - rate) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return (
        min(rate, max(0.0, center - half_width)),
        max(rate, min(1.0, center + half_width)),
    )


def _rate_summary(values: np.ndarray, *, confidence: float) -> dict[str, float | int]:
    boolean = np.asarray(values, dtype=bool)
    successes = int(boolean.sum())
    total = int(boolean.size)
    low, high = _wilson_interval(successes, total, confidence)
    return {
        "successes": successes,
        "replications": total,
        "rate": successes / total,
        "monte_carlo_confidence": confidence,
        "monte_carlo_ci_method": "wilson_score",
        "monte_carlo_ci_low": low,
        "monte_carlo_ci_high": high,
    }


def _scenario_report(
    config: OperatingCharacteristicsConfig,
    scenario: _Scenario,
    arrays: _SimulationArrays,
) -> dict[str, Any]:
    candidate_index = ACTIVE_ARMS.index(CANDIDATE_ARM)
    probabilities = _probability_matrices(config, scenario)
    candidate_effect = arrays.primary.risk_difference[:, candidate_index]
    true_effects = (
        probabilities["funded_14d"][:, 1:].mean(axis=0)
        - probabilities["funded_14d"][:, [0]].mean(axis=0)
    )
    true_effect = float(true_effects[candidate_index])
    candidate_coverage = (
        arrays.primary.valid[:, candidate_index]
        & (arrays.primary.ci_low[:, candidate_index] <= true_effect)
        & (arrays.primary.ci_high[:, candidate_index] >= true_effect)
    )
    guardrail_family_any_noninferiority = np.logical_or.reduce(
        tuple(
            arrays.guardrails[outcome].noninferiority_pass.any(axis=1)
            for outcome in DEFAULT_GUARDRAILS
        )
    )
    negative_control_reject = arrays.negative_control.holm_reject.any(axis=1)
    metrics: dict[str, Any] = {
        "invalid_replicate": _rate_summary(
            arrays.invalid, confidence=config.monte_carlo_confidence
        ),
        "primary_family_any_holm_rejection": _rate_summary(
            (~arrays.invalid) & arrays.primary.holm_reject.any(axis=1),
            confidence=config.monte_carlo_confidence,
        ),
        "primary_family_any_superiority": _rate_summary(
            (~arrays.invalid) & arrays.primary.superiority_pass.any(axis=1),
            confidence=config.monte_carlo_confidence,
        ),
        "candidate_superiority": _rate_summary(
            (~arrays.invalid) & arrays.primary.superiority_pass[:, candidate_index],
            confidence=config.monte_carlo_confidence,
        ),
        "candidate_nominal_95_ci_coverage": _rate_summary(
            (~arrays.invalid) & candidate_coverage,
            confidence=config.monte_carlo_confidence,
        ),
        "nominal_95_ci_coverage_by_active_arm": {
            arm: _rate_summary(
                (~arrays.invalid)
                & arrays.primary.valid[:, arm_index]
                & (arrays.primary.ci_low[:, arm_index] <= true_effects[arm_index])
                & (arrays.primary.ci_high[:, arm_index] >= true_effects[arm_index]),
                confidence=config.monte_carlo_confidence,
            )
            for arm_index, arm in enumerate(ACTIVE_ARMS)
        },
        "negative_control_family_any_holm_rejection": _rate_summary(
            (~arrays.invalid) & negative_control_reject,
            confidence=config.monte_carlo_confidence,
        ),
        "negative_control_gate_pass": _rate_summary(
            (~arrays.invalid) & ~negative_control_reject,
            confidence=config.monte_carlo_confidence,
        ),
        "guardrail_family_any_noninferiority_pass": _rate_summary(
            (~arrays.invalid) & guardrail_family_any_noninferiority,
            confidence=config.monte_carlo_confidence,
        ),
        "policy_criteria_met": _rate_summary(
            arrays.criteria_met, confidence=config.monte_carlo_confidence
        ),
        "policy_continue_testing": _rate_summary(
            ~arrays.criteria_met, confidence=config.monte_carlo_confidence
        ),
        "candidate_guardrail_noninferiority": {
            outcome: _rate_summary(
                (~arrays.invalid)
                & arrays.guardrails[outcome].noninferiority_pass[:, candidate_index],
                confidence=config.monte_carlo_confidence,
            )
            for outcome in DEFAULT_GUARDRAILS
        },
        "policy_blocking_reason_frequency": {
            "invalid_inference": _rate_summary(
                arrays.invalid, confidence=config.monte_carlo_confidence
            ),
            "pre_period_negative_control": _rate_summary(
                (~arrays.invalid) & negative_control_reject,
                confidence=config.monte_carlo_confidence,
            ),
            "primary_superiority_not_established": _rate_summary(
                (~arrays.invalid)
                & ~arrays.primary.superiority_pass[:, candidate_index],
                confidence=config.monte_carlo_confidence,
            ),
            **{
                f"guardrail_noninferiority_not_established:{outcome}": _rate_summary(
                    (~arrays.invalid)
                    & ~arrays.guardrails[outcome].noninferiority_pass[:, candidate_index],
                    confidence=config.monte_carlo_confidence,
                )
                for outcome in DEFAULT_GUARDRAILS
            },
        },
    }
    return {
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "replications_requested": config.replications,
        "replications_in_every_denominator": config.replications,
        "data_generating_process": {
            "candidate_arm": CANDIDATE_ARM,
            "candidate_true_funding_risk_difference": true_effect,
            "funding_probability_range": [
                float(probabilities["funded_14d"].min()),
                float(probabilities["funded_14d"].max()),
            ],
            "pre_period_probability_range": [
                float(probabilities["pre_period_engaged"].min()),
                float(probabilities["pre_period_engaged"].max()),
            ],
            "candidate_true_guardrail_risk_differences": {
                outcome: float(
                    probabilities[outcome][:, 1 + candidate_index].mean()
                    - probabilities[outcome][:, 0].mean()
                )
                for outcome in DEFAULT_GUARDRAILS
            },
            "centered_funding_block_heterogeneity": (
                scenario.centered_funding_heterogeneity
            ),
            "all_active_guardrails_at_noninferiority_margins": (
                scenario.guardrails_at_margins
            ),
        },
        "candidate_effect_diagnostics": {
            "mean_estimated_risk_difference": float(candidate_effect.mean()),
            "monte_carlo_bias": float(candidate_effect.mean() - true_effect),
            "empirical_standard_deviation": float(candidate_effect.std(ddof=1)),
        },
        "operating_characteristics": metrics,
    }


def _frame_from_counts(
    config: OperatingCharacteristicsConfig,
    primary_counts: np.ndarray,
    negative_counts: np.ndarray,
    guardrail_counts: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    blocks = _block_values(config)
    n = config.units_per_arm_per_block
    for block_index, block in enumerate(blocks):
        for arm_index, arm in enumerate(ALL_ARMS):
            if arm == HOLDOUT_ARM:
                content: str | None = None
                cadence: str | None = None
            else:
                content, cadence = arm.split("__", maxsplit=1)
            counts = {
                "funded_14d": int(primary_counts[block_index, arm_index]),
                "pre_period_engaged": int(negative_counts[block_index, arm_index]),
                **{
                    outcome: int(guardrail_counts[outcome][block_index, arm_index])
                    for outcome in DEFAULT_GUARDRAILS
                },
            }
            for row_index in range(n):
                rows.append(
                    {
                        "participant_id": (
                            f"parity_{block_index:02d}_{arm_index:02d}_{row_index:04d}"
                        ),
                        "lifecycle_segment": block[0],
                        "tenure_band": block[1],
                        "assignment_wave": block[2],
                        "arm": arm,
                        "content": content,
                        "cadence": cadence,
                        **{
                            outcome: int(row_index < successes)
                            for outcome, successes in counts.items()
                        },
                    }
                )
    return pd.DataFrame(rows)


def production_api_parity_check(
    config: OperatingCharacteristicsConfig | None = None,
    *,
    replications: int = 12,
) -> dict[str, Any]:
    """Compare fixed vectorized replicates with the production decision APIs.

    The check reconstructs only ephemeral synthetic rows.  Its returned value is
    aggregate: maximum numeric error and exact Boolean/status agreement.
    """

    base = config or OperatingCharacteristicsConfig()
    parity_config = OperatingCharacteristicsConfig(
        **{
            **asdict(base),
            "replications": replications,
            "units_per_arm_per_block": max(base.units_per_arm_per_block, 100),
        }
    )
    _validate_config(parity_config)
    scenario = next(
        item
        for item in _scenario_definitions(parity_config)
        if item.scenario_id == "configured_small_effects"
    )
    arrays = _simulate_scenario(parity_config, scenario, retain_counts=True)
    if (
        arrays.primary_counts is None
        or arrays.negative_control_counts is None
        or arrays.guardrail_counts is None
    ):
        raise AssertionError("parity simulation did not retain counts")

    numeric_errors: list[float] = []
    exact_booleans = True
    exact_decisions = True
    candidate_index = ACTIVE_ARMS.index(CANDIDATE_ARM)
    margins = {
        "unsubscribe_14d": parity_config.unsubscribe_noninferiority_margin,
        "complaint_14d": parity_config.complaint_noninferiority_margin,
    }
    for replicate in range(replications):
        frame = _frame_from_counts(
            parity_config,
            arrays.primary_counts[replicate],
            arrays.negative_control_counts[replicate],
            {
                outcome: counts[replicate]
                for outcome, counts in arrays.guardrail_counts.items()
            },
        )
        primary = evaluate_primary_family(frame, block_cols=BLOCK_COLUMNS)
        negative = evaluate_primary_family(
            frame,
            outcome_col="pre_period_engaged",
            block_cols=BLOCK_COLUMNS,
        )
        guardrails = evaluate_guardrail_family(frame, margins=margins)

        for production, vectorized in (
            (primary, arrays.primary),
            (negative, arrays.negative_control),
        ):
            production = production.set_index("active_arm").loc[list(ACTIVE_ARMS)]
            for column, values in (
                ("risk_difference", vectorized.risk_difference[replicate]),
                (
                    "risk_difference_standard_error",
                    vectorized.standard_error[replicate],
                ),
                ("risk_difference_ci_low_nominal", vectorized.ci_low[replicate]),
                ("risk_difference_ci_high_nominal", vectorized.ci_high[replicate]),
                ("p_value_two_sided", vectorized.raw_p[replicate]),
                ("p_value_holm", vectorized.holm_p[replicate]),
            ):
                numeric_errors.append(
                    float(np.max(np.abs(production[column].to_numpy(dtype=float) - values)))
                )
            exact_booleans &= np.array_equal(
                production["holm_significant"].to_numpy(dtype=bool),
                vectorized.holm_reject[replicate],
            )
            exact_booleans &= np.array_equal(
                production["superiority_pass"].to_numpy(dtype=bool),
                vectorized.superiority_pass[replicate],
            )

        for outcome in DEFAULT_GUARDRAILS:
            production_outcome = (
                guardrails.loc[guardrails["outcome"] == outcome]
                .set_index("active_arm")
                .loc[list(ACTIVE_ARMS)]
            )
            vectorized = arrays.guardrails[outcome]
            numeric_errors.append(
                float(
                    np.max(
                        np.abs(
                            production_outcome[
                                "simultaneous_one_sided_upper_bound"
                            ].to_numpy(dtype=float)
                            - vectorized.upper_bound[replicate]
                        )
                    )
                )
            )
            exact_booleans &= np.array_equal(
                production_outcome["noninferiority_pass"].to_numpy(dtype=bool),
                vectorized.noninferiority_pass[replicate],
            )

        negative_pass = not bool(negative["holm_significant"].any())
        quality_gates = {name: True for name in REQUIRED_QUALITY_GATES}
        quality_gates["pre_period_negative_control"] = negative_pass
        production_decision = make_launch_decision(
            primary,
            guardrails,
            quality_gates,
            candidate_arm=CANDIDATE_ARM,
        )
        expected_status = "criteria_met" if arrays.criteria_met[replicate] else "continue_testing"
        expected_reasons: list[str] = []
        if not negative_pass:
            expected_reasons.append("quality_gate_failed:pre_period_negative_control")
        if not arrays.primary.superiority_pass[replicate, candidate_index]:
            expected_reasons.append("primary_superiority_not_established")
        expected_reasons.extend(
            f"guardrail_noninferiority_not_established:{outcome}"
            for outcome in sorted(DEFAULT_GUARDRAILS)
            if not arrays.guardrails[outcome].noninferiority_pass[
                replicate, candidate_index
            ]
        )
        exact_decisions &= production_decision.status == expected_status
        exact_decisions &= production_decision.reasons == tuple(expected_reasons)

    maximum_error = max(numeric_errors, default=0.0)
    tolerance = 1e-12
    return {
        "scenario_id": scenario.scenario_id,
        "replications": replications,
        "units_per_arm_per_block": parity_config.units_per_arm_per_block,
        "numeric_tolerance": tolerance,
        "maximum_absolute_numeric_error": maximum_error,
        "numeric_parity_passed": maximum_error <= tolerance,
        "boolean_parity_passed": bool(exact_booleans),
        "decision_parity_passed": bool(exact_decisions),
        "passed": bool(maximum_error <= tolerance and exact_booleans and exact_decisions),
    }


def _calibration_gates(
    reports: dict[str, dict[str, Any]],
    parity: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    global_metrics = reports["global_null"]["operating_characteristics"]
    strong_safe = reports["strong_effect_safe_guardrails"]["operating_characteristics"]
    boundary = reports["strong_effect_guardrails_at_margins"][
        "operating_characteristics"
    ]
    negative_upper = max(
        report["operating_characteristics"][
            "negative_control_family_any_holm_rejection"
        ]["monte_carlo_ci_high"]
        for report in reports.values()
    )
    boundary_family_upper = boundary["guardrail_family_any_noninferiority_pass"][
        "monte_carlo_ci_high"
    ]
    global_coverage_lower = min(
        metric["monte_carlo_ci_low"]
        for metric in global_metrics["nominal_95_ci_coverage_by_active_arm"].values()
    )
    invalid_count = sum(
        report["operating_characteristics"]["invalid_replicate"]["successes"]
        for report in reports.values()
    )
    return {
        "global_null_primary_fwer": {
            "criterion": "99% Monte Carlo upper bound <= 0.06",
            "observed": global_metrics["primary_family_any_holm_rejection"][
                "monte_carlo_ci_high"
            ],
            "passed": global_metrics["primary_family_any_holm_rejection"][
                "monte_carlo_ci_high"
            ]
            <= 0.06,
        },
        "negative_control_fwer_all_scenarios": {
            "criterion": "largest 99% Monte Carlo upper bound <= 0.06",
            "observed": negative_upper,
            "passed": negative_upper <= 0.06,
        },
        "global_null_nominal_interval_coverage": {
            "criterion": "smallest active-arm 99% Monte Carlo lower bound >= 0.93",
            "observed": global_coverage_lower,
            "passed": global_coverage_lower >= 0.93,
        },
        "strong_effect_candidate_detection": {
            "criterion": "99% Monte Carlo lower bound >= 0.99",
            "observed": strong_safe["candidate_superiority"]["monte_carlo_ci_low"],
            "passed": strong_safe["candidate_superiority"][
                "monte_carlo_ci_low"
            ]
            >= 0.99,
        },
        "guardrail_margin_boundary_family_false_noninferiority": {
            "criterion": "family-any 99% Monte Carlo upper bound <= 0.06",
            "observed": boundary_family_upper,
            "passed": boundary_family_upper <= 0.06,
        },
        "no_invalid_replicates": {
            "criterion": "zero invalid replicates across every requested denominator",
            "observed": invalid_count,
            "passed": invalid_count == 0,
        },
        "production_api_parity": {
            "criterion": "numeric error <= 1e-12 with exact Boolean and decision parity",
            "observed": parity["maximum_absolute_numeric_error"],
            "passed": bool(parity["passed"]),
        },
    }


def build_operating_characteristics_benchmark(
    config: OperatingCharacteristicsConfig | None = None,
) -> dict[str, Any]:
    """Run the five-scenario aggregate operating-characteristics benchmark."""

    active_config = config or OperatingCharacteristicsConfig()
    _validate_config(active_config)
    scenario_reports: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for scenario in _scenario_definitions(active_config):
        arrays = _simulate_scenario(active_config, scenario)
        report = _scenario_report(active_config, scenario, arrays)
        scenario_reports.append(report)
        by_id[scenario.scenario_id] = report

    parity = production_api_parity_check(active_config)
    gates = _calibration_gates(by_id, parity)
    planning_power = by_id["planning_reference"]["operating_characteristics"][
        "candidate_superiority"
    ]
    strong_policy = by_id["strong_effect_safe_guardrails"][
        "operating_characteristics"
    ]["policy_criteria_met"]
    readiness_reasons = [
        "This synthetic operating-characteristics artifact cannot authorize a rollout.",
    ]
    if strong_policy["monte_carlo_ci_low"] < 0.80:
        readiness_reasons.append(
            "Even an eight-point funding effect with safe true guardrail risks does not "
            "reach an 80% lower-bound policy-success target because the rare-event "
            "guardrail bounds remain imprecise."
        )
    native_config = {
        name: value.item() if isinstance(value, np.generic) else value
        for name, value in asdict(active_config).items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "synthetic_prospective_operating_characteristics_benchmark",
        "data_classification": "synthetic",
        "report_scope": "aggregate_only_monte_carlo",
        "artifact_scope": "aggregate_synthetic_design_diagnostic_not_campaign_evidence",
        "config": native_config,
        "simulation_contract": {
            "version": SIMULATION_CONTRACT_VERSION,
            "bit_generator": "PCG64",
            "stream_derivation": (
                "first 128 bits of SHA-256 over contract|seed|scenario|outcome"
            ),
            "shared_holdout": True,
            "active_arms": list(ACTIVE_ARMS),
            "block_columns": list(BLOCK_COLUMNS),
            "block_count": len(_block_values(active_config)),
            "primary_multiplicity": "six active-vs-holdout two-sided Holm family",
            "negative_control_multiplicity": (
                "six active-vs-holdout two-sided Holm family"
            ),
            "guardrail_uncertainty": (
                "pooled Newcombe-style one-sided bounds with 12-way Bonferroni"
            ),
            "invalid_replicate_policy": (
                "fail closed and retain every requested replicate in every rate denominator"
            ),
            "monte_carlo_intervals": "two-sided 99% Wilson score",
        },
        "scenarios": scenario_reports,
        "production_api_parity": parity,
        "calibration_gates": gates,
        "all_calibration_gates_passed": all(gate["passed"] for gate in gates.values()),
        "calibration_status": (
            "passed" if all(gate["passed"] for gate in gates.values()) else "failed"
        ),
        "planning_diagnostic": {
            "reference_effect": PLANNING_REFERENCE_EFFECT,
            "reference_effect_percentage_points": 100.0 * PLANNING_REFERENCE_EFFECT,
            "original_approximation": (
                "4% baseline-variance normal approximation with alpha/6 Bonferroni "
                "planning allocation and nominal 80% target power"
            ),
            "empirical_candidate_holm_superiority": planning_power,
            "interpretation": (
                "The empirical Holm power is the design diagnostic. The analytic value is "
                "a planning reference, not an achieved 80%-power MDE."
            ),
        },
        "design_readiness": {
            "status": "continue_testing",
            "reasons": readiness_reasons,
            "interpretation": (
                "Calibration checks method behavior; design readiness additionally requires "
                "adequate end-to-end decision probability and external data validation."
            ),
        },
        "limitations": [
            (
                "Results are conditional on the declared independent Bernoulli "
                "superpopulation data-generating process; they are not an arbitrary "
                "finite-population coverage guarantee."
            ),
            (
                "Every simulation assumes exact block allocation, complete follow-up, no "
                "contamination, and all non-statistical quality gates passing."
            ),
            (
                "The benchmark does not simulate assignment-ledger omissions, delayed or "
                "missing event feeds, censoring, join amplification, or real-world drift."
            ),
            "No result is evidence about a real campaign or a production rollout probability.",
        ],
    }


def write_operating_characteristics_benchmark(
    output_path: str | Path,
    config: OperatingCharacteristicsConfig | None = None,
) -> Path:
    """Write one deterministic, aggregate-only, synthetic JSON artifact."""

    destination = Path(output_path)
    lowered_name = destination.name.lower()
    if (
        destination.suffix.lower() != ".json"
        or "synthetic" not in lowered_name
        or "operating" not in lowered_name
    ):
        raise ContractError(
            "operating-characteristics output must be a synthetic operating JSON file"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = build_operating_characteristics_benchmark(config)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


__all__ = [
    "CANDIDATE_ARM",
    "OperatingCharacteristicsConfig",
    "PLANNING_REFERENCE_EFFECT",
    "SCENARIO_IDS",
    "STRONG_EFFECT",
    "build_operating_characteristics_benchmark",
    "production_api_parity_check",
    "write_operating_characteristics_benchmark",
]
