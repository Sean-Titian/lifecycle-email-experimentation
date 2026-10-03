"""Dependency-light statistics for randomized binary experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erfc, exp, log, sqrt
from statistics import NormalDist
from typing import Literal

import numpy as np
import pandas as pd

from .contracts import ContractError, require_columns, validate_binary

AdjustmentMethod = Literal["holm", "benjamini-hochberg"]


@dataclass(frozen=True)
class BinaryEffect:
    """Treatment-minus-control comparison for a binary outcome."""

    treatment_successes: int
    treatment_total: int
    control_successes: int
    control_total: int
    treatment_rate: float
    control_rate: float
    absolute_effect: float
    absolute_ci_low: float
    absolute_ci_high: float
    relative_risk: float
    relative_risk_ci_low: float
    relative_risk_ci_high: float
    z_statistic: float
    p_value_two_sided: float


@dataclass(frozen=True)
class BlockStandardizedEffect:
    """Block-standardized difference in binary-outcome risks.

    The variance is the conservative Neyman estimator for a stratified
    randomized design.  It does not assume one common risk difference across
    blocks and omits the unidentified finite-population treatment-effect variance term.
    """

    treatment_successes: int
    treatment_total: int
    control_successes: int
    control_total: int
    treatment_rate: float
    control_rate: float
    absolute_effect: float
    absolute_standard_error: float
    absolute_ci_low: float
    absolute_ci_high: float
    z_statistic: float
    p_value_two_sided: float
    block_count: int


def _validate_count(successes: int, total: int, label: str) -> None:
    if isinstance(successes, bool) or isinstance(total, bool):
        raise ContractError(f"{label} counts must be integers")
    if int(successes) != successes or int(total) != total:
        raise ContractError(f"{label} counts must be integers")
    if total <= 0 or successes < 0 or successes > total:
        raise ContractError(
            f"{label} requires 0 <= successes <= total and total > 0"
        )


def _wilson_interval(successes: int, total: int, z: float) -> tuple[float, float]:
    rate = successes / total
    denominator = 1.0 + z**2 / total
    center = (rate + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * sqrt(rate * (1.0 - rate) / total + z**2 / (4.0 * total**2))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def compare_binary_proportions(
    treatment_successes: int,
    treatment_total: int,
    control_successes: int,
    control_total: int,
    *,
    confidence: float = 0.95,
) -> BinaryEffect:
    """Compare two independent proportions using transparent approximations.

    The two-sided p-value uses the pooled score test.  The risk-difference
    interval uses the Newcombe hybrid-Wilson construction.  The risk-ratio
    interval uses the log method with a 0.5 continuity correction whenever a
    cell is zero.  These are large-sample tools, not a replacement for an exact
    small-sample analysis.
    """

    _validate_count(treatment_successes, treatment_total, "treatment")
    _validate_count(control_successes, control_total, "control")
    if not 0.0 < confidence < 1.0:
        raise ContractError("confidence must be between 0 and 1")

    treatment_rate = treatment_successes / treatment_total
    control_rate = control_successes / control_total
    difference = treatment_rate - control_rate
    alpha = 1.0 - confidence
    z_critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    pooled = (treatment_successes + control_successes) / (
        treatment_total + control_total
    )
    pooled_se = sqrt(
        pooled
        * (1.0 - pooled)
        * (1.0 / treatment_total + 1.0 / control_total)
    )
    if pooled_se == 0.0:
        z_statistic = 0.0
        p_value = 1.0
    else:
        z_statistic = difference / pooled_se
        p_value = erfc(abs(z_statistic) / sqrt(2.0))

    treatment_low, treatment_high = _wilson_interval(
        treatment_successes, treatment_total, z_critical
    )
    control_low, control_high = _wilson_interval(
        control_successes, control_total, z_critical
    )
    difference_low = treatment_rate - control_rate - sqrt(
        (treatment_rate - treatment_low) ** 2
        + (control_high - control_rate) ** 2
    )
    difference_high = treatment_rate - control_rate + sqrt(
        (treatment_high - treatment_rate) ** 2
        + (control_rate - control_low) ** 2
    )

    correction_needed = treatment_successes == 0 or control_successes == 0
    t_success = treatment_successes + (0.5 if correction_needed else 0.0)
    c_success = control_successes + (0.5 if correction_needed else 0.0)
    t_total = treatment_total + (1.0 if correction_needed else 0.0)
    c_total = control_total + (1.0 if correction_needed else 0.0)
    relative_risk = (t_success / t_total) / (c_success / c_total)
    log_rr_se = sqrt(
        max(0.0, 1.0 / t_success - 1.0 / t_total)
        + max(0.0, 1.0 / c_success - 1.0 / c_total)
    )
    relative_low = exp(log(relative_risk) - z_critical * log_rr_se)
    relative_high = exp(log(relative_risk) + z_critical * log_rr_se)

    return BinaryEffect(
        treatment_successes=int(treatment_successes),
        treatment_total=int(treatment_total),
        control_successes=int(control_successes),
        control_total=int(control_total),
        treatment_rate=treatment_rate,
        control_rate=control_rate,
        absolute_effect=difference,
        absolute_ci_low=max(-1.0, difference_low),
        absolute_ci_high=min(1.0, difference_high),
        relative_risk=relative_risk,
        relative_risk_ci_low=relative_low,
        relative_risk_ci_high=relative_high,
        z_statistic=z_statistic,
        p_value_two_sided=min(1.0, max(0.0, p_value)),
    )


def compare_block_standardized_binary(
    frame: pd.DataFrame,
    *,
    group_col: str,
    treatment_label: object,
    control_label: object,
    outcome_col: str,
    block_cols: tuple[str, ...],
    confidence: float = 0.95,
) -> BlockStandardizedEffect:
    """Compare two exactly allocated arms using their declared randomization blocks.

    Each block must contain the same number of treatment and control rows and at
    least two rows per arm.  Block weights are each block's share of the two-arm
    analysis population.  Under the repository's common seven-arm allocation
    ratio, these are also the full-population block weights.
    """

    if isinstance(block_cols, (str, bytes)) or not isinstance(block_cols, tuple):
        raise ContractError("block_cols must be a non-empty tuple of unique column names")
    if (
        not block_cols
        or len(set(block_cols)) != len(block_cols)
        or any(not isinstance(column, str) or not column for column in block_cols)
        or group_col == outcome_col
        or group_col in block_cols
        or outcome_col in block_cols
    ):
        raise ContractError("block_cols must be a non-empty tuple of unique column names")
    if treatment_label == control_label:
        raise ContractError("treatment and control labels must be distinct")
    if not 0.0 < confidence < 1.0:
        raise ContractError("confidence must be between 0 and 1")

    require_columns(
        frame,
        [group_col, outcome_col, *block_cols],
        frame_name="block-standardized binary data",
    )
    if frame.empty:
        raise ContractError("block-standardized binary data must contain rows")
    if frame[list(block_cols)].isna().any().any():
        raise ContractError("randomization block labels must be complete")
    for column in block_cols:
        values = frame[column]
        if not values.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ContractError("randomization block labels must be non-empty strings")

    observed_groups = set(frame[group_col])
    if observed_groups != {treatment_label, control_label}:
        raise ContractError("block-standardized comparison requires exactly two declared arms")
    working = frame.loc[:, [*block_cols, group_col, outcome_col]].copy()
    working[outcome_col] = validate_binary(
        working[outcome_col], name=outcome_col, allow_missing=False
    ).astype(int)

    grouped = list(working.groupby(list(block_cols), sort=True, dropna=False))
    if not grouped:
        raise ContractError("block-standardized comparison requires at least one block")
    invalid_blocks = 0
    too_small_blocks = 0
    for _, block in grouped:
        counts = block[group_col].value_counts()
        treatment_n = int(counts.get(treatment_label, 0))
        control_n = int(counts.get(control_label, 0))
        if treatment_n != control_n or treatment_n == 0:
            invalid_blocks += 1
        elif treatment_n < 2:
            too_small_blocks += 1
    if invalid_blocks:
        raise ContractError(
            "block-standardized comparison violates exact two-arm allocation in "
            f"{invalid_blocks} blocks"
        )
    if too_small_blocks:
        raise ContractError(
            "block-standardized comparison has fewer than two units per arm in "
            f"{too_small_blocks} blocks"
        )

    total_rows = len(working)
    treatment_rate = 0.0
    control_rate = 0.0
    variance = 0.0
    for _, block in grouped:
        treatment = block.loc[block[group_col] == treatment_label, outcome_col]
        control = block.loc[block[group_col] == control_label, outcome_col]
        weight = len(block) / total_rows
        treatment_rate += weight * float(treatment.mean())
        control_rate += weight * float(control.mean())
        variance += weight**2 * (
            float(treatment.var(ddof=1)) / len(treatment)
            + float(control.var(ddof=1)) / len(control)
        )

    if not np.isfinite(variance) or variance <= 0.0:
        raise ContractError(
            "block-standardized comparison requires positive finite Neyman variance"
        )
    standard_error = sqrt(variance)
    difference = treatment_rate - control_rate
    z_statistic = difference / standard_error
    p_value = erfc(abs(z_statistic) / sqrt(2.0))
    alpha = 1.0 - confidence
    z_critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    ci_low = max(-1.0, difference - z_critical * standard_error)
    ci_high = min(1.0, difference + z_critical * standard_error)

    treatment_rows = working.loc[working[group_col] == treatment_label, outcome_col]
    control_rows = working.loc[working[group_col] == control_label, outcome_col]
    return BlockStandardizedEffect(
        treatment_successes=int(treatment_rows.sum()),
        treatment_total=int(treatment_rows.size),
        control_successes=int(control_rows.sum()),
        control_total=int(control_rows.size),
        treatment_rate=treatment_rate,
        control_rate=control_rate,
        absolute_effect=difference,
        absolute_standard_error=standard_error,
        absolute_ci_low=ci_low,
        absolute_ci_high=ci_high,
        z_statistic=z_statistic,
        p_value_two_sided=min(1.0, max(0.0, p_value)),
        block_count=len(grouped),
    )


def adjust_pvalues(
    p_values: list[float] | np.ndarray | pd.Series,
    method: AdjustmentMethod,
) -> np.ndarray:
    """Return multiplicity-adjusted p-values in the original order."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ContractError("p-values must be a finite one-dimensional sequence")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ContractError("p-values must be between 0 and 1")
    if values.size == 0:
        return values.copy()

    order = np.argsort(values, kind="stable")
    ranked = values[order]
    count = values.size
    if method == "holm":
        adjusted_ranked = np.maximum.accumulate(
            ranked * np.arange(count, 0, -1, dtype=float)
        )
    elif method == "benjamini-hochberg":
        raw = ranked * count / np.arange(1, count + 1, dtype=float)
        adjusted_ranked = np.minimum.accumulate(raw[::-1])[::-1]
    else:
        raise ContractError(f"unsupported adjustment method: {method}")

    adjusted = np.empty(count, dtype=float)
    adjusted[order] = np.minimum(1.0, adjusted_ranked)
    return adjusted


def compare_experiment_groups(
    frame: pd.DataFrame,
    *,
    experiment_col: str = "experiment",
    arm_col: str = "arm",
    outcome_col: str = "outcome",
    treatment_value: str = "treatment",
    control_value: str = "control",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compare treatment/control for every experiment and adjust all p-values."""

    require_columns(
        frame,
        [experiment_col, arm_col, outcome_col],
        frame_name="experiment data",
    )
    if frame[experiment_col].isna().any() or frame[arm_col].isna().any():
        raise ContractError("experiment and arm labels must be complete")
    outcomes = validate_binary(frame[outcome_col], name=outcome_col)
    working = frame[[experiment_col, arm_col]].copy()
    working[outcome_col] = outcomes

    rows: list[dict[str, float | int | str]] = []
    for experiment, group in working.groupby(experiment_col, sort=True, dropna=False):
        observed_arms = set(group[arm_col].unique())
        expected_arms = {treatment_value, control_value}
        if observed_arms != expected_arms:
            raise ContractError(
                "each experiment must contain exactly one treatment and one control arm"
            )
        treatment = group.loc[group[arm_col] == treatment_value, outcome_col]
        control = group.loc[group[arm_col] == control_value, outcome_col]
        result = compare_binary_proportions(
            int(treatment.sum()),
            int(treatment.size),
            int(control.sum()),
            int(control.size),
            confidence=confidence,
        )
        row = asdict(result)
        row[experiment_col] = str(experiment)
        rows.append(row)

    if not rows:
        raise ContractError("experiment data must contain at least one row")

    results = pd.DataFrame(rows)
    p_values = results["p_value_two_sided"].to_numpy()
    results["p_value_holm"] = adjust_pvalues(p_values, "holm")
    results["p_value_bh"] = adjust_pvalues(p_values, "benjamini-hochberg")
    return results[[experiment_col, *[c for c in results.columns if c != experiment_col]]]


def approximate_mde(
    baseline_rate: float,
    control_total: int,
    treatment_total: int,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Approximate the detectable absolute lift for a two-sided z-test.

    This planning approximation holds variance at the baseline rate.  It should
    be rounded and sensitivity-tested rather than reported as exact power.
    """

    if not 0.0 < baseline_rate < 1.0:
        raise ContractError("baseline_rate must be between 0 and 1")
    if control_total <= 0 or treatment_total <= 0:
        raise ContractError("arm sizes must be positive")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ContractError("alpha and power must be between 0 and 1")

    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(power)
    standard_error = sqrt(
        baseline_rate
        * (1.0 - baseline_rate)
        * (1.0 / control_total + 1.0 / treatment_total)
    )
    return min(1.0 - baseline_rate, (z_alpha + z_power) * standard_error)
