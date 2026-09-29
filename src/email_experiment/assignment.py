"""Auditable stratified randomization for the prospective synthetic study.

The public design uses one concurrent holdout and a three-by-two active
factorial.  Assignments are balanced independently inside every
``lifecycle_segment`` x ``tenure_band`` x ``assignment_wave`` block.  The
implementation is deterministic for a given seed and participant set, but the
seed is still part of the recorded randomization contract rather than a claim
of cryptographic unpredictability.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import exp
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    ContractError,
    parse_aware_utc_series,
    require_exact_columns,
    validate_unique_key,
)
from .statistics import adjust_pvalues

CONTENT_LEVELS = ("current", "challenger_a", "challenger_b")
CADENCE_LEVELS = ("daily", "twice_weekly")
HOLDOUT_ARM = "holdout"
ACTIVE_ARMS = tuple(
    f"{content}__{cadence}"
    for content in CONTENT_LEVELS
    for cadence in CADENCE_LEVELS
)
ALL_ARMS = (HOLDOUT_ARM, *ACTIVE_ARMS)
EXPECTED_PROBABILITY = 1.0 / len(ALL_ARMS)
RANDOMIZATION_VERSION = "stratified-permuted-block-v1"

ELIGIBLE_COLUMNS = (
    "participant_id",
    "lifecycle_segment",
    "tenure_band",
    "assignment_wave",
    "eligible_at",
    "assigned_at",
)
ASSIGNMENT_COLUMNS = (
    *ELIGIBLE_COLUMNS,
    "arm",
    "content",
    "cadence",
    "expected_probability",
    "randomization_version",
)
BLOCK_COLUMNS = ("lifecycle_segment", "tenure_band", "assignment_wave")


@dataclass(frozen=True)
class SampleRatioCheck:
    """One seven-cell Pearson sample-ratio check."""

    sample_size: int
    observed_counts: dict[str, int]
    expected_count_per_arm: float
    chi_square: float
    degrees_of_freedom: int
    p_value: float
    p_value_holm: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "observed_counts": self.observed_counts,
            "expected_count_per_arm": self.expected_count_per_arm,
            "chi_square": self.chi_square,
            "degrees_of_freedom": self.degrees_of_freedom,
            "p_value": self.p_value,
            "p_value_holm": self.p_value_holm,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class BlockSampleRatioCheck:
    """SRM result for a non-identifying synthetic assignment block."""

    lifecycle_segment: str
    tenure_band: str
    assignment_wave: str
    check: SampleRatioCheck

    def to_dict(self) -> dict[str, Any]:
        return {
            "lifecycle_segment": self.lifecycle_segment,
            "tenure_band": self.tenure_band,
            "assignment_wave": self.assignment_wave,
            **self.check.to_dict(),
        }


@dataclass(frozen=True)
class SampleRatioAudit:
    """Overall and multiplicity-adjusted block-level SRM diagnostics."""

    alpha: float
    overall: SampleRatioCheck
    by_block: tuple[BlockSampleRatioCheck, ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "overall": self.overall.to_dict(),
            "by_block": [result.to_dict() for result in self.by_block],
            "passed": self.passed,
        }


def _complete_strings(series: pd.Series, *, name: str) -> pd.Series:
    if series.isna().any() or not series.map(lambda value: isinstance(value, str)).all():
        raise ContractError(f"{name} must contain complete non-empty strings")
    converted = series.astype("string")
    if converted.str.strip().eq("").any():
        raise ContractError(f"{name} must contain complete non-empty strings")
    return converted


def _validate_eligible(eligible: pd.DataFrame) -> pd.DataFrame:
    require_exact_columns(eligible, ELIGIBLE_COLUMNS, frame_name="eligible population")
    validate_unique_key(eligible, "participant_id", frame_name="eligible population")
    output = eligible.loc[:, ELIGIBLE_COLUMNS].copy()
    for column in ("participant_id", *BLOCK_COLUMNS):
        output[column] = _complete_strings(output[column], name=column)
    output["eligible_at"] = parse_aware_utc_series(
        output["eligible_at"], name="eligible_at"
    )
    output["assigned_at"] = parse_aware_utc_series(
        output["assigned_at"], name="assigned_at"
    )
    after_assignment = int((output["eligible_at"] > output["assigned_at"]).sum())
    if after_assignment:
        raise ContractError(
            f"eligibility occurs after assignment for {after_assignment} rows"
        )
    return output


def _block_seed(seed: int, block_values: tuple[str, str, str]) -> int:
    payload = "|".join((str(seed), *block_values)).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], byteorder="big")


def stratified_factorial_assignment(
    eligible: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Assign the seven study cells exactly within each declared block.

    Every block must contain a multiple of seven participants.  Failing rather
    than silently creating an imbalanced remainder keeps the public fixture's
    intended allocation and its expected probabilities unambiguous.
    """

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ContractError("randomization seed must be an integer")
    working = _validate_eligible(eligible)
    assignments: list[pd.DataFrame] = []
    grouped = working.groupby(list(BLOCK_COLUMNS), sort=True, dropna=False)
    for block_values, group in grouped:
        if len(group) % len(ALL_ARMS):
            raise ContractError(
                "each stratification-by-wave block must contain a multiple of seven rows"
            )
        ordered = group.sort_values("participant_id", kind="stable").reset_index(drop=True)
        rng = np.random.default_rng(_block_seed(int(seed), block_values))
        shuffled_positions = rng.permutation(len(ordered))
        arm_labels = np.empty(len(ordered), dtype=object)
        for start in range(0, len(ordered), len(ALL_ARMS)):
            permuted_arms = np.asarray(ALL_ARMS, dtype=object)[
                rng.permutation(len(ALL_ARMS))
            ]
            positions = shuffled_positions[start : start + len(ALL_ARMS)]
            arm_labels[positions] = permuted_arms
        ordered["arm"] = pd.Series(arm_labels, dtype="string")
        assignments.append(ordered)

    if not assignments:
        raise ContractError("eligible population must contain at least one complete block")
    output = pd.concat(assignments, ignore_index=True)
    output["content"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output["cadence"] = pd.Series(pd.NA, index=output.index, dtype="string")
    active = output["arm"] != HOLDOUT_ARM
    split = output.loc[active, "arm"].str.split("__", n=1, expand=True)
    output.loc[active, "content"] = split[0].to_numpy()
    output.loc[active, "cadence"] = split[1].to_numpy()
    output["expected_probability"] = EXPECTED_PROBABILITY
    output["randomization_version"] = RANDOMIZATION_VERSION
    output = output.loc[:, ASSIGNMENT_COLUMNS].sort_values(
        "participant_id", kind="stable"
    )
    return output.reset_index(drop=True)


def validate_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, clocks, factors, and exact allocation inside every block."""

    require_exact_columns(assignments, ASSIGNMENT_COLUMNS, frame_name="assignments")
    working = _validate_eligible(assignments.loc[:, ELIGIBLE_COLUMNS])
    for column in ("arm", "randomization_version"):
        working[column] = _complete_strings(assignments[column], name=column)
    invalid_arms = int((~working["arm"].isin(ALL_ARMS)).sum())
    if invalid_arms:
        raise ContractError(f"assignments contain {invalid_arms} unsupported arm labels")
    if not working["randomization_version"].eq(RANDOMIZATION_VERSION).all():
        raise ContractError("assignments contain an unsupported randomization version")

    content = assignments["content"].astype("string")
    cadence = assignments["cadence"].astype("string")
    holdout = working["arm"] == HOLDOUT_ARM
    invalid_holdout_factors = int((holdout & (content.notna() | cadence.notna())).sum())
    if invalid_holdout_factors:
        raise ContractError(
            f"holdout assignments contain {invalid_holdout_factors} non-null factor rows"
        )
    active_missing = int((~holdout & (content.isna() | cadence.isna())).sum())
    if active_missing:
        raise ContractError(
            f"active assignments contain {active_missing} rows with missing factors"
        )
    expected_arm = content.fillna("") + "__" + cadence.fillna("")
    factor_mismatch = int((~holdout & working["arm"].ne(expected_arm)).sum())
    if factor_mismatch:
        raise ContractError(
            f"active assignments contain {factor_mismatch} arm-to-factor mismatches"
        )
    invalid_content = int((~holdout & ~content.isin(CONTENT_LEVELS)).sum())
    invalid_cadence = int((~holdout & ~cadence.isin(CADENCE_LEVELS)).sum())
    if invalid_content or invalid_cadence:
        raise ContractError(
            f"active assignments contain {invalid_content} invalid content and "
            f"{invalid_cadence} invalid cadence values"
        )

    probabilities = pd.to_numeric(assignments["expected_probability"], errors="coerce")
    invalid_probability = int(
        (
            probabilities.isna()
            | ~np.isfinite(probabilities)
            | ~np.isclose(probabilities, EXPECTED_PROBABILITY, rtol=0.0, atol=1e-15)
        ).sum()
    )
    if invalid_probability:
        raise ContractError(
            f"assignments contain {invalid_probability} invalid expected probabilities"
        )

    block_arm_counts = (
        working.groupby([*BLOCK_COLUMNS, "arm"], sort=True, dropna=False)
        .size()
        .unstack("arm", fill_value=0)
        .reindex(columns=ALL_ARMS, fill_value=0)
    )
    if block_arm_counts.empty:
        raise ContractError("assignments must contain at least one complete block")
    exact_blocks = block_arm_counts.gt(0).all(axis=1) & block_arm_counts.nunique(
        axis=1
    ).eq(1)
    invalid_blocks = int((~exact_blocks).sum())
    if invalid_blocks:
        raise ContractError(
            "assignments violate exact randomized block allocation in "
            f"{invalid_blocks} blocks"
        )
    working["content"] = content
    working["cadence"] = cadence
    working["expected_probability"] = probabilities.astype(float)
    return working.loc[:, ASSIGNMENT_COLUMNS]


def assignment_sha256(assignments: pd.DataFrame) -> str:
    """Hash the canonical assignment ledger without exposing unit identifiers."""

    working = validate_assignments(assignments).sort_values(
        "participant_id", kind="stable"
    )
    rows: list[str] = []
    for row in working.itertuples(index=False):
        values = []
        for column, value in zip(ASSIGNMENT_COLUMNS, row, strict=True):
            if pd.isna(value):
                rendered = ""
            elif column in {"eligible_at", "assigned_at"}:
                rendered = pd.Timestamp(value).isoformat()
            elif column == "expected_probability":
                rendered = format(float(value), ".17g")
            else:
                rendered = str(value)
            values.append(rendered)
        rows.append("\x1f".join(values))
    return sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _chi_square_survival_df6(statistic: float) -> float:
    """Survival function for chi-square(6), exact for this seven-cell design."""

    half = statistic / 2.0
    return min(1.0, max(0.0, exp(-half) * (1.0 + half + half**2 / 2.0)))


def _srm_components(frame: pd.DataFrame) -> tuple[int, dict[str, int], float, float]:
    counts = frame["arm"].value_counts().reindex(ALL_ARMS, fill_value=0).astype(int)
    sample_size = int(counts.sum())
    if sample_size == 0:
        raise ContractError("sample-ratio checks require at least one assignment")
    expected = sample_size * EXPECTED_PROBABILITY
    statistic = float((((counts - expected) ** 2) / expected).sum())
    p_value = _chi_square_survival_df6(statistic)
    return sample_size, {arm: int(counts[arm]) for arm in ALL_ARMS}, statistic, p_value


def audit_sample_ratio(
    assignments: pd.DataFrame,
    *,
    alpha: float = 0.01,
) -> SampleRatioAudit:
    """Audit overall and block-level assignment ratios.

    Block p-values form one diagnostic family and are Holm adjusted.  The
    overall test is a separate pre-specified gate.  Assignment hashing remains
    necessary because an SRM test cannot detect label swaps that preserve cell
    counts.
    """

    if not 0.0 < alpha < 1.0:
        raise ContractError("SRM alpha must be between zero and one")
    working = validate_assignments(assignments)
    total, counts, statistic, overall_p = _srm_components(working)

    raw_blocks: list[tuple[tuple[str, str, str], int, dict[str, int], float, float]] = []
    for values, block in working.groupby(list(BLOCK_COLUMNS), sort=True, dropna=False):
        block_total, block_counts, block_statistic, block_p = _srm_components(block)
        raw_blocks.append((values, block_total, block_counts, block_statistic, block_p))
    adjusted = adjust_pvalues([result[4] for result in raw_blocks], "holm")

    block_results: list[BlockSampleRatioCheck] = []
    for result, adjusted_p in zip(raw_blocks, adjusted, strict=True):
        values, block_total, block_counts, block_statistic, block_p = result
        check = SampleRatioCheck(
            sample_size=block_total,
            observed_counts=block_counts,
            expected_count_per_arm=block_total * EXPECTED_PROBABILITY,
            chi_square=block_statistic,
            degrees_of_freedom=len(ALL_ARMS) - 1,
            p_value=block_p,
            p_value_holm=float(adjusted_p),
            passed=bool(adjusted_p >= alpha),
        )
        block_results.append(
            BlockSampleRatioCheck(
                lifecycle_segment=values[0],
                tenure_band=values[1],
                assignment_wave=values[2],
                check=check,
            )
        )

    overall = SampleRatioCheck(
        sample_size=total,
        observed_counts=counts,
        expected_count_per_arm=total * EXPECTED_PROBABILITY,
        chi_square=statistic,
        degrees_of_freedom=len(ALL_ARMS) - 1,
        p_value=overall_p,
        p_value_holm=overall_p,
        passed=overall_p >= alpha,
    )
    passed = overall.passed and all(result.check.passed for result in block_results)
    return SampleRatioAudit(
        alpha=alpha,
        overall=overall,
        by_block=tuple(block_results),
        passed=passed,
    )
