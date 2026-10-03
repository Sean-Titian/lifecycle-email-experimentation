import numpy as np
import pandas as pd
import pytest

from email_experiment.contracts import ContractError
from email_experiment.statistics import (
    adjust_pvalues,
    approximate_mde,
    compare_binary_proportions,
    compare_block_standardized_binary,
    compare_experiment_groups,
)


def test_binary_effect_reports_absolute_relative_ci_and_two_sided_p() -> None:
    result = compare_binary_proportions(120, 1000, 100, 1000)
    assert result.absolute_effect == pytest.approx(0.02)
    assert result.relative_risk == pytest.approx(1.2)
    assert result.absolute_ci_low < 0 < result.absolute_ci_high
    assert 0 < result.p_value_two_sided < 1
    assert result.relative_risk_ci_low < result.relative_risk_ci_high


def test_binary_effect_zero_cells_remain_finite() -> None:
    result = compare_binary_proportions(1, 100, 0, 100)
    values = [
        result.relative_risk,
        result.relative_risk_ci_low,
        result.relative_risk_ci_high,
    ]
    assert np.isfinite(values).all()

    all_zero = compare_binary_proportions(0, 100, 0, 100)
    assert all_zero.p_value_two_sided == 1.0


def _manual_block_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "block": ["small"] * 4 + ["large"] * 8,
            "arm": ["active"] * 2
            + ["holdout"] * 2
            + ["active"] * 4
            + ["holdout"] * 4,
            "outcome": [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        }
    )


def test_block_standardized_binary_matches_hand_calculation_and_row_order() -> None:
    frame = _manual_block_frame()
    expected_variance = (2.0 / 3.0) ** 2 * ((1.0 / 3.0) / 4.0)
    expected = compare_block_standardized_binary(
        frame,
        group_col="arm",
        treatment_label="active",
        control_label="holdout",
        outcome_col="outcome",
        block_cols=("block",),
    )
    assert expected.treatment_rate == pytest.approx(1.0 / 3.0)
    assert expected.control_rate == pytest.approx(1.0 / 3.0)
    assert expected.absolute_effect == pytest.approx(0.0)
    assert expected.absolute_standard_error == pytest.approx(np.sqrt(expected_variance))
    assert expected.p_value_two_sided == pytest.approx(1.0)
    assert expected.block_count == 2

    shuffled = frame.sample(frac=1.0, random_state=41).reset_index(drop=True)
    actual = compare_block_standardized_binary(
        shuffled,
        group_col="arm",
        treatment_label="active",
        control_label="holdout",
        outcome_col="outcome",
        block_cols=("block",),
    )
    assert actual == expected


def test_block_standardized_binary_fails_closed_on_invalid_blocks() -> None:
    frame = _manual_block_frame()

    missing = frame.copy()
    missing.loc[0, "block"] = None
    with pytest.raises(ContractError, match="block labels must be complete"):
        compare_block_standardized_binary(
            missing,
            group_col="arm",
            treatment_label="active",
            control_label="holdout",
            outcome_col="outcome",
            block_cols=("block",),
        )

    swapped = frame.copy()
    swapped.loc[0, "block"] = "large"
    swapped.loc[8, "block"] = "small"
    with pytest.raises(ContractError, match="exact two-arm allocation"):
        compare_block_standardized_binary(
            swapped,
            group_col="arm",
            treatment_label="active",
            control_label="holdout",
            outcome_col="outcome",
            block_cols=("block",),
        )

    one_per_arm = pd.DataFrame(
        {"block": ["one", "one"], "arm": ["active", "holdout"], "outcome": [1, 0]}
    )
    with pytest.raises(ContractError, match="fewer than two"):
        compare_block_standardized_binary(
            one_per_arm,
            group_col="arm",
            treatment_label="active",
            control_label="holdout",
            outcome_col="outcome",
            block_cols=("block",),
        )


def test_block_standardized_binary_rejects_zero_variance_and_ambiguous_schema() -> None:
    constant = _manual_block_frame()
    constant["outcome"] = 0
    with pytest.raises(ContractError, match="positive finite Neyman variance"):
        compare_block_standardized_binary(
            constant,
            group_col="arm",
            treatment_label="active",
            control_label="holdout",
            outcome_col="outcome",
            block_cols=("block",),
        )

    with pytest.raises(ContractError, match="block_cols must be"):
        compare_block_standardized_binary(
            _manual_block_frame(),
            group_col="arm",
            treatment_label="active",
            control_label="holdout",
            outcome_col="outcome",
            block_cols=["block"],  # type: ignore[arg-type]
        )

    with pytest.raises(ContractError, match="block_cols must be"):
        compare_block_standardized_binary(
            _manual_block_frame(),
            group_col="outcome",
            treatment_label=1,
            control_label=0,
            outcome_col="outcome",
            block_cols=("block",),
        )


def test_holm_and_bh_are_monotone_in_rank_and_restore_input_order() -> None:
    p_values = [0.01, 0.04, 0.03]
    assert adjust_pvalues(p_values, "holm") == pytest.approx([0.03, 0.06, 0.06])
    assert adjust_pvalues(p_values, "benjamini-hochberg") == pytest.approx(
        [0.03, 0.04, 0.04]
    )


def test_group_comparison_adjusts_one_declared_family() -> None:
    frame = pd.DataFrame(
        {
            "experiment": ["experiment_01"] * 8 + ["experiment_02"] * 8,
            "arm": (["control"] * 4 + ["treatment"] * 4) * 2,
            "outcome": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
        }
    )
    result = compare_experiment_groups(frame)
    assert list(result["experiment"]) == ["experiment_01", "experiment_02"]
    assert {"absolute_effect", "relative_risk", "p_value_holm", "p_value_bh"} <= set(
        result.columns
    )
    assert (result["p_value_holm"] >= result["p_value_two_sided"]).all()


def test_group_comparison_rejects_incomplete_or_extra_arms() -> None:
    frame = pd.DataFrame(
        {
            "experiment": ["experiment_01", "experiment_01"],
            "arm": ["control", "control"],
            "outcome": [0, 1],
        }
    )
    with pytest.raises(ContractError, match="exactly one treatment"):
        compare_experiment_groups(frame)


def test_approximate_mde_is_plausible_and_shrinks_with_sample_size() -> None:
    small = approximate_mde(0.03, 2_000, 2_000)
    large = approximate_mde(0.03, 20_000, 20_000)
    assert 0 < large < small < 0.03
