import numpy as np
import pandas as pd
import pytest

from email_experiment.contracts import ContractError
from email_experiment.statistics import (
    adjust_pvalues,
    approximate_mde,
    compare_binary_proportions,
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
