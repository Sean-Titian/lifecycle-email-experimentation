from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import pytest

from email_experiment.contracts import ContractError
from email_experiment.decision import (
    evaluate_guardrail_family,
    evaluate_primary_family,
    make_launch_decision,
)

ACTIVE_CELLS = (
    ("content_a_daily", "content_a", "daily"),
    ("content_a_twice_weekly", "content_a", "twice_weekly"),
    ("content_b_daily", "content_b", "daily"),
    ("content_b_twice_weekly", "content_b", "twice_weekly"),
    ("content_c_daily", "content_c", "daily"),
    ("content_c_twice_weekly", "content_c", "twice_weekly"),
)


def _factorial_frame(
    *,
    total_per_arm: int = 5_000,
    funded: Mapping[str, int] | None = None,
    unsubscribed: Mapping[str, int] | None = None,
    complained: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    arms = [("holdout", None, None), *ACTIVE_CELLS]
    funded = dict(funded or {})
    unsubscribed = dict(unsubscribed or {})
    complained = dict(complained or {})
    rows: list[pd.DataFrame] = []
    offset = 0
    for arm, content, cadence in arms:
        funding_successes = funded.get(arm, total_per_arm // 20)
        unsubscribe_successes = unsubscribed.get(arm, total_per_arm // 100)
        complaint_successes = complained.get(arm, total_per_arm // 200)
        for value, label in (
            (funding_successes, "funded"),
            (unsubscribe_successes, "unsubscribed"),
            (complaint_successes, "complained"),
        ):
            if not 0 <= value <= total_per_arm:
                raise AssertionError(f"invalid {label} fixture count")
        rows.append(
            pd.DataFrame(
                {
                    "participant_id": [
                        f"synthetic_{index:08d}"
                        for index in range(offset, offset + total_per_arm)
                    ],
                    "arm": arm,
                    "content": content,
                    "cadence": cadence,
                    "funded_14d": [1] * funding_successes
                    + [0] * (total_per_arm - funding_successes),
                    "unsubscribe_14d": [1] * unsubscribe_successes
                    + [0] * (total_per_arm - unsubscribe_successes),
                    "complaint_14d": [1] * complaint_successes
                    + [0] * (total_per_arm - complaint_successes),
                }
            )
        )
        offset += total_per_arm
    return pd.concat(rows, ignore_index=True)


def _margins() -> dict[str, float]:
    return {"unsubscribe_14d": 0.01, "complaint_14d": 0.01}


def test_primary_family_is_six_cell_vs_holdout_holm_family() -> None:
    frame = _factorial_frame(
        funded={
            "holdout": 250,
            "content_a_daily": 500,
            "content_c_daily": 800,
        }
    )
    result = evaluate_primary_family(frame)

    assert len(result) == 6
    assert set(result["comparator_arm"]) == {"holdout"}
    assert set(result["family_size"]) == {6}
    assert set(result["family_alpha"]) == {0.05}
    assert (result["p_value_holm"] >= result["p_value_two_sided"]).all()
    candidate = result.set_index("active_arm").loc["content_a_daily"]
    assert candidate["risk_difference"] == pytest.approx(0.05)
    assert candidate["relative_risk"] == pytest.approx(2.0)
    assert bool(candidate["superiority_pass"])


def test_primary_family_is_row_order_invariant() -> None:
    frame = _factorial_frame(total_per_arm=200)
    expected = evaluate_primary_family(frame)
    shuffled = frame.sample(frac=1.0, random_state=77).reset_index(drop=True)
    actual = evaluate_primary_family(shuffled)
    pd.testing.assert_frame_equal(actual, expected)


def test_factorial_contract_rejects_nonempty_holdout_factors_and_duplicate_units() -> None:
    frame = _factorial_frame(total_per_arm=20)
    invalid_holdout = frame.copy()
    invalid_holdout.loc[invalid_holdout["arm"] == "holdout", "content"] = "content_a"
    with pytest.raises(ContractError, match="holdout rows"):
        evaluate_primary_family(invalid_holdout)

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ContractError, match="duplicate keys"):
        evaluate_primary_family(duplicate)


def test_factorial_contract_requires_exact_three_by_two_active_cells() -> None:
    frame = _factorial_frame(total_per_arm=20)
    reduced = frame.loc[frame["arm"] != "content_c_twice_weekly"].copy()
    with pytest.raises(ContractError, match="six active arms"):
        evaluate_primary_family(reduced)

    bad_mapping = frame.copy()
    bad_mapping.loc[
        bad_mapping["arm"] == "content_c_twice_weekly", "cadence"
    ] = "daily"
    with pytest.raises(ContractError, match="factorial cell"):
        evaluate_primary_family(bad_mapping)


def test_guardrails_use_itt_denominators_and_twelve_way_bonferroni_bounds() -> None:
    frame = _factorial_frame()
    result = evaluate_guardrail_family(frame, margins=_margins())

    assert len(result) == 12
    assert set(result["family_size"]) == {12}
    assert result["bonferroni_alpha_each"].nunique() == 1
    assert result["bonferroni_alpha_each"].iloc[0] == pytest.approx(0.05 / 12)
    assert set(result["estimand"]) == {
        "intention_to_treat_unique_randomized_user_risk"
    }
    assert set(result["active_total"]) == {5_000}
    assert set(result["holdout_total"]) == {5_000}
    assert result["noninferiority_pass"].all()


def test_guardrail_precision_is_required_and_upper_bound_equality_does_not_pass() -> None:
    small = _factorial_frame(total_per_arm=100)
    imprecise = evaluate_guardrail_family(small, margins=_margins())
    row = imprecise.loc[
        (imprecise["active_arm"] == "content_a_daily")
        & (imprecise["outcome"] == "unsubscribe_14d")
    ].iloc[0]
    assert row["risk_difference"] < row["noninferiority_margin"]
    assert not bool(row["noninferiority_pass"])
    assert row["conclusion"] == "inconclusive"

    wide_margins = {"unsubscribe_14d": 0.5, "complaint_14d": 0.5}
    initial = evaluate_guardrail_family(small, margins=wide_margins)
    exact_bound = float(
        initial.loc[
            (initial["active_arm"] == "content_a_daily")
            & (initial["outcome"] == "unsubscribe_14d"),
            "simultaneous_one_sided_upper_bound",
        ].iloc[0]
    )
    equality = evaluate_guardrail_family(
        small,
        margins={"unsubscribe_14d": exact_bound, "complaint_14d": 0.5},
    )
    equality_row = equality.loc[
        (equality["active_arm"] == "content_a_daily")
        & (equality["outcome"] == "unsubscribe_14d")
    ].iloc[0]
    assert equality_row["simultaneous_one_sided_upper_bound"] == pytest.approx(
        exact_bound
    )
    assert not bool(equality_row["noninferiority_pass"])


def test_guardrail_over_margin_is_harmful_and_missing_margin_fails_closed() -> None:
    frame = _factorial_frame(
        unsubscribed={"holdout": 50, "content_a_daily": 150}
    )
    result = evaluate_guardrail_family(frame, margins=_margins())
    harmful = result.loc[
        (result["active_arm"] == "content_a_daily")
        & (result["outcome"] == "unsubscribe_14d")
    ].iloc[0]
    assert harmful["risk_difference"] == pytest.approx(0.02)
    assert harmful["conclusion"] == "harmful_or_over_margin"
    assert not bool(harmful["noninferiority_pass"])

    with pytest.raises(ContractError, match="explicit margin"):
        evaluate_guardrail_family(frame, margins={"unsubscribe_14d": 0.01})
    with pytest.raises(ContractError, match="must be finite"):
        evaluate_guardrail_family(
            frame,
            margins={"unsubscribe_14d": None, "complaint_14d": 0.01},  # type: ignore[dict-item]
        )


def test_invalid_alpha_fails_with_a_contract_error() -> None:
    frame = _factorial_frame(total_per_arm=20)
    with pytest.raises(ContractError, match="alpha must be a finite number"):
        evaluate_primary_family(frame, alpha="not-a-number")  # type: ignore[arg-type]


def test_decision_never_selects_the_observed_largest_arm() -> None:
    frame = _factorial_frame(
        funded={
            "holdout": 250,
            "content_a_daily": 500,
            "content_c_daily": 800,
        }
    )
    primary = evaluate_primary_family(frame)
    guardrails = evaluate_guardrail_family(frame, margins=_margins())

    no_candidate = make_launch_decision(
        primary,
        guardrails,
        {"srm": True, "followup_complete": True, "negative_control": True},
        candidate_arm=None,
    )
    assert no_candidate.status == "continue_testing"
    assert no_candidate.reasons == ("pre_specified_candidate_required",)

    specified = make_launch_decision(
        primary,
        guardrails,
        {"srm": True, "followup_complete": True, "negative_control": True},
        candidate_arm="content_a_daily",
    )
    assert specified.status == "criteria_met"
    assert specified.candidate_arm == "content_a_daily"
    assert specified.scope == "synthetic_method_check_not_a_rollout_recommendation"
    assert specified.to_dict()["candidate_arm"] != "content_c_daily"


def test_any_failed_quality_or_guardrail_gate_means_continue_testing() -> None:
    frame = _factorial_frame(
        funded={"holdout": 250, "content_a_daily": 500},
        unsubscribed={"holdout": 50, "content_a_daily": 150},
    )
    primary = evaluate_primary_family(frame)
    guardrails = evaluate_guardrail_family(frame, margins=_margins())

    result = make_launch_decision(
        primary,
        guardrails,
        {"srm": False, "followup_complete": True, "negative_control": True},
        candidate_arm="content_a_daily",
    )
    assert result.status == "continue_testing"
    assert "quality_gate_failed:srm" in result.reasons
    assert (
        "guardrail_noninferiority_not_established:unsubscribe_14d"
        in result.reasons
    )


def test_decision_rejects_empty_or_nonboolean_quality_gates() -> None:
    frame = _factorial_frame(total_per_arm=100)
    primary = evaluate_primary_family(frame)
    guardrails = evaluate_guardrail_family(
        frame,
        margins={"unsubscribe_14d": 0.5, "complaint_14d": 0.5},
    )
    with pytest.raises(ContractError, match="at least one quality gate"):
        make_launch_decision(
            primary, guardrails, {}, candidate_arm="content_a_daily"
        )
    with pytest.raises(ContractError, match="boolean values"):
        make_launch_decision(
            primary,
            guardrails,
            {"srm": 1},  # type: ignore[dict-item]
            candidate_arm="content_a_daily",
        )


def test_decision_rejects_tampered_pass_indicators() -> None:
    frame = _factorial_frame(total_per_arm=5_000)
    primary = evaluate_primary_family(frame)
    guardrails = evaluate_guardrail_family(frame, margins=_margins())

    bad_primary = primary.copy()
    bad_primary.loc[bad_primary.index[0], "superiority_pass"] = True
    with pytest.raises(ContractError, match="primary pass indicators"):
        make_launch_decision(
            bad_primary,
            guardrails,
            {"srm": True},
            candidate_arm="content_a_daily",
        )

    bad_guardrail = guardrails.copy()
    bad_guardrail.loc[bad_guardrail.index[0], "noninferiority_pass"] = False
    with pytest.raises(ContractError, match="guardrail pass indicators"):
        make_launch_decision(
            primary,
            bad_guardrail,
            {"srm": True},
            candidate_arm="content_a_daily",
        )
