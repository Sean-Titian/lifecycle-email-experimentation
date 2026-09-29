import pandas as pd
import pytest

from email_experiment.assignment import (
    ALL_ARMS,
    ASSIGNMENT_COLUMNS,
    EXPECTED_PROBABILITY,
    HOLDOUT_ARM,
    assignment_sha256,
    audit_sample_ratio,
    stratified_factorial_assignment,
    validate_assignments,
)
from email_experiment.contracts import ContractError


def _eligible(*, per_block: int = 140, naive_time: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    timestamp = "2026-02-01T12:00:00" if naive_time else "2026-02-01T12:00:00Z"
    index = 0
    for segment in ("new", "established"):
        for tenure in ("0_30d", "31_90d"):
            for wave in ("wave_01", "wave_02"):
                for _ in range(per_block):
                    rows.append(
                        {
                            "participant_id": f"synthetic_participant_{index:08d}",
                            "lifecycle_segment": segment,
                            "tenure_band": tenure,
                            "assignment_wave": wave,
                            "eligible_at": (
                                "2026-01-31T12:00:00Z"
                                if not naive_time
                                else "2026-01-31T12:00:00"
                            ),
                            "assigned_at": timestamp,
                        }
                    )
                    index += 1
    return pd.DataFrame(rows)


def test_assignment_is_deterministic_row_order_invariant_and_exactly_balanced() -> None:
    eligible = _eligible()
    first = stratified_factorial_assignment(eligible, seed=20260908)
    shuffled = eligible.sample(frac=1.0, random_state=91).reset_index(drop=True)
    second = stratified_factorial_assignment(shuffled, seed=20260908)

    pd.testing.assert_frame_equal(first, second)
    assert tuple(first.columns) == ASSIGNMENT_COLUMNS
    assert first["participant_id"].is_unique
    assert first["expected_probability"].eq(EXPECTED_PROBABILITY).all()
    by_block = first.groupby(
        ["lifecycle_segment", "tenure_band", "assignment_wave", "arm"]
    ).size()
    assert by_block.nunique() == 1
    assert set(first["arm"]) == set(ALL_ARMS)

    holdout = first[first["arm"] == HOLDOUT_ARM]
    assert holdout["content"].isna().all()
    assert holdout["cadence"].isna().all()
    active = first[first["arm"] != HOLDOUT_ARM]
    assert active["content"].notna().all()
    assert active["cadence"].notna().all()
    assert assignment_sha256(first) == assignment_sha256(second)


def test_assignment_seed_changes_ledger_but_not_block_balance() -> None:
    eligible = _eligible()
    first = stratified_factorial_assignment(eligible, seed=3)
    second = stratified_factorial_assignment(eligible, seed=4)
    assert not first["arm"].equals(second["arm"])
    assert assignment_sha256(first) != assignment_sha256(second)
    assert audit_sample_ratio(first).passed
    assert audit_sample_ratio(second).passed


def test_srm_audits_overall_and_each_stratification_wave_block() -> None:
    assignments = stratified_factorial_assignment(_eligible(), seed=8)
    clean = audit_sample_ratio(assignments)
    assert clean.passed
    assert clean.overall.p_value == pytest.approx(1.0)
    assert len(clean.by_block) == 8
    assert all(result.check.p_value_holm == pytest.approx(1.0) for result in clean.by_block)


def test_assignment_contract_rejects_block_imbalance_with_balanced_global_counts() -> None:
    assignments = stratified_factorial_assignment(_eligible(), seed=8)
    before = assignments["arm"].value_counts().sort_index()
    block_groups = list(
        assignments.groupby(
            ["lifecycle_segment", "tenure_band", "assignment_wave"], sort=True
        ).groups.values()
    )
    first_block = assignments.loc[block_groups[0]]
    second_block = assignments.loc[block_groups[1]]
    first_index = first_block.index[first_block["arm"].eq(HOLDOUT_ARM)][0]
    second_index = second_block.index[
        second_block["arm"].eq("current__daily")
    ][0]
    tampered = assignments.copy()
    factor_columns = ["arm", "content", "cadence"]
    first_values = tampered.loc[first_index, factor_columns].copy()
    tampered.loc[first_index, factor_columns] = tampered.loc[
        second_index, factor_columns
    ].to_numpy()
    tampered.loc[second_index, factor_columns] = first_values.to_numpy()

    pd.testing.assert_series_equal(
        tampered["arm"].value_counts().sort_index(), before
    )
    with pytest.raises(ContractError, match="exact randomized block allocation"):
        validate_assignments(tampered)
    with pytest.raises(ContractError, match="exact randomized block allocation"):
        audit_sample_ratio(tampered)


def test_assignment_contract_rejects_naive_time_and_post_assignment_eligibility() -> None:
    with pytest.raises(ContractError, match="timezone-naive"):
        stratified_factorial_assignment(_eligible(naive_time=True), seed=1)

    eligible = _eligible()
    eligible.loc[0, "eligible_at"] = "2026-02-02T00:00:00Z"
    with pytest.raises(ContractError, match="eligibility occurs after assignment"):
        stratified_factorial_assignment(eligible, seed=1)


def test_assignment_requires_complete_blocks_and_exact_schema() -> None:
    with pytest.raises(ContractError, match="multiple of seven"):
        stratified_factorial_assignment(_eligible(per_block=8), seed=1)

    eligible = _eligible()
    eligible["post_treatment_feature"] = 1
    with pytest.raises(ContractError, match="schema mismatch"):
        stratified_factorial_assignment(eligible, seed=1)

    duplicated_column = pd.concat(
        [_eligible(), _eligible()[["participant_id"]]], axis=1
    )
    with pytest.raises(ContractError, match="duplicate names"):
        stratified_factorial_assignment(duplicated_column, seed=1)

    non_string_id = _eligible()
    non_string_id["participant_id"] = non_string_id["participant_id"].astype("object")
    non_string_id.loc[0, "participant_id"] = 1
    with pytest.raises(ContractError, match="non-empty strings"):
        stratified_factorial_assignment(non_string_id, seed=1)


def test_assignment_validation_rejects_holdout_factors_and_arm_mismatch() -> None:
    assignments = stratified_factorial_assignment(_eligible(), seed=5)
    holdout_index = assignments.index[assignments["arm"] == HOLDOUT_ARM][0]
    bad_holdout = assignments.copy()
    bad_holdout.loc[holdout_index, "content"] = "current"
    with pytest.raises(ContractError, match="holdout assignments"):
        validate_assignments(bad_holdout)

    active_index = assignments.index[assignments["arm"] != HOLDOUT_ARM][0]
    mismatch = assignments.copy()
    mismatch.loc[active_index, "content"] = "challenger_b"
    with pytest.raises(ContractError, match="arm-to-factor mismatches"):
        validate_assignments(mismatch)
