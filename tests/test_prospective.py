from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from email_experiment.assignment import (
    ALL_ARMS,
    BLOCK_COLUMNS,
    HOLDOUT_ARM,
    stratified_factorial_assignment,
)
from email_experiment.contracts import ContractError
from email_experiment.decision import REQUIRED_QUALITY_GATES
from email_experiment.prospective import (
    EVENT_COLUMNS,
    ContaminationAudit,
    ProspectiveSyntheticConfig,
    ProspectiveSyntheticData,
    audit_contamination,
    audit_followup,
    build_prospective_synthetic_benchmark,
    construct_prospective_analysis,
    generate_prospective_synthetic,
    validate_events,
    write_prospective_synthetic_benchmark,
)


def _small_config() -> ProspectiveSyntheticConfig:
    return ProspectiveSyntheticConfig(
        seed=77,
        units_per_arm_per_block=2,
        assignment_waves=1,
    )


def _one_block_assignments() -> pd.DataFrame:
    assigned_at = pd.Timestamp("2026-04-01T00:00:00Z")
    eligible = pd.DataFrame(
        {
            "participant_id": [f"synthetic_boundary_{index:02d}" for index in range(7)],
            "lifecycle_segment": ["new"] * 7,
            "tenure_band": ["0_30d"] * 7,
            "assignment_wave": ["wave_01"] * 7,
            "eligible_at": [assigned_at - pd.Timedelta(days=1)] * 7,
            "assigned_at": [assigned_at] * 7,
        }
    )
    return stratified_factorial_assignment(eligible, seed=9)


def _event(
    *,
    event_id: str,
    participant_id: str,
    event_type: str,
    event_at: object,
    available_at: object,
    delivered_content: str | None = None,
    delivered_cadence: str | None = None,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "participant_id": participant_id,
        "event_type": event_type,
        "event_at": event_at,
        "available_at": available_at,
        "delivered_content": delivered_content,
        "delivered_cadence": delivered_cadence,
    }


def test_contamination_audit_preserves_the_legacy_constructor_shape() -> None:
    audit = ContaminationAudit(10, 0, 0, 0, 0, 0, 0, True)
    assert audit.passed
    assert audit.active_participants_without_in_window_delivery == 0
    assert not audit.active_delivery_coverage_passed


def test_generator_is_deterministic_seven_arm_concurrent_and_public_safe() -> None:
    config = _small_config()
    first = generate_prospective_synthetic(config)
    second = generate_prospective_synthetic(config)
    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    pd.testing.assert_frame_equal(first.events, second.events)
    assert first.data_freeze_at == second.data_freeze_at
    assert set(first.assignments["arm"]) == set(ALL_ARMS)
    assert tuple(first.events.columns) == EVENT_COLUMNS
    assert first.events["event_id"].is_unique
    assert first.events["event_at"].dt.tz is not None
    assert first.events["available_at"].dt.tz is not None

    block_counts = first.assignments.groupby(
        ["lifecycle_segment", "tenure_band", "assignment_wave", "arm"]
    ).size()
    assert block_counts.nunique() == 1
    assigned_times = first.assignments.groupby(
        ["lifecycle_segment", "tenure_band", "assignment_wave"]
    )["assigned_at"].nunique()
    assert assigned_times.eq(1).all()
    assert not first.events.loc[
        first.events["event_type"] == "delivered", "participant_id"
    ].isin(
        first.assignments.loc[
            first.assignments["arm"] == HOLDOUT_ARM, "participant_id"
        ]
    ).any()


def test_generator_rejects_invalid_seed_and_unclippable_probabilities() -> None:
    with pytest.raises(ContractError, match="seed must be non-negative"):
        generate_prospective_synthetic(replace(_small_config(), seed=-1))
    with pytest.raises(ContractError, match="funding effects"):
        generate_prospective_synthetic(
            replace(_small_config(), baseline_funding_rate=0.001)
        )
    with pytest.raises(ContractError, match="guardrail effects"):
        generate_prospective_synthetic(
            replace(_small_config(), daily_unsubscribe_effect=-0.01)
        )
    with pytest.raises(ContractError, match="pre-period effects"):
        generate_prospective_synthetic(
            replace(_small_config(), baseline_pre_period_engagement_rate=0.01)
        )
    with pytest.raises(ContractError, match="at least two units per arm per block"):
        generate_prospective_synthetic(
            replace(_small_config(), units_per_arm_per_block=1)
        )


def test_fourteen_day_window_is_half_open_at_exact_boundary() -> None:
    assignments = _one_block_assignments()
    participants = assignments["participant_id"].iloc[:3].tolist()
    start = assignments["assigned_at"].iloc[0]
    events = pd.DataFrame(
        [
            _event(
                event_id="synthetic_event_boundary_01",
                participant_id=participants[0],
                event_type="funded",
                event_at=start,
                available_at=start + pd.Timedelta(minutes=1),
            ),
            _event(
                event_id="synthetic_event_boundary_02",
                participant_id=participants[1],
                event_type="funded",
                event_at=start + pd.Timedelta(days=14) - pd.Timedelta(microseconds=1),
                available_at=start + pd.Timedelta(days=14),
            ),
            _event(
                event_id="synthetic_event_boundary_03",
                participant_id=participants[2],
                event_type="funded",
                event_at=start + pd.Timedelta(days=14),
                available_at=start + pd.Timedelta(days=14, minutes=1),
            ),
        ],
        columns=EVENT_COLUMNS,
    )
    data = ProspectiveSyntheticData(
        assignments=assignments,
        events=events,
        data_freeze_at=start + pd.Timedelta(days=16),
    )
    result = construct_prospective_analysis(data).set_index("participant_id")
    assert result.loc[participants[0], "funded_14d"] == 1
    assert result.loc[participants[1], "funded_14d"] == 1
    assert result.loc[participants[2], "funded_14d"] == 0
    assert len(result) == len(assignments)


def test_followup_gate_detects_late_arrival_and_incomplete_freeze() -> None:
    data = generate_prospective_synthetic(_small_config())
    participant = data.assignments.iloc[0]
    late = pd.DataFrame(
        [
            _event(
                event_id="synthetic_event_late_arrival",
                participant_id=participant["participant_id"],
                event_type="funded",
                event_at=participant["assigned_at"] + pd.Timedelta(days=1),
                available_at=data.data_freeze_at + pd.Timedelta(minutes=1),
            )
        ],
        columns=EVENT_COLUMNS,
    )
    events = pd.concat([data.events, late], ignore_index=True)
    late_audit = audit_followup(
        data.assignments,
        events,
        data_freeze_at=data.data_freeze_at,
        latency_buffer_hours=_small_config().latency_buffer_hours,
    )
    assert late_audit.late_arriving_in_window_rows == 1
    assert not late_audit.passed
    late_data = ProspectiveSyntheticData(
        assignments=data.assignments,
        events=events,
        data_freeze_at=data.data_freeze_at,
    )
    with pytest.raises(ContractError, match="follow-up audit failed"):
        construct_prospective_analysis(late_data)

    early_freeze = data.assignments["assigned_at"].max() + pd.Timedelta(days=14)
    censored = audit_followup(
        data.assignments,
        data.events,
        data_freeze_at=early_freeze,
        latency_buffer_hours=_small_config().latency_buffer_hours,
    )
    assert censored.censored_units > 0
    assert not censored.passed
    censored_data = ProspectiveSyntheticData(
        assignments=data.assignments,
        events=data.events,
        data_freeze_at=early_freeze,
    )
    with pytest.raises(ContractError, match="follow-up audit failed"):
        construct_prospective_analysis(censored_data)


def test_preperiod_negative_control_must_be_available_before_assignment() -> None:
    data = generate_prospective_synthetic(_small_config())
    event_index = data.events.index[
        data.events["event_type"].eq("pre_period_engaged")
    ][0]
    unavailable = data.events.copy()
    participant = unavailable.loc[event_index, "participant_id"]
    assigned_at = data.assignments.set_index("participant_id").loc[
        participant, "assigned_at"
    ]
    unavailable.loc[event_index, "available_at"] = assigned_at + pd.Timedelta(minutes=1)
    audit = audit_followup(
        data.assignments,
        unavailable,
        data_freeze_at=data.data_freeze_at,
        latency_buffer_hours=_small_config().latency_buffer_hours,
    )
    assert audit.negative_control_not_available_at_assignment_rows == 1
    assert not audit.passed
    unavailable_data = ProspectiveSyntheticData(
        assignments=data.assignments,
        events=unavailable,
        data_freeze_at=data.data_freeze_at,
    )
    with pytest.raises(ContractError, match="unavailable negative controls"):
        construct_prospective_analysis(unavailable_data)


def test_contamination_gate_catches_holdout_cross_factor_and_early_delivery() -> None:
    data = generate_prospective_synthetic(_small_config())
    holdout = data.assignments.loc[data.assignments["arm"] == HOLDOUT_ARM].iloc[0]
    active = data.assignments.loc[data.assignments["arm"] != HOLDOUT_ARM].iloc[0]
    holdout_delivery = pd.DataFrame(
        [
            _event(
                event_id="synthetic_event_holdout_delivery",
                participant_id=holdout["participant_id"],
                event_type="delivered",
                event_at=holdout["assigned_at"] + pd.Timedelta(hours=1),
                available_at=holdout["assigned_at"] + pd.Timedelta(hours=2),
                delivered_content="current",
                delivered_cadence="daily",
            )
        ],
        columns=EVENT_COLUMNS,
    )
    holdout_events = pd.concat([data.events, holdout_delivery], ignore_index=True)
    holdout_audit = audit_contamination(data.assignments, holdout_events)
    assert holdout_audit.holdout_delivery_rows == 1
    assert not holdout_audit.passed

    active_delivery_index = data.events.index[
        data.events["participant_id"].eq(active["participant_id"])
        & data.events["event_type"].eq("delivered")
    ][0]
    mismatch_events = data.events.copy()
    replacement_content = (
        "challenger_b" if active["content"] != "challenger_b" else "current"
    )
    mismatch_events.loc[active_delivery_index, "delivered_content"] = replacement_content
    mismatch_audit = audit_contamination(data.assignments, mismatch_events)
    assert mismatch_audit.cross_content_delivery_rows == 1
    assert not mismatch_audit.passed

    early_events = data.events.copy()
    early_events.loc[active_delivery_index, "event_at"] = (
        active["assigned_at"] - pd.Timedelta(minutes=1)
    )
    early_audit = audit_contamination(data.assignments, early_events)
    assert early_audit.pre_assignment_delivery_rows == 1
    assert not early_audit.passed


def test_active_delivery_coverage_requires_a_correct_in_window_delivery() -> None:
    data = generate_prospective_synthetic(_small_config())
    active = data.assignments.loc[data.assignments["arm"] != HOLDOUT_ARM].iloc[0]
    participant = active["participant_id"]
    without_delivery = data.events.loc[
        ~(
            data.events["participant_id"].eq(participant)
            & data.events["event_type"].eq("delivered")
        )
    ].copy()
    missing_audit = audit_contamination(data.assignments, without_delivery)
    assert missing_audit.active_participants_without_delivery == 1
    assert missing_audit.active_participants_without_in_window_delivery == 1
    assert not missing_audit.active_delivery_coverage_passed
    assert missing_audit.passed

    outside_window = pd.concat(
        [
            without_delivery,
            pd.DataFrame(
                [
                    _event(
                        event_id="synthetic_event_outside_delivery_window",
                        participant_id=participant,
                        event_type="delivered",
                        event_at=active["assigned_at"] + pd.Timedelta(days=14),
                        available_at=(
                            active["assigned_at"]
                            + pd.Timedelta(days=14, minutes=1)
                        ),
                        delivered_content=active["content"],
                        delivered_cadence=active["cadence"],
                    )
                ],
                columns=EVENT_COLUMNS,
            ),
        ],
        ignore_index=True,
    )
    outside_audit = audit_contamination(data.assignments, outside_window)
    assert outside_audit.active_participants_without_delivery == 0
    assert outside_audit.active_participants_without_in_window_delivery == 1
    assert not outside_audit.active_delivery_coverage_passed


def test_event_contract_rejects_orphans_exact_payload_duplicates_and_naive_time() -> None:
    data = generate_prospective_synthetic(_small_config())
    orphan = data.events.copy()
    orphan.loc[0, "participant_id"] = "synthetic_unknown"
    with pytest.raises(ContractError, match="outside assignments"):
        validate_events(data.assignments, orphan)

    duplicate = data.events.iloc[[0]].copy()
    duplicate.loc[:, "event_id"] = "synthetic_event_new_identifier"
    duplicate.loc[:, "available_at"] = (
        duplicate["available_at"] + pd.Timedelta(minutes=1)
    )
    duplicated_events = pd.concat([data.events, duplicate], ignore_index=True)
    with pytest.raises(ContractError, match="duplicate payloads"):
        validate_events(data.assignments, duplicated_events)

    naive = data.events.copy()
    naive["event_at"] = naive["event_at"].astype("object")
    naive.loc[0, "event_at"] = "2026-03-01 12:00:00"
    with pytest.raises(ContractError, match="timezone-naive"):
        validate_events(data.assignments, naive)

    non_string_id = data.events.copy()
    non_string_id["event_id"] = non_string_id["event_id"].astype("object")
    non_string_id.loc[0, "event_id"] = 1
    with pytest.raises(ContractError, match="non-empty strings"):
        validate_events(data.assignments, non_string_id)

    reversed_availability = data.events.copy()
    reversed_availability.loc[0, "available_at"] = (
        reversed_availability.loc[0, "event_at"] - pd.Timedelta(minutes=1)
    )
    with pytest.raises(ContractError, match="before occurrence"):
        validate_events(data.assignments, reversed_availability)


def test_analysis_is_invariant_to_assignment_and_event_row_order() -> None:
    data = generate_prospective_synthetic(_small_config())
    expected = (
        construct_prospective_analysis(data)
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    shuffled = ProspectiveSyntheticData(
        assignments=data.assignments.sample(frac=1.0, random_state=3).reset_index(drop=True),
        events=data.events.sample(frac=1.0, random_state=4).reset_index(drop=True),
        data_freeze_at=data.data_freeze_at,
    )
    actual = (
        construct_prospective_analysis(shuffled)
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert set(BLOCK_COLUMNS) <= set(actual.columns)


def test_canonical_aggregate_is_deterministic_byte_stable_and_contains_no_rows(
    tmp_path,
) -> None:
    config = replace(_small_config(), units_per_arm_per_block=5)
    first = build_prospective_synthetic_benchmark(config)
    second = build_prospective_synthetic_benchmark(config)
    assert first == second
    assert first["artifact_type"] == "synthetic_prospective_factorial_benchmark"
    assert first["data_classification"] == "synthetic"
    assert first["report_scope"] == "aggregate_only"
    assert first["population"]["eligible"] == first["population"]["itt_analyzed"]
    assert tuple(first["quality_gates"]) == REQUIRED_QUALITY_GATES
    assert all(first["quality_gates"].values())
    assert first["pre_period_negative_control"]["passed"]
    assert first["decision"]["status"] == "continue_testing"
    assert first["prospective_power_plan"]["uses_observed_control_rate"] is False
    assert first["schema_version"] == "1.2.0"
    assert first["analysis_contract"]["primary_estimator"] == (
        "block_standardized_difference_in_means"
    )
    assert first["analysis_contract"]["primary_variance"] == (
        "stratified_neyman_conservative"
    )
    assert first["analysis_contract"]["guardrail_bound"].endswith(
        "not_block_adjusted"
    )
    assert all(
        row["estimator"] == "block_standardized_difference_in_means"
        for row in first["primary_funding_family"]
    )
    assert all(
        row["block_adjusted_uncertainty"] is False
        for row in first["guardrail_noninferiority_family"]
    )
    serialized = json.dumps(first, sort_keys=True)
    assert "synthetic_participant_" not in serialized
    assert "synthetic_event_" not in serialized
    assert "participant_id" not in serialized
    assert "event_id" not in serialized

    first_path = write_prospective_synthetic_benchmark(
        tmp_path / "prospective-synthetic-benchmark.json", config
    )
    first_bytes = first_path.read_bytes()
    second_path = write_prospective_synthetic_benchmark(
        tmp_path / "second-prospective-synthetic-benchmark.json", config
    )
    assert second_path.read_bytes() == first_bytes
    assert json.loads(first_bytes) == first

    with pytest.raises(ContractError, match="synthetic-labeled"):
        write_prospective_synthetic_benchmark(tmp_path / "benchmark.json", config)


def test_default_fixture_honestly_returns_continue_testing() -> None:
    report = build_prospective_synthetic_benchmark()
    assert report["decision"]["status"] == "continue_testing"
    assert "primary_superiority_not_established" in report["decision"]["reasons"]
    assert all(report["quality_gates"].values())
    assert tuple(report["quality_gates"]) == REQUIRED_QUALITY_GATES
    candidate = report["config"]["pre_specified_candidate_arm"]
    candidate_result = next(
        row
        for row in report["primary_funding_family"]
        if row["active_arm"] == candidate
    )
    assert not candidate_result["superiority_pass"]
    assert candidate_result["risk_difference"] == pytest.approx(0.015)
    assert candidate_result["risk_difference_standard_error"] == pytest.approx(
        0.00815551,
        rel=1e-5,
    )
    assert candidate_result["risk_difference_ci_low_nominal"] == pytest.approx(
        -0.000985,
        abs=1e-5,
    )
    assert candidate_result["risk_difference_ci_high_nominal"] == pytest.approx(
        0.030985,
        abs=1e-5,
    )
    assert candidate_result["p_value_holm"] == pytest.approx(0.395273, rel=1e-5)
