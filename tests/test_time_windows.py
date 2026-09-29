import pandas as pd
import pytest

from email_experiment.contracts import ContractError
from email_experiment.time_windows import (
    censoring_summary,
    construct_windowed_outcome,
)


def test_fixed_window_excludes_pre_treatment_and_preserves_censoring() -> None:
    assignments = pd.DataFrame(
        {
            "participant_id": ["synthetic_01", "synthetic_02", "synthetic_03"],
            "arm": ["control", "treatment", "treatment"],
            "assigned_at": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "2026-01-04T00:00:00Z",
            ],
        }
    )
    events = pd.DataFrame(
        {
            "participant_id": ["synthetic_01", "synthetic_01", "synthetic_03"],
            "event_type": ["converted", "converted", "converted"],
            "event_at": [
                "2025-12-31T23:00:00Z",
                "2026-01-02T00:00:00Z",
                "2026-01-05T00:00:00Z",
            ],
        }
    )
    result = construct_windowed_outcome(
        assignments,
        events,
        observation_end="2026-01-08T00:00:00Z",
    ).set_index("participant_id")

    assert result.loc["synthetic_01", "outcome"] == True  # noqa: E712
    assert bool(result.loc["synthetic_01", "has_pre_assignment_outcome"])
    assert result.loc["synthetic_02", "outcome"] == False  # noqa: E712
    assert pd.isna(result.loc["synthetic_03", "outcome"])
    assert not bool(result.loc["synthetic_03", "followup_complete"])

    summary = censoring_summary(result.reset_index())
    treatment = summary.set_index("arm").loc["treatment"]
    assert treatment["complete_followup_rate"] == pytest.approx(0.5)


def test_window_is_half_open_at_exact_deadline() -> None:
    assignments = pd.DataFrame(
        {
            "participant_id": ["synthetic_01"],
            "assigned_at": ["2026-01-01T00:00:00Z"],
        }
    )
    events = pd.DataFrame(
        {
            "participant_id": ["synthetic_01"],
            "event_type": ["converted"],
            "event_at": ["2026-01-08T00:00:00Z"],
        }
    )
    result = construct_windowed_outcome(
        assignments, events, observation_end="2026-01-08T00:00:00Z"
    )
    assert result.loc[0, "outcome"] == False  # noqa: E712


def test_window_rejects_orphan_event_rows() -> None:
    assignments = pd.DataFrame(
        {
            "participant_id": ["synthetic_01"],
            "assigned_at": ["2026-01-01T00:00:00Z"],
        }
    )
    events = pd.DataFrame(
        {
            "participant_id": ["synthetic_unknown"],
            "event_type": ["converted"],
            "event_at": ["2026-01-02T00:00:00Z"],
        }
    )
    with pytest.raises(ContractError, match="unknown participants"):
        construct_windowed_outcome(
            assignments, events, observation_end="2026-01-08T00:00:00Z"
        )


@pytest.mark.parametrize(
    ("assignment_time", "event_time", "observation_end"),
    [
        ("2026-01-01T00:00:00", "2026-01-02T00:00:00Z", "2026-01-08T00:00:00Z"),
        ("2026-01-01T00:00:00Z", "2026-01-02T00:00:00", "2026-01-08T00:00:00Z"),
        ("2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-08T00:00:00"),
    ],
)
def test_window_rejects_timezone_naive_inputs(
    assignment_time: str,
    event_time: str,
    observation_end: str,
) -> None:
    assignments = pd.DataFrame(
        {"participant_id": ["synthetic_01"], "assigned_at": [assignment_time]}
    )
    events = pd.DataFrame(
        {
            "participant_id": ["synthetic_01"],
            "event_type": ["converted"],
            "event_at": [event_time],
        }
    )
    with pytest.raises(ContractError, match="timezone"):
        construct_windowed_outcome(
            assignments, events, observation_end=observation_end
        )


def test_window_normalizes_explicit_timezone_offsets_to_utc() -> None:
    assignments = pd.DataFrame(
        {
            "participant_id": ["synthetic_01"],
            "assigned_at": ["2026-01-01T02:00:00+02:00"],
        }
    )
    events = pd.DataFrame(
        {
            "participant_id": ["synthetic_01"],
            "event_type": ["converted"],
            "event_at": ["2026-01-01T03:00:00+02:00"],
        }
    )
    result = construct_windowed_outcome(
        assignments,
        events,
        observation_end="2026-01-08T00:00:00Z",
    )
    assert result.loc[0, "assigned_at"] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert result.loc[0, "outcome"] == True  # noqa: E712
