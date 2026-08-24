import pandas as pd
import pytest

from email_experiment.contracts import ContractError
from email_experiment.funnel import (
    new_link_funnel_membership,
    ordered_funnel_membership,
    summarize_ordered_funnel,
    unsubscribe_metrics,
)


def test_funnel_requires_temporal_order_and_excludes_pre_entry_events() -> None:
    cohort = pd.DataFrame(
        {
            "participant_id": ["synthetic_01", "synthetic_02", "synthetic_03"],
            "assigned_at": ["2026-01-01T00:00:00Z"] * 3,
        }
    )
    events = pd.DataFrame(
        {
            "participant_id": [
                "synthetic_01",
                "synthetic_01",
                "synthetic_01",
                "synthetic_01",
                "synthetic_02",
                "synthetic_03",
                "synthetic_03",
            ],
            "event_type": [
                "linked",
                "opened",
                "linked",
                "converted",
                "linked",
                "opened",
                "linked",
            ],
            "event_at": [
                "2026-01-01T01:00Z",
                "2026-01-01T02:00Z",
                "2026-01-01T03:00Z",
                "2026-01-01T04:00Z",
                "2026-01-01T01:00Z",
                "2025-12-31T23:00Z",
                "2026-01-01T02:00Z",
            ],
        }
    )
    stages = ["opened", "linked", "converted"]
    membership = ordered_funnel_membership(events, stages, cohort=cohort).set_index(
        "participant_id"
    )
    assert membership.loc["synthetic_01", "converted_reached"]
    assert membership.loc["synthetic_01", "linked_at"] == pd.Timestamp(
        "2026-01-01T03:00Z"
    )
    assert not membership.loc["synthetic_02", "opened_reached"]
    assert not membership.loc["synthetic_03", "linked_reached"]

    summary = summarize_ordered_funnel(membership.reset_index(), stages)
    assert list(summary["users"]) == [1, 1, 1]


def test_same_timestamp_does_not_establish_stage_order() -> None:
    events = pd.DataFrame(
        {
            "participant_id": ["synthetic_01", "synthetic_01"],
            "event_type": ["opened", "linked"],
            "event_at": ["2026-01-01T01:00Z", "2026-01-01T01:00Z"],
        }
    )
    result = ordered_funnel_membership(events, ["opened", "linked"])
    assert result.loc[0, "opened_reached"]
    assert not result.loc[0, "linked_reached"]


def test_new_link_estimand_excludes_prelinked_and_prefunded_population() -> None:
    cohort = pd.DataFrame(
        {
            "participant_id": ["synthetic_01", "synthetic_02", "synthetic_03"],
            "assigned_at": ["2026-01-01T00:00:00Z"] * 3,
            "linked_before_entry": [False, True, False],
            "converted_before_entry": [False, False, True],
        }
    )
    events = pd.DataFrame(
        {
            "participant_id": ["synthetic_01"] * 3 + ["synthetic_02"] * 3,
            "event_type": ["opened", "linked", "converted"] * 2,
            "event_at": [
                "2026-01-01T01:00Z",
                "2026-01-01T02:00Z",
                "2026-01-01T03:00Z",
            ]
            * 2,
        }
    )
    result = new_link_funnel_membership(events, cohort)
    assert list(result["participant_id"]) == ["synthetic_01"]
    assert result.loc[0, "converted_reached"]


def test_unsubscribe_user_risk_and_event_rate_keep_distinct_denominators() -> None:
    cohort = pd.DataFrame(
        {
            "participant_id": [
                "synthetic_01",
                "synthetic_02",
                "synthetic_03",
                "synthetic_04",
            ],
            "arm": ["control", "control", "treatment", "control"],
            "messages_delivered": [2, 3, 4, 0],
        }
    )
    events = pd.DataFrame(
        {
            "participant_id": ["synthetic_01", "synthetic_01", "synthetic_03"],
            "event_type": ["unsubscribed", "unsubscribed", "opened"],
        }
    )
    result = unsubscribe_metrics(cohort, events).set_index("arm")
    assert result.loc["control", "unique_user_risk"] == pytest.approx(0.5)
    assert result.loc["control", "actual_recipients"] == 2
    assert result.loc["control", "unsubscribe_events"] == 2
    assert result.loc["control", "events_per_1000_deliveries"] == pytest.approx(400)
    assert result.loc["treatment", "unique_user_risk"] == 0


def test_unsubscribe_rejects_events_for_nonrecipients() -> None:
    cohort = pd.DataFrame(
        {
            "participant_id": ["synthetic_01"],
            "arm": ["control"],
            "messages_delivered": [0],
        }
    )
    events = pd.DataFrame(
        {
            "participant_id": ["synthetic_01"],
            "event_type": ["unsubscribed"],
        }
    )
    with pytest.raises(ContractError, match="no delivered exposure"):
        unsubscribe_metrics(cohort, events)
