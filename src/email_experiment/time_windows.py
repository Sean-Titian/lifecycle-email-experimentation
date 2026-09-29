"""Point-in-time outcome construction with explicit censoring."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from .contracts import (
    ContractError,
    parse_aware_utc_series,
    parse_aware_utc_timestamp,
    require_columns,
    validate_unique_key,
)


def construct_windowed_outcome(
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    *,
    id_col: str = "participant_id",
    assignment_time_col: str = "assigned_at",
    event_time_col: str = "event_at",
    event_type_col: str = "event_type",
    outcome_event: str = "converted",
    window: timedelta | pd.Timedelta = timedelta(days=7),
    observation_end: str | pd.Timestamp,
    fail_on_orphan_events: bool = True,
) -> pd.DataFrame:
    """Build one binary outcome per assignment using a half-open follow-up window.

    The eligible interval is ``[assigned_at, assigned_at + window)``.  Outcomes
    for units without complete follow-up are returned as nullable booleans rather
    than being silently treated as zero.
    """

    require_columns(
        assignments,
        [id_col, assignment_time_col],
        frame_name="assignments",
    )
    require_columns(
        events,
        [id_col, event_time_col, event_type_col],
        frame_name="events",
    )
    validate_unique_key(assignments, id_col, frame_name="assignments")
    if events[id_col].isna().any():
        raise ContractError("events contain missing participant identifiers")

    duration = pd.Timedelta(window)
    if duration <= pd.Timedelta(0):
        raise ContractError("window must be positive")
    end = parse_aware_utc_timestamp(observation_end, name="observation_end")

    output = assignments.copy()
    output[assignment_time_col] = parse_aware_utc_series(
        output[assignment_time_col], name=assignment_time_col
    )
    output["window_end"] = output[assignment_time_col] + duration
    output["followup_complete"] = output["window_end"] <= end

    event_work = events[[id_col, event_time_col, event_type_col]].copy()
    event_work[event_time_col] = parse_aware_utc_series(
        event_work[event_time_col], name=event_time_col
    )
    known_ids = set(output[id_col])
    orphan_count = int((~event_work[id_col].isin(known_ids)).sum())
    if fail_on_orphan_events and orphan_count:
        raise ContractError(f"events contain {orphan_count} rows with unknown participants")
    event_work = event_work[event_work[id_col].isin(known_ids)]

    joined = event_work.merge(
        output[[id_col, assignment_time_col, "window_end"]],
        on=id_col,
        how="left",
        validate="many_to_one",
    )
    joined = joined[joined[event_type_col] == outcome_event]
    joined["is_pre_assignment_outcome"] = (
        joined[event_time_col] < joined[assignment_time_col]
    )
    joined["is_in_window_outcome"] = (
        (joined[event_time_col] >= joined[assignment_time_col])
        & (joined[event_time_col] < joined["window_end"])
        & (joined[event_time_col] <= end)
    )

    first_outcome = (
        joined.loc[joined["is_in_window_outcome"]]
        .groupby(id_col, sort=False)[event_time_col]
        .min()
        .rename("first_outcome_at")
    )
    pre_assignment = joined.groupby(id_col, sort=False)[
        "is_pre_assignment_outcome"
    ].any().rename("has_pre_assignment_outcome")
    # A merge preserves timezone-aware datetime dtype even when no outcome falls
    # inside the window. Series.map on an empty timezone-aware Series regressed
    # in pandas 3.0 by trying to coerce datetimes to float.
    outcome_summary = pd.concat([first_outcome, pre_assignment], axis=1)
    output = output.merge(
        outcome_summary,
        left_on=id_col,
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    output["has_pre_assignment_outcome"] = (
        output["has_pre_assignment_outcome"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    observed = output["first_outcome_at"].notna()
    outcome = pd.Series(pd.NA, index=output.index, dtype="boolean")
    outcome.loc[output["followup_complete"]] = observed.loc[
        output["followup_complete"]
    ]
    output["outcome"] = outcome
    return output


def censoring_summary(
    outcomes: pd.DataFrame,
    *,
    arm_col: str = "arm",
    complete_col: str = "followup_complete",
) -> pd.DataFrame:
    """Summarize complete follow-up by arm to reveal differential censoring."""

    require_columns(outcomes, [arm_col, complete_col], frame_name="outcomes")
    if outcomes[arm_col].isna().any() or outcomes[complete_col].isna().any():
        raise ContractError("arm and follow-up status must be complete")
    if not outcomes[complete_col].isin([True, False, 0, 1]).all():
        raise ContractError("follow-up status must be binary")

    summary = (
        outcomes.groupby(arm_col, sort=True)[complete_col]
        .agg(total="size", complete="sum")
        .reset_index()
    )
    summary["censored"] = summary["total"] - summary["complete"]
    summary["complete_followup_rate"] = summary["complete"] / summary["total"]
    return summary
