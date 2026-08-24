"""Temporal funnels and unsubscribe guardrail metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .contracts import ContractError, require_columns, validate_unique_key


def ordered_funnel_membership(
    events: pd.DataFrame,
    stages: Sequence[str],
    *,
    id_col: str = "participant_id",
    event_col: str = "event_type",
    time_col: str = "event_at",
    cohort: pd.DataFrame | None = None,
    entry_time_col: str = "assigned_at",
) -> pd.DataFrame:
    """Find the first sequential occurrence of each stage for every participant.

    A later stage only counts if it occurs strictly after the previously reached
    stage; same-time events are conservatively treated as unordered.  When a
    cohort is provided, pre-entry events are excluded and cohort members with no
    events remain in the denominator.
    """

    stage_list = list(stages)
    if not stage_list or len(stage_list) != len(set(stage_list)):
        raise ContractError("stages must be a non-empty sequence of unique labels")
    require_columns(events, [id_col, event_col, time_col], frame_name="events")
    if events[[id_col, event_col, time_col]].isna().any().any():
        raise ContractError("funnel events must have complete id, type, and time")

    work = events[[id_col, event_col, time_col]].copy()
    try:
        work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ContractError("funnel event times must be valid timestamps") from exc
    work = work[work[event_col].isin(stage_list)]

    if cohort is None:
        population = pd.DataFrame({id_col: pd.unique(work[id_col])})
    else:
        require_columns(cohort, [id_col, entry_time_col], frame_name="cohort")
        validate_unique_key(cohort, id_col, frame_name="cohort")
        population = cohort.copy()
        try:
            population[entry_time_col] = pd.to_datetime(
                population[entry_time_col], utc=True, errors="raise"
            )
        except (TypeError, ValueError) as exc:
            raise ContractError("cohort entry times must be valid timestamps") from exc
        if population[entry_time_col].isna().any():
            raise ContractError("cohort entry times must be complete")
        unknown = int((~work[id_col].isin(population[id_col])).sum())
        if unknown:
            raise ContractError(f"funnel events contain {unknown} rows outside the cohort")
        work = work.merge(
            population[[id_col, entry_time_col]],
            on=id_col,
            how="left",
            validate="many_to_one",
        )
        work = work[work[time_col] >= work[entry_time_col]]

    work = work.sort_values([id_col, time_col, event_col], kind="stable")
    stage_times: dict[object, list[pd.Timestamp | pd.NaT]] = {}
    for participant, group in work.groupby(id_col, sort=False):
        previous: pd.Timestamp | None = None
        participant_times: list[pd.Timestamp | pd.NaT] = []
        for stage in stage_list:
            candidates = group.loc[group[event_col] == stage, time_col]
            if previous is not None:
                candidates = candidates[candidates > previous]
            if candidates.empty:
                # Once a stage is missed, all later stages remain unreached.
                participant_times.extend(
                    [pd.NaT] * (len(stage_list) - len(participant_times))
                )
                break
            reached_at = candidates.iloc[0]
            participant_times.append(reached_at)
            previous = reached_at
        stage_times[participant] = participant_times

    output = population.copy()
    for index, stage in enumerate(stage_list):
        mapping = {
            participant: times[index]
            for participant, times in stage_times.items()
            if len(times) > index
        }
        output[f"{stage}_at"] = output[id_col].map(mapping)
        output[f"{stage}_reached"] = output[f"{stage}_at"].notna()
    return output


def new_link_funnel_membership(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    id_col: str = "participant_id",
    event_col: str = "event_type",
    time_col: str = "event_at",
    entry_time_col: str = "assigned_at",
    prelinked_col: str = "linked_before_entry",
    preconverted_col: str = "converted_before_entry",
    stages: Sequence[str] = ("opened", "linked", "converted"),
) -> pd.DataFrame:
    """Build a new-link funnel after excluding ineligible pre-existing states.

    Someone already linked or converted before entry cannot contribute to a
    post-entry *new-link* estimand.  Requiring both baseline flags prevents a
    population-state funnel from being mislabeled as acquisition progression.
    """

    require_columns(
        cohort,
        [id_col, entry_time_col, prelinked_col, preconverted_col],
        frame_name="cohort",
    )
    validate_unique_key(cohort, id_col, frame_name="cohort")
    for column in (prelinked_col, preconverted_col):
        if cohort[column].isna().any() or not cohort[column].isin(
            [True, False, 0, 1]
        ).all():
            raise ContractError(f"{column} must be complete and binary")
    eligible = cohort.loc[
        ~cohort[prelinked_col].astype(bool)
        & ~cohort[preconverted_col].astype(bool)
    ].copy()
    eligible_ids = set(eligible[id_col])
    eligible_events = events.loc[events[id_col].isin(eligible_ids)].copy()
    return ordered_funnel_membership(
        eligible_events,
        stages,
        id_col=id_col,
        event_col=event_col,
        time_col=time_col,
        cohort=eligible,
        entry_time_col=entry_time_col,
    )


def summarize_ordered_funnel(
    membership: pd.DataFrame,
    stages: Sequence[str],
) -> pd.DataFrame:
    """Summarize ordered membership using participant counts, not event counts."""

    stage_list = list(stages)
    reached_columns = [f"{stage}_reached" for stage in stage_list]
    require_columns(membership, reached_columns, frame_name="funnel membership")
    total = len(membership)
    if total == 0:
        raise ContractError("funnel membership must contain at least one participant")

    rows: list[dict[str, float | int | str]] = []
    previous = total
    for stage, column in zip(stage_list, reached_columns, strict=True):
        if membership[column].isna().any() or not membership[column].isin(
            [True, False, 0, 1]
        ).all():
            raise ContractError("funnel reached columns must be complete and binary")
        users = int(membership[column].sum())
        rows.append(
            {
                "stage": stage,
                "users": users,
                "rate_from_population": users / total,
                "rate_from_previous": users / previous if previous else np.nan,
            }
        )
        previous = users
    return pd.DataFrame(rows)


def unsubscribe_metrics(
    cohort: pd.DataFrame,
    events: pd.DataFrame,
    *,
    id_col: str = "participant_id",
    group_col: str = "arm",
    exposure_col: str = "messages_delivered",
    event_col: str = "event_type",
    unsubscribe_event: str = "unsubscribed",
) -> pd.DataFrame:
    """Report both unique-user risk and events per delivered-message exposure."""

    require_columns(
        cohort,
        [id_col, group_col, exposure_col],
        frame_name="cohort",
    )
    require_columns(events, [id_col, event_col], frame_name="events")
    validate_unique_key(cohort, id_col, frame_name="cohort")
    if cohort[group_col].isna().any():
        raise ContractError("cohort group labels must be complete")
    exposures = pd.to_numeric(cohort[exposure_col], errors="coerce")
    if exposures.isna().any() or (~np.isfinite(exposures)).any() or (exposures < 0).any():
        raise ContractError("message exposures must be finite and non-negative")
    if not np.equal(exposures, np.floor(exposures)).all():
        raise ContractError("message exposures must be integer counts")

    event_work = events[[id_col, event_col]].copy()
    if event_work[id_col].isna().any():
        raise ContractError("events contain missing participant identifiers")
    unknown = int((~event_work[id_col].isin(cohort[id_col])).sum())
    if unknown:
        raise ContractError(f"events contain {unknown} rows outside the cohort")
    unsubscribe_counts = (
        event_work.loc[event_work[event_col] == unsubscribe_event]
        .groupby(id_col)
        .size()
    )

    working = cohort[[id_col, group_col]].copy()
    working["exposures"] = exposures.astype(float)
    working["unsubscribe_events"] = (
        working[id_col].map(unsubscribe_counts).fillna(0).astype(int)
    )
    invalid_events = (working["exposures"] == 0) & (
        working["unsubscribe_events"] > 0
    )
    if invalid_events.any():
        raise ContractError(
            "unsubscribe events include participants with no delivered exposure"
        )
    # Person-level campaign risk is defined only among actual recipients. Keeping
    # zero-delivery assignments in this denominator would dilute the guardrail.
    working = working.loc[working["exposures"] > 0].copy()
    if working.empty:
        raise ContractError("unsubscribe risk requires at least one actual recipient")
    working["unsubscribed"] = working["unsubscribe_events"] > 0
    summary = (
        working.groupby(group_col, sort=True)
        .agg(
            actual_recipients=(id_col, "size"),
            unique_unsubscribers=("unsubscribed", "sum"),
            unsubscribe_events=("unsubscribe_events", "sum"),
            delivered_exposures=("exposures", "sum"),
        )
        .reset_index()
    )
    summary["unique_user_risk"] = (
        summary["unique_unsubscribers"] / summary["actual_recipients"]
    )
    summary["events_per_1000_deliveries"] = np.where(
        summary["delivered_exposures"] > 0,
        1000.0 * summary["unsubscribe_events"] / summary["delivered_exposures"],
        np.nan,
    )
    return summary
