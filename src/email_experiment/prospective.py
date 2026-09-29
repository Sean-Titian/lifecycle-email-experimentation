"""Prospective, public-safe synthetic lifecycle experiment harness.

This module is intentionally separate from the retrospective source benchmark.
It generates a generic seven-arm randomized study, applies a fixed 14-day
intention-to-treat clock, and emits aggregate method evidence only.  None of the
synthetic effects are evidence about a real campaign.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .assignment import (
    ACTIVE_ARMS,
    ALL_ARMS,
    CADENCE_LEVELS,
    CONTENT_LEVELS,
    HOLDOUT_ARM,
    assignment_sha256,
    audit_sample_ratio,
    stratified_factorial_assignment,
    validate_assignments,
)
from .contracts import (
    ContractError,
    parse_aware_utc_series,
    parse_aware_utc_timestamp,
    require_exact_columns,
    validate_unique_key,
)
from .decision import (
    REQUIRED_QUALITY_GATES,
    evaluate_guardrail_family,
    evaluate_primary_family,
    make_launch_decision,
)
from .statistics import approximate_mde
from .time_windows import construct_windowed_outcome

EVENT_COLUMNS = (
    "event_id",
    "participant_id",
    "event_type",
    "event_at",
    "available_at",
    "delivered_content",
    "delivered_cadence",
)
EVENT_TYPES = (
    "delivered",
    "funded",
    "unsubscribed",
    "complained",
    "pre_period_engaged",
)
OUTCOME_EVENT_MAP = {
    "funded_14d": "funded",
    "unsubscribe_14d": "unsubscribed",
    "complaint_14d": "complained",
}
LIFECYCLE_SEGMENTS = ("new", "established", "reactivation")
TENURE_BANDS = ("0_30d", "31_90d")
FOLLOWUP_DAYS = 14


@dataclass(frozen=True)
class ProspectiveSyntheticConfig:
    """Declared parameters for the generic prospective simulation."""

    seed: int = 20260908
    units_per_arm_per_block: int = 100
    assignment_waves: int = 2
    days_between_waves: int = 7
    followup_days: int = FOLLOWUP_DAYS
    latency_buffer_hours: int = 48
    maximum_arrival_lag_hours: int = 36
    baseline_funding_rate: float = 0.040
    challenger_a_funding_effect: float = 0.002
    challenger_b_funding_effect: float = -0.001
    daily_funding_effect: float = 0.0005
    baseline_unsubscribe_rate: float = 0.004
    active_unsubscribe_effect: float = 0.0002
    daily_unsubscribe_effect: float = 0.0003
    baseline_complaint_rate: float = 0.001
    active_complaint_effect: float = 0.0001
    daily_complaint_effect: float = 0.0001
    baseline_pre_period_engagement_rate: float = 0.25
    srm_alpha: float = 0.01
    family_alpha: float = 0.05
    unsubscribe_noninferiority_margin: float = 0.005
    complaint_noninferiority_margin: float = 0.003
    pre_specified_candidate_arm: str = "challenger_a__twice_weekly"


@dataclass(frozen=True)
class ProspectiveSyntheticData:
    assignments: pd.DataFrame
    events: pd.DataFrame
    data_freeze_at: pd.Timestamp


@dataclass(frozen=True)
class ContaminationAudit:
    delivery_rows: int
    active_participants_without_delivery: int
    holdout_delivery_rows: int
    cross_content_delivery_rows: int
    cross_cadence_delivery_rows: int
    pre_assignment_delivery_rows: int
    affected_participants: int
    passed: bool
    active_participants_without_in_window_delivery: int = 0
    active_delivery_coverage_passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FollowupAudit:
    followup_days: int
    latency_buffer_hours: int
    randomized_units: int
    censored_units: int
    late_arriving_in_window_rows: int
    pre_assignment_outcome_rows: int
    negative_control_after_assignment_rows: int
    negative_control_not_available_at_assignment_rows: int
    aligned_14_day_followup: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_config(config: ProspectiveSyntheticConfig) -> None:
    integer_fields = {
        "seed": config.seed,
        "units_per_arm_per_block": config.units_per_arm_per_block,
        "assignment_waves": config.assignment_waves,
        "days_between_waves": config.days_between_waves,
        "followup_days": config.followup_days,
        "latency_buffer_hours": config.latency_buffer_hours,
        "maximum_arrival_lag_hours": config.maximum_arrival_lag_hours,
    }
    for name, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ContractError(f"{name} must be an integer")
    if config.units_per_arm_per_block < 1 or config.assignment_waves < 1:
        raise ContractError("the prospective design requires positive block and wave sizes")
    if config.seed < 0:
        raise ContractError("seed must be non-negative")
    if config.days_between_waves < 1:
        raise ContractError("days_between_waves must be positive")
    if config.followup_days != FOLLOWUP_DAYS:
        raise ContractError("the prospective primary follow-up must be exactly 14 days")
    if config.latency_buffer_hours < 1 or config.maximum_arrival_lag_hours < 0:
        raise ContractError("latency durations must be non-negative with a positive buffer")
    if config.maximum_arrival_lag_hours >= config.latency_buffer_hours:
        raise ContractError("maximum event lag must be shorter than the data-freeze buffer")

    base_rates = {
        "baseline_funding_rate": config.baseline_funding_rate,
        "baseline_unsubscribe_rate": config.baseline_unsubscribe_rate,
        "baseline_complaint_rate": config.baseline_complaint_rate,
        "baseline_pre_period_engagement_rate": config.baseline_pre_period_engagement_rate,
    }
    for name, value in base_rates.items():
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            raise ContractError(f"{name} must be finite and between zero and one")
    margins = {
        "unsubscribe_noninferiority_margin": config.unsubscribe_noninferiority_margin,
        "complaint_noninferiority_margin": config.complaint_noninferiority_margin,
    }
    for name, value in margins.items():
        if not np.isfinite(value) or not 0.0 <= value < 1.0:
            raise ContractError(f"{name} must be finite and in [0, 1)")
    for name, value in {
        "srm_alpha": config.srm_alpha,
        "family_alpha": config.family_alpha,
    }.items():
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            raise ContractError(f"{name} must be finite and between zero and one")
    if config.family_alpha != 0.05:
        raise ContractError("the pre-registered primary and guardrail family alpha must be 0.05")
    if config.pre_specified_candidate_arm not in ACTIVE_ARMS:
        raise ContractError("pre_specified_candidate_arm must be an active factorial cell")

    content_effects = (
        0.0,
        config.challenger_a_funding_effect,
        config.challenger_b_funding_effect,
    )
    cadence_effects = (config.daily_funding_effect, 0.0)
    funding_probabilities = [
        config.baseline_funding_rate
        + segment_effect
        + tenure_effect
        + content_effect
        + cadence_effect
        for segment_effect in (-0.006, 0.0, 0.008)
        for tenure_effect in (-0.002, 0.002)
        for content_effect in content_effects
        for cadence_effect in cadence_effects
    ]
    guardrail_probabilities = [
        baseline + active * active_effect + daily * daily_effect
        for baseline, active_effect, daily_effect in (
            (
                config.baseline_unsubscribe_rate,
                config.active_unsubscribe_effect,
                config.daily_unsubscribe_effect,
            ),
            (
                config.baseline_complaint_rate,
                config.active_complaint_effect,
                config.daily_complaint_effect,
            ),
        )
        for active, daily in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    ]
    preperiod_probabilities = [
        config.baseline_pre_period_engagement_rate + segment_effect + tenure_effect
        for segment_effect in (-0.08, 0.0, 0.08)
        for tenure_effect in (-0.02, 0.02)
    ]
    if any(not 0.0 <= probability <= 1.0 for probability in funding_probabilities):
        raise ContractError("declared funding effects produce an invalid probability")
    if any(not 0.0 <= probability <= 1.0 for probability in guardrail_probabilities):
        raise ContractError("declared guardrail effects produce an invalid probability")
    if any(not 0.0 <= probability <= 1.0 for probability in preperiod_probabilities):
        raise ContractError("declared pre-period effects produce an invalid probability")


def _eligible_population(config: ProspectiveSyntheticConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = 0
    start = pd.Timestamp("2026-03-01T12:00:00Z")
    block_size = config.units_per_arm_per_block * len(ALL_ARMS)
    for segment in LIFECYCLE_SEGMENTS:
        for tenure in TENURE_BANDS:
            for wave_index in range(config.assignment_waves):
                assigned_at = start + pd.Timedelta(
                    days=wave_index * config.days_between_waves
                )
                for _ in range(block_size):
                    rows.append(
                        {
                            "participant_id": f"synthetic_participant_{index:08d}",
                            "lifecycle_segment": segment,
                            "tenure_band": tenure,
                            "assignment_wave": f"wave_{wave_index + 1:02d}",
                            "eligible_at": assigned_at - pd.Timedelta(days=1),
                            "assigned_at": assigned_at,
                        }
                    )
                    index += 1
    return pd.DataFrame(rows)


def _probabilities(assignments: pd.DataFrame, config: ProspectiveSyntheticConfig) -> pd.DataFrame:
    segment_funding = assignments["lifecycle_segment"].map(
        {"new": -0.006, "established": 0.0, "reactivation": 0.008}
    )
    tenure_funding = assignments["tenure_band"].map({"0_30d": -0.002, "31_90d": 0.002})
    content_funding = assignments["content"].map(
        {
            "current": 0.0,
            "challenger_a": config.challenger_a_funding_effect,
            "challenger_b": config.challenger_b_funding_effect,
        }
    ).fillna(0.0)
    cadence_funding = assignments["cadence"].map(
        {"daily": config.daily_funding_effect, "twice_weekly": 0.0}
    ).fillna(0.0)
    active = assignments["arm"].ne(HOLDOUT_ARM).astype(float)
    daily = assignments["cadence"].eq("daily").fillna(False).astype(float)
    segment_preperiod = assignments["lifecycle_segment"].map(
        {"new": -0.08, "established": 0.0, "reactivation": 0.08}
    )
    tenure_preperiod = assignments["tenure_band"].map({"0_30d": -0.02, "31_90d": 0.02})
    return pd.DataFrame(
        {
            "funded": np.clip(
                config.baseline_funding_rate
                + segment_funding
                + tenure_funding
                + content_funding
                + cadence_funding,
                0.0,
                1.0,
            ),
            "unsubscribed": np.clip(
                config.baseline_unsubscribe_rate
                + active * config.active_unsubscribe_effect
                + daily * config.daily_unsubscribe_effect,
                0.0,
                1.0,
            ),
            "complained": np.clip(
                config.baseline_complaint_rate
                + active * config.active_complaint_effect
                + daily * config.daily_complaint_effect,
                0.0,
                1.0,
            ),
            "pre_period_engaged": np.clip(
                config.baseline_pre_period_engagement_rate
                + segment_preperiod
                + tenure_preperiod,
                0.0,
                1.0,
            ),
        }
    )


def _generate_events(
    assignments: pd.DataFrame,
    config: ProspectiveSyntheticConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + 1)
    probabilities = _probabilities(assignments, config)
    draws = {
        outcome: rng.random(len(assignments)) < probabilities[outcome].to_numpy()
        for outcome in probabilities
    }
    maximum_lag_minutes = config.maximum_arrival_lag_hours * 60
    followup_minutes = config.followup_days * 24 * 60
    event_rows: list[dict[str, object]] = []

    def append_event(
        participant_id: str,
        event_type: str,
        event_at: pd.Timestamp,
        *,
        delivered_content: str | None = None,
        delivered_cadence: str | None = None,
    ) -> None:
        lag = int(rng.integers(0, maximum_lag_minutes + 1))
        event_rows.append(
            {
                "event_id": f"synthetic_event_{len(event_rows):09d}",
                "participant_id": participant_id,
                "event_type": event_type,
                "event_at": event_at,
                "available_at": event_at + pd.Timedelta(minutes=lag),
                "delivered_content": delivered_content,
                "delivered_cadence": delivered_cadence,
            }
        )

    for index, row in enumerate(assignments.itertuples(index=False)):
        participant_id = row.participant_id
        assigned_at = row.assigned_at
        if row.arm != HOLDOUT_ARM:
            delivery_days = range(config.followup_days) if row.cadence == "daily" else (0, 3, 7, 10)
            for day in delivery_days:
                append_event(
                    participant_id,
                    "delivered",
                    assigned_at + pd.Timedelta(days=day, hours=2),
                    delivered_content=row.content,
                    delivered_cadence=row.cadence,
                )
        if draws["funded"][index]:
            append_event(
                participant_id,
                "funded",
                assigned_at
                + pd.Timedelta(minutes=int(rng.integers(1, followup_minutes))),
            )
        if draws["unsubscribed"][index]:
            append_event(
                participant_id,
                "unsubscribed",
                assigned_at
                + pd.Timedelta(minutes=int(rng.integers(1, followup_minutes))),
            )
        if draws["complained"][index]:
            append_event(
                participant_id,
                "complained",
                assigned_at
                + pd.Timedelta(minutes=int(rng.integers(1, followup_minutes))),
            )
        if draws["pre_period_engaged"][index]:
            append_event(
                participant_id,
                "pre_period_engaged",
                assigned_at - pd.Timedelta(days=int(rng.integers(3, 31))),
            )

    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    for column in ("event_id", "participant_id", "event_type"):
        events[column] = events[column].astype("string")
    for column in ("delivered_content", "delivered_cadence"):
        events[column] = events[column].astype("string")
    events["event_at"] = pd.to_datetime(events["event_at"], utc=True)
    events["available_at"] = pd.to_datetime(events["available_at"], utc=True)
    return events.sort_values("event_id", kind="stable").reset_index(drop=True)


def generate_prospective_synthetic(
    config: ProspectiveSyntheticConfig | None = None,
) -> ProspectiveSyntheticData:
    """Generate a deterministic, independently authored prospective fixture."""

    if config is None:
        config = ProspectiveSyntheticConfig()
    _validate_config(config)
    eligible = _eligible_population(config)
    assignments = stratified_factorial_assignment(eligible, seed=config.seed)
    events = _generate_events(assignments, config)
    data_freeze_at = (
        assignments["assigned_at"].max()
        + pd.Timedelta(days=config.followup_days)
        + pd.Timedelta(hours=config.latency_buffer_hours)
    )
    return ProspectiveSyntheticData(assignments, events, data_freeze_at)


def validate_events(
    assignments: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the exact event contract without silently repairing rows."""

    assignment_work = validate_assignments(assignments)
    require_exact_columns(events, EVENT_COLUMNS, frame_name="events")
    validate_unique_key(events, "event_id", frame_name="events")
    working = events.loc[:, EVENT_COLUMNS].copy()
    for column in ("event_id", "participant_id", "event_type"):
        if working[column].isna().any() or not working[column].map(
            lambda value: isinstance(value, str)
        ).all():
            raise ContractError(f"{column} must contain complete non-empty strings")
        working[column] = working[column].astype("string")
        if working[column].str.strip().eq("").any():
            raise ContractError(f"{column} must contain complete non-empty strings")
    invalid_types = int((~working["event_type"].isin(EVENT_TYPES)).sum())
    if invalid_types:
        raise ContractError(f"events contain {invalid_types} unsupported event types")
    working["event_at"] = parse_aware_utc_series(
        working["event_at"], name="event_at"
    )
    working["available_at"] = parse_aware_utc_series(
        working["available_at"], name="available_at"
    )
    before_occurrence = int((working["available_at"] < working["event_at"]).sum())
    if before_occurrence:
        raise ContractError(
            f"events contain {before_occurrence} availability times before occurrence"
        )
    duplicate_payload_rows = int(
        working.duplicated(
            [
                "participant_id",
                "event_type",
                "event_at",
                "delivered_content",
                "delivered_cadence",
            ],
            keep=False,
        ).sum()
    )
    if duplicate_payload_rows:
        raise ContractError(
            f"events contain {duplicate_payload_rows} rows with exact duplicate payloads"
        )
    orphan_rows = int(
        (~working["participant_id"].isin(assignment_work["participant_id"])).sum()
    )
    if orphan_rows:
        raise ContractError(f"events contain {orphan_rows} rows outside assignments")

    content = working["delivered_content"].astype("string")
    cadence = working["delivered_cadence"].astype("string")
    delivery = working["event_type"].eq("delivered")
    missing_delivery_factors = int((delivery & (content.isna() | cadence.isna())).sum())
    non_delivery_factors = int((~delivery & (content.notna() | cadence.notna())).sum())
    invalid_content = int((delivery & ~content.isin(CONTENT_LEVELS)).sum())
    invalid_cadence = int((delivery & ~cadence.isin(CADENCE_LEVELS)).sum())
    if missing_delivery_factors or non_delivery_factors or invalid_content or invalid_cadence:
        raise ContractError(
            "event factor contract failed with "
            f"{missing_delivery_factors} missing delivery, {non_delivery_factors} "
            f"non-delivery, {invalid_content} invalid content, and "
            f"{invalid_cadence} invalid cadence rows"
        )
    working["delivered_content"] = content
    working["delivered_cadence"] = cadence
    return working.loc[:, EVENT_COLUMNS]


def audit_contamination(
    assignments: pd.DataFrame,
    events: pd.DataFrame,
) -> ContaminationAudit:
    """Audit contamination and minimum in-window active delivery coverage."""

    assignment_work = validate_assignments(assignments)
    event_work = validate_events(assignment_work, events)
    deliveries = event_work.loc[event_work["event_type"] == "delivered"].merge(
        assignment_work[
            ["participant_id", "arm", "content", "cadence", "assigned_at"]
        ],
        on="participant_id",
        how="left",
        validate="many_to_one",
    )
    holdout = deliveries["arm"].eq(HOLDOUT_ARM)
    content_mismatch = ~holdout & deliveries["delivered_content"].ne(
        deliveries["content"]
    )
    cadence_mismatch = ~holdout & deliveries["delivered_cadence"].ne(
        deliveries["cadence"]
    )
    pre_assignment = deliveries["event_at"] < deliveries["assigned_at"]
    contaminated = holdout | content_mismatch | cadence_mismatch | pre_assignment
    active_ids = set(
        assignment_work.loc[
            assignment_work["arm"] != HOLDOUT_ARM, "participant_id"
        ]
    )
    in_window = deliveries["event_at"] < (
        deliveries["assigned_at"] + pd.Timedelta(days=FOLLOWUP_DAYS)
    )
    valid_active_delivery = (
        ~holdout
        & ~content_mismatch
        & ~cadence_mismatch
        & ~pre_assignment
        & in_window
    )
    any_active_delivery_ids = set(deliveries.loc[~holdout, "participant_id"])
    valid_delivery_ids = set(
        deliveries.loc[valid_active_delivery, "participant_id"]
    )
    missing_any_active_deliveries = len(active_ids - any_active_delivery_ids)
    missing_valid_active_deliveries = len(active_ids - valid_delivery_ids)
    audit = ContaminationAudit(
        delivery_rows=len(deliveries),
        active_participants_without_delivery=missing_any_active_deliveries,
        active_participants_without_in_window_delivery=(
            missing_valid_active_deliveries
        ),
        active_delivery_coverage_passed=missing_valid_active_deliveries == 0,
        holdout_delivery_rows=int(holdout.sum()),
        cross_content_delivery_rows=int(content_mismatch.sum()),
        cross_cadence_delivery_rows=int(cadence_mismatch.sum()),
        pre_assignment_delivery_rows=int(pre_assignment.sum()),
        affected_participants=int(deliveries.loc[contaminated, "participant_id"].nunique()),
        passed=not bool(contaminated.any()),
    )
    return audit


def audit_followup(
    assignments: pd.DataFrame,
    events: pd.DataFrame,
    *,
    data_freeze_at: object,
    followup_days: int = FOLLOWUP_DAYS,
    latency_buffer_hours: int = 48,
) -> FollowupAudit:
    """Audit aligned clocks, censoring, availability lag, and temporal direction."""

    if followup_days != FOLLOWUP_DAYS:
        raise ContractError("the prospective primary follow-up must be exactly 14 days")
    if latency_buffer_hours < 0:
        raise ContractError("latency_buffer_hours must be non-negative")
    assignment_work = validate_assignments(assignments)
    event_work = validate_events(assignment_work, events)
    freeze = parse_aware_utc_timestamp(data_freeze_at, name="data_freeze_at")
    assignment_times = assignment_work[["participant_id", "assigned_at"]].copy()
    assignment_times["window_end"] = assignment_times["assigned_at"] + pd.Timedelta(
        days=followup_days
    )
    assignment_times["complete_at"] = assignment_times["window_end"] + pd.Timedelta(
        hours=latency_buffer_hours
    )
    censored = assignment_times["complete_at"] > freeze
    joined = event_work.merge(
        assignment_times,
        on="participant_id",
        how="left",
        validate="many_to_one",
    )
    in_window = (
        (joined["event_at"] >= joined["assigned_at"])
        & (joined["event_at"] < joined["window_end"])
    )
    late = in_window & (joined["available_at"] > freeze)
    outcome = joined["event_type"].isin(OUTCOME_EVENT_MAP.values())
    pre_assignment_outcome = outcome & (joined["event_at"] < joined["assigned_at"])
    negative_control_after = joined["event_type"].eq("pre_period_engaged") & (
        joined["event_at"] >= joined["assigned_at"]
    )
    negative_control_unavailable = joined["event_type"].eq("pre_period_engaged") & (
        joined["available_at"] > joined["assigned_at"]
    )
    aligned = bool(
        (
            assignment_times["window_end"] - assignment_times["assigned_at"]
            == pd.Timedelta(days=FOLLOWUP_DAYS)
        ).all()
    )
    passed = bool(
        aligned
        and not censored.any()
        and not late.any()
        and not pre_assignment_outcome.any()
        and not negative_control_after.any()
        and not negative_control_unavailable.any()
    )
    return FollowupAudit(
        followup_days=followup_days,
        latency_buffer_hours=latency_buffer_hours,
        randomized_units=len(assignment_work),
        censored_units=int(censored.sum()),
        late_arriving_in_window_rows=int(late.sum()),
        pre_assignment_outcome_rows=int(pre_assignment_outcome.sum()),
        negative_control_after_assignment_rows=int(negative_control_after.sum()),
        negative_control_not_available_at_assignment_rows=int(
            negative_control_unavailable.sum()
        ),
        aligned_14_day_followup=aligned,
        passed=passed,
    )


def construct_prospective_analysis(
    data: ProspectiveSyntheticData,
    *,
    followup_days: int = FOLLOWUP_DAYS,
    latency_buffer_hours: int = 48,
) -> pd.DataFrame:
    """Construct one complete 14-day ITT row per randomized synthetic unit."""

    if followup_days != FOLLOWUP_DAYS:
        raise ContractError("the prospective primary follow-up must be exactly 14 days")
    followup = audit_followup(
        data.assignments,
        data.events,
        data_freeze_at=data.data_freeze_at,
        followup_days=followup_days,
        latency_buffer_hours=latency_buffer_hours,
    )
    if not followup.passed:
        raise ContractError(
            "prospective follow-up audit failed with "
            f"{followup.censored_units} censored units, "
            f"{followup.late_arriving_in_window_rows} late rows, "
            f"{followup.pre_assignment_outcome_rows} pre-assignment outcomes, and "
            f"{followup.negative_control_after_assignment_rows} mistimed negative controls, "
            "with "
            f"{followup.negative_control_not_available_at_assignment_rows} unavailable "
            "negative controls"
        )
    assignments = validate_assignments(data.assignments)
    events = validate_events(assignments, data.events)
    freeze = parse_aware_utc_timestamp(data.data_freeze_at, name="data_freeze_at")
    available = events.loc[events["available_at"] <= freeze].copy()
    analysis = assignments.loc[:, ["participant_id", "arm", "content", "cadence"]].copy()
    for outcome_column, event_type in OUTCOME_EVENT_MAP.items():
        outcome = construct_windowed_outcome(
            assignments,
            available,
            outcome_event=event_type,
            window=pd.Timedelta(days=followup_days),
            observation_end=freeze,
        )[["participant_id", "outcome"]]
        analysis = analysis.merge(
            outcome.rename(columns={"outcome": outcome_column}),
            on="participant_id",
            how="left",
            validate="one_to_one",
        )
    outcome_columns = list(OUTCOME_EVENT_MAP)
    incomplete = int(analysis[outcome_columns].isna().any(axis=1).sum())
    if incomplete:
        raise ContractError(
            f"prospective ITT analysis has {incomplete} units without complete follow-up"
        )
    for column in outcome_columns:
        analysis[column] = analysis[column].astype(int)

    preperiod = available.loc[
        available["event_type"] == "pre_period_engaged",
        ["participant_id", "event_at", "available_at"],
    ].merge(
        assignments[["participant_id", "assigned_at"]],
        on="participant_id",
        how="left",
        validate="many_to_one",
    )
    preperiod = preperiod.loc[
        (preperiod["event_at"] < preperiod["assigned_at"])
        & (preperiod["available_at"] <= preperiod["assigned_at"])
    ]
    preperiod_ids = set(preperiod["participant_id"])
    analysis["pre_period_engaged"] = analysis["participant_id"].isin(preperiod_ids).astype(int)
    if len(analysis) != len(assignments) or not analysis["participant_id"].is_unique:
        raise ContractError("prospective analysis changed the randomized ITT population")
    return analysis


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def _event_type_counts(events: pd.DataFrame) -> dict[str, int]:
    counts = events["event_type"].value_counts().reindex(EVENT_TYPES, fill_value=0)
    return {event_type: int(counts[event_type]) for event_type in EVENT_TYPES}


def build_prospective_synthetic_benchmark(
    config: ProspectiveSyntheticConfig | None = None,
) -> dict[str, object]:
    """Run the full prospective method check and return aggregate evidence."""

    if config is None:
        config = ProspectiveSyntheticConfig()
    _validate_config(config)
    data = generate_prospective_synthetic(config)
    assignments = validate_assignments(data.assignments)
    events = validate_events(assignments, data.events)
    srm = audit_sample_ratio(assignments, alpha=config.srm_alpha)
    contamination = audit_contamination(assignments, events)
    followup = audit_followup(
        assignments,
        events,
        data_freeze_at=data.data_freeze_at,
        followup_days=config.followup_days,
        latency_buffer_hours=config.latency_buffer_hours,
    )
    analysis = construct_prospective_analysis(
        data,
        followup_days=config.followup_days,
        latency_buffer_hours=config.latency_buffer_hours,
    )
    primary = evaluate_primary_family(analysis, alpha=config.family_alpha)
    guardrails = evaluate_guardrail_family(
        analysis,
        margins={
            "unsubscribe_14d": config.unsubscribe_noninferiority_margin,
            "complaint_14d": config.complaint_noninferiority_margin,
        },
        alpha=config.family_alpha,
    )
    negative_control = evaluate_primary_family(
        analysis,
        outcome_col="pre_period_engaged",
        alpha=config.family_alpha,
    )
    negative_control = negative_control.copy()
    negative_control["family"] = "pre_period_negative_control"
    negative_control_passed = not bool(negative_control["holm_significant"].any())
    concurrent_holdout = bool(
        assignments.groupby(
            ["lifecycle_segment", "tenure_band", "assignment_wave"], sort=True
        )["assigned_at"]
        .nunique()
        .eq(1)
        .all()
    )
    quality_gates = {
        "assignment_contract": True,
        "exact_block_allocation": True,
        "sample_ratio": bool(srm.passed),
        "concurrent_holdout": concurrent_holdout,
        "aligned_14_day_followup": bool(followup.aligned_14_day_followup),
        "complete_followup_and_latency_buffer": bool(followup.passed),
        "no_contamination": bool(contamination.passed),
        "active_delivery_coverage": bool(
            contamination.active_delivery_coverage_passed
        ),
        "pre_period_negative_control": negative_control_passed,
        "itt_population_preserved": len(analysis) == len(assignments),
    }
    if tuple(quality_gates) != REQUIRED_QUALITY_GATES:
        raise ContractError(
            "canonical quality-gate schema does not match the decision contract"
        )
    decision = make_launch_decision(
        primary,
        guardrails,
        quality_gates,
        candidate_arm=config.pre_specified_candidate_arm,
    )
    per_arm_size = int(assignments["arm"].value_counts().min())
    planning_alpha = config.family_alpha / len(ACTIVE_ARMS)
    planned_mde = approximate_mde(
        config.baseline_funding_rate,
        per_arm_size,
        per_arm_size,
        alpha=planning_alpha,
        power=0.80,
    )

    report: dict[str, object] = {
        "schema_version": "1.1.0",
        "artifact_type": "synthetic_prospective_factorial_benchmark",
        "data_classification": "synthetic",
        "report_scope": "aggregate_only",
        "derivation": "parameterized simulation; no source rows used",
        "analysis_contract": {
            "randomization_unit": "one fictional eligible lifecycle subject",
            "design": "stratified three-content by two-cadence factorial plus concurrent holdout",
            "estimand": "intention_to_treat",
            "primary_endpoint": "unique subject funding within 14 days",
            "followup_interval": "[assigned_at, assigned_at + 14 days)",
            "factorial_cells": 6,
            "total_arms": 7,
            "assignment_completeness": (
                "synthetic generator reconciled internally; external ledgers require "
                "a frozen eligibility snapshot"
            ),
            "primary_multiplicity": "six active-vs-holdout comparisons; Holm FWER",
            "guardrail_multiplicity": (
                "twelve one-sided active-vs-holdout bounds; Bonferroni family"
            ),
            "active_delivery_coverage": (
                "minimum one correct in-window delivery per active assignment; "
                "intention-to-treat population retained"
            ),
            "event_source_completeness": (
                "synthetic generator only; external sparse feeds require independent "
                "completeness watermarks"
            ),
            "required_quality_gates": list(REQUIRED_QUALITY_GATES),
            "estimator_note": (
                "Unadjusted intention-to-treat cell differences with nominal large-sample "
                "intervals; exact within-block allocation balances the canonical fixture, "
                "but a deployment should pre-specify block-adjusted inference."
            ),
        },
        "config": asdict(config),
        "data_freeze_at": data.data_freeze_at.isoformat(),
        "population": {
            "eligible": len(assignments),
            "randomized": len(assignments),
            "itt_analyzed": len(analysis),
            "assignment_blocks": len(srm.by_block),
        },
        "event_counts": {
            "total": len(events),
            "by_type": _event_type_counts(events),
        },
        "prospective_power_plan": {
            "baseline_funding_rate": config.baseline_funding_rate,
            "planned_units_per_arm": per_arm_size,
            "power": 0.80,
            "family_alpha": config.family_alpha,
            "planned_comparisons": len(ACTIVE_ARMS),
            "planning_alpha_per_comparison_bonferroni": planning_alpha,
            "approximate_mde_absolute": planned_mde,
            "uses_observed_control_rate": False,
            "interpretation": (
                "Conservative planning approximation for the pre-specified Holm family; "
                "not a post-hoc power calculation."
            ),
        },
        "randomization": {
            "assignment_sha256": assignment_sha256(assignments),
            "srm": srm.to_dict(),
        },
        "followup_audit": followup.to_dict(),
        "contamination_audit": contamination.to_dict(),
        "primary_funding_family": _records(primary),
        "guardrail_noninferiority_family": _records(guardrails),
        "pre_period_negative_control": {
            "passed": negative_control_passed,
            "comparisons": _records(negative_control),
        },
        "quality_gates": quality_gates,
        "decision": decision.to_dict(),
        "interpretation": (
            "Synthetic method check only. The declared candidate remains in testing unless "
            "corrected funding, simultaneous guardrail, and all quality gates pass."
        ),
    }
    return report


def write_prospective_synthetic_benchmark(
    output_path: str | Path,
    config: ProspectiveSyntheticConfig | None = None,
) -> Path:
    """Write one deterministic aggregate JSON artifact to a synthetic-labeled path."""

    destination = Path(output_path)
    if destination.suffix.lower() != ".json" or "synthetic" not in destination.name.lower():
        raise ContractError("prospective output must be a synthetic-labeled JSON file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = build_prospective_synthetic_benchmark(config)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


__all__ = [
    "EVENT_COLUMNS",
    "FOLLOWUP_DAYS",
    "ContaminationAudit",
    "FollowupAudit",
    "ProspectiveSyntheticConfig",
    "ProspectiveSyntheticData",
    "audit_contamination",
    "audit_followup",
    "build_prospective_synthetic_benchmark",
    "construct_prospective_analysis",
    "generate_prospective_synthetic",
    "validate_events",
    "write_prospective_synthetic_benchmark",
]
