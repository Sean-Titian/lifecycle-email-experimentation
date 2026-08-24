"""Reproducible synthetic lifecycle experiment data.

The generator models generic behavior from declared parameters.  It does not
sample, hash, perturb, or otherwise derive records from a private dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import ContractError


@dataclass(frozen=True)
class SyntheticExperimentConfig:
    n_participants: int = 8_000
    n_experiments: int = 4
    seed: int = 20260824
    assignment_days: int = 21
    outcome_window_days: int = 7
    baseline_conversion_rate: float = 0.035
    treatment_relative_lift: float = 0.12
    baseline_unsubscribe_rate: float = 0.006
    treatment_unsubscribe_relative_lift: float = 0.05


@dataclass(frozen=True)
class SyntheticExperimentData:
    assignments: pd.DataFrame
    events: pd.DataFrame
    observation_end: pd.Timestamp


def _validate_config(config: SyntheticExperimentConfig) -> None:
    if config.n_experiments < 1:
        raise ContractError("synthetic data requires participants and experiments")
    if config.n_participants < 2 * config.n_experiments:
        raise ContractError(
            "n_participants must provide at least two units per experiment"
        )
    if config.assignment_days < 1 or config.outcome_window_days < 1:
        raise ContractError("synthetic time spans must be positive")
    for name, value in (
        ("baseline_conversion_rate", config.baseline_conversion_rate),
        ("baseline_unsubscribe_rate", config.baseline_unsubscribe_rate),
    ):
        if not 0.0 < value < 1.0:
            raise ContractError(f"{name} must be between 0 and 1")
    if config.treatment_relative_lift <= -1.0:
        raise ContractError("treatment_relative_lift must be greater than -1")
    if config.treatment_unsubscribe_relative_lift <= -1.0:
        raise ContractError(
            "treatment_unsubscribe_relative_lift must be greater than -1"
        )


def generate_synthetic_experiment(
    config: SyntheticExperimentConfig | None = None,
) -> SyntheticExperimentData:
    """Generate generic assignments and temporally ordered lifecycle events."""

    if config is None:
        config = SyntheticExperimentConfig()
    _validate_config(config)
    rng = np.random.default_rng(config.seed)
    count = config.n_participants
    participant_ids = [f"synthetic_{index:08d}" for index in range(count)]
    experiments = np.array(
        [f"experiment_{index + 1:02d}" for index in range(config.n_experiments)]
    )
    experiment = experiments[np.arange(count) % config.n_experiments]
    arm = np.empty(count, dtype=object)
    for experiment_name in experiments:
        indices = np.flatnonzero(experiment == experiment_name)
        labels = np.where(np.arange(indices.size) % 2, "treatment", "control")
        rng.shuffle(labels)
        arm[indices] = labels
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    assigned_offsets = rng.integers(
        0, config.assignment_days * 24 * 60, size=count
    )
    assigned_at = start + pd.to_timedelta(assigned_offsets, unit="m")
    delivered = rng.integers(1, 5, size=count)
    linked_before_entry = rng.random(count) < 0.10
    converted_before_entry = rng.random(count) < 0.01

    assignments = pd.DataFrame(
        {
            "participant_id": participant_ids,
            "experiment": experiment,
            "arm": arm,
            "assigned_at": assigned_at,
            "messages_delivered": delivered,
            "linked_before_entry": linked_before_entry,
            "converted_before_entry": converted_before_entry,
        }
    )

    event_rows: list[dict[str, object]] = []
    conversion_probability = np.full(count, config.baseline_conversion_rate)
    unsubscribe_probability = np.full(count, config.baseline_unsubscribe_rate)
    treated = arm == "treatment"
    conversion_probability[treated] *= 1.0 + config.treatment_relative_lift
    unsubscribe_probability[treated] *= (
        1.0 + config.treatment_unsubscribe_relative_lift
    )
    conversion_probability = np.clip(conversion_probability, 0.0, 1.0)
    unsubscribe_probability = np.clip(unsubscribe_probability, 0.0, 1.0)

    opened = rng.random(count) < 0.43
    linked = opened & (rng.random(count) < 0.12)
    converted = rng.random(count) < conversion_probability
    unsubscribed = rng.random(count) < unsubscribe_probability

    for index, participant_id in enumerate(participant_ids):
        base = assigned_at[index]
        for delivery_index in range(int(delivered[index])):
            event_rows.append(
                {
                    "participant_id": participant_id,
                    "event_type": "delivered",
                    "event_at": base + pd.Timedelta(minutes=delivery_index * 720 + 1),
                }
            )
        open_time = base + pd.Timedelta(minutes=int(rng.integers(5, 480)))
        if opened[index]:
            event_rows.append(
                {
                    "participant_id": participant_id,
                    "event_type": "opened",
                    "event_at": open_time,
                }
            )
        link_time = open_time + pd.Timedelta(minutes=int(rng.integers(1, 240)))
        if linked[index]:
            event_rows.append(
                {
                    "participant_id": participant_id,
                    "event_type": "linked",
                    "event_at": link_time,
                }
            )
        if converted[index]:
            lower_bound = link_time if linked[index] else base
            event_rows.append(
                {
                    "participant_id": participant_id,
                    "event_type": "converted",
                    "event_at": lower_bound
                    + pd.Timedelta(minutes=int(rng.integers(5, 4 * 24 * 60))),
                }
            )
        if unsubscribed[index]:
            event_rows.append(
                {
                    "participant_id": participant_id,
                    "event_type": "unsubscribed",
                    "event_at": base
                    + pd.Timedelta(minutes=int(rng.integers(10, 3 * 24 * 60))),
                }
            )

    events = pd.DataFrame(event_rows).sort_values(
        ["participant_id", "event_at", "event_type"], kind="stable"
    )
    events = events.reset_index(drop=True)
    observation_end = (
        assignments["assigned_at"].max()
        + pd.Timedelta(days=config.outcome_window_days)
    )
    return SyntheticExperimentData(assignments, events, observation_end)
