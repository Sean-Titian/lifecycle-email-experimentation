"""Command-line entry points limited to explicitly synthetic artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .contracts import ContractError
from .funnel import (
    new_link_funnel_membership,
    summarize_ordered_funnel,
    unsubscribe_metrics,
)
from .prospective import (
    ProspectiveSyntheticConfig,
    write_prospective_synthetic_benchmark,
)
from .statistics import (
    approximate_mde,
    compare_binary_proportions,
    compare_experiment_groups,
)
from .synthetic import SyntheticExperimentConfig, generate_synthetic_experiment
from .time_windows import construct_windowed_outcome


def _synthetic_output_directory(value: str | Path) -> Path:
    path = Path(value)
    if "synthetic" not in {part.lower() for part in path.parts}:
        raise ContractError(
            "synthetic artifacts must be written beneath a directory named 'synthetic'"
        )
    return path


def write_synthetic_bundle(
    output_dir: str | Path,
    config: SyntheticExperimentConfig,
) -> tuple[Path, Path, Path]:
    """Write generated rows and a classification manifest to a synthetic path."""

    destination = _synthetic_output_directory(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    bundle = generate_synthetic_experiment(config)
    assignments_path = destination / "assignments.csv"
    events_path = destination / "events.csv"
    manifest_path = destination / "manifest.json"
    bundle.assignments.to_csv(assignments_path, index=False)
    bundle.events.to_csv(events_path, index=False)
    manifest = {
        "data_classification": "synthetic",
        "derivation": "parameterized simulation; no source rows used",
        "observation_end": bundle.observation_end.isoformat(),
        "assignment_rows": len(bundle.assignments),
        "event_rows": len(bundle.events),
        "config": asdict(config),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return assignments_path, events_path, manifest_path


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a frame to JSON-native records with non-finite values as null."""

    return json.loads(frame.to_json(orient="records", date_format="iso"))


def build_synthetic_aggregate(
    config: SyntheticExperimentConfig,
) -> dict[str, object]:
    """Run the public analysis entirely in memory and return aggregates only."""

    bundle = generate_synthetic_experiment(config)
    outcomes = construct_windowed_outcome(
        bundle.assignments,
        bundle.events,
        outcome_event="converted",
        window=pd.Timedelta(days=config.outcome_window_days),
        observation_end=bundle.observation_end,
    )
    excluded_preexisting = outcomes["converted_before_entry"].astype(bool)
    excluded_pre_assignment = (
        ~excluded_preexisting & outcomes["has_pre_assignment_outcome"].astype(bool)
    )
    eligible = ~excluded_preexisting & ~excluded_pre_assignment
    complete = outcomes.loc[eligible & outcomes["followup_complete"]].copy()
    complete["outcome"] = complete["outcome"].astype(int)
    effects = compare_experiment_groups(complete)
    smoothed_baseline = (effects["control_successes"] + 0.5) / (
        effects["control_total"] + 1.0
    )
    effects["mde_absolute_80pct_power"] = [
        approximate_mde(rate, int(control), int(treatment))
        for rate, control, treatment in zip(
            smoothed_baseline,
            effects["control_total"],
            effects["treatment_total"],
            strict=True,
        )
    ]

    censoring = (
        outcomes.loc[eligible]
        .groupby(["experiment", "arm"], sort=True)["followup_complete"]
        .agg(total="size", complete="sum")
        .reset_index()
    )
    censoring["censored"] = censoring["total"] - censoring["complete"]
    censoring["complete_followup_rate"] = (
        censoring["complete"] / censoring["total"]
    )

    funnel_membership = new_link_funnel_membership(bundle.events, bundle.assignments)
    funnel = summarize_ordered_funnel(
        funnel_membership, ["opened", "linked", "converted"]
    )
    guardrail = unsubscribe_metrics(bundle.assignments, bundle.events)
    guardrail_by_arm = guardrail.set_index("arm")
    guardrail_effect = compare_binary_proportions(
        int(guardrail_by_arm.loc["treatment", "unique_unsubscribers"]),
        int(guardrail_by_arm.loc["treatment", "actual_recipients"]),
        int(guardrail_by_arm.loc["control", "unique_unsubscribers"]),
        int(guardrail_by_arm.loc["control", "actual_recipients"]),
    )
    negative_control = compare_experiment_groups(
        bundle.assignments.rename(columns={"linked_before_entry": "negative_outcome"}),
        outcome_col="negative_outcome",
    )

    return {
        "data_classification": "synthetic",
        "report_scope": "aggregate_only",
        "derivation": "parameterized simulation; no source rows used",
        "config": asdict(config),
        "observation_end": bundle.observation_end.isoformat(),
        "population": {
            "assigned": len(bundle.assignments),
            "excluded_preexisting_conversion": int(excluded_preexisting.sum()),
            "excluded_pre_assignment_event": int(excluded_pre_assignment.sum()),
            "censored_after_eligibility": int(
                (eligible & ~outcomes["followup_complete"]).sum()
            ),
            "eligible_primary_analysis": len(complete),
            "strict_new_link_eligible": len(funnel_membership),
        },
        "effects": _records(effects),
        "pre_treatment_negative_control": {
            "outcome": "linked_before_entry",
            "interpretation": (
                "A pre-treatment balance diagnostic; significance is investigated, "
                "but no seed is guaranteed to produce p > 0.05."
            ),
            "comparisons": _records(negative_control),
        },
        "censoring": _records(censoring),
        "strict_new_link_funnel": _records(funnel),
        "unsubscribe_guardrail": _records(guardrail),
        "unsubscribe_guardrail_effect": asdict(guardrail_effect),
    }


def write_synthetic_aggregate(
    output_dir: str | Path,
    config: SyntheticExperimentConfig,
) -> Path:
    """Write one deterministic aggregate report and no participant-level rows."""

    destination = _synthetic_output_directory(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "aggregate.json"
    output_path.write_text(
        json.dumps(build_synthetic_aggregate(config), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="email-experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser(
        "generate-synthetic", help="generate privacy-safe demonstration data"
    )
    generate.add_argument("--output-dir", default="data/synthetic")
    generate.add_argument("--participants", type=int, default=8_000)
    generate.add_argument("--experiments", type=int, default=4)
    generate.add_argument("--seed", type=int, default=20260824)
    run = subparsers.add_parser(
        "run-synthetic", help="run the complete synthetic aggregate analysis"
    )
    run.add_argument("--output-dir", default="reports/synthetic")
    run.add_argument("--participants", type=int, default=8_000)
    run.add_argument("--experiments", type=int, default=4)
    run.add_argument("--seed", type=int, default=20260824)
    prospective = subparsers.add_parser(
        "run-prospective-synthetic",
        help="run the seven-arm prospective synthetic method check",
    )
    prospective.add_argument(
        "--output",
        default="reports/prospective-synthetic-benchmark.json",
        help="synthetic-labeled aggregate JSON path",
    )
    prospective.add_argument("--seed", type=int, default=20260908)
    prospective.add_argument("--units-per-arm-per-block", type=int, default=100)
    prospective.add_argument("--waves", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"generate-synthetic", "run-synthetic"}:
        config = SyntheticExperimentConfig(
            n_participants=args.participants,
            n_experiments=args.experiments,
            seed=args.seed,
        )
        if args.command == "generate-synthetic":
            write_synthetic_bundle(args.output_dir, config)
        else:
            write_synthetic_aggregate(args.output_dir, config)
        return 0
    if args.command == "run-prospective-synthetic":
        config = ProspectiveSyntheticConfig(
            seed=args.seed,
            units_per_arm_per_block=args.units_per_arm_per_block,
            assignment_waves=args.waves,
        )
        write_prospective_synthetic_benchmark(args.output, config)
        return 0
    raise AssertionError("argparse accepted an unknown command")
