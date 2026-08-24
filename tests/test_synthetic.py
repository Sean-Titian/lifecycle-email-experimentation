import json

import pandas as pd
import pytest

from email_experiment.cli import (
    build_synthetic_aggregate,
    write_synthetic_aggregate,
    write_synthetic_bundle,
)
from email_experiment.contracts import ContractError
from email_experiment.synthetic import (
    SyntheticExperimentConfig,
    generate_synthetic_experiment,
)


def test_synthetic_generator_is_deterministic_and_uses_safe_ids() -> None:
    config = SyntheticExperimentConfig(n_participants=100, seed=42)
    first = generate_synthetic_experiment(config)
    second = generate_synthetic_experiment(config)
    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    pd.testing.assert_frame_equal(first.events, second.events)
    assert first.observation_end == second.observation_end
    assert first.assignments["participant_id"].str.fullmatch(
        r"synthetic_\d{8}"
    ).all()
    assert not first.assignments.astype(str).apply(
        lambda column: column.str.contains("@", regex=False).any()
    ).any()


def test_synthetic_generator_changes_with_seed() -> None:
    config = SyntheticExperimentConfig(n_participants=100, seed=42)
    other = SyntheticExperimentConfig(n_participants=100, seed=43)
    assert not generate_synthetic_experiment(config).assignments.equals(
        generate_synthetic_experiment(other).assignments
    )


def test_smallest_valid_design_contains_both_arms_in_each_experiment() -> None:
    config = SyntheticExperimentConfig(n_participants=4, n_experiments=2, seed=3)
    assignments = generate_synthetic_experiment(config).assignments
    arm_sets = assignments.groupby("experiment")["arm"].agg(set)
    assert all(arms == {"control", "treatment"} for arms in arm_sets)


def test_invalid_design_cannot_leave_an_experiment_without_an_arm() -> None:
    config = SyntheticExperimentConfig(n_participants=3, n_experiments=2)
    with pytest.raises(ContractError, match="at least two units"):
        generate_synthetic_experiment(config)


def test_writer_requires_explicit_synthetic_path_and_labels_manifest(tmp_path) -> None:
    config = SyntheticExperimentConfig(n_participants=20, seed=7)
    with pytest.raises(ContractError, match="directory named 'synthetic'"):
        write_synthetic_bundle(tmp_path / "output", config)

    assignments, events, manifest = write_synthetic_bundle(
        tmp_path / "synthetic", config
    )
    assert assignments.exists() and events.exists() and manifest.exists()
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    assert metadata["data_classification"] == "synthetic"
    assert metadata["assignment_rows"] == 20
    assert "no source rows" in metadata["derivation"]


def test_end_to_end_aggregate_is_deterministic_and_contains_no_unit_rows(
    tmp_path,
) -> None:
    config = SyntheticExperimentConfig(n_participants=800, n_experiments=2, seed=11)
    first = build_synthetic_aggregate(config)
    second = build_synthetic_aggregate(config)
    assert first == second
    assert first["data_classification"] == "synthetic"
    assert first["report_scope"] == "aggregate_only"
    assert {
        "effects",
        "censoring",
        "strict_new_link_funnel",
        "unsubscribe_guardrail",
        "unsubscribe_guardrail_effect",
        "pre_treatment_negative_control",
    } <= first.keys()
    assert all("mde_absolute_80pct_power" in row for row in first["effects"])
    negative_control = first["pre_treatment_negative_control"]
    assert negative_control["outcome"] == "linked_before_entry"
    assert all("p_value_holm" in row for row in negative_control["comparisons"])
    assert {
        "absolute_effect",
        "absolute_ci_low",
        "absolute_ci_high",
        "relative_risk",
        "p_value_two_sided",
    } <= first["unsubscribe_guardrail_effect"].keys()
    population = first["population"]
    assert (
        population["eligible_primary_analysis"]
        + population["excluded_preexisting_conversion"]
        + population["excluded_pre_assignment_event"]
        + population["censored_after_eligibility"]
        == population["assigned"]
    )
    serialized = json.dumps(first)
    assert "synthetic_000" not in serialized
    assert "participant_id" not in serialized

    output = write_synthetic_aggregate(tmp_path / "reports" / "synthetic", config)
    assert output.name == "aggregate.json"
    assert sorted(path.name for path in output.parent.iterdir()) == ["aggregate.json"]
