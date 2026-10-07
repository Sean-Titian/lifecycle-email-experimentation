from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import email_experiment
from email_experiment.assignment import ACTIVE_ARMS
from email_experiment.cli import build_parser
from email_experiment.contracts import ContractError
from email_experiment.operating_characteristics import (
    CANDIDATE_ARM,
    PLANNING_REFERENCE_EFFECT,
    SCENARIO_IDS,
    OperatingCharacteristicsConfig,
    _probability_matrices,
    _rng,
    _scenario_definitions,
    _simulate_scenario,
    build_operating_characteristics_benchmark,
    production_api_parity_check,
    write_operating_characteristics_benchmark,
)


def _fast_config(**changes: object) -> OperatingCharacteristicsConfig:
    config = OperatingCharacteristicsConfig(
        seed=20261006,
        replications=256,
        units_per_arm_per_block=20,
    )
    return replace(config, **changes)


def _scenario(
    config: OperatingCharacteristicsConfig,
    scenario_id: str,
):
    return next(
        scenario
        for scenario in _scenario_definitions(config)
        if scenario.scenario_id == scenario_id
    )


def test_default_contract_freezes_twenty_thousand_replicates_and_five_scenarios() -> None:
    config = OperatingCharacteristicsConfig()
    assert config.replications == 20_000
    assert config.units_per_arm_per_block == 100
    assert config.assignment_waves == 2
    assert config.pre_specified_candidate_arm == CANDIDATE_ARM
    assert tuple(item.scenario_id for item in _scenario_definitions(config)) == SCENARIO_IDS


def test_public_package_exports_the_operating_characteristics_api() -> None:
    expected = {
        "OperatingCharacteristicsConfig",
        "build_operating_characteristics_benchmark",
        "production_api_parity_check",
        "write_operating_characteristics_benchmark",
    }
    assert expected <= set(email_experiment.__all__)
    assert email_experiment.__version__ == "0.5.0"


def test_cli_exposes_the_frozen_operating_characteristics_contract() -> None:
    args = build_parser().parse_args(
        [
            "run-prospective-operating-characteristics",
            "--output",
            "reports/prospective-synthetic-operating-characteristics.json",
        ]
    )
    assert args.seed == 20261006
    assert args.replications == 20_000
    assert args.units_per_arm_per_block == 100
    assert args.waves == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", -1, "seed"),
        ("replications", 1, "at least two"),
        ("replications", True, "integer"),
        ("units_per_arm_per_block", 1, "at least two"),
        ("assignment_waves", 0, "positive"),
        ("family_alpha", 0.04, "family alpha"),
        ("monte_carlo_confidence", 0.95, "confidence"),
        ("unsubscribe_noninferiority_margin", -0.01, "margin"),
        ("pre_specified_candidate_arm", "current__daily", "candidate"),
    ],
)
def test_invalid_simulation_contract_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        build_operating_characteristics_benchmark(_fast_config(**{field: value}))


def test_configured_probabilities_preserve_the_prospective_data_generating_process() -> None:
    config = _fast_config(replications=2)
    probabilities = _probability_matrices(
        config,
        _scenario(config, "configured_small_effects"),
    )
    candidate_index = 1 + (
        (
            "current__daily",
            "current__twice_weekly",
            "challenger_a__daily",
            "challenger_a__twice_weekly",
            "challenger_b__daily",
            "challenger_b__twice_weekly",
        ).index(CANDIDATE_ARM)
    )
    assert probabilities["funded_14d"].min() == pytest.approx(0.031)
    assert probabilities["funded_14d"].max() == pytest.approx(0.0525)
    assert np.allclose(
        probabilities["funded_14d"][:, candidate_index]
        - probabilities["funded_14d"][:, 0],
        0.002,
    )
    assert np.allclose(
        probabilities["unsubscribe_14d"][:, candidate_index]
        - probabilities["unsubscribe_14d"][:, 0],
        0.0002,
    )
    assert np.allclose(
        probabilities["complaint_14d"][:, candidate_index]
        - probabilities["complaint_14d"][:, 0],
        0.0001,
    )
    assert probabilities["pre_period_engaged"].min() == pytest.approx(0.15)
    assert probabilities["pre_period_engaged"].max() == pytest.approx(0.35)


def test_planning_reference_centers_block_heterogeneity_and_uses_frozen_effect() -> None:
    config = _fast_config(replications=2)
    scenario = _scenario(config, "planning_reference")
    probabilities = _probability_matrices(config, scenario)
    candidate_index = 1 + 3
    holdout = probabilities["funded_14d"][:, 0]
    candidate = probabilities["funded_14d"][:, candidate_index]
    assert holdout.mean() == pytest.approx(0.04)
    assert np.allclose(candidate - holdout, PLANNING_REFERENCE_EFFECT)
    other_active = np.delete(probabilities["funded_14d"][:, 1:], 3, axis=1)
    assert np.allclose(other_active - holdout[:, None], 0.0)


def test_sha256_derived_pcg64_streams_are_stable_and_namespaced() -> None:
    first = _rng(17, "global_null", "funded_14d").random(8)
    second = _rng(17, "global_null", "funded_14d").random(8)
    different_outcome = _rng(17, "global_null", "complaint_14d").random(8)
    different_scenario = _rng(17, "configured_small_effects", "funded_14d").random(8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different_outcome)
    assert not np.array_equal(first, different_scenario)


def test_vectorized_results_match_production_apis_to_frozen_tolerance() -> None:
    result = production_api_parity_check(_fast_config(), replications=3)
    assert result["replications"] == 3
    assert result["units_per_arm_per_block"] == 100
    assert result["maximum_absolute_numeric_error"] <= 1e-12
    assert result["numeric_parity_passed"]
    assert result["boolean_parity_passed"]
    assert result["decision_parity_passed"]
    assert result["passed"]


def test_report_is_deterministic_aggregate_only_and_keeps_every_denominator() -> None:
    config = _fast_config(replications=128)
    first = build_operating_characteristics_benchmark(config)
    second = build_operating_characteristics_benchmark(config)
    assert first == second
    assert [item["scenario_id"] for item in first["scenarios"]] == list(SCENARIO_IDS)
    assert first["design_readiness"]["status"] == "continue_testing"
    assert first["simulation_contract"]["shared_holdout"] is True
    assert first["simulation_contract"]["monte_carlo_intervals"].startswith(
        "two-sided 99% Wilson"
    )
    rendered = repr(first).lower()
    assert "participant_id" not in rendered
    assert "replicate_id" not in rendered
    for scenario in first["scenarios"]:
        assert scenario["replications_requested"] == config.replications
        assert scenario["replications_in_every_denominator"] == config.replications
        metrics = scenario["operating_characteristics"]
        for name, metric in metrics.items():
            if name in {
                "candidate_guardrail_noninferiority",
                "nominal_95_ci_coverage_by_active_arm",
                "policy_blocking_reason_frequency",
            }:
                assert all(
                    outcome_metric["replications"] == config.replications
                    for outcome_metric in metric.values()
                )
            else:
                assert metric["replications"] == config.replications
                assert metric["monte_carlo_confidence"] == 0.99
                assert (
                    metric["monte_carlo_ci_low"]
                    <= metric["rate"]
                    <= metric["monte_carlo_ci_high"]
                )


def test_invalid_replicates_fail_closed_without_denominator_drops() -> None:
    config = _fast_config(replications=128, units_per_arm_per_block=2)
    arrays = _simulate_scenario(config, _scenario(config, "global_null"))
    assert arrays.invalid.any()
    assert not arrays.criteria_met[arrays.invalid].any()
    report = build_operating_characteristics_benchmark(config)
    global_null = next(
        scenario for scenario in report["scenarios"] if scenario["scenario_id"] == "global_null"
    )
    metrics = global_null["operating_characteristics"]
    assert metrics["invalid_replicate"]["successes"] == int(arrays.invalid.sum())
    reasons = metrics["policy_blocking_reason_frequency"]
    assert reasons["invalid_inference"]["successes"] == int(arrays.invalid.sum())
    assert reasons["pre_period_negative_control"]["successes"] == int(
        ((~arrays.invalid) & arrays.negative_control.holm_reject.any(axis=1)).sum()
    )
    candidate_index = ACTIVE_ARMS.index(CANDIDATE_ARM)
    assert reasons["primary_superiority_not_established"]["successes"] == int(
        (
            (~arrays.invalid)
            & ~arrays.primary.superiority_pass[:, candidate_index]
        ).sum()
    )
    assert all(
        metric["replications"] == config.replications
        for name, metric in metrics.items()
        if name
        not in {
            "candidate_guardrail_noninferiority",
            "nominal_95_ci_coverage_by_active_arm",
            "policy_blocking_reason_frequency",
        }
    )
    assert not report["calibration_gates"]["no_invalid_replicates"]["passed"]


def test_margin_boundary_scenario_places_every_active_guardrail_at_its_margin() -> None:
    config = _fast_config(replications=2)
    probabilities = _probability_matrices(
        config,
        _scenario(config, "strong_effect_guardrails_at_margins"),
    )
    for outcome, margin in {
        "unsubscribe_14d": config.unsubscribe_noninferiority_margin,
        "complaint_14d": config.complaint_noninferiority_margin,
    }.items():
        assert np.allclose(
            probabilities[outcome][:, 1:] - probabilities[outcome][:, [0]],
            margin,
        )


def test_writer_is_deterministic_strict_json_and_requires_a_safe_name(tmp_path: Path) -> None:
    config = _fast_config(replications=64)
    first = write_operating_characteristics_benchmark(
        tmp_path / "prospective-synthetic-operating-characteristics.json",
        config,
    )
    second = write_operating_characteristics_benchmark(
        tmp_path / "second-synthetic-operating-characteristics.json",
        config,
    )
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["report_scope"] == "aggregate_only_monte_carlo"
    assert payload["design_readiness"]["status"] == "continue_testing"
    with pytest.raises(ContractError, match="synthetic operating"):
        write_operating_characteristics_benchmark(tmp_path / "benchmark.json", config)
