import json
import re
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".txt",
    ".csv",
}


def _public_files() -> list[Path]:
    excluded_parts = {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "__pycache__",
        "build",
        "dist",
    }
    files: list[Path] = []
    ignored_output_roots = {
        ("data", "raw"),
        ("data", "synthetic"),
        ("reports", "generated"),
        ("reports", "synthetic"),
    }
    for path in REPOSITORY.rglob("*"):
        if not path.is_file() or excluded_parts.intersection(path.parts):
            continue
        if any(part.endswith(".egg-info") for part in path.parts):
            continue
        relative = path.relative_to(REPOSITORY)
        if tuple(relative.parts[:2]) in ignored_output_roots and path.name != ".gitkeep":
            continue
        files.append(path)
    return files


def test_public_tree_has_no_oversized_artifact() -> None:
    oversized = [
        path.relative_to(REPOSITORY)
        for path in _public_files()
        if path.stat().st_size > 1_000_000
    ]
    assert oversized == []


def test_public_text_has_no_local_paths_private_brand_or_secret_assignment() -> None:
    private_brand = "".join(["直通", "硅谷"])
    local_path = re.compile(r"[A-Za-z]:[\\/](?:Users|Documents)[\\/] | /(?:Users|home)/", re.X)
    secret_assignment = re.compile(
        r"(?:api[_-]?key|client[_-]?secret|access[_-]?token)\s*[:=]\s*['\"][^'\"]{12,}",
        re.IGNORECASE,
    )
    findings: list[str] = []
    for path in _public_files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if private_brand in text or local_path.search(text) or secret_assignment.search(text):
            findings.append(str(path.relative_to(REPOSITORY)))
    assert findings == []


def test_tabular_demo_files_live_only_under_synthetic_directory() -> None:
    misplaced = [
        str(path.relative_to(REPOSITORY))
        for path in _public_files()
        if path.suffix.lower() in {".csv", ".parquet"}
        and "synthetic" not in {part.lower() for part in path.parts}
    ]
    assert misplaced == []


def test_private_source_benchmark_uses_a_minimal_allowlisted_schema() -> None:
    benchmark = json.loads(
        (REPOSITORY / "reports" / "source-benchmark.json").read_text(encoding="utf-8")
    )
    assert set(benchmark) == {
        "schema_version",
        "artifact_type",
        "privacy",
        "study_scale",
        "template_recorded_open",
        "funding_aggregate_control_snapshot",
        "direct_cadence_comparisons",
        "unsubscribe_guardrail",
        "strict_temporal_funnel_approx",
        "validity_flags",
        "claim",
    }
    assert benchmark["artifact_type"] == "minimal_anonymous_aggregate_benchmark"
    assert benchmark["privacy"]["contains_exact_operational_counts"] is False
    assert benchmark["privacy"]["source_data_redistributed"] is False
    assert benchmark["study_scale"]["participant_count_is_rounded"] is True
    assert benchmark["strict_temporal_funnel_approx"]["counts_are_rounded"] is True
    assert set(benchmark["unsubscribe_guardrail"]) == {
        "unique_recipient_risk_approx",
        "preferred_denominator",
    }
    assert set(benchmark["validity_flags"]) == {
        "pre_send_outcomes_detected",
        "full_followup_complete_for_all_arms",
        "control_has_user_level_timestamps",
        "matched_treatment_control_window_possible",
        "assignment_block_reused_across_groups",
        "pre_treatment_negative_control_issue_detected",
    }
