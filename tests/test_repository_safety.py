import json
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

from email_experiment.decision import REQUIRED_QUALITY_GATES

REPOSITORY = Path(__file__).resolve().parents[1]

PUBLIC_JSON_ALLOWLIST = {
    "reports/prospective-synthetic-benchmark.json",
    "reports/source-benchmark.json",
}
PLACEHOLDER_ALLOWLIST = {
    "data/raw/.gitkeep",
    "data/synthetic/.gitkeep",
}
SAFE_TEXT_ALLOWLIST = {"requirements-benchmark.txt"}
ALLOWED_SUFFIXES = {".json", ".md", ".py", ".toml", ".yaml", ".yml"}
ALLOWED_EXTENSIONLESS_FILES = {".gitattributes", ".gitignore", "LICENSE"}
RISKY_SUFFIXES = {
    ".7z",
    ".arrow",
    ".avi",
    ".avro",
    ".bin",
    ".bz2",
    ".ckpt",
    ".csv",
    ".db",
    ".doc",
    ".docx",
    ".dta",
    ".duckdb",
    ".feather",
    ".flac",
    ".gif",
    ".gz",
    ".h5",
    ".hdf5",
    ".ico",
    ".ipynb",
    ".jar",
    ".joblib",
    ".jpeg",
    ".jpg",
    ".jsonl",
    ".keras",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ndjson",
    ".npy",
    ".npz",
    ".onnx",
    ".orc",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".ppt",
    ".pptx",
    ".pt",
    ".pth",
    ".rar",
    ".rdata",
    ".rds",
    ".safetensors",
    ".sas7bdat",
    ".sav",
    ".sqlite",
    ".sqlite3",
    ".svg",
    ".tar",
    ".tgz",
    ".tsv",
    ".txt",
    ".wav",
    ".webm",
    ".xls",
    ".xlsx",
    ".xz",
    ".zip",
}
RESTRICTED_PREFIXES = {
    ("data", "raw"),
    ("data", "synthetic"),
    ("reports", "generated"),
    ("reports", "synthetic"),
}
RESTRICTED_PATH_PARTS = {
    "course_materials",
    "notebooks",
    "reference_answers",
    "teacher_materials",
    "videos",
}
TEXT_SUFFIXES = {".json", ".md", ".py", ".toml", ".yaml", ".yml"}

EMAIL_ADDRESS = re.compile(
    r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+\b"
)
LOCAL_PATHS = (
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),
    re.compile(r"(?<![:A-Za-z0-9])/(?:Users|home|tmp|var/tmp)/", re.IGNORECASE),
    re.compile("file:" + r"//", re.IGNORECASE),
)
DRIVE_LINK = re.compile(r"(?:drive|docs)" + r"\.google\.com/", re.IGNORECASE)
SECRET_ASSIGNMENT = re.compile(
    r"(?:api[_-]?key|client[_-]?secret|access[_-]?token|password|passwd|secret)"
    r"\s*[:=]\s*['\"][^'\"]{8,}",
    re.IGNORECASE,
)
KNOWN_SECRET = re.compile(
    r"(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{20,})"
)
SYNTHETIC_ROW_ID = re.compile(r"\bsynthetic[_-]\d{4,}\b", re.IGNORECASE)
FORBIDDEN_REPORT_KEYS = {
    "email",
    "email_address",
    "free_text",
    "input_path",
    "local_path",
    "message_body",
    "participant_id",
    "recipient_id",
    "records",
    "row_data",
    "rows",
    "source_path",
    "subject_line",
    "user_id",
}


def _git_candidates() -> list[Path]:
    """Return every tracked or non-ignored untracked file, including force-adds."""

    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=REPOSITORY,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    try:
        names = completed.stdout.decode("utf-8").split("\0")
    except UnicodeDecodeError as error:
        raise AssertionError("Git candidate paths must be valid UTF-8") from error

    repository_root = REPOSITORY.resolve()
    candidates: list[Path] = []
    for name in names:
        if not name:
            continue
        candidate = (REPOSITORY / name).resolve()
        assert candidate.is_relative_to(repository_root), f"path escapes repository: {name}"
        assert candidate.is_file(), f"candidate is missing or not a file: {name}"
        candidates.append(candidate)
    return candidates


def _relative(path: Path) -> str:
    return path.relative_to(REPOSITORY).as_posix()


def _json_items(value: object) -> Iterator[tuple[str, object]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _json_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _json_items(child)


def test_git_candidate_inventory_is_unique() -> None:
    relative = [_relative(path) for path in _git_candidates()]
    assert len(relative) == len(set(relative))


def test_only_explicit_public_file_types_are_candidates() -> None:
    invalid: list[str] = []
    for path in _git_candidates():
        relative = _relative(path)
        if relative in PLACEHOLDER_ALLOWLIST or relative in SAFE_TEXT_ALLOWLIST:
            continue
        if path.name in ALLOWED_EXTENSIONLESS_FILES:
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            invalid.append(relative)
    assert invalid == []


def test_no_risky_artifact_suffix_is_a_candidate() -> None:
    risky = [
        _relative(path)
        for path in _git_candidates()
        if path.suffix.lower() in RISKY_SUFFIXES
        and _relative(path) not in SAFE_TEXT_ALLOWLIST
    ]
    assert risky == []


def test_restricted_output_roots_contain_only_placeholders() -> None:
    findings: list[str] = []
    for path in _git_candidates():
        relative = _relative(path)
        parts = tuple(part.lower() for part in Path(relative).parts)
        if parts[:2] in RESTRICTED_PREFIXES and relative not in PLACEHOLDER_ALLOWLIST:
            findings.append(relative)
        if RESTRICTED_PATH_PARTS.intersection(parts):
            findings.append(relative)
    assert findings == []


def test_only_allowlisted_json_artifacts_are_candidates() -> None:
    json_files = {
        _relative(path) for path in _git_candidates() if path.suffix.lower() == ".json"
    }
    assert json_files <= PUBLIC_JSON_ALLOWLIST
    assert "reports/source-benchmark.json" in json_files


def test_public_candidates_have_no_oversized_artifact() -> None:
    oversized = [
        _relative(path) for path in _git_candidates() if path.stat().st_size > 1_000_000
    ]
    assert oversized == []


def test_public_text_has_no_private_material_path_address_or_secret() -> None:
    private_brand = "".join(["直通", "硅谷"])
    private_key_marker = "-" * 5 + "BEGIN " + "PRIVATE KEY" + "-" * 5
    findings: list[str] = []
    for path in _git_candidates():
        if (
            path.suffix.lower() not in TEXT_SUFFIXES
            and _relative(path) not in SAFE_TEXT_ALLOWLIST
            and path.name not in {
                ".gitattributes",
                ".gitignore",
                "LICENSE",
            }
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        unsafe = (
            private_brand in text
            or private_key_marker in text
            or EMAIL_ADDRESS.search(text)
            or DRIVE_LINK.search(text)
            or SECRET_ASSIGNMENT.search(text)
            or KNOWN_SECRET.search(text)
            or any(pattern.search(text) for pattern in LOCAL_PATHS)
        )
        if unsafe:
            findings.append(_relative(path))
    assert findings == []


def test_benchmark_dependency_lock_is_minimal_and_exact() -> None:
    lines = (REPOSITORY / "requirements-benchmark.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert lines == ["numpy==2.5.2", "pandas==3.0.5"]


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


def test_prospective_report_is_canonical_aggregate_only_when_present() -> None:
    report_path = REPOSITORY / "reports" / "prospective-synthetic-benchmark.json"
    if not report_path.exists():
        return

    raw = report_path.read_bytes().decode("utf-8")
    report = json.loads(raw)
    assert report["artifact_type"] == "synthetic_prospective_factorial_benchmark"
    assert report["data_classification"] == "synthetic"
    assert report["report_scope"] == "aggregate_only"
    assert report["schema_version"] == "1.1.0"
    assert isinstance(report.get("quality_gates"), dict)
    assert set(report["quality_gates"]) == set(REQUIRED_QUALITY_GATES)
    assert all(value is True for value in report["quality_gates"].values())

    report_items = list(_json_items(report))
    forbidden_keys = sorted(
        key for key, _ in report_items if key.lower() in FORBIDDEN_REPORT_KEYS
    )
    assert forbidden_keys == []
    assert SYNTHETIC_ROW_ID.search(raw) is None
    assert EMAIL_ADDRESS.search(raw) is None
    assert not any(pattern.search(raw) for pattern in LOCAL_PATHS)

    canonical = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert raw == canonical
