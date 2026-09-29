"""Fail-closed data contracts for experiment analysis.

The helpers in this module deliberately report counts rather than example keys so
that validation errors do not copy user identifiers into logs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd


class ContractError(ValueError):
    """Raised when input data cannot safely support the requested analysis."""


def require_unique_column_names(
    frame: pd.DataFrame,
    *,
    frame_name: str = "frame",
) -> None:
    """Reject ambiguous schemas without copying column names into errors."""

    duplicate_positions = int(frame.columns.duplicated(keep=False).sum())
    if duplicate_positions:
        raise ContractError(
            f"{frame_name} has {duplicate_positions} column positions with duplicate names"
        )


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    frame_name: str = "frame",
) -> None:
    """Require a set of columns without exposing row values in errors."""

    require_unique_column_names(frame, frame_name=frame_name)
    required = tuple(dict.fromkeys(columns))
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ContractError(f"{frame_name} is missing required columns: {missing}")


def require_exact_columns(
    frame: pd.DataFrame,
    expected: Iterable[str],
    *,
    frame_name: str = "frame",
) -> None:
    """Require an exact, unambiguous schema while reporting counts only."""

    require_unique_column_names(frame, frame_name=frame_name)
    expected_columns = tuple(expected)
    missing = set(expected_columns) - set(frame.columns)
    unexpected = set(frame.columns) - set(expected_columns)
    if missing or unexpected:
        raise ContractError(
            f"{frame_name} schema mismatch: {len(missing)} missing and "
            f"{len(unexpected)} unexpected columns"
        )


def parse_aware_utc_series(series: pd.Series, *, name: str) -> pd.Series:
    """Parse timestamps, requiring an explicit timezone on every value."""

    converted: list[pd.Timestamp] = []
    invalid = 0
    naive = 0
    for value in series:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            invalid += 1
            continue
        if pd.isna(timestamp):
            invalid += 1
        elif timestamp.tzinfo is None:
            naive += 1
        else:
            try:
                converted.append(timestamp.tz_convert("UTC"))
            except (TypeError, ValueError, OverflowError):
                invalid += 1
    if invalid or naive or len(converted) != len(series):
        raise ContractError(
            f"{name} has {invalid} invalid or missing and {naive} timezone-naive values"
        )
    try:
        return pd.Series(converted, index=series.index, dtype="datetime64[ns, UTC]")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(
            f"{name} contains timestamps outside the supported range"
        ) from exc


def parse_aware_utc_timestamp(value: object, *, name: str) -> pd.Timestamp:
    """Parse one timestamp and reject missing or timezone-naive values."""

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{name} must be a valid timezone-aware timestamp") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ContractError(f"{name} must be a valid timezone-aware timestamp")
    try:
        converted = timestamp.tz_convert("UTC")
        normalized = pd.Series([converted], dtype="datetime64[ns, UTC]")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(
            f"{name} is outside the supported timestamp range"
        ) from exc
    return normalized.iloc[0]


def validate_unique_key(
    frame: pd.DataFrame,
    key: str | Sequence[str],
    *,
    frame_name: str = "frame",
) -> None:
    """Require a complete, unique key.

    Any duplicate fails, including byte-for-byte duplicate rows.  Silently
    deduplicating those rows would make conflicting records and join expansion
    indistinguishable, so the caller must resolve the source data explicitly.
    """

    keys = [key] if isinstance(key, str) else list(key)
    if not keys:
        raise ContractError("key must contain at least one column")
    require_columns(frame, keys, frame_name=frame_name)

    null_rows = int(frame[keys].isna().any(axis=1).sum())
    if null_rows:
        raise ContractError(
            f"{frame_name} has {null_rows} rows with an incomplete key"
        )

    duplicate_rows = int(frame.duplicated(keys, keep=False).sum())
    if duplicate_rows:
        raise ContractError(
            f"{frame_name} has {duplicate_rows} rows participating in duplicate keys"
        )


def validate_binary(
    series: pd.Series,
    *,
    name: str = "outcome",
    allow_missing: bool = False,
) -> pd.Series:
    """Return a nullable integer binary series after strict validation."""

    if not allow_missing and series.isna().any():
        raise ContractError(f"{name} contains missing values")

    non_missing = series.dropna()
    invalid = ~non_missing.isin([0, 1, False, True])
    if invalid.any():
        raise ContractError(f"{name} must contain only binary values")
    return series.astype("Int64")


def one_to_one_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str | Sequence[str],
    how: str = "inner",
    require_match: str = "both",
    suffixes: tuple[str, str] = ("_left", "_right"),
) -> pd.DataFrame:
    """Join two frames only after verifying one-to-one cardinality.

    ``require_match`` can be ``"both"``, ``"left"``, ``"right"`` or ``"none"``.
    A required side having an unmatched key raises instead of silently changing
    the analysis population.
    """

    keys = [on] if isinstance(on, str) else list(on)
    validate_unique_key(left, keys, frame_name="left")
    validate_unique_key(right, keys, frame_name="right")

    valid_how = {"inner", "left", "right", "outer"}
    if how not in valid_how:
        raise ContractError(f"how must be one of {sorted(valid_how)}")
    valid_requirement = {"both", "left", "right", "none"}
    if require_match not in valid_requirement:
        raise ContractError(
            f"require_match must be one of {sorted(valid_requirement)}"
        )

    audited = left.merge(
        right,
        on=keys,
        how="outer",
        suffixes=suffixes,
        validate="one_to_one",
        indicator=True,
    )
    unmatched_left = int((audited["_merge"] == "left_only").sum())
    unmatched_right = int((audited["_merge"] == "right_only").sum())
    if require_match in {"both", "left"} and unmatched_left:
        raise ContractError(f"join has {unmatched_left} unmatched left keys")
    if require_match in {"both", "right"} and unmatched_right:
        raise ContractError(f"join has {unmatched_right} unmatched right keys")

    return left.merge(
        right,
        on=keys,
        how=how,
        suffixes=suffixes,
        validate="one_to_one",
    )
