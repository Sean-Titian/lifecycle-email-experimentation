import pandas as pd
import pytest

from email_experiment.contracts import (
    ContractError,
    one_to_one_join,
    parse_aware_utc_series,
    parse_aware_utc_timestamp,
    require_columns,
    validate_unique_key,
)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"id": [1, 1], "value": ["same", "same"]}),
        pd.DataFrame({"id": [1, 1], "value": ["first", "conflict"]}),
    ],
)
def test_unique_key_fails_closed_for_every_duplicate(frame: pd.DataFrame) -> None:
    with pytest.raises(ContractError, match="duplicate keys"):
        validate_unique_key(frame, "id")


def test_unique_key_rejects_missing_id_without_logging_values() -> None:
    frame = pd.DataFrame({"id": [1, None], "value": ["a", "b"]})
    with pytest.raises(ContractError, match="incomplete key"):
        validate_unique_key(frame, "id")


def test_one_to_one_join_rejects_many_to_many_risk() -> None:
    left = pd.DataFrame({"id": [1, 1], "arm": ["control", "treatment"]})
    right = pd.DataFrame({"id": [1, 1], "outcome": [0, 1]})
    with pytest.raises(ContractError, match="duplicate keys"):
        one_to_one_join(left, right, on="id")


def test_one_to_one_join_requires_declared_population_match() -> None:
    left = pd.DataFrame({"id": [1, 2], "arm": ["control", "treatment"]})
    right = pd.DataFrame({"id": [1], "outcome": [0]})
    with pytest.raises(ContractError, match="unmatched left"):
        one_to_one_join(left, right, on="id", how="left")

    joined = one_to_one_join(
        left,
        right,
        on="id",
        how="left",
        require_match="none",
    )
    assert len(joined) == 2
    assert joined["outcome"].isna().sum() == 1


def test_duplicate_column_names_fail_closed_without_logging_names() -> None:
    sensitive_name = "private_identifier_column"
    frame = pd.DataFrame([[1, 2]], columns=[sensitive_name, sensitive_name])
    with pytest.raises(ContractError, match="duplicate names") as exc_info:
        require_columns(frame, [sensitive_name], frame_name="input")
    assert sensitive_name not in str(exc_info.value)

    irrelevant_duplicate = pd.DataFrame(
        [[1, "left", "right"]], columns=["id", "unused", "unused"]
    )
    with pytest.raises(ContractError, match="duplicate names"):
        require_columns(irrelevant_duplicate, ["id"], frame_name="input")


def test_out_of_supported_range_timestamp_raises_contract_error() -> None:
    with pytest.raises(ContractError):
        parse_aware_utc_series(
            pd.Series(["9999-01-01T00:00:00Z"]), name="event_at"
        )
    with pytest.raises(ContractError):
        parse_aware_utc_timestamp(
            "9999-01-01T00:00:00Z", name="observation_end"
        )
