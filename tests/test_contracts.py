import pandas as pd
import pytest

from email_experiment.contracts import (
    ContractError,
    one_to_one_join,
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
