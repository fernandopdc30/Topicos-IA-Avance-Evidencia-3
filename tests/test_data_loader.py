import pandas as pd
import pytest

from src.data_loader import load_raw_tickets


def _write_pair(directory, split, ids, texts, labels):
    pd.DataFrame({"id": ids, "text": texts}).to_csv(directory / f"X_{split}.csv", index=False)
    pd.DataFrame({"id": ids, "category_truth": labels}).to_csv(directory / f"y_{split}.csv", index=False)


def test_load_raw_tickets_merges_and_deduplicates(tmp_path):
    _write_pair(tmp_path, "train", [1, 2], ["a", "b"], ["x", "y"])
    _write_pair(tmp_path, "test", [1, 3], ["a", "c"], ["x", "z"])

    tickets = load_raw_tickets(raw_dir=tmp_path)
    assert len(tickets) == 3
    assert set(tickets["id"]) == {1, 2, 3}
    assert set(tickets.columns) >= {"id", "text", "category_truth", "split"}


def test_load_raw_tickets_detects_misalignment(tmp_path):
    pd.DataFrame({"id": [1, 2], "text": ["a", "b"]}).to_csv(tmp_path / "X_train.csv", index=False)
    pd.DataFrame({"id": [2, 1], "category_truth": ["x", "y"]}).to_csv(
        tmp_path / "y_train.csv", index=False
    )
    pd.DataFrame({"id": [3], "text": ["c"]}).to_csv(tmp_path / "X_test.csv", index=False)
    pd.DataFrame({"id": [3], "category_truth": ["z"]}).to_csv(tmp_path / "y_test.csv", index=False)

    with pytest.raises(ValueError):
        load_raw_tickets(raw_dir=tmp_path)


def test_load_raw_tickets_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_tickets(raw_dir=tmp_path)
