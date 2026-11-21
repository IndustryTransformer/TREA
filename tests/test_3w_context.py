from types import SimpleNamespace

import pandas as pd
import pytest

from examples import benchmark_3w_context as context


def test_context_runner_refuses_consumed_confirmation():
    with pytest.raises(ValueError, match="consumed confirmation"):
        context.run(SimpleNamespace(evaluate_test=True))


def test_chronology_audit_rejects_unsorted_timestamp_index(monkeypatch):
    protocol = {
        s: {"data_path": "/unused", "files": [f"{s}.parquet"]} for s in ["train", "val"]
    }
    valid = pd.DataFrame(index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    monkeypatch.setattr(context.pd, "read_parquet", lambda *args, **kwargs: valid)
    result = context.audit_chronology(protocol)
    assert result["files"] == 2
    assert result["all_datetime_monotonic"]
    monkeypatch.setattr(
        context.pd, "read_parquet", lambda *args, **kwargs: valid.iloc[::-1]
    )
    with pytest.raises(ValueError, match="chronological order"):
        context.audit_chronology(protocol)
