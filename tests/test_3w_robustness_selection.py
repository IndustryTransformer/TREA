import json

import pytest

from examples.freeze_3w_robustness import candidate_rows, verify_history_gate


def test_robustness_selection_requires_exact_three_fit_ensemble():
    rows = [{"model": "single_row", "seed": s} for s in [42, 43, 44]]
    with pytest.raises(ValueError, match="ensemble"):
        candidate_rows(rows, "single_row")
    rows.append({"model": "single_row_ensemble", "seeds": [42, 43, 44]})
    fits, ensemble = candidate_rows(rows, "single_row")
    assert len(fits) == 3
    assert ensemble["seeds"] == [42, 43, 44]
    with pytest.raises(ValueError, match="duplicated"):
        candidate_rows([*rows, rows[0]], "single_row")


def test_unverified_history_cannot_pass_selection_gate(tmp_path):
    (tmp_path / "history_equivalence.json").write_text(json.dumps({"passed": False}))
    with pytest.raises(ValueError, match="not verified"):
        verify_history_gate(tmp_path)
