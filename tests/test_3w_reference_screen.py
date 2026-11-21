import pytest

from examples.collect_3w_reference_screen import summarize


def sample_rows():
    return [
        {"model": "xgb", "seed": seed, "validation": {"auc": value, "macro_f1": value}}
        for seed, value in [(42, 0.8), (43, 0.9), (44, 1.0)]
    ] + [{"model": "xgb_ensemble", "validation": {"auc": 0.99, "macro_f1": 0.98}}]


def test_separates_single_fits_from_ensemble():
    result = summarize(sample_rows(), "xgb")
    assert result["single_fit_auc"]["mean"] == pytest.approx(0.9)
    assert result["ensemble"]["auc"] == 0.99
    assert result["single_fit_auc"]["population_std"] == pytest.approx(0.081649658)


def test_rejects_partial_or_duplicate_seeds_and_missing_ensemble():
    for rows in [
        sample_rows()[1:],
        sample_rows() + [sample_rows()[0]],
        sample_rows()[:-1],
    ]:
        with pytest.raises(ValueError):
            summarize(rows, "xgb")
