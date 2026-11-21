import pytest

from examples.collect_3w_regularization import verify_contrast


def test_allows_only_planned_contrast():
    baseline = dict(
        weight_decay=0.0003, lr=0.0003, models="single_row", output_dir="old"
    )
    candidate = baseline | dict(
        weight_decay=0.03, models="single_row,tabm_ple", output_dir="new"
    )
    verify_contrast(baseline, candidate)
    for modification in [dict(lr=0.001), dict(weight_decay=0.3), dict(extra_flag=True)]:
        with pytest.raises(ValueError):
            verify_contrast(baseline, candidate | modification)
