import pytest

from examples.benchmark_3w_regularization import contrast_config


def config():
    return dict(
        evaluate_test=False,
        value_policy="preserve_range",
        feature_family="all",
        relative_stats=False,
        preprocess="quantile",
        group_manifest="frozen.json",
        models="single_row",
        weight_decay=0.0003,
        seeds="42,43,44",
        width=128,
        layers=2,
        lr=0.0003,
        group_loss_power=0.5,
        output_dir="original",
    )


def test_only_declared_contrast_changes():
    baseline = config()
    modified = vars(contrast_config(baseline, "new"))
    assert {key for key in baseline if baseline[key] != modified[key]} == {
        "weight_decay",
        "models",
        "output_dir",
    }
    assert baseline["weight_decay"] == 0.0003


def test_forbids_held_out_loading_or_unexpected_baseline():
    for updates in [
        dict(evaluate_test=True),
        dict(weight_decay=0.01),
        dict(seeds="42"),
    ]:
        with pytest.raises(ValueError):
            contrast_config(config() | updates, "new")
