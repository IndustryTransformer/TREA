import copy
import hashlib
import json

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.benchmark_3w_binary import binary_metrics, extract_engineered_features
from examples.benchmark_3w_grouped import (
    TemporalForward,
    build_model,
    choose_threshold,
    group_loss_weights,
    load_data,
    summary_features,
)
from examples.evaluate_3w_reserved import validate_evaluation_plan
from trea.models import TriplePatchTransformer
from utils.three_w import (
    frozen_group_split,
    grouped_file_split,
    sanitize_numeric_values,
)


def test_frozen_groups_validate_inventory_coverage_and_disjointness():
    files = [f"WELL-{i:04d}_{j}.parquet" for i in range(4) for j in range(2)]
    manifest = {
        "file_inventory_sha256": hashlib.sha256(
            "\n".join(sorted(files)).encode()
        ).hexdigest(),
        "partitions": dict(
            zip(
                ["train", "val", "test", "excluded"],
                [[f"WELL-{i:04d}"] for i in range(4)],
                strict=True,
            )
        ),
    }
    groups, indices = frozen_group_split(files, manifest)
    assert set(groups[indices["train"]]) == {"WELL-0000"}
    assert set(groups[indices["excluded"]]) == {"WELL-0003"}
    with pytest.raises(ValueError, match="inventory"):
        frozen_group_split(files[:-1], manifest)
    with pytest.raises(ValueError, match="canonical file order"):
        frozen_group_split(files[::-1], manifest)
    bad = copy.deepcopy(manifest)
    bad["partitions"]["test"].append("WELL-0000")
    with pytest.raises(ValueError, match="overlapping"):
        frozen_group_split(files, bad)
    bad = copy.deepcopy(manifest)
    bad["partitions"]["excluded"] = []
    with pytest.raises(ValueError, match="cover"):
        frozen_group_split(files, bad)


def test_reserved_plan_rejects_incomplete_fits_and_duplicate_blend_members(tmp_path):
    (tmp_path / "protocol.json").write_text(json.dumps({"train": {}, "val": {}}))
    (tmp_path / "config.json").write_text("{}")
    rows = [
        {"model": name, "seed": seed}
        for name in ["xgb", "single_row"]
        for seed in [42, 43]
    ]
    (tmp_path / "results.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="requires seeds"):
        validate_evaluation_plan([str(tmp_path)], ["xgb", "single_row"])
    rows += [{"model": name, "seed": 44} for name in ["xgb", "single_row"]]
    (tmp_path / "results.json").write_text(json.dumps(rows))
    validate_evaluation_plan([str(tmp_path)], ["xgb", "single_row"])
    with pytest.raises(ValueError, match="blend"):
        validate_evaluation_plan(
            [str(tmp_path)], ["xgb", "single_row"], ["bad=single_row,single_row"]
        )


def test_grouped_three_way_is_disjoint_reproducible_and_preserves_outer_holdout():
    files = [f"WELL-{i:04d}_{j}.parquet" for i in range(30) for j in range(3)]
    groups, split = grouped_file_split(files, 0.2, 0.25, 42)
    _, legacy = grouped_file_split(files, 0.2, 0, 42)
    np.testing.assert_array_equal(split["test"], legacy["val"])
    np.testing.assert_array_equal(
        split["val"], grouped_file_split(files, 0.2, 0.25, 42)[1]["val"]
    )
    sets = {k: set(groups[v]) for k, v in split.items()}
    assert len(np.concatenate(list(split.values()))) == len(files)
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]


def test_threshold_is_exact_with_ties_and_extreme_class_imbalance():
    y = np.array([0, 1, 1, 0, 1, 0, 0])
    p = np.array([0.1, 0.1, 0.3, 0.7, 0.8, 0.9, 0.9])
    threshold = choose_threshold(y, p)
    actual = binary_metrics((p >= threshold).astype(int), y)["macro_f1"]
    brute = max(binary_metrics((p >= t).astype(int), y)["macro_f1"] for t in [0, *p, 1])
    assert actual == pytest.approx(brute)


def test_group_weighting_preserves_source_mass_and_balances_group_influence():
    groups = np.array(["WELL-1"] * 8 + ["WELL-2"] * 2 + ["sim-1"] * 3 + ["sim-2"])
    assert group_loss_weights(groups, 0) is None
    weights = group_loss_weights(groups, 1)
    assert weights[:10].sum() == pytest.approx(10)
    assert weights[10:].sum() == pytest.approx(4)
    assert weights[:8].sum() == pytest.approx(weights[8:10].sum())
    assert weights[10:13].sum() == pytest.approx(weights[13:].sum())
    assert np.isfinite(group_loss_weights(groups[:10], 0.5)).all()


def test_relative_summaries_preserve_base_features_and_are_unit_invariant():
    windows = np.array([[[1, 2, 3, 4], [np.nan] * 4]], dtype=np.float32)
    data = {"windows": windows, "features": extract_engineered_features(windows)}
    norm = {"mean": [10, 20], "std": [2, 3]}
    augmented = summary_features(data, norm, relative=True)
    np.testing.assert_array_equal(augmented[:, :16], data["features"])
    assert augmented.shape == (1, 24)
    assert np.isfinite(augmented).all()
    np.testing.assert_array_equal(augmented[:, 17::2], 0)
    changed_units = {"mean": [10000, 20], "std": [2000, 3]}
    np.testing.assert_allclose(
        augmented, summary_features(data, changed_units, relative=True), rtol=1e-6
    )


def test_piecewise_attention_model_forward_backward_and_checkpoint():
    args = SimpleNamespace(width=16, layers=2)
    x = np.random.default_rng(42).normal(size=(32, 4)).astype(np.float32)
    net = build_model("single_row_ple", x, args)
    logits = net(torch.from_numpy(x[:4]))
    assert logits.shape == (4, 2)
    logits.square().mean().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None
    )
    restored = build_model("single_row_ple", x, args)
    restored.load_state_dict(net.state_dict(), strict=True)
    torch.testing.assert_close(restored(torch.from_numpy(x[:4])), logits)


def model(**kwargs):
    return TriplePatchTransformer(
        C_num=3,
        C_cat=0,
        cat_cardinalities=[],
        T=8,
        d_model=16,
        n_head=4,
        num_layers=1,
        patch_len=4,
        stride=4,
        num_classes=2,
        use_stat_tokens=True,
        **kwargs,
    )


def test_exact_stat_values_match_xgb_including_missing_endpoints():
    x = torch.tensor(
        [
            [
                [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, float("nan")],
                [float("nan")] * 8,
                [2.0] * 8,
            ]
        ]
    )
    m = model(stat_exact_engineered=True, stat_include_delta=True)
    captured = []
    m.stat_token_proj.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach())
    )
    m.compute_stat_tokens(torch.nan_to_num(x), torch.isnan(x).float())
    # Native token ordering: mean,std,missing,trend,energy,min,max,delta.
    actual = captured[0][:, :, [0, 1, 5, 6, 2, 7, 3, 4]].transpose(1, 2).reshape(1, -1)
    expected = extract_engineered_features(x.numpy())
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)


@pytest.mark.parametrize("kind", ["legacy", "direct", "only"])
def test_stat_variants_have_finite_forward_and_gradients(kind):
    m = model(
        stat_sensor_identity=kind != "legacy",
        stat_include_delta=kind != "legacy",
        direct_stat_pooling=kind == "direct",
        stats_only=kind == "only",
        stat_exact_engineered=kind != "legacy",
    )
    x = torch.randn(4, 3, 8)
    x[0] = float("nan")
    out = m(x, torch.empty(4, 0, 8, dtype=torch.long))
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None
    )


def test_explicit_disabled_flags_keep_legacy_checkpoint_compatible():
    torch.manual_seed(42)
    old = model()
    torch.manual_seed(42)
    new = model(
        stat_sensor_identity=False,
        stat_include_delta=False,
        direct_stat_pooling=False,
        stats_only=False,
        stat_exact_engineered=False,
    )
    new.load_state_dict(old.state_dict(), strict=True)
    assert old.state_dict().keys() == new.state_dict().keys()
    old.eval()
    new.eval()
    x = torch.randn(2, 3, 8)
    cat = torch.empty(2, 0, 8, dtype=torch.long)
    torch.testing.assert_close(old(x, cat), new(x, cat), rtol=0, atol=0)


def test_invalid_stat_flags_fail_early():
    with pytest.raises(ValueError, match="Choose"):
        model(stats_only=True, direct_stat_pooling=True)


def test_preserve_range_keeps_pressure_variation_and_masks_extremes():
    x = np.array([[2e7, 3e7, -1.180116e42, -1.711613e38, np.inf, np.nan, -40.0]])
    original = x.copy()
    legacy = sanitize_numeric_values(x)
    repaired = sanitize_numeric_values(x, "preserve_range")
    np.testing.assert_array_equal(legacy[0, :2], [1e6, 1e6])
    np.testing.assert_array_equal(repaired[0, :2], [2e7, 3e7])
    assert np.isnan(repaired[0, 2:6]).all()
    assert repaired[0, -1] == -40
    assert repaired.dtype == np.float32
    np.testing.assert_array_equal(x, original)


def test_legacy_value_policy_matches_original_clipping_exactly():
    x = np.array([np.nan, np.inf, -np.inf, 3e7, -4e6, 10.0, -20.0])
    expected = np.clip(x, -1e6, 1e6).astype(np.float32)
    np.testing.assert_array_equal(sanitize_numeric_values(x), expected)


def test_development_loader_never_reads_test_and_reuses_train_stats(
    tmp_path, monkeypatch
):
    calls = []

    class FakeDataset:
        def __init__(self, **kwargs):
            split = kwargs["split"]
            calls.append((split, kwargs["normalization_stats"]))
            self.windows = np.arange(32, dtype=np.float32).reshape(2, 2, 8)
            self.labels = np.array([0, 1])
            self.file_ids = np.array([0, 1])
            self.group_ids = np.array([split, split])
            self.selected_groups = [split]
            self.selected_files = [f"{split}.parquet"]
            self.normalization_stats = {
                "mean": np.array([1.0, 2.0]),
                "std": np.array([3.0, 4.0]),
            }

        def __len__(self):
            return len(self.labels)

    monkeypatch.setattr("examples.benchmark_3w_grouped.ThreeWDataset", FakeDataset)
    args = SimpleNamespace(
        split_seed=42,
        holdout_fraction=0.2,
        internal_val_fraction=0.25,
        data_path="fake",
        toolkit_path="fake",
        cache_dir=str(tmp_path),
        evaluate_test=False,
    )
    data, _ = load_data(args)
    assert set(data) == {"train", "val"}
    assert [c[0] for c in calls] == ["train", "val"]
    assert calls[0][1] is None
    np.testing.assert_array_equal(calls[1][1]["mean"], [1.0, 2.0])
    args.evaluate_test = True
    data, _ = load_data(args)
    assert set(data) == {"train", "val", "test"}
    assert [c[0] for c in calls] == ["train", "val", "test"]
    np.testing.assert_array_equal(calls[-1][1]["mean"], [1.0, 2.0])


def test_shuffle_preserves_simultaneous_sensors_and_is_deterministic_at_eval():
    class Capture(torch.nn.Module):
        def forward(self, x, cat):
            return x

    wrapper = TemporalForward(Capture(), shuffle=True).eval()
    x = torch.arange(8.0).repeat(4, 2, 1)
    x[:, 1] += 100
    shuffled = wrapper(x)
    torch.testing.assert_close(
        shuffled[:, 1] - shuffled[:, 0], torch.full((4, 8), 100.0)
    )
    torch.testing.assert_close(shuffled.sort(dim=-1).values, x.sort(dim=-1).values)
    torch.testing.assert_close(shuffled, wrapper(x), rtol=0, atol=0)
    assert not torch.equal(shuffled[0], shuffled[1])
