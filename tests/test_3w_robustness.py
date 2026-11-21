from examples.prepare_3w_robustness import make_folds


def test_outer_folds_cover_training_only_and_keep_validation_and_old_tests_separate():
    parent = {
        "file_inventory_sha256": "test",
        "partitions": {
            "train": [
                *[f"WELL-{i:03}" for i in range(20)],
                *[f"sim-{i}" for i in range(24)],
            ],
            "val": ["WELL-validation", "sim-validation"],
            "test": ["WELL-consumed", "sim-consumed"],
            "excluded": ["WELL-earlier"],
        },
    }
    folds = make_folds(parent)
    assert folds == make_folds(parent)
    outer_groups = []
    for fold in folds:
        parts = {s: set(ids) for s, ids in fold["partitions"].items()}
        assert parts["val"] == set(parent["partitions"]["val"])
        assert parts["excluded"] == {"WELL-consumed", "sim-consumed", "WELL-earlier"}
        for a in parts:
            for b in parts:
                if a != b:
                    assert not parts[a] & parts[b]
        assert fold["counts"]["test"]["physical_wells"] == 5
        outer_groups += list(parts["test"])
    assert len(outer_groups) == len(set(outer_groups))
    assert set(outer_groups) == set(parent["partitions"]["train"])
