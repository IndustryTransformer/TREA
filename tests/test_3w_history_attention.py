import numpy as np
import pytest
import torch

from examples.three_w_history_attention import (
    HistoryAttentionClassifier,
    HistoryInputs,
    difference_scale,
    history_indices,
    shuffled_history_indices,
    to_sensor_stats,
)


FILES = np.array([0] * 6 + [1] * 2 + [2] * 5 + [3] * 4)
PHYSICAL = np.array([True] * 8 + [False] * 9)


def test_context_is_strictly_earlier_and_file_local():
    indices = history_indices(FILES, history=3)
    assert indices.shape == (len(FILES), 5)
    np.testing.assert_array_equal(indices[:, -1], np.arange(len(FILES)))
    context = indices[:, :-1]
    present = context >= 0
    queries = np.broadcast_to(np.arange(len(FILES))[:, None], context.shape)
    assert (context[present] < queries[present]).all()
    assert (FILES[context[present]] == FILES[queries[present]]).all()
    np.testing.assert_array_equal(indices[0], [-1, -1, -1, -1, 0])
    np.testing.assert_array_equal(indices[3], [-1, 0, 1, 2, 3])
    # The anchor appears only once the first row has left the recent context.
    np.testing.assert_array_equal(indices[4], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(indices[6], [-1, -1, -1, -1, 6])


def test_no_context_arm_and_contiguity_guard():
    np.testing.assert_array_equal(
        history_indices(FILES, history=0), np.arange(len(FILES))[:, None]
    )
    with pytest.raises(ValueError, match="contiguous"):
        history_indices(np.array([0, 1, 0]))


def test_shuffled_control_uses_another_file_of_the_same_source():
    indices = history_indices(FILES, history=3)
    shuffled = shuffled_history_indices(indices, FILES, PHYSICAL, seed=0)
    np.testing.assert_array_equal(shuffled[:, -1], indices[:, -1])
    context = shuffled[:, :-1]
    for query in range(len(FILES)):
        rows = context[query][context[query] >= 0]
        assert (FILES[rows] != FILES[query]).all()
        assert (PHYSICAL[rows] == PHYSICAL[query]).all()


def make_inputs(raw, indices):
    scale = np.ones((raw.shape[1] // 8, 8), dtype=np.float32)
    return HistoryInputs(raw, raw, indices, scale, torch.device("cpu"))


def test_prediction_ignores_future_rows_other_files_and_absent_slots():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(len(FILES), 8 * 3)).astype(np.float32)
    raw[:, 12:15] = 0  # missing-rate block: every sensor observed
    indices = history_indices(FILES, history=3)
    model = HistoryAttentionClassifier(3, indices.shape[1], d_model=16, layers=2).eval()
    rows = torch.tensor([2, 9])
    with torch.no_grad():
        reference = model(*make_inputs(raw, indices)(rows))
        changed = raw.copy()
        changed[3:8] += 100  # later rows of file 0 and all of file 1
        changed[10:] -= 100  # later rows of file 2 and all of file 3
        altered = model(*make_inputs(changed, indices)(rows))
        tokens, absent = make_inputs(raw, indices)(rows)
        tokens = tokens + absent[:, :, None, None] * 50.0
        padded = model(tokens, absent)
    torch.testing.assert_close(reference, altered)
    torch.testing.assert_close(reference, padded, atol=1e-5, rtol=1e-5)


def test_prediction_depends_on_own_history():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(len(FILES), 8 * 3)).astype(np.float32)
    raw[:, 12:15] = 0
    indices = history_indices(FILES, history=3)
    model = HistoryAttentionClassifier(3, indices.shape[1], d_model=16, layers=2).eval()
    changed = raw.copy()
    changed[1] += 5
    with torch.no_grad():
        a = model(*make_inputs(raw, indices)(torch.tensor([2])))
        b = model(*make_inputs(changed, indices)(torch.tensor([2])))
    assert not torch.allclose(a, b)


def test_missing_sensor_pairs_have_no_difference_signal():
    raw = np.ones((2, 8 * 2), dtype=np.float32)
    raw[1, :2] = 9
    raw[:, 8:10] = [[1, 0], [0, 0]]  # sensor 0 missing in the context row
    tokens, absent = make_inputs(raw, history_indices(np.array([0, 0]), history=1))(
        torch.tensor([1])
    )
    # Columns are [anchor, previous, query]; the anchor is absent this early.
    np.testing.assert_array_equal(absent[0], [True, False, False])
    context = tokens[0, 1]
    assert context[0, 8] == 0 and context[0, 17] == 0
    np.testing.assert_array_equal(context[0, 9:17], 0)
    assert context[1, 17] == 1


def test_difference_scale_uses_requested_rows_only():
    raw = to_sensor_stats(
        np.repeat(np.arange(8, dtype=np.float32)[:, None] ** 2, 8, axis=1)
    )
    files = np.zeros(8, dtype=int)
    scale = difference_scale(raw, np.array([1, 2, 3]), files)
    assert scale.shape == (1, 8)
    np.testing.assert_allclose(scale, 2.0)
    raw[6:] = 1e6
    np.testing.assert_allclose(difference_scale(raw, np.array([1, 2, 3]), files), 2.0)


def encoder_model(slots, seed=0):
    import pandas as pd

    from examples.three_w_history_attention import EncoderHistoryClassifier
    from trea.models.single_row import TabularClassifier
    from trea.utils.single_row_data import SingleRowConfig

    torch.manual_seed(seed)
    frame = pd.DataFrame(
        np.random.default_rng(0).normal(size=(8, 6)),
        columns=[f"feature_{i}" for i in range(6)],
    )
    base = TabularClassifier(
        SingleRowConfig.generate(frame), num_classes=2, d_model=16, n_heads=4
    )
    return EncoderHistoryClassifier(base, d_model=16, slots=slots).eval()


def test_offset_indices_are_earlier_file_local_and_anchor_is_not_duplicated():
    from examples.three_w_history_attention import offset_history_indices

    indices = offset_history_indices(FILES, offsets=(4, 2, 1))
    np.testing.assert_array_equal(indices[0], [-1, -1, -1, -1, 0])
    np.testing.assert_array_equal(indices[1], [-1, -1, -1, 0, 1])
    np.testing.assert_array_equal(indices[3], [0, -1, 1, 2, 3])
    np.testing.assert_array_equal(indices[4], [-1, 0, 2, 3, 4])
    np.testing.assert_array_equal(indices[5], [0, 1, 3, 4, 5])
    context = indices[:, :-1]
    present = context >= 0
    queries = np.broadcast_to(np.arange(len(FILES))[:, None], context.shape)
    assert (context[present] < queries[present]).all()
    assert (FILES[context[present]] == FILES[queries[present]]).all()


def test_encoder_model_starts_as_the_base_classifier_then_uses_context():
    from examples.three_w_history_attention import (
        EncoderHistoryInputs,
        offset_history_indices,
    )

    z = np.random.default_rng(2).normal(size=(len(FILES), 6)).astype(np.float32)
    indices = offset_history_indices(FILES, offsets=(2, 1))
    model = encoder_model(slots=indices.shape[1] - 1)
    rows = torch.tensor([0, 2, 9])
    device = torch.device("cpu")
    with torch.no_grad():
        batch = EncoderHistoryInputs(z, indices, device)(rows)
        base = model.base(batch[0], None)
        torch.testing.assert_close(model(*batch), base)
        torch.nn.init.normal_(model.out.weight, std=0.5)
        reference = model(*batch)
        assert torch.isfinite(reference).all()
        assert not torch.allclose(reference[1:], base[1:])
        changed = z.copy()
        changed[3:8] += 10  # later rows of file 0, all of file 1
        changed[10:] -= 10  # later rows of file 2, all of file 3
        altered = model(*EncoderHistoryInputs(changed, indices, device)(rows))
        torch.testing.assert_close(reference, altered)
        changed = z.copy()
        changed[1] += 10  # an earlier row of the query at row 2
        moved = model(*EncoderHistoryInputs(changed, indices, device)(rows))
        torch.testing.assert_close(reference[[0, 2]], moved[[0, 2]])
        assert not torch.allclose(reference[1], moved[1])
        # Absent slots must not leak whatever row id 0 happens to hold.
        query, context, absent = batch
        noisy = context + absent[:, :, None] * 100.0
        torch.testing.assert_close(reference, model(query, noisy, absent))


def test_encoder_context_rows_receive_no_gradient_path():
    from examples.three_w_history_attention import (
        EncoderHistoryInputs,
        offset_history_indices,
    )

    z = np.random.default_rng(3).normal(size=(len(FILES), 6)).astype(np.float32)
    indices = offset_history_indices(FILES, offsets=(2, 1))
    model = encoder_model(slots=indices.shape[1] - 1)
    torch.nn.init.normal_(model.out.weight, std=0.5)
    query, context, absent = EncoderHistoryInputs(z, indices, torch.device("cpu"))(
        torch.tensor([2, 5])
    )
    context = context.clone().requires_grad_(True)
    query = query.clone().requires_grad_(True)
    model(query, context, absent).sum().backward()
    assert context.grad is None
    assert query.grad.abs().sum() > 0


def test_no_context_encoder_arm_is_the_base_model():
    model = encoder_model(slots=0)
    x = torch.randn(5, 6)
    with torch.no_grad():
        torch.testing.assert_close(
            model(x, x.new_zeros(5, 0, 6), torch.zeros(5, 0, dtype=torch.bool)),
            model.base(x, None),
        )
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in model.base.parameters()
    )


def _correction(absolute):
    from examples.three_w_history_attention import HistoryCorrection

    torch.manual_seed(0)
    model = HistoryCorrection(
        16, slots=3, mean=torch.zeros(16), scale=torch.ones(16), absolute=absolute
    )
    h, encoded = torch.randn(5, 16), torch.randn(5, 3, 16)
    absent = torch.zeros(5, 3, dtype=torch.bool)
    absent[0] = True
    absent[1, 0] = True
    return model.eval(), h, encoded, absent


def test_history_correction_is_zero_at_init_and_without_context():
    for absolute in [False, True]:
        model, h, encoded, absent = _correction(absolute)
        assert torch.equal(model(h, encoded, absent), torch.zeros(5))
        torch.nn.init.normal_(model.out.weight)
        delta = model(h, encoded, absent)
        assert delta[0] == 0 and torch.isfinite(delta).all()
        assert (delta[1:] != 0).all()
        changed = encoded.clone()
        changed[1, 0] += 5.0  # absent slot must not matter
        assert torch.allclose(model(h, changed, absent), delta, atol=1e-6)


def test_difference_only_correction_ignores_overall_level():
    model, h, encoded, absent = _correction(False)
    torch.nn.init.normal_(model.out.weight)
    shift = torch.randn(5, 16)
    moved = model(h + shift, encoded + shift[:, None], absent)
    assert torch.allclose(moved, model(h, encoded, absent), atol=1e-5)
    model, h, encoded, absent = _correction(True)
    torch.nn.init.normal_(model.out.weight)
    moved = model(h + shift, encoded + shift[:, None], absent)
    assert not torch.allclose(moved[1:], model(h, encoded, absent)[1:], atol=1e-5)
