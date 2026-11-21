"""Protect the replication's job grid, label draws, and pre-registered verdict."""

import numpy as np

from scripts.nox_ssl_replication import (
    INIT_SEEDS,
    LOW,
    REPRO_DRAW,
    plan_jobs,
    verdict,
)
from scripts.nox_temporal_pretraining import draw_indices


def test_job_grid():
    supervised, ssl, pretrained = plan_jobs([101, 102], [42, 43], [0.01, 0.1])
    assert len(set(supervised)) == len(supervised)
    assert len(set(pretrained)) == len(pretrained)
    # Random-encoder control only at <=5% labels, one per SSL seed.
    random = [j for j in supervised if j[0] == "random"]
    assert {j[1] for j in random} == {0.01}
    assert len(random) == 2 * 2 * len(INIT_SEEDS) + len(INIT_SEEDS)
    # The reproduction cell uses only the original SSL seed.
    assert {j[3] for j in pretrained if j[2] == REPRO_DRAW} == {42}
    assert sorted(ssl) == sorted(
        {(a, 1.0, 0, s, 0) for a in ["ssl_pre", "ssl_pre_target"] for s in [42, 43]}
    )
    _, _, no_repro = plan_jobs([101], [42], [0.01], repro=False)
    assert all(j[2] == 101 for j in no_repro)


def test_draw_seeds_give_distinct_nested_chronological_draws():
    historical = np.arange(22191)
    fits = []
    for seed in [101, 102]:
        previous = set()
        for fraction in [0.01, 0.02, 0.05, 0.10]:
            fit, val = draw_indices(historical, fraction, seed)
            assert fit.max() < val.min()
            assert len(fit) + len(val) == round(len(historical) * fraction)
            assert previous <= set(fit)
            previous = set(fit)
        fits.append(previous)
    assert fits[0] != fits[1]
    # The default still reproduces the original seed-42 draw.
    assert np.array_equal(
        draw_indices(historical, 0.01)[0], draw_indices(historical, 0.01, 42)[0]
    )


def make_ens(draws, seeds, pre_delta, random_delta=0.5):
    """pre = scratch + pre_delta[fraction][draw]; random = pre + random_delta."""
    ens = {}
    for f in map(str, LOW):
        for i, d in enumerate(draws):
            ens[("scratch", f, d, 0)] = 12.0
            for s in seeds:
                pre = 12.0 + pre_delta[f][i]
                ens[("pre", f, d, s)] = pre
                ens[("random", f, d, s)] = pre + random_delta
    return ens


def test_verdict_rules():
    draws, seeds = [1, 2, 3, 4, 5], [42, 43, 44]
    better = [-0.3] * 5
    ens = make_ens(draws, seeds, {str(f): better for f in LOW})
    assert verdict(ens, draws, seeds)["status"] == "REPLICATED"

    # One fraction where only 3/5 draws improve fails criterion B.
    mixed = {str(f): better for f in LOW}
    mixed["0.05"] = [-1.0, -1.0, -1.0, 0.1, 0.1]
    v = verdict(make_ens(draws, seeds, mixed), draws, seeds)
    assert v["status"] == "PARTIAL"
    assert v["tests"]["0.05"]["criteria"]["A_mean_beats_scratch"]
    assert not v["tests"]["0.05"]["criteria"]["B_draws_improved"]

    # Beating scratch but not the random-encoder control fails criterion C.
    v = verdict(
        make_ens(draws, seeds, {str(f): better for f in LOW}, -0.1), draws, seeds
    )
    assert v["status"] == "NOT REPLICATED"

    del ens[("pre", "0.01", 1, 42)]
    assert verdict(ens, draws, seeds)["status"] == "INCOMPLETE"
