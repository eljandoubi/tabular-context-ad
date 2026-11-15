import numpy as np
import pytest
import torch

from contextad.detectors import knn_to_context
from contextad.evaluate import (
    context_net_scores,
    group_contexts,
    reference_scores,
    summarize,
    summarize_table,
)
from contextad.metrics import auprc, auroc
from contextad.models import ContextNet
from contextad.scoring import score_context_net, to_blocks

DIM = 4


def test_to_blocks_pads_with_the_last_row():
    x = np.arange(14, dtype=float).reshape(7, 2)
    out = to_blocks(x, 3, 3)
    assert out.shape == (3, 3, 2)
    assert np.array_equal(out.reshape(9, 2)[:7], x)
    assert (out.reshape(9, 2)[7:] == x[-1]).all()
    with pytest.raises(ValueError):
        to_blocks(x, 2, 3)


def test_group_contexts_come_from_the_pool_of_their_group():
    pools = {0: np.random.default_rng(0).uniform(0, 1, (40, DIM))}
    pools[1] = pools[0] + 10.0
    group = np.array([0] * 120 + [1] * 60)
    ctx = group_contexts(pools, group, 8, np.random.default_rng(1), block=50)
    assert ctx[0].shape == (3, 8, DIM) and ctx[1].shape == (2, 8, DIM)
    assert ctx[0].max() <= 1.0 and ctx[1].min() >= 10.0


def test_context_net_scores_follow_the_group_labels():
    torch.manual_seed(0)
    net = ContextNet(DIM).eval()
    rng = np.random.default_rng(2)
    x = rng.normal(size=(90, DIM))
    group = rng.integers(0, 2, size=90)
    pools = {0: rng.normal(size=(30, DIM)), 1: rng.normal(size=(30, DIM)) + 3}
    ctx = group_contexts(pools, group, 6, rng, block=20)
    cols = np.arange(DIM)
    scores = context_net_scores(net, x, group, ctx, cols, block=20)
    for g in (0, 1):
        direct = score_context_net(net, x[group == g], ctx[g], cols, block=20)
        assert np.allclose(scores[group == g], direct)


def test_reference_scores_match_a_direct_call():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(35, DIM))
    group = np.zeros(35, dtype=int)
    ctx = {0: rng.normal(size=(2, 10, DIM))}
    cols = np.array([0, 2])
    out = reference_scores("knn_context", x, group, ctx, cols, block=20)
    rows = to_blocks(x[:, cols], 2, 20)
    expected = knn_to_context(ctx[0][..., cols], rows).reshape(-1)[:35]
    assert out.shape == (35,) and np.allclose(out, expected)
    maha = reference_scores("mahalanobis_context", x, group, ctx, cols, block=20)
    assert maha.shape == (35,) and (maha >= 0).all()


def test_summarize_separates_anomaly_types_and_averages_over_groups():
    # group 0 ranks the contextual rows first, group 1 ranks the ordinary ones first
    kind = np.array([0, 0, 1, 2, 0, 0, 1, 2])
    scores = np.array([0.1, 0.2, 0.9, 0.5, 0.1, 0.2, 0.5, 0.9])
    group = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    out = summarize(kind, scores, group)
    assert out["auroc"] == 1.0 and out["auprc"] == 1.0
    assert out["auroc_contextual"] == pytest.approx(1.0)
    assert out["auroc_ordinary"] == pytest.approx(1.0)
    flipped = summarize(kind, -scores, group)
    assert flipped["auroc"] == 0.0
    per_group = [auroc(kind[:4] > 0, scores[:4]), auroc(kind[4:] > 0, scores[4:])]
    assert out["auroc"] == pytest.approx(np.mean(per_group))
    assert out["auroc_pooled"] == auroc(kind > 0, scores)


def test_summarize_with_a_single_anomaly_type():
    kind = np.array([0, 0, 0, 1, 0, 0, 1, 0])
    scores = np.array([1.0, 2.0, 3.0, 9.0, 1.0, 2.0, 0.0, 3.0])
    out = summarize(kind, scores, np.array([0] * 4 + [1] * 4))
    assert out["auroc_contextual"] == pytest.approx(0.5)
    assert np.isnan(out["auroc_ordinary"])


def test_summarize_table():
    y = np.array([0, 0, 1, 1, 0])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.2])
    out = summarize_table(y, s)
    assert out == {"auroc": auroc(y, s), "auprc": auprc(y, s)}
