import numpy as np
import pytest
import torch

from contextad.masking import column_mask, hide_groups
from contextad.models import ContextNet, MaskedMLP
from contextad.scoring import (
    draw_contexts,
    group_scores,
    n_blocks,
    score_context_net,
    score_masked_mlp,
)
from contextad.training import train_masked_mlp

DIM = 6


def test_group_scores_is_the_mean_squared_error_over_available_columns():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(7, DIM, generator=gen)
    columns = np.array([0, 2, 3, 5])
    groups = hide_groups(columns, max_groups=3)

    def zero(a, obs):
        return torch.zeros_like(a), torch.zeros_like(a)

    scores = group_scores(zero, x, column_mask(columns, DIM), groups)
    assert torch.allclose(scores, (x[:, columns] ** 2).mean(dim=1))


def test_group_scores_divides_by_the_predicted_variance():
    x = torch.randn(5, DIM)
    avail = column_mask(np.arange(DIM), DIM)
    groups = hide_groups(np.arange(DIM))

    def wide(a, obs):
        return torch.zeros_like(a), torch.full_like(a, np.log(4.0))

    raw = group_scores(wide, x, avail, groups, normalise=False)
    scaled = group_scores(wide, x, avail, groups, normalise=True)
    assert torch.allclose(scaled, raw / 4.0, atol=1e-6)


def test_group_scores_hides_exactly_one_group_per_call():
    x = torch.randn(3, DIM)
    columns = np.array([1, 2, 4, 5])
    avail = column_mask(columns, DIM)
    groups = hide_groups(columns, max_groups=2)
    seen = []

    def spy(a, obs):
        seen.append(obs[0].clone())
        return torch.zeros_like(a), torch.zeros_like(a)

    group_scores(spy, x, avail, groups)
    assert len(seen) == 2
    for obs, group in zip(seen, groups, strict=True):
        assert not obs[torch.as_tensor(group)].any()
        assert obs.sum() == len(columns) - len(group)
        assert not obs[[0, 3]].any()


def test_draw_contexts_distinct_rows_from_the_pool():
    pool = np.arange(60, dtype=float).reshape(30, 2)
    ctx = draw_contexts(pool, 8, 5, np.random.default_rng(0))
    assert ctx.shape == (8, 5, 2)
    for c in ctx:
        assert len({tuple(r) for r in c}) == 5
    assert np.isin(ctx[..., 0], pool[:, 0]).all()
    again = draw_contexts(pool, 8, 5, np.random.default_rng(0))
    assert np.array_equal(ctx, again)


def test_draw_contexts_rejects_a_context_larger_than_the_pool():
    with pytest.raises(ValueError):
        draw_contexts(np.zeros((4, 2)), 2, 5, np.random.default_rng(0))


def test_n_blocks():
    assert n_blocks(100, 50) == 2 and n_blocks(101, 50) == 3 and n_blocks(1, 50) == 1


@pytest.fixture(scope="module")
def nets():
    torch.manual_seed(0)
    return ContextNet(DIM).eval(), MaskedMLP(DIM).eval()


def test_context_scores_for_rows_not_filling_the_last_block(nets):
    net, _ = nets
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(23, DIM, generator=gen)
    ctx = torch.randn(5, 8, DIM, generator=gen)
    cols = np.arange(DIM)
    scores = score_context_net(net, x, ctx, cols, block=5)
    assert scores.shape == (23,) and np.isfinite(scores).all()
    padded = torch.cat([x, x[-1:], x[-1:]])
    full = score_context_net(net, padded, ctx, cols, block=5)
    assert np.allclose(scores, full[:23], atol=1e-5)


def test_context_scores_do_not_depend_on_the_chunking(nets):
    net, _ = nets
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(40, DIM, generator=gen)
    ctx = torch.randn(4, 6, DIM, generator=gen)
    cols = np.arange(DIM)
    a = score_context_net(net, x, ctx, cols, block=10, chunk=1)
    b = score_context_net(net, x, ctx, cols, block=10, chunk=32)
    assert np.allclose(a, b, atol=1e-5)


def test_context_scores_work_without_context_rows(nets):
    net, _ = nets
    x = torch.randn(12, DIM)
    scores = score_context_net(net, x, torch.zeros(2, 0, DIM), np.arange(DIM), block=6)
    assert scores.shape == (12,) and np.isfinite(scores).all()


def test_context_scores_need_a_context_per_block(nets):
    net, _ = nets
    with pytest.raises(ValueError):
        score_context_net(net, torch.randn(30, DIM), torch.randn(2, 4, DIM), [0, 1], 10)


def test_columns_outside_the_schema_do_not_change_the_scores(nets):
    net, mlp = nets
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(20, DIM, generator=gen)
    ctx = torch.randn(2, 8, DIM, generator=gen)
    cols = np.array([0, 1, 4])
    x2, ctx2 = x.clone(), ctx.clone()
    x2[:, [2, 3, 5]] = 40.0
    ctx2[..., [2, 3, 5]] = -40.0
    a = score_context_net(net, x, ctx, cols, block=10)
    b = score_context_net(net, x2, ctx2, cols, block=10)
    assert np.allclose(a, b, atol=1e-6)
    assert np.allclose(score_masked_mlp(mlp, x, cols), score_masked_mlp(mlp, x2, cols))


def test_masked_mlp_scores_flag_a_row_that_breaks_the_learned_relation():
    gen = torch.Generator().manual_seed(4)
    base = torch.randn(1, 600, 1, generator=gen)
    bank = torch.cat([base, 2 * base, -base + 0.05 * torch.randn_like(base)], dim=2)
    mlp, _ = train_masked_mlp(bank, steps=400, rows=256, seed=0)
    typical = torch.tensor([[1.0, 2.0, -1.0]])
    broken = torch.tensor([[1.0, -2.0, -1.0]])
    cols = np.arange(3)
    good = score_masked_mlp(mlp.eval(), typical, cols)
    bad = score_masked_mlp(mlp.eval(), broken, cols)
    assert bad[0] > 10 * good[0]
