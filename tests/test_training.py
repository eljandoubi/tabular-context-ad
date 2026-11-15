import numpy as np
import torch

from contextad.training import (
    sample_episodes,
    train_context_net,
    train_masked_mlp,
)


def toy_bank(n_pop=4, n_rows=64, dim=6, seed=0):
    gen = torch.Generator().manual_seed(seed)
    means = 3 * torch.randn(n_pop, 1, dim, generator=gen)
    return means + 0.1 * torch.randn(n_pop, n_rows, dim, generator=gen)


def test_episodes_hold_distinct_rows_of_one_population():
    n_pop, n_rows = 5, 40
    ids = torch.arange(n_pop)[:, None] * 1000 + torch.arange(n_rows)[None, :]
    bank = ids[:, :, None].float().expand(n_pop, n_rows, 3).contiguous()
    gen = torch.Generator().manual_seed(0)
    context, targets = sample_episodes(bank, 12, 30, 5, gen)
    assert context.shape == (30, 12, 3) and targets.shape == (30, 5, 3)
    for c, t in zip(context[..., 0], targets[..., 0], strict=True):
        rows = torch.cat([c, t]).long()
        assert len(set(rows.tolist())) == 17
        assert len(set((rows // 1000).tolist())) == 1


def test_episodes_without_context():
    bank = toy_bank()
    context, targets = sample_episodes(bank, 0, 8, 3, torch.Generator().manual_seed(1))
    assert context.shape == (8, 0, 6) and targets.shape == (8, 3, 6)


def test_context_net_training_lowers_the_loss():
    model, losses = train_context_net(
        toy_bank(),
        steps=150,
        episodes=16,
        targets=4,
        sizes=(0, 4, 8),
        probs=(0.2, 0.4, 0.4),
    )
    assert np.mean(losses[-20:]) < 0.5 * np.mean(losses[:20])


def test_masked_mlp_training_lowers_the_loss():
    model, losses = train_masked_mlp(toy_bank(), steps=150, rows=64)
    assert np.mean(losses[-20:]) < 0.5 * np.mean(losses[:20])


def test_same_seed_gives_the_same_weights():
    kwargs = dict(steps=4, episodes=4, targets=2, sizes=(2, 4), probs=(0.5, 0.5))
    a, _ = train_context_net(toy_bank(), seed=3, **kwargs)
    b, _ = train_context_net(toy_bank(), seed=3, **kwargs)
    c, _ = train_context_net(toy_bank(), seed=4, **kwargs)
    pa, pb, pc = (list(m.parameters()) for m in (a, b, c))
    assert all(torch.equal(x, y) for x, y in zip(pa, pb, strict=True))
    assert not all(torch.equal(x, y) for x, y in zip(pa, pc, strict=True))


def test_context_size_is_capped_by_the_rows_available():
    bank = toy_bank(n_rows=10)
    model, losses = train_context_net(
        bank, steps=3, episodes=4, targets=4, sizes=(64,), probs=(1.0,)
    )
    assert len(losses) == 3 and np.isfinite(losses).all()
