import numpy as np
import pytest
import torch

from contextad.training import (
    _validation_loss_context,
    _validation_loss_mlp,
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
    fit = train_context_net(
        toy_bank(),
        steps=150,
        episodes=16,
        targets=4,
        sizes=(0, 4, 8),
        probs=(0.2, 0.4, 0.4),
    )
    assert np.mean(fit.losses[-20:]) < 0.5 * np.mean(fit.losses[:20])
    assert fit.val_losses == [] and fit.best_step is None


def test_masked_mlp_training_lowers_the_loss():
    fit = train_masked_mlp(toy_bank(), steps=150, rows=64)
    assert np.mean(fit.losses[-20:]) < 0.5 * np.mean(fit.losses[:20])


def test_same_seed_gives_the_same_weights():
    kwargs = dict(steps=4, episodes=4, targets=2, sizes=(2, 4), probs=(0.5, 0.5))
    a = train_context_net(toy_bank(), seed=3, **kwargs).model
    b = train_context_net(toy_bank(), seed=3, **kwargs).model
    c = train_context_net(toy_bank(), seed=4, **kwargs).model
    pa, pb, pc = (list(m.parameters()) for m in (a, b, c))
    assert all(torch.equal(x, y) for x, y in zip(pa, pb, strict=True))
    assert not all(torch.equal(x, y) for x, y in zip(pa, pc, strict=True))


def test_context_size_is_capped_by_the_rows_available():
    bank = toy_bank(n_rows=10)
    fit = train_context_net(
        bank, steps=3, episodes=4, targets=4, sizes=(64,), probs=(1.0,)
    )
    assert len(fit.losses) == 3 and np.isfinite(fit.losses).all()


def test_a_bank_function_is_asked_for_a_new_bank_every_refresh():
    for train, kwargs in (
        (train_context_net, dict(episodes=4, targets=2, sizes=(2,), probs=(1.0,))),
        (train_masked_mlp, dict(rows=16)),
    ):
        calls = []

        def make(round_, calls=calls):
            calls.append(round_)
            return toy_bank(seed=round_)

        fit = train(make, steps=10, refresh_every=4, **kwargs)
        assert calls == [0, 1, 2]
        assert len(fit.losses) == 10


def test_context_net_keeps_the_weights_with_the_best_validation_loss():
    rows = toy_bank(n_pop=1, n_rows=96)
    fit = train_context_net(
        rows[:, :72],
        steps=60,
        episodes=8,
        targets=4,
        sizes=(4, 8),
        probs=(0.5, 0.5),
        val=rows[0, 72:],
        val_every=20,
    )
    assert [step for step, _ in fit.val_losses] == [20, 40, 60]
    step, best = min(fit.val_losses, key=lambda item: item[1])
    assert fit.best_step == step
    kept = _validation_loss_context(fit.model, rows[0, :72], rows[0, 72:])
    assert kept == pytest.approx(best, abs=1e-5)


def test_masked_mlp_validation_uses_the_trimmed_loss():
    rows = toy_bank(n_pop=1, n_rows=96)
    fit = train_masked_mlp(
        rows[:, :72], steps=40, rows=32, trim=0.2, val=rows[0, 72:], val_every=20
    )
    step, best = min(fit.val_losses, key=lambda item: item[1])
    kept = _validation_loss_mlp(fit.model, rows[0, 72:], trim=0.2)
    assert fit.best_step == step and kept == pytest.approx(best, abs=1e-5)
    assert kept != pytest.approx(_validation_loss_mlp(fit.model, rows[0, 72:]))


def test_masked_mlp_keeps_the_weights_with_the_best_validation_loss():
    rows = toy_bank(n_pop=1, n_rows=96)
    fit = train_masked_mlp(
        rows[:, :72], steps=60, rows=32, val=rows[0, 72:], val_every=20
    )
    step, best = min(fit.val_losses, key=lambda item: item[1])
    assert fit.best_step == step
    assert _validation_loss_mlp(fit.model, rows[0, 72:]) == pytest.approx(
        best, abs=1e-5
    )
