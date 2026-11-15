"""Self-supervised training: predict hidden columns of rows, with or without context."""

from __future__ import annotations

import math

import torch

from .masking import hide_mask, schema_mask
from .models import ContextNet, MaskedMLP, gaussian_nll

CONTEXT_SIZES = (0, 4, 8, 16, 32, 64)
CONTEXT_PROBS = (0.1, 0.1, 0.15, 0.15, 0.3, 0.2)


def sample_episodes(bank, k: int, episodes: int, targets: int, gen: torch.Generator):
    """Per episode: one population, k context rows and `targets` other rows of it.

    bank has shape (populations, rows, dim). Returns (context, targets).
    """
    n_pop, n_rows, _ = bank.shape
    pop = torch.randint(n_pop, (episodes,), generator=gen)
    picks = torch.rand(episodes, n_rows, generator=gen).argsort(dim=1)
    rows = bank[pop[:, None], picks[:, : k + targets]]
    return rows[:, :k], rows[:, k:]


def _optimizer(model, lr: float, steps: int, warmup: int = 100):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, factor)


def _update(model, opt, sched, loss):
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()


def train_context_net(
    bank: torch.Tensor,
    steps: int = 3000,
    episodes: int = 32,
    targets: int = 16,
    lr: float = 2e-3,
    sizes: tuple[int, ...] = CONTEXT_SIZES,
    probs: tuple[float, ...] = CONTEXT_PROBS,
    seed: int = 0,
):
    """Train a ContextNet on episodes drawn from bank; returns (model, losses)."""
    dim = bank.shape[2]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = ContextNet(dim)
    opt, sched = _optimizer(model, lr, steps)
    weights = torch.tensor(probs, dtype=torch.float)
    losses = []
    for _ in range(steps):
        k = sizes[int(torch.multinomial(weights, 1, generator=gen))]
        k = min(k, bank.shape[1] - targets)
        context, x = sample_episodes(bank, k, episodes, targets, gen)
        avail = schema_mask(episodes, dim, gen)
        avail_rows = avail[:, None, :].expand(-1, targets, -1)
        hidden = hide_mask(avail_rows, gen)
        mean, logvar = model(x, avail_rows & ~hidden, context, avail)
        loss = gaussian_nll(mean, logvar, x, hidden)
        _update(model, opt, sched, loss)
        losses.append(loss.item())
    return model, losses


def train_masked_mlp(
    bank: torch.Tensor,
    steps: int = 3000,
    rows: int = 512,
    lr: float = 2e-3,
    seed: int = 0,
):
    """Train a MaskedMLP on rows pooled from bank; returns (model, losses)."""
    dim = bank.shape[2]
    flat = bank.reshape(-1, dim)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = MaskedMLP(dim)
    opt, sched = _optimizer(model, lr, steps)
    losses = []
    for _ in range(steps):
        x = flat[torch.randint(len(flat), (rows,), generator=gen)]
        avail = schema_mask(rows, dim, gen)
        hidden = hide_mask(avail, gen)
        mean, logvar = model(x, avail & ~hidden)
        loss = gaussian_nll(mean, logvar, x, hidden)
        _update(model, opt, sched, loss)
        losses.append(loss.item())
    return model, losses
