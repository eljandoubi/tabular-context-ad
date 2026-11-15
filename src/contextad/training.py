"""Self-supervised training: predict hidden columns of rows, with or without context."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from .masking import hide_mask, schema_mask
from .models import ContextNet, MaskedMLP, gaussian_nll

CONTEXT_SIZES = (0, 4, 8, 16, 32, 64)
CONTEXT_PROBS = (0.1, 0.1, 0.15, 0.15, 0.3, 0.2)
VALIDATION_SEED = 12345

Bank = torch.Tensor | Callable[[int], torch.Tensor]


@dataclass
class Fit:
    """A trained model with its training loss and, if used, the validation curve."""

    model: torch.nn.Module
    losses: list[float]
    val_losses: list[tuple[int, float]] = field(default_factory=list)
    best_step: int | None = None


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


@torch.no_grad()
def _validation_loss_mlp(model, val, trim: float = 0.0, repeats: int = 4):
    gen = torch.Generator().manual_seed(VALIDATION_SEED)
    x = val.repeat(repeats, 1)
    avail = schema_mask(len(x), val.shape[1], gen)
    hidden = hide_mask(avail, gen)
    mean, logvar = model(x, avail & ~hidden)
    return gaussian_nll(mean, logvar, x, hidden, trim).item()


@torch.no_grad()
def _validation_loss_context(model, pool, val, trim: float = 0.0, k: int = 32):
    gen = torch.Generator().manual_seed(VALIDATION_SEED)
    repeats = 4
    k = min(k, len(pool))
    picks = torch.rand(repeats, len(pool), generator=gen).argsort(dim=1)[:, :k]
    x = val.unsqueeze(0).expand(repeats, -1, -1)
    avail = schema_mask(repeats, val.shape[1], gen)
    avail_rows = avail[:, None, :].expand(-1, len(val), -1)
    hidden = hide_mask(avail_rows, gen)
    mean, logvar = model(x, avail_rows & ~hidden, pool[picks], avail)
    return gaussian_nll(mean, logvar, x, hidden, trim).item()


def train_context_net(
    bank: Bank,
    steps: int = 3000,
    episodes: int = 32,
    targets: int = 16,
    lr: float = 2e-3,
    sizes: tuple[int, ...] = CONTEXT_SIZES,
    probs: tuple[float, ...] = CONTEXT_PROBS,
    seed: int = 0,
    refresh_every: int = 250,
    trim: float = 0.0,
    val: torch.Tensor | None = None,
    val_every: int = 50,
) -> Fit:
    """Train a ContextNet on episodes drawn from a bank (populations, rows, dim).

    bank may be a function of the refresh round that returns a new tensor every
    refresh_every steps. trim is the share of target rows left out of each loss.
    With val (rows, dim) held out of a one-population bank, the weights with the
    lowest validation loss (context of 32 rows) are kept.
    """
    current = bank(0) if callable(bank) else bank
    dim = current.shape[2]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = ContextNet(dim)
    opt, sched = _optimizer(model, lr, steps)
    weights = torch.tensor(probs, dtype=torch.float)
    fit = Fit(model, [])
    best, state = math.inf, None
    for step in range(steps):
        if callable(bank) and step and step % refresh_every == 0:
            current = bank(step // refresh_every)
        k = sizes[int(torch.multinomial(weights, 1, generator=gen))]
        k = min(k, current.shape[1] - targets)
        context, x = sample_episodes(current, k, episodes, targets, gen)
        avail = schema_mask(episodes, dim, gen)
        avail_rows = avail[:, None, :].expand(-1, targets, -1)
        hidden = hide_mask(avail_rows, gen)
        mean, logvar = model(x, avail_rows & ~hidden, context, avail)
        loss = gaussian_nll(mean, logvar, x, hidden, trim)
        _update(model, opt, sched, loss)
        fit.losses.append(loss.item())
        if val is not None and (step + 1) % val_every == 0:
            score = _validation_loss_context(model, current[0], val, trim)
            fit.val_losses.append((step + 1, score))
            if score < best:
                best, fit.best_step = score, step + 1
                state = copy.deepcopy(model.state_dict())
    if state is not None:
        model.load_state_dict(state)
    return fit


def train_masked_mlp(
    bank: Bank,
    steps: int = 3000,
    rows: int = 512,
    lr: float = 2e-3,
    seed: int = 0,
    refresh_every: int = 250,
    trim: float = 0.0,
    val: torch.Tensor | None = None,
    val_every: int = 50,
) -> Fit:
    """Train a MaskedMLP on rows pooled from bank; arguments as for ContextNet."""
    current = bank(0) if callable(bank) else bank
    dim = current.shape[2]
    flat = current.reshape(-1, dim)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = MaskedMLP(dim)
    opt, sched = _optimizer(model, lr, steps)
    fit = Fit(model, [])
    best, state = math.inf, None
    for step in range(steps):
        if callable(bank) and step and step % refresh_every == 0:
            flat = bank(step // refresh_every).reshape(-1, dim)
        x = flat[torch.randint(len(flat), (rows,), generator=gen)]
        avail = schema_mask(rows, dim, gen)
        hidden = hide_mask(avail, gen)
        mean, logvar = model(x, avail & ~hidden)
        loss = gaussian_nll(mean, logvar, x, hidden, trim)
        _update(model, opt, sched, loss)
        fit.losses.append(loss.item())
        if val is not None and (step + 1) % val_every == 0:
            score = _validation_loss_mlp(model, val, trim)
            fit.val_losses.append((step + 1, score))
            if score < best:
                best, fit.best_step = score, step + 1
                state = copy.deepcopy(model.state_dict())
    if state is not None:
        model.load_state_dict(state)
    return fit
