"""Anomaly scores from the masked predictors: one group of columns hidden at a time."""

from __future__ import annotations

import numpy as np
import torch

from .masking import column_mask, hide_groups


def group_scores(predict, x, avail, groups, normalise: bool = True):
    """Mean squared prediction error over the available columns.

    Every group of columns is hidden in turn and predicted from the others.
    predict(x, obs) returns mean and log variance shaped like x. With normalise the
    error of each column is divided by its predicted variance.
    """
    total = torch.zeros(x.shape[:-1])
    for cols in groups:
        cols = torch.as_tensor(np.asarray(cols), dtype=torch.long)
        obs = avail.clone()
        obs[cols] = False
        mean, logvar = predict(x, obs.expand(x.shape))
        err = (x - mean) ** 2
        if normalise:
            err = err * torch.exp(-logvar)
        total += err[..., cols].sum(dim=-1)
    return total / avail.sum()


def draw_contexts(pool: np.ndarray, n: int, k: int, rng: np.random.Generator):
    """n context sets of k distinct rows each, drawn from pool (m, d)."""
    if k > len(pool):
        raise ValueError("context larger than the pool")
    picks = np.stack([rng.choice(len(pool), size=k, replace=False) for _ in range(n)])
    return pool[picks].reshape(n, k, pool.shape[1])


def n_blocks(n_rows: int, block: int) -> int:
    return -(-n_rows // block)


@torch.no_grad()
def score_masked_mlp(model, x, columns, normalise: bool = True, max_groups: int = 12):
    """Scores of rows x (n, d) using only the given columns."""
    x = torch.as_tensor(x, dtype=torch.float32)
    avail = column_mask(columns, x.shape[-1])
    groups = hide_groups(columns, max_groups)
    return group_scores(model, x, avail, groups, normalise).numpy()


@torch.no_grad()
def score_context_net(
    model,
    x,
    context,
    columns,
    block: int = 50,
    max_groups: int = 12,
    chunk: int = 32,
):
    """Scores of rows x (n, d); block b of `block` consecutive rows uses context[b]."""
    x = torch.as_tensor(x, dtype=torch.float32)
    context = torch.as_tensor(context, dtype=torch.float32)
    n, dim = x.shape
    blocks = context.shape[0]
    if blocks < n_blocks(n, block):
        raise ValueError("not enough context sets for the rows")
    pad = blocks * block - n
    rows = torch.cat([x, x[-1:].expand(pad, dim)]) if pad else x
    rows = rows.reshape(blocks, block, dim)
    avail = column_mask(columns, dim)
    groups = hide_groups(columns, max_groups)
    out = []
    for start in range(0, blocks, chunk):
        ctx = context[start : start + chunk]
        part = rows[start : start + chunk]
        memory = None
        if ctx.shape[1] > 0:
            memory = model.encode(ctx, avail.expand(len(ctx), dim))

        def predict(a, o, memory=memory):
            return model.decode(a, o, memory)

        out.append(group_scores(predict, part, avail, groups))
    return torch.cat(out).reshape(-1)[:n].numpy()
