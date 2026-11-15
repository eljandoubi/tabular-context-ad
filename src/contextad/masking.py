"""Column masks: which columns a table has and which ones a model must predict."""

from __future__ import annotations

import numpy as np
import torch


def schema_mask(
    n: int,
    dim: int,
    gen: torch.Generator,
    p_full: float = 0.5,
    keep_range: tuple[float, float] = (0.3, 0.9),
    min_keep: int = 2,
) -> torch.Tensor:
    """Columns present in n episodes: all with probability p_full, else a subset."""
    low, high = keep_range
    keep_prob = low + (high - low) * torch.rand(n, 1, generator=gen)
    u = torch.rand(n, dim, generator=gen)
    mask = u < keep_prob
    mask |= u.argsort(dim=1).argsort(dim=1) < min_keep
    mask[torch.rand(n, generator=gen) < p_full] = True
    return mask


def hide_mask(
    avail: torch.Tensor, gen: torch.Generator, max_fraction: float = 0.5
) -> torch.Tensor:
    """Hide between one and max_fraction of the available columns of every row."""
    count_avail = avail.sum(dim=-1)
    most = (count_avail * max_fraction).floor().long().clamp(min=1)
    count = 1 + (torch.rand(avail.shape[:-1], generator=gen) * most).long()
    u = torch.rand(avail.shape, generator=gen).masked_fill(~avail, 2.0)
    rank = u.argsort(dim=-1).argsort(dim=-1)
    return (rank < count.unsqueeze(-1)) & avail


def hide_groups(columns, max_groups: int = 12, seed: int = 0) -> list[np.ndarray]:
    """Split columns into at most max_groups sets that are hidden one set at a time."""
    columns = np.asarray(columns)
    order = np.random.default_rng(seed).permutation(len(columns))
    n_groups = min(len(columns), max_groups)
    return [np.sort(columns[order[g::n_groups]]) for g in range(n_groups)]


def random_columns(
    dim: int, fraction: float, rng: np.random.Generator, min_keep: int = 2
) -> np.ndarray:
    """Sorted indices of a random subset holding the given fraction of the columns."""
    keep = max(min_keep, int(round(fraction * dim)))
    return np.sort(rng.choice(dim, size=min(keep, dim), replace=False))


def column_mask(columns, dim: int) -> torch.Tensor:
    """Boolean vector of length dim that is True on the given columns."""
    mask = torch.zeros(dim, dtype=torch.bool)
    mask[torch.as_tensor(np.asarray(columns), dtype=torch.long)] = True
    return mask
