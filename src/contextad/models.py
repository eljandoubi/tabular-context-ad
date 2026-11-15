"""Models that predict hidden columns of a row: a masked MLP and a context set model."""

from __future__ import annotations

import torch
from torch import nn

LOGVAR_MIN, LOGVAR_MAX = -8.0, 4.0


def _embed(width_in: int, dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width_in, dim), nn.GELU(), nn.Linear(dim, dim))


def gaussian_nll(mean, logvar, x, hidden, trim: float = 0.0):
    """Negative log likelihood (up to a constant) over the hidden entries.

    With trim > 0 the rows with the largest loss are left out of the average, so a
    few anomalies in the training rows do not teach the model to expect them.
    """
    nll = 0.5 * (logvar + (x - mean) ** 2 * torch.exp(-logvar))
    if trim <= 0:
        return (nll * hidden).sum() / hidden.sum()
    per_row = ((nll * hidden).sum(dim=-1) / hidden.sum(dim=-1)).reshape(-1)
    keep = per_row.numel() - int(trim * per_row.numel())
    return per_row.topk(keep, largest=False).values.mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class MaskedMLP(nn.Module):
    """Mean and log variance of every column given the observed columns of one row."""

    def __init__(self, dim: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        width = 2 * dim
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.GELU()]
            width = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2 * dim)

    def forward(self, x, obs):
        o = obs.to(x.dtype)
        mean, logvar = self.head(self.body(torch.cat([x * o, o], dim=-1))).chunk(2, -1)
        return mean, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)


class _SelfBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Linear(ff, width))

    def forward(self, h):
        q = self.norm1(h)
        h = h + self.attn(q, q, q, need_weights=False)[0]
        return h + self.ff(self.norm2(h))


class _CrossBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Linear(ff, width))

    def forward(self, h, memory=None):
        if memory is not None:
            h = h + self.attn(self.norm1(h), memory, memory, need_weights=False)[0]
        return h + self.ff(self.norm2(h))


class ContextNet(nn.Module):
    """Predicts hidden columns of a row from its observed columns and a context set.

    The context rows go through self-attention without positions, so the output does
    not depend on their order. Target rows attend to the context rows and never to
    each other. Without a context the same network is a plain masked predictor.
    """

    def __init__(
        self,
        dim: int,
        width: int = 64,
        heads: int = 4,
        ff: int = 128,
        context_layers: int = 2,
        decoder_layers: int = 2,
    ):
        super().__init__()
        self.row_embed = _embed(2 * dim, width)
        self.target_embed = _embed(2 * dim, width)
        self.context_blocks = nn.ModuleList(
            _SelfBlock(width, heads, ff) for _ in range(context_layers)
        )
        self.context_norm = nn.LayerNorm(width)
        self.decoder_blocks = nn.ModuleList(
            _CrossBlock(width, heads, ff) for _ in range(decoder_layers)
        )
        self.out_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2 * dim)

    def encode(self, context, avail):
        """Embed context rows (B, k, d) as seen through the columns in avail (B, d)."""
        a = avail.to(context.dtype).unsqueeze(1).expand_as(context)
        h = self.row_embed(torch.cat([context * a, a], dim=-1))
        for block in self.context_blocks:
            h = block(h)
        return self.context_norm(h)

    def decode(self, x, obs, memory=None):
        """Predict from target rows (B, T, d) and their observed mask, given memory."""
        o = obs.to(x.dtype)
        h = self.target_embed(torch.cat([x * o, o], dim=-1))
        for block in self.decoder_blocks:
            h = block(h, memory)
        mean, logvar = self.head(self.out_norm(h)).chunk(2, dim=-1)
        return mean, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)

    def forward(self, x, obs, context=None, avail=None):
        memory = None
        if context is not None and context.shape[1] > 0:
            memory = self.encode(context, avail)
        return self.decode(x, obs, memory)
