"""Evaluation helpers shared by the experiment scripts."""

from __future__ import annotations

import numpy as np

from .detectors import knn_to_context, mahalanobis_to_context
from .metrics import auprc, auroc, grouped_mean
from .scoring import draw_contexts, n_blocks, score_context_net, to_blocks
from .synthetic import CONTEXTUAL, NORMAL, ORDINARY

REFERENCES = {
    "knn_context": knn_to_context,
    "mahalanobis_context": mahalanobis_to_context,
}


def group_contexts(pools, group, k: int, rng: np.random.Generator, block: int = 50):
    """For every group, one context set per block of its rows, drawn from its pool."""
    out = {}
    for g in np.unique(group):
        blocks = n_blocks(int((group == g).sum()), block)
        out[int(g)] = draw_contexts(pools[int(g)], blocks, k, rng)
    return out


def context_net_scores(net, x, group, contexts, columns, block: int = 50):
    """Scores of the context model; rows of a group are cut into consecutive blocks."""
    out = np.zeros(len(x))
    for g, ctx in contexts.items():
        m = group == g
        out[m] = score_context_net(net, x[m], ctx, columns, block)
    return out


def reference_scores(name, x, group, contexts, columns, block: int = 50):
    """Scores of a context reference ("knn_context" or "mahalanobis_context")."""
    out = np.zeros(len(x))
    for g, ctx in contexts.items():
        m = group == g
        rows = to_blocks(x[m][:, columns], len(ctx), block)
        out[m] = REFERENCES[name](ctx[..., columns], rows).reshape(-1)[: m.sum()]
    return out


def summarize(kind, scores, group) -> dict:
    """Metrics averaged over groups, with AUROC per anomaly type and pooled AUROC."""
    kind, scores, group = map(np.asarray, (kind, scores, group))
    y = kind > 0
    contextual = (kind == NORMAL) | (kind == CONTEXTUAL)
    ordinary = (kind == NORMAL) | (kind == ORDINARY)
    return {
        "auroc": grouped_mean(auroc, y, scores, group),
        "auprc": grouped_mean(auprc, y, scores, group),
        "auroc_contextual": grouped_mean(
            auroc, y[contextual], scores[contextual], group[contextual]
        ),
        "auroc_ordinary": grouped_mean(
            auroc, y[ordinary], scores[ordinary], group[ordinary]
        ),
        "auroc_pooled": auroc(y, scores),
    }


def summarize_table(y, scores) -> dict:
    """AUROC and AUPRC of one table."""
    return {"auroc": auroc(y, scores), "auprc": auprc(y, scores)}
