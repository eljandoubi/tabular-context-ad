"""Ranking metrics and their average over groups of rows."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve; tied scores count one half. NaN with one class."""
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision: precision at each distinct score, weighted by recall steps."""
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    y_sorted = y[order]
    s_sorted = score[order]
    last = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]
    tp = np.cumsum(y_sorted)[last]
    precision = tp / (last + 1)
    recall = tp / n_pos
    steps = np.diff(np.r_[0.0, recall])
    return float((steps * precision).sum())


def grouped_mean(metric, y, score, group) -> float:
    """Mean of metric(y, score) over the groups, skipping groups where it is NaN."""
    y, score, group = map(np.asarray, (y, score, group))
    values = [metric(y[group == g], score[group == g]) for g in np.unique(group)]
    return float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
