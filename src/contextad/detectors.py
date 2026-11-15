"""Detectors without a neural model: scikit-learn baselines and context references."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

BASELINES = ("iforest", "lof", "knn", "pca")


def fit_baselines(train: np.ndarray, seed: int = 0, max_rows: int = 10000) -> dict:
    """Fit the four scikit-learn detectors; each entry maps rows to scores."""
    if len(train) > max_rows:
        train = train[np.random.default_rng(seed).choice(len(train), max_rows, False)]
    forest = IsolationForest(n_estimators=200, random_state=seed).fit(train)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(train)
    neighbours = NearestNeighbors(n_neighbors=5).fit(train)
    pca = PCA(n_components=0.9, svd_solver="full").fit(train)
    if pca.n_components_ >= train.shape[1]:
        pca = PCA(n_components=train.shape[1] - 1).fit(train)

    def reconstruction_error(x):
        return ((x - pca.inverse_transform(pca.transform(x))) ** 2).sum(axis=1)

    return {
        "iforest": lambda x: -forest.score_samples(x),
        "lof": lambda x: -lof.score_samples(x),
        "knn": lambda x: neighbours.kneighbors(x)[0].mean(axis=1),
        "pca": reconstruction_error,
    }


def knn_to_context(context: np.ndarray, rows: np.ndarray, neighbours: int = 5):
    """Mean distance to the nearest context rows. context (B, k, d), rows (B, T, d)."""
    dist = np.linalg.norm(rows[:, :, None, :] - context[:, None, :, :], axis=-1)
    nearest = np.sort(dist, axis=-1)[..., : min(neighbours, dist.shape[-1])]
    return nearest.mean(axis=-1)


def mahalanobis_to_context(context: np.ndarray, rows: np.ndarray):
    """Squared Mahalanobis distance to each context set under a shrunk covariance."""
    out = np.empty(rows.shape[:2])
    for b in range(len(context)):
        out[b] = LedoitWolf().fit(context[b]).mahalanobis(rows[b])
    return out


def oracle_scores(world, index: int, rows: np.ndarray):
    """Negative log density of rows under the true population (knows the population)."""
    return -world.populations[index].log_density(rows)
