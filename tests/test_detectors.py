import numpy as np
from sklearn.covariance import LedoitWolf

from contextad.detectors import (
    BASELINES,
    fit_baselines,
    knn_to_context,
    mahalanobis_to_context,
    oracle_scores,
)
from contextad.metrics import auroc
from contextad.synthetic import CONTEXTUAL, make_world


def test_baselines_rank_a_far_row_above_a_typical_one():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(600, 5))
    detectors = fit_baselines(train, seed=0)
    assert set(detectors) == set(BASELINES)
    rows = np.vstack([np.zeros(5), np.full(5, 7.0)])
    for name, score in detectors.items():
        s = score(rows)
        assert s.shape == (2,), name
        assert s[1] > s[0], name


def test_baselines_are_deterministic_and_respect_the_row_cap():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(500, 4))
    a = fit_baselines(train, seed=2, max_rows=100)
    b = fit_baselines(train, seed=2, max_rows=100)
    rows = rng.normal(size=(20, 4))
    for name in BASELINES:
        assert np.allclose(a[name](rows), b[name](rows))


def test_pca_keeps_fewer_components_than_columns():
    rng = np.random.default_rng(3)
    train = rng.normal(size=(800, 5))
    error = fit_baselines(train)["pca"](rng.normal(size=(50, 5)))
    assert (error > 0).all()


def test_pca_error_is_small_on_the_learned_subspace():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(800, 2))
    basis = 3.0 * np.linalg.qr(rng.normal(size=(6, 2)))[0].T
    train = z @ basis + 0.01 * rng.normal(size=(800, 6))
    pca = fit_baselines(train)["pca"]
    on_plane = rng.normal(size=(30, 2)) @ basis
    off_plane = on_plane + 2.0 * rng.normal(size=(30, 6))
    assert pca(off_plane).mean() > 100 * pca(on_plane).mean()


def test_knn_to_context_on_a_line():
    context = np.array([[[0.0], [1.0], [2.0]]])
    rows = np.array([[[0.5], [10.0]]])
    out = knn_to_context(context, rows, neighbours=2)
    assert out.shape == (1, 2)
    assert np.allclose(out[0], [0.5, 8.5])


def test_knn_to_context_caps_neighbours_at_the_context_size():
    context = np.zeros((2, 3, 4))
    rows = np.ones((2, 5, 4))
    assert np.allclose(knn_to_context(context, rows, neighbours=10), 2.0)


def test_mahalanobis_matches_the_shrunk_estimator_and_flags_far_rows():
    rng = np.random.default_rng(5)
    context = rng.normal(size=(3, 64, 4)) * [1.0, 2.0, 0.5, 1.0]
    rows = rng.normal(size=(3, 6, 4)) * [1.0, 2.0, 0.5, 1.0]
    out = mahalanobis_to_context(context, rows)
    assert out.shape == (3, 6)
    expected = LedoitWolf().fit(context[1]).mahalanobis(rows[1])
    assert np.allclose(out[1], expected)
    far = rows.copy()
    far[:, 0] += 12.0
    assert (mahalanobis_to_context(context, far)[:, 0] > out.max()).all()


def test_oracle_scores_rank_contextual_anomalies_above_normal_rows():
    world = make_world(np.random.default_rng(6), 4)
    x, kind = world.sample_rows(0, 6000, np.random.default_rng(7), rate=0.1)
    keep = (kind == 0) | (kind == CONTEXTUAL)
    score = oracle_scores(world, 0, x)
    assert np.allclose(score, -world.populations[0].log_density(x))
    assert auroc(kind[keep] > 0, score[keep]) > 0.95
