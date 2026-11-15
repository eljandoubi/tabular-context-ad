import numpy as np
import pytest
from scipy.stats import multivariate_normal

from contextad.synthetic import (
    CONTEXTUAL,
    NORMAL,
    ORDINARY,
    make_population,
    make_world,
)


@pytest.fixture(scope="module")
def world():
    return make_world(np.random.default_rng(0), 5)


def test_same_seed_gives_same_world_and_rows():
    a = make_world(np.random.default_rng(3), 4)
    b = make_world(np.random.default_rng(3), 4)
    for pa, pb in zip(a.populations, b.populations, strict=True):
        assert np.array_equal(pa.means, pb.means)
        assert np.array_equal(pa.factors, pb.factors)
    xa, ka = a.sample_rows(1, 500, np.random.default_rng(9))
    xb, kb = b.sample_rows(1, 500, np.random.default_rng(9))
    assert np.array_equal(xa, xb) and np.array_equal(ka, kb)


def test_different_seeds_give_different_worlds():
    a = make_world(np.random.default_rng(1), 3)
    b = make_world(np.random.default_rng(2), 3)
    centre_a = a.populations[0].means.mean(axis=0)
    centre_b = b.populations[0].means.mean(axis=0)
    assert not np.allclose(centre_a, centre_b)


def test_prior_ranges(world):
    assert world.dim == 10
    for pop in world.populations:
        assert 1 <= len(pop.weights) <= 3
        assert pop.factors.shape[2] in (2, 3)
        assert 0.02 <= pop.rate <= 0.05
        assert np.isclose(pop.weights.sum(), 1.0)


def test_log_density_matches_scipy():
    rng = np.random.default_rng(4)
    pop = make_population(rng, max_components=1)
    cov = pop.factors[0] @ pop.factors[0].T + np.diag(pop.noise[0])
    x = pop.sample(50, rng)
    expected = multivariate_normal(pop.means[0], cov).logpdf(x)
    assert np.allclose(pop.log_density(x), expected)


def test_log_density_all_matches_each_population(world):
    x, _ = world.sample_rows(0, 600, np.random.default_rng(15), rate=0.3)
    stacked = np.stack([p.log_density(x) for p in world.populations], axis=1)
    assert np.allclose(world.log_density_all(x), stacked)


def test_sample_moments_match_the_parameters():
    rng = np.random.default_rng(5)
    pop = make_population(rng, max_components=1)
    cov = pop.factors[0] @ pop.factors[0].T + np.diag(pop.noise[0])
    x = pop.sample(200000, rng)
    assert np.abs(x.mean(axis=0) - pop.means[0]).max() < 0.03
    assert np.abs(np.cov(x.T) - cov).max() < 0.05


def test_scale_multiplies_the_deviation():
    rng = np.random.default_rng(6)
    pop = make_population(rng, max_components=1)
    x = pop.sample(100000, rng, scale=2.0)
    cov = pop.factors[0] @ pop.factors[0].T + np.diag(pop.noise[0])
    assert np.abs(np.cov(x.T) - 4.0 * cov).max() < 0.3


def test_normal_rows_hit_the_calibrated_quantile(world):
    pop = world.populations[0]
    x, kind = world.sample_rows(0, 60000, np.random.default_rng(7), rate=0.0)
    assert (kind == NORMAL).all()
    below = (pop.log_density(x) < pop.unlikely).mean()
    assert 0.014 < below < 0.026


def test_anomaly_rate_and_mix(world):
    rate = 0.04
    _, kind = world.sample_rows(2, 60000, np.random.default_rng(8), rate=rate)
    assert abs((kind > 0).mean() - rate) < 0.004
    share = (kind == CONTEXTUAL).sum() / (kind > 0).sum()
    assert abs(share - 0.5) < 0.06


def test_rate_defaults_to_the_population_rate(world):
    pop = world.populations[3]
    _, kind = world.sample_rows(3, 60000, np.random.default_rng(10))
    assert abs((kind > 0).mean() - pop.rate) < 0.004


def test_contextual_rows_are_unlikely_here_and_typical_elsewhere(world):
    rng = np.random.default_rng(11)
    x, kind = world.sample_rows(1, 20000, rng, rate=0.2)
    rows = x[kind == CONTEXTUAL]
    assert len(rows) > 500
    logp = world.log_density_all(rows)
    assert (logp[:, 1] < world.unlikely[1]).all()
    typical = np.array([p.typical for p in world.populations])
    elsewhere = np.delete(logp - typical, 1, axis=1)
    assert (elsewhere.max(axis=1) >= 0).all()


def test_ordinary_rows_are_unlikely_everywhere(world):
    x, kind = world.sample_rows(4, 20000, np.random.default_rng(12), rate=0.2)
    rows = x[kind == ORDINARY]
    assert len(rows) > 500
    assert (world.log_density_all(rows) < world.unlikely).all()


def test_rate_extremes(world):
    rng = np.random.default_rng(13)
    _, none = world.sample_rows(0, 300, rng, rate=0.0)
    _, every = world.sample_rows(0, 300, rng, rate=1.0)
    assert (none == NORMAL).all() and (every > 0).all()


def test_contextual_rows_need_other_populations():
    world = make_world(np.random.default_rng(14), 1)
    with pytest.raises(ValueError):
        world.sample_rows(0, 50, np.random.default_rng(0), rate=1.0, contextual=1.0)
