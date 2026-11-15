"""Synthetic populations: Gaussian mixtures with their own mean and covariance."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

NORMAL, CONTEXTUAL, ORDINARY = 0, 1, 2
UNLIKELY_QUANTILE = 0.02
TYPICAL_QUANTILE = 0.05
OUTLIER_SCALE = 3.0
CALIBRATION_ROWS = 20000


@dataclass
class Population:
    """Mixture of Gaussians whose covariances are low rank plus diagonal."""

    weights: np.ndarray
    means: np.ndarray
    factors: np.ndarray
    noise: np.ndarray
    rate: float = 0.03
    unlikely: float = -np.inf
    typical: float = -np.inf
    chols: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        covs = np.einsum("cdr,cer->cde", self.factors, self.factors)
        covs = covs + np.einsum("cd,de->cde", self.noise, np.eye(self.means.shape[1]))
        self.chols = [np.linalg.cholesky(c) for c in covs]

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    def sample(self, n: int, rng: np.random.Generator, scale: float = 1.0):
        """Draw n rows; scale multiplies the deviation from the component mean."""
        comp = rng.choice(len(self.weights), size=n, p=self.weights)
        rank = self.factors.shape[2]
        low = np.einsum(
            "ndr,nr->nd", self.factors[comp], rng.standard_normal((n, rank))
        )
        diag = np.sqrt(self.noise[comp]) * rng.standard_normal((n, self.dim))
        return self.means[comp] + scale * (low + diag)

    def log_density(self, x: np.ndarray) -> np.ndarray:
        """Log density of the rows of x under the mixture."""
        x = np.atleast_2d(x)
        parts = []
        for w, mean, chol in zip(self.weights, self.means, self.chols, strict=True):
            z = solve_triangular(chol, (x - mean).T, lower=True)
            logdet = 2.0 * np.log(np.diag(chol)).sum()
            quad = (z**2).sum(axis=0)
            parts.append(
                np.log(w) - 0.5 * (self.dim * np.log(2 * np.pi) + logdet + quad)
            )
        return logsumexp(np.stack(parts, axis=1), axis=1)

    def calibrate(self, rng: np.random.Generator) -> None:
        """Set the unlikely and typical log density levels from own samples."""
        logp = self.log_density(self.sample(CALIBRATION_ROWS, rng))
        self.unlikely = float(np.quantile(logp, UNLIKELY_QUANTILE))
        self.typical = float(np.quantile(logp, TYPICAL_QUANTILE))


def make_population(
    rng: np.random.Generator,
    dim: int = 10,
    centre_std: float = 1.0,
    offset_std: float = 0.7,
    max_components: int = 3,
    ranks: tuple[int, ...] = (2, 3),
    rate_range: tuple[float, float] = (0.02, 0.05),
) -> Population:
    """Draw one population: up to max_components low rank plus diagonal Gaussians."""
    n_comp = int(rng.integers(1, max_components + 1))
    weights = rng.dirichlet(np.full(n_comp, 4.0))
    means = rng.normal(0.0, centre_std, dim) + rng.normal(
        0.0, offset_std, (n_comp, dim)
    )
    rank = int(rng.choice(ranks))
    factors = rng.normal(0.0, 1.0 / np.sqrt(rank), (n_comp, dim, rank))
    noise = rng.uniform(0.05, 0.3, (n_comp, dim))
    pop = Population(
        weights, means, factors, noise, rate=float(rng.uniform(*rate_range))
    )
    pop.calibrate(rng)
    return pop


class World:
    """Populations drawn together: the anomalies of one come from the others."""

    def __init__(self, populations: list[Population]):
        self.populations = populations
        self.unlikely = np.array([p.unlikely for p in populations])

    def __len__(self) -> int:
        return len(self.populations)

    @property
    def dim(self) -> int:
        return self.populations[0].dim

    def log_density_all(self, x: np.ndarray) -> np.ndarray:
        """Log density of every row under every population, shape (n, P)."""
        return np.stack([p.log_density(x) for p in self.populations], axis=1)

    def contextual_rows(
        self, index: int, n: int, rng: np.random.Generator, tries: int = 100
    ):
        """Rows typical for another population and unlikely under population index."""
        own = self.populations[index]
        others = np.array([j for j in range(len(self)) if j != index])
        if len(others) == 0:
            raise ValueError("contextual anomalies need at least two populations")
        found: list[np.ndarray] = []
        total = 0
        for _ in range(tries):
            if total >= n:
                break
            source = rng.choice(others, size=max(2 * (n - total), 16))
            for j in np.unique(source):
                other = self.populations[j]
                cand = other.sample(int((source == j).sum()), rng)
                ok = (other.log_density(cand) >= other.typical) & (
                    own.log_density(cand) < own.unlikely
                )
                found.append(cand[ok])
                total += int(ok.sum())
        if total < n:
            raise RuntimeError("populations overlap too much for contextual anomalies")
        return np.concatenate(found)[:n]

    def ordinary_rows(
        self, index: int, n: int, rng: np.random.Generator, tries: int = 100
    ):
        """Tail rows of population index that are unlikely under every population."""
        own = self.populations[index]
        found: list[np.ndarray] = []
        total = 0
        for _ in range(tries):
            if total >= n:
                break
            cand = own.sample(max(2 * (n - total), 16), rng, scale=OUTLIER_SCALE)
            ok = (self.log_density_all(cand) < self.unlikely).all(axis=1)
            found.append(cand[ok])
            total += int(ok.sum())
        if total < n:
            raise RuntimeError("could not draw ordinary outliers")
        return np.concatenate(found)[:n]

    def sample_rows(
        self,
        index: int,
        n: int,
        rng: np.random.Generator,
        rate: float | None = None,
        contextual: float = 0.5,
    ):
        """Rows of population index and their kind (NORMAL, CONTEXTUAL or ORDINARY)."""
        pop = self.populations[index]
        rate = pop.rate if rate is None else rate
        anomalous = rng.random(n) < rate
        is_contextual = anomalous & (rng.random(n) < contextual)
        is_ordinary = anomalous & ~is_contextual
        x = pop.sample(n, rng)
        kind = np.zeros(n, dtype=np.int8)
        if is_contextual.any():
            x[is_contextual] = self.contextual_rows(
                index, int(is_contextual.sum()), rng
            )
            kind[is_contextual] = CONTEXTUAL
        if is_ordinary.any():
            x[is_ordinary] = self.ordinary_rows(index, int(is_ordinary.sum()), rng)
            kind[is_ordinary] = ORDINARY
        return x, kind


def make_world(rng: np.random.Generator, n_populations: int, **prior) -> World:
    """Draw n_populations populations from the prior."""
    return World([make_population(rng, **prior) for _ in range(n_populations)])
