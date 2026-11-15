"""ADBench tables: loading, stratified splits and standardisation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

TABLES = {
    "cardio": "6_cardio",
    "thyroid": "38_thyroid",
    "satellite": "30_satellite",
    "shuttle": "32_shuttle",
    "pima": "29_Pima",
    "wbc": "42_WBC",
    "vowels": "40_vowels",
    "breastw": "4_breastw",
}
CLIP = 8.0
MIN_VALIDATION = 16


def load_table(name: str, root: str | Path = "data/adbench"):
    """Features (n, d) as float64 and labels (n,) with 1 for an anomaly."""
    with np.load(Path(root) / f"{TABLES[name]}.npz") as data:
        return data["X"].astype(np.float64), data["y"].astype(np.int64)


def follow_first_row(side: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Side of every row when each set of identical rows takes the side of its first.

    side is a boolean per row of x (n, d); rows count as identical if all columns match.
    """
    _, first, group = np.unique(x, axis=0, return_index=True, return_inverse=True)
    return side[first][group.reshape(-1)]


def split_table(x, y, seed: int, test_size: float = 0.3, whole_groups: bool = True):
    """Stratified random split; returns x_train, x_test, y_train, y_test.

    With whole_groups, rows with identical feature vectors stay on one side: a set of
    identical rows that the draw cut goes to the side of its first row in the table.
    The rows of a table without repeated rows split exactly as the plain draw does.
    """
    rows = np.arange(len(x))
    train, test = train_test_split(
        rows, test_size=test_size, stratify=y, random_state=seed
    )
    if whole_groups:
        in_test = np.zeros(len(x), dtype=bool)
        in_test[test] = True
        moved = follow_first_row(in_test, x)
        if (moved != in_test).any():
            # Shuffle both sides so that blocks of consecutive test rows stay mixed.
            order = np.random.default_rng([seed, 1]).permutation(len(x))
            train, test = order[~moved[order]], order[moved[order]]
    return x[train], x[test], y[train], y[test]


def has_copy(rows: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """For every row of rows (n, d), whether an identical row exists in pool (m, d)."""
    both = np.concatenate([pool, rows])
    group = np.unique(both, axis=0, return_inverse=True)[1].reshape(-1)
    return np.isin(group[len(pool) :], group[: len(pool)])


def validation_split(
    x: np.ndarray, order: np.ndarray, n_val: int, minimum: int = MIN_VALIDATION
):
    """Row numbers (bank, validation): first n_val of order, identical rows together.

    A set of identical rows that the cut separated goes to the side of its first row
    in x. If that leaves fewer than minimum validation rows, whole sets of identical
    rows are added in the order of the permutation until there are minimum. Without
    repeated rows and with n_val >= minimum this is order[n_val:] and order[:n_val].
    """
    in_val = np.zeros(len(x), dtype=bool)
    in_val[order[:n_val]] = True
    in_val = follow_first_row(in_val, x)
    if in_val.sum() < minimum:
        group = np.unique(x, axis=0, return_inverse=True)[1].reshape(-1)
        for row in order[n_val:]:
            in_val |= group == group[row]
            if in_val.sum() >= minimum:
                break
    return order[~in_val[order]], order[in_val[order]]


def standardize(train: np.ndarray, *others: np.ndarray, clip: float = CLIP):
    """Scale every column by the training mean and std, then clip to +-clip.

    Columns that are constant in the training rows keep a scale of one.
    """
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return tuple(np.clip((a - mean) / std, -clip, clip) for a in (train, *others))
