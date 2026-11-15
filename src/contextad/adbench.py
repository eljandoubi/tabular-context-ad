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


def load_table(name: str, root: str | Path = "data/adbench"):
    """Features (n, d) as float64 and labels (n,) with 1 for an anomaly."""
    with np.load(Path(root) / f"{TABLES[name]}.npz") as data:
        return data["X"].astype(np.float64), data["y"].astype(np.int64)


def split_table(x, y, seed: int, test_size: float = 0.3):
    """Stratified random split; returns x_train, x_test, y_train, y_test."""
    return train_test_split(x, y, test_size=test_size, stratify=y, random_state=seed)


def standardize(train: np.ndarray, *others: np.ndarray, clip: float = CLIP):
    """Scale every column by the training mean and std, then clip to +-clip.

    Columns that are constant in the training rows keep a scale of one.
    """
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return tuple(np.clip((a - mean) / std, -clip, clip) for a in (train, *others))
