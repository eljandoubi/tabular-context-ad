import numpy as np
import pytest

from contextad.adbench import TABLES, load_table, split_table, standardize


@pytest.fixture
def npz_root(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4)).astype(np.float32)
    y = np.zeros(200, dtype=np.int32)
    y[:20] = 1
    np.savez(tmp_path / "6_cardio.npz", X=x, y=y)
    return tmp_path


def test_table_names_cover_the_eight_tables():
    assert sorted(TABLES) == [
        "breastw",
        "cardio",
        "pima",
        "satellite",
        "shuttle",
        "thyroid",
        "vowels",
        "wbc",
    ]


def test_load_table_returns_float_features_and_integer_labels(npz_root):
    x, y = load_table("cardio", npz_root)
    assert x.shape == (200, 4) and x.dtype == np.float64
    assert y.dtype == np.int64 and y.sum() == 20


def test_split_is_stratified_disjoint_and_seeded(npz_root):
    x, y = load_table("cardio", npz_root)
    x[:, 0] = np.arange(200)
    a = split_table(x, y, seed=1)
    b = split_table(x, y, seed=1)
    c = split_table(x, y, seed=2)
    x_train, x_test, y_train, y_test = a
    assert len(x_test) == 60 and len(x_train) == 140
    assert y_train.sum() == 14 and y_test.sum() == 6
    assert not set(x_train[:, 0]) & set(x_test[:, 0])
    assert np.array_equal(a[1], b[1]) and not np.array_equal(a[1], c[1])


def test_standardize_uses_the_training_statistics():
    rng = np.random.default_rng(1)
    train = rng.normal(5.0, 3.0, size=(500, 3))
    test = rng.normal(5.0, 3.0, size=(50, 3))
    z_train, z_test = standardize(train, test)
    assert np.allclose(z_train.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(z_train.std(axis=0), 1, atol=0.02)
    expected = np.clip((test - train.mean(0)) / train.std(0), -8, 8)
    assert np.allclose(z_test, expected)


def test_standardize_clips_and_handles_constant_columns():
    train = np.column_stack([np.arange(100.0), np.full(100, 3.0)])
    far = np.array([[1e6, 3.0], [-1e6, 4.0]])
    z_train, z_far = standardize(train, far, clip=5.0)
    assert np.abs(z_far).max() == 5.0
    assert (z_train[:, 1] == 0).all()
    assert z_far[1, 1] == 1.0
