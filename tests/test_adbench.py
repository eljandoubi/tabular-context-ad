import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from contextad.adbench import (
    TABLES,
    follow_first_row,
    has_copy,
    load_table,
    split_table,
    standardize,
    validation_split,
)
from contextad.scoring import draw_contexts


@pytest.fixture
def npz_root(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4)).astype(np.float32)
    y = np.zeros(200, dtype=np.int32)
    y[:20] = 1
    np.savez(tmp_path / "6_cardio.npz", X=x, y=y)
    return tmp_path


@pytest.fixture
def repeated_table():
    """120 distinct rows, each present 1 to 5 times, 12 of them anomalies."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(120, 4))
    counts = rng.integers(1, 6, size=120)
    x = np.repeat(base, counts, axis=0)
    y = np.repeat((np.arange(120) < 12).astype(int), counts)
    order = rng.permutation(len(x))
    return x[order], y[order]


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


def same_split(a, b):
    return all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))


def test_split_is_the_plain_stratified_split_without_repeated_rows(npz_root):
    x, y = load_table("cardio", npz_root)
    for seed in range(3):
        old = train_test_split(x, y, test_size=0.3, stratify=y, random_state=seed)
        assert same_split(split_table(x, y, seed), old)
        assert same_split(split_table(x, y, seed, whole_groups=False), old)


def test_split_is_deterministic_and_keeps_every_group_whole(repeated_table):
    x, y = repeated_table
    assert same_split(split_table(x, y, seed=1), split_table(x, y, seed=1))
    assert not np.array_equal(
        split_table(x, y, seed=1)[1], split_table(x, y, seed=2)[1]
    )
    for seed in range(5):
        x_train, x_test, y_train, y_test = split_table(x, y, seed)
        assert not has_copy(x_test, x_train).any()
        kept = np.concatenate([x_train, x_test])
        assert np.array_equal(np.unique(kept, axis=0), np.unique(x, axis=0))
        assert len(kept) == len(x) and y_train.sum() + y_test.sum() == y.sum()
        assert 0.2 < len(x_test) / len(x) < 0.4 and y_test.sum() > 0


def test_no_test_row_has_a_copy_in_the_training_rows_or_a_context(repeated_table):
    x, y = repeated_table
    plain = split_table(x, y, 0, whole_groups=False)
    assert has_copy(plain[1], plain[0]).any()
    rng = np.random.default_rng(0)
    for seed in range(3):
        x_train, x_test, _, _ = split_table(x, y, seed)
        z_train, z_test = standardize(x_train, x_test)
        contexts = draw_contexts(z_train, 20, 32, rng)
        assert not has_copy(z_test, z_train).any()
        assert not has_copy(z_test, contexts.reshape(-1, x.shape[1])).any()


def test_follow_first_row_moves_every_set_of_identical_rows_with_its_first():
    x = np.array([[1.0], [1.0], [2.0], [3.0], [3.0], [3.0]])
    side = np.array([False, True, True, False, True, True])
    assert follow_first_row(side, x).tolist() == [False, False, True] + [False] * 3
    assert follow_first_row(~side, x).tolist() == [True, True, False] + [True] * 3


def test_has_copy_compares_whole_rows():
    pool = np.array([[3.0, 4.0], [5.0, 6.0]])
    rows = np.array([[1.0, 2.0], [3.0, 4.0], [3.0, 6.0]])
    assert has_copy(rows, pool).tolist() == [False, True, False]
    assert not has_copy(rows, pool[:0]).any()


def test_validation_split_is_the_permutation_cut_without_repeated_rows():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 3))
    order = rng.permutation(100)
    bank, val = validation_split(x, order, 10)
    assert np.array_equal(bank, order[10:]) and np.array_equal(val, order[:10])


def test_validation_split_keeps_identical_rows_together(repeated_table):
    x, _ = repeated_table
    order = np.random.default_rng(1).permutation(len(x))
    bank, val = validation_split(x, order, 30)
    assert sorted(np.concatenate([bank, val])) == list(range(len(x)))
    assert len(val) > 0 and not has_copy(x[val], x[bank]).any()


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
