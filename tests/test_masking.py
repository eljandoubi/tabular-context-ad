import numpy as np
import torch

from contextad.masking import (
    column_mask,
    hide_groups,
    hide_mask,
    random_columns,
    schema_mask,
)


def test_schema_mask_keeps_at_least_two_columns_and_sometimes_all():
    gen = torch.Generator().manual_seed(0)
    mask = schema_mask(2000, 10, gen)
    assert mask.shape == (2000, 10) and mask.dtype == torch.bool
    assert mask.sum(dim=1).min() >= 2
    full = mask.all(dim=1).float().mean()
    assert 0.45 < full < 0.6
    assert mask[~mask.all(dim=1)].sum(dim=1).float().mean() < 9


def test_schema_mask_is_deterministic_per_generator_seed():
    a = schema_mask(50, 8, torch.Generator().manual_seed(1))
    b = schema_mask(50, 8, torch.Generator().manual_seed(1))
    c = schema_mask(50, 8, torch.Generator().manual_seed(2))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_hide_mask_stays_inside_the_available_columns():
    gen = torch.Generator().manual_seed(3)
    avail = schema_mask(500, 10, gen)[:, None, :].expand(500, 4, 10)
    hidden = hide_mask(avail, gen)
    assert not (hidden & ~avail).any()
    n_hidden = hidden.sum(dim=-1)
    n_avail = avail.sum(dim=-1)
    assert (n_hidden >= 1).all()
    assert (n_hidden <= (n_avail // 2).clamp(min=1)).all()
    assert (n_avail - n_hidden >= 1).all()


def test_hide_mask_covers_every_count_and_column():
    gen = torch.Generator().manual_seed(4)
    avail = torch.ones(4000, 10, dtype=torch.bool)
    hidden = hide_mask(avail, gen)
    counts = hidden.sum(dim=-1)
    assert set(counts.tolist()) == {1, 2, 3, 4, 5}
    assert hidden.any(dim=0).all()


def test_hide_groups_partition_the_columns():
    columns = np.array([1, 2, 4, 5, 6, 8, 9])
    groups = hide_groups(columns, max_groups=3, seed=0)
    assert len(groups) == 3
    joined = np.concatenate(groups)
    assert sorted(joined.tolist()) == columns.tolist()
    sizes = sorted(len(g) for g in groups)
    assert sizes[-1] - sizes[0] <= 1


def test_hide_groups_is_leave_one_out_for_few_columns():
    groups = hide_groups(np.arange(5), max_groups=12)
    assert [len(g) for g in groups] == [1] * 5


def test_hide_groups_is_deterministic():
    a = hide_groups(np.arange(36), seed=5)
    b = hide_groups(np.arange(36), seed=5)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert len(a) == 12


def test_random_columns_fraction_and_floor():
    rng = np.random.default_rng(0)
    cols = random_columns(10, 0.5, rng)
    assert len(cols) == 5 and len(set(cols.tolist())) == 5
    assert (np.diff(cols) > 0).all()
    assert len(random_columns(3, 0.1, rng)) == 2
    assert len(random_columns(2, 0.5, rng)) == 2


def test_column_mask():
    mask = column_mask([0, 3], 5)
    assert mask.tolist() == [True, False, False, True, False]
