"""Count test rows that have an exact copy in the training rows or in a scored context.

usage: python scripts/count_duplicates.py [--root data/adbench] [--out results]
Replays the split and the context draws of scripts/run_adbench.py for the plain
stratified split and for the split that keeps identical rows together. A copy is a row
with the same value in every column. Writes duplicates.csv (one row per table, seed,
split and label) and duplicate_rows.csv (repeated feature vectors per table, and how
many of them carry both labels) to --out.
The clipped_ columns repeat the counts on the standardized columns, where clipping at
8 standard deviations can make two different rows equal.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from contextad.adbench import TABLES, has_copy, load_table, split_table, standardize
from contextad.evaluate import group_contexts
from contextad.report import COPY_FIELDS
from contextad.training import CONTEXT_SIZES

SPLITS = {"plain": False, "grouped": True}
COUNT_FIELDS = ["table", "seed", "split", "label", "test_rows"]
COUNT_FIELDS += COPY_FIELDS + ["clipped_" + name for name in COPY_FIELDS]
TABLE_FIELDS = ["table", "rows", "distinct_rows", "rows_in_repeats", "largest_group"]
TABLE_FIELDS += ["mixed_label_vectors"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tables", nargs="+", choices=sorted(TABLES), default=sorted(TABLES)
    )
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--block", type=int, default=50)
    p.add_argument("--root", default="data/adbench")
    p.add_argument("--out", default="results")
    return p.parse_args()


def context_hits(x_test, contexts, block):
    """Whether each test row has an exact copy in the context of its own block."""
    hits = np.zeros(len(x_test), dtype=bool)
    for b, context in enumerate(contexts):
        rows = slice(b * block, (b + 1) * block)
        hits[rows] = has_copy(x_test[rows], context)
    return hits


def copies(x_train, x_test, seed, k, block):
    """Per test row: copy in the training rows, in the main context, in any context."""
    # Same random stream as run_adbench.py: the permutation of the training rows, the
    # draws of the main table (k=0, then k), then one draw per size of the sweep.
    rng = np.random.default_rng(seed)
    rng.permutation(len(x_train))
    pools, group = {0: x_train}, np.zeros(len(x_test), dtype=int)

    def draw(size):
        contexts = group_contexts(pools, group, size, rng, block)[0]
        return context_hits(x_test, contexts, block)

    draw(0)
    main = draw(k)
    anywhere = main | np.any([draw(size) for size in CONTEXT_SIZES], axis=0)
    return [has_copy(x_test, x_train), main, anywhere]


def count_split(table, x, y, seed, split, k, block):
    x_train, x_test, _, y_test = split_table(x, y, seed, whole_groups=SPLITS[split])
    z_train, z_test = standardize(x_train, x_test)
    found = copies(x_train, x_test, seed, k, block)
    found += copies(z_train, z_test, seed, k, block)
    rows = []
    for label in (0, 1):
        m = y_test == label
        rows.append([table, seed, split, label, int(m.sum())])
        rows[-1] += [int(f[m].sum()) for f in found]
    return rows


def repeated_rows(table, x, y):
    _, group, count = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    anomalies = np.bincount(group.reshape(-1), weights=y)
    mixed = int(((anomalies > 0) & (anomalies < count)).sum())
    repeated = int(count[count > 1].sum())
    return [table, len(x), len(count), repeated, int(count.max()), mixed]


def write_csv(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    args = parse_args()
    counts, tables = [], []
    for table in args.tables:
        x, y = load_table(table, args.root)
        tables.append(repeated_rows(table, x, y))
        for seed in range(args.seeds):
            for split in SPLITS:
                counts += count_split(table, x, y, seed, split, args.k, args.block)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "duplicate_rows.csv", TABLE_FIELDS, tables)
    write_csv(out / "duplicates.csv", COUNT_FIELDS, counts)
    print(out / "duplicates.csv")


if __name__ == "__main__":
    main()
