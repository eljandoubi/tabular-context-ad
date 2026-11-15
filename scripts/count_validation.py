"""Validation rows of every ADBench run, with and without the minimum.

usage: python scripts/count_validation.py [--root data/adbench] [--out results]
Replays the split and the permutation of scripts/run_adbench.py and writes
validation_rows.csv, one row per table and seed: the rows asked for (the larger of the
validation share and the minimum), the rows left once identical rows are kept together
(grouped), the rows kept once the minimum applies, and how many of those have an exact
copy among the bank rows.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from contextad.adbench import (
    MIN_VALIDATION,
    TABLES,
    has_copy,
    load_table,
    split_table,
    validation_split,
)

FIELDS = ["table", "seed", "train_rows", "asked", "grouped", "kept", "copy_in_bank"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tables", nargs="+", choices=sorted(TABLES), default=sorted(TABLES)
    )
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--root", default="data/adbench")
    p.add_argument("--out", default="results")
    return p.parse_args()


def validation_sizes(table, x, y, seed, val_fraction):
    x_train = split_table(x, y, seed)[0]
    order = np.random.default_rng(seed).permutation(len(x_train))
    asked = max(MIN_VALIDATION, int(val_fraction * len(x_train)))
    grouped = validation_split(x_train, order, asked, minimum=0)[1]
    bank, kept = validation_split(x_train, order, asked)
    copies = int(has_copy(x_train[kept], x_train[bank]).sum())
    return [table, seed, len(x_train), asked, len(grouped), len(kept), copies]


def main():
    args = parse_args()
    rows = []
    for table in args.tables:
        x, y = load_table(table, args.root)
        for seed in range(args.seeds):
            rows.append(validation_sizes(table, x, y, seed, args.val_fraction))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "validation_rows.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
