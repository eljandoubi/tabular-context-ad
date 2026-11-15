"""ADBench experiment for one table and one split: main scores, context size, schema.

usage: python scripts/run_adbench.py --dataset cardio --seed 0
Writes results/adbench/<dataset>_seed<seed>.json and prints its path.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from contextad.adbench import TABLES, load_table, split_table, standardize
from contextad.detectors import fit_baselines
from contextad.evaluate import (
    context_net_scores,
    group_contexts,
    reference_scores,
    summarize_table,
)
from contextad.masking import random_columns
from contextad.models import count_parameters
from contextad.scoring import score_masked_mlp
from contextad.training import (
    CONTEXT_SIZES,
    train_context_net,
    train_masked_mlp,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(TABLES), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--trim", type=float, default=0.10)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--block", type=int, default=50)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--schema-repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--root", default="data/adbench")
    p.add_argument("--out", default="results/adbench")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()
    x, y = load_table(args.dataset, args.root)
    x_train, x_test, y_train, y_test = split_table(x, y, args.seed)
    z_train, z_test = standardize(x_train, x_test)
    dim = z_train.shape[1]
    all_columns = np.arange(dim)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(z_train))
    n_val = max(16, int(args.val_fraction * len(z_train)))
    bank = torch.tensor(z_train[order[n_val:]], dtype=torch.float32)[None]
    val = torch.tensor(z_train[order[:n_val]], dtype=torch.float32)
    t0 = time.time()
    fit_args = {"steps": args.steps, "seed": args.seed, "trim": args.trim, "val": val}
    mlp_fit = train_masked_mlp(bank, **fit_args)
    net_fit = train_context_net(bank, **fit_args)
    mlp, net = mlp_fit.model.eval(), net_fit.model.eval()
    train = {
        "seconds": time.time() - t0,
        "load_end": os.getloadavg()[0],
        "mlp_best_step": mlp_fit.best_step,
        "net_best_step": net_fit.best_step,
        "parameters": {
            "mlp": count_parameters(mlp),
            "context_net": count_parameters(net),
        },
    }

    group = np.zeros(len(z_test), dtype=int)
    pools = {0: z_train}
    block = args.block

    def contexts_of(k):
        return group_contexts(pools, group, k, rng, block)

    def context_scores(columns, contexts, with_references=True, squared=False):
        scores = {
            "context_net": context_net_scores(
                net, z_test, group, contexts, columns, block
            )
        }
        if squared:
            scores["context_net_mse"] = context_net_scores(
                net, z_test, group, contexts, columns, block, normalise=False
            )
        if with_references:
            for name in ("knn_context", "mahalanobis_context"):
                scores[name] = reference_scores(
                    name, z_test, group, contexts, columns, block
                )
        return scores

    def summaries(scores):
        return {n: summarize_table(y_test, s) for n, s in scores.items()}

    def all_methods(columns):
        scores = {}
        for name, fn in fit_baselines(z_train[:, columns], args.seed).items():
            scores[name] = fn(z_test[:, columns])
        scores["masked_mlp"] = score_masked_mlp(mlp, z_test, columns)
        scores["masked_mlp_mse"] = score_masked_mlp(
            mlp, z_test, columns, normalise=False
        )
        scores["context_net_k0"] = context_scores(columns, contexts_of(0), False)[
            "context_net"
        ]
        scores.update(context_scores(columns, contexts_of(args.k), squared=True))
        return scores

    result = {
        "config": vars(args),
        "shape": {"train": list(z_train.shape), "test": list(z_test.shape)},
        "anomaly_rate": {"train": float(y_train.mean()), "test": float(y_test.mean())},
        "train": train,
    }
    result["main"] = summaries(all_methods(all_columns))
    result["context_size"] = {
        str(k): summaries(context_scores(all_columns, contexts_of(k), k > 1))
        for k in CONTEXT_SIZES
    }
    cols_rng = np.random.default_rng([args.seed, 7])
    result["schema"] = []
    for _ in range(args.schema_repeats):
        columns = random_columns(dim, 0.5, cols_rng)
        result["schema"].append(
            {"columns": columns.tolist(), "methods": summaries(all_methods(columns))}
        )
    result["seconds"] = time.time() - started

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.dataset}_seed{args.seed}.json"
    path.write_text(json.dumps(result, indent=1))
    print(path)


if __name__ == "__main__":
    main()
