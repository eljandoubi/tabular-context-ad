"""Synthetic experiment for one seed: main table, context size, contamination, schema.

usage: python scripts/run_synthetic.py --seed 0
Writes results/synthetic/seed<seed>.json and prints its path.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from contextad.detectors import fit_baselines, oracle_scores
from contextad.evaluate import (
    context_net_scores,
    group_contexts,
    reference_scores,
    summarize,
)
from contextad.masking import random_columns
from contextad.models import count_parameters
from contextad.scoring import score_masked_mlp
from contextad.synthetic import make_world
from contextad.training import (
    CONTEXT_SIZES,
    train_context_net,
    train_masked_mlp,
)

CONTAMINATION = (0.0, 0.05, 0.10, 0.20)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--train-pops", type=int, default=256)
    p.add_argument("--train-rows", type=int, default=128)
    p.add_argument("--refresh-every", type=int, default=250)
    p.add_argument("--trim", type=float, default=0.10)
    p.add_argument("--test-pops", type=int, default=8)
    p.add_argument("--test-rows", type=int, default=2000)
    p.add_argument("--pool-rows", type=int, default=512)
    p.add_argument("--block", type=int, default=50)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--schema-repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default="results/synthetic")
    return p.parse_args()


def thin(losses, every=100):
    return [float(np.mean(losses[i : i + every])) for i in range(0, len(losses), every)]


def bank_maker(args):
    """Function of the refresh round: standardised rows of new training populations."""
    stats = {}

    def make(round_):
        rng = np.random.default_rng([args.seed, 11, round_])
        world = make_world(rng, args.train_pops)
        rows = np.stack(
            [
                world.sample_rows(i, args.train_rows, rng)[0]
                for i in range(args.train_pops)
            ]
        )
        if not stats:
            flat = rows.reshape(-1, rows.shape[-1])
            stats["mean"], stats["std"] = flat.mean(axis=0), flat.std(axis=0)
        return (rows - stats["mean"]) / stats["std"]

    return make, stats


def build_data(args, make_bank, stats, rngs):
    """First training bank, held-out test rows and context pools, all standardised."""
    test_world_rng, test_rng, pool_rng = rngs[:3]
    bank = make_bank(0)
    mean, std = stats["mean"], stats["std"]
    test_world = make_world(test_world_rng, args.test_pops)
    rows, kinds, groups, pools = [], [], [], {}
    for g in range(args.test_pops):
        x, kind = test_world.sample_rows(g, args.test_rows, test_rng)
        rows.append(x)
        kinds.append(kind)
        groups.append(np.full(len(x), g))
        pools[g] = test_world.sample_rows(g, args.pool_rows, pool_rng)[0]
    return {
        "world": test_world,
        "bank": bank,
        "raw": np.concatenate(rows),
        "x": (np.concatenate(rows) - mean) / std,
        "kind": np.concatenate(kinds),
        "group": np.concatenate(groups),
        "pools": {g: (p - mean) / std for g, p in pools.items()},
        "mean": mean,
        "std": std,
    }


def fixed_methods(columns, data, mlp, seed):
    """Scores of the methods that never see a context set."""
    x, bank = data["x"], data["bank"]
    train_rows = bank.reshape(-1, bank.shape[-1])[:, columns]
    test_pool = np.concatenate([data["pools"][g] for g in sorted(data["pools"])])
    scores = {}
    for name, fn in fit_baselines(train_rows, seed).items():
        scores[name] = fn(x[:, columns])
    for name, fn in fit_baselines(test_pool[:, columns], seed).items():
        scores[f"{name}_testpool"] = fn(x[:, columns])
    scores["masked_mlp"] = score_masked_mlp(mlp, x, columns)
    scores["masked_mlp_mse"] = score_masked_mlp(mlp, x, columns, normalise=False)
    oracle = np.zeros(len(x))
    for g in sorted(data["pools"]):
        m = data["group"] == g
        oracle[m] = oracle_scores(data["world"], g, data["raw"][m], columns)
    scores["oracle"] = oracle
    return scores


def context_methods(columns, data, net, contexts, block, with_references=True):
    """Scores of the context model and of the two context references."""
    x, group = data["x"], data["group"]
    scores = {
        "context_net": context_net_scores(net, x, group, contexts, columns, block)
    }
    if with_references:
        for name in ("knn_context", "mahalanobis_context"):
            scores[name] = reference_scores(name, x, group, contexts, columns, block)
    return scores


def summarize_all(scores, data):
    return {n: summarize(data["kind"], s, data["group"]) for n, s in scores.items()}


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    seeds = np.random.SeedSequence(args.seed).spawn(6)
    rngs = [np.random.default_rng(s) for s in seeds]
    ctx_rng, cols_rng, extra_rng = rngs[3:]
    started = time.time()
    make_bank, stats = bank_maker(args)
    data = build_data(args, make_bank, stats, rngs)
    dim = data["x"].shape[1]
    all_columns = np.arange(dim)
    print(f"data ready in {time.time() - started:.0f} s", flush=True)

    def make_tensor(round_):
        return torch.tensor(make_bank(round_), dtype=torch.float32)

    train = {"load_start": os.getloadavg()[0]}
    fit_args = {
        "steps": args.steps,
        "seed": args.seed,
        "refresh_every": args.refresh_every,
        "trim": args.trim,
    }
    t0 = time.time()
    mlp_fit = train_masked_mlp(make_tensor, **fit_args)
    train["mlp_seconds"] = time.time() - t0
    t0 = time.time()
    net_fit = train_context_net(make_tensor, **fit_args)
    train["net_seconds"] = time.time() - t0
    mlp, net = mlp_fit.model, net_fit.model
    train.update(
        load_end=os.getloadavg()[0],
        mlp_loss=thin(mlp_fit.losses),
        net_loss=thin(net_fit.losses),
        parameters={"mlp": count_parameters(mlp), "context_net": count_parameters(net)},
    )
    mlp.eval()
    net.eval()
    print(f"trained in {train['mlp_seconds'] + train['net_seconds']:.0f} s", flush=True)

    result = {"config": vars(args), "train": train}
    block = args.block

    sizes = {
        k: group_contexts(data["pools"], data["group"], k, ctx_rng, block)
        for k in CONTEXT_SIZES
    }
    sweep = {}
    for k, contexts in sizes.items():
        sweep[str(k)] = summarize_all(
            context_methods(all_columns, data, net, contexts, block, k > 1), data
        )
    result["context_size"] = sweep
    main_scores = fixed_methods(all_columns, data, mlp, args.seed)
    main_scores.update(context_methods(all_columns, data, net, sizes[args.k], block))
    result["main"] = summarize_all(main_scores, data)
    print("main table done", flush=True)

    contamination = {}
    for rate in CONTAMINATION:
        pools = {
            g: (
                data["world"].sample_rows(g, args.pool_rows, extra_rng, rate=rate)[0]
                - data["mean"]
            )
            / data["std"]
            for g in data["pools"]
        }
        contexts = group_contexts(pools, data["group"], args.k, ctx_rng, block)
        contamination[str(rate)] = summarize_all(
            context_methods(all_columns, data, net, contexts, block), data
        )
    result["contamination"] = contamination
    print("contamination done", flush=True)

    schema = []
    for _ in range(args.schema_repeats):
        columns = random_columns(dim, 0.5, cols_rng)
        contexts = group_contexts(data["pools"], data["group"], args.k, ctx_rng, block)
        empty = group_contexts(data["pools"], data["group"], 0, ctx_rng, block)
        scores = fixed_methods(columns, data, mlp, args.seed)
        scores.update(context_methods(columns, data, net, contexts, block))
        free = context_methods(columns, data, net, empty, block, False)
        scores["context_net_k0"] = free["context_net"]
        schema.append(
            {"columns": columns.tolist(), "methods": summarize_all(scores, data)}
        )
    result["schema"] = schema
    result["seconds"] = time.time() - started

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"seed{args.seed}.json"
    path.write_text(json.dumps(result, indent=1))
    print(path)


if __name__ == "__main__":
    main()
