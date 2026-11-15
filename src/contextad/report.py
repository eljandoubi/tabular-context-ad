"""Tables of the study built from the per-seed JSON files in results/."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

SYNTHETIC_ROWS = [
    ("iforest", "Isolation Forest"),
    ("lof", "Local Outlier Factor"),
    ("knn", "kNN distance"),
    ("pca", "PCA reconstruction error"),
    ("masked_mlp", "Masked MLP, standardized error"),
    ("masked_mlp_mse", "Masked MLP, squared error"),
    ("context_net_k0", "Context model without context (k=0)"),
    ("context_net", "Context model (k=32)"),
    ("context_net_mse", "Context model, squared error (k=32)"),
    ("knn_context", "kNN to the context rows (k=32)"),
    ("mahalanobis_context", "Mahalanobis to the context rows (k=32)"),
    ("oracle", "Oracle density of the true population"),
]
TESTPOOL_ROWS = [
    ("iforest_testpool", "Isolation Forest, fit on test populations"),
    ("lof_testpool", "Local Outlier Factor, fit on test populations"),
    ("knn_testpool", "kNN distance, fit on test populations"),
    ("pca_testpool", "PCA reconstruction error, fit on test populations"),
]
ADBENCH_ROWS = [
    ("iforest", "Isolation Forest"),
    ("lof", "LOF"),
    ("knn", "kNN"),
    ("pca", "PCA"),
    ("masked_mlp", "Masked MLP, standardized error"),
    ("masked_mlp_mse", "Masked MLP, squared error"),
    ("context_net_k0", "Context model k=0"),
    ("context_net", "Context model k=32"),
    ("context_net_mse", "Context model k=32, squared error"),
    ("knn_context", "kNN to context"),
    ("mahalanobis_context", "Mahalanobis to context"),
]
SIZES = ["0", "4", "8", "16", "32", "64"]
RATES = ["0.0", "0.05", "0.1", "0.2"]
CONTEXT_METHODS = [
    ("context_net", "Context model"),
    ("knn_context", "kNN to context"),
    ("mahalanobis_context", "Mahalanobis to context"),
]
ADBENCH_PAIRS = [
    ("context_net", "context_net_k0", "standardized error: context k=32 minus k=0"),
    ("context_net", "masked_mlp", "standardized error: context k=32 minus masked MLP"),
    (
        "context_net",
        "knn_context",
        "standardized error: context k=32 minus kNN to context",
    ),
    (
        "context_net",
        "iforest",
        "standardized error: context k=32 minus Isolation Forest",
    ),
    (
        "context_net_mse",
        "masked_mlp_mse",
        "squared error: context k=32 minus masked MLP",
    ),
    (
        "context_net_mse",
        "knn_context",
        "squared error: context k=32 minus kNN to context",
    ),
    (
        "context_net_mse",
        "iforest",
        "squared error: context k=32 minus Isolation Forest",
    ),
    (
        "context_net_mse",
        "context_net",
        "context k=32: squared minus standardized error",
    ),
]
COPY_FIELDS = ["copy_in_train", "copy_in_context", "copy_in_any_context"]


def load(directory: Path, pattern: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(directory.glob(pattern))]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_std(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean()), std


def fmt(values, digits: int = 3) -> str:
    mean, std = mean_std(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def synthetic_value(run, method, metric, setting="main"):
    if method == "context_net_k0":
        return run["context_size"]["0"]["context_net"][metric]
    return run[setting][method][metric]


def synthetic_table(runs, rows):
    header = ["method", "AUROC", "AUROC contextual", "AUROC ordinary", "AUPRC"]
    metrics = ["auroc", "auroc_contextual", "auroc_ordinary", "auprc"]
    body = []
    for key, name in rows:
        cells = [fmt([synthetic_value(r, key, m) for r in runs]) for m in metrics]
        body.append([name, *cells])
    return header, body


def adbench_by_dataset(runs) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for run in runs:
        out.setdefault(run["config"]["dataset"], []).append(run)
    return dict(sorted(out.items()))


def adbench_value(run, method, metric="auroc", setting="main"):
    if setting == "main":
        return run["main"][method][metric]
    return float(np.mean([s["methods"][method][metric] for s in run["schema"]]))


def adbench_table(runs, metric="auroc"):
    by_dataset = adbench_by_dataset(runs)
    header = ["method", *by_dataset, "mean"]
    body = []
    for key, name in ADBENCH_ROWS:
        cells, means = [], []
        for dataset_runs in by_dataset.values():
            values = [adbench_value(r, key, metric) for r in dataset_runs]
            cells.append(fmt(values, 2))
            means.append(np.mean(values))
        body.append([name, *cells, f"{np.mean(means):.3f}"])
    return header, body


def dataset_table(adbench):
    """Size and anomaly rate of each ADBench table, from the first split of each."""
    header = ["table", "rows", "columns", "anomalies"]
    body = []
    for name, runs in adbench_by_dataset(adbench).items():
        run = runs[0]
        n_train, dim = run["shape"]["train"]
        n_test = run["shape"]["test"][0]
        rate = run["anomaly_rate"]
        share = (rate["train"] * n_train + rate["test"] * n_test) / (n_train + n_test)
        body.append([name, f"{n_train + n_test:,}", str(dim), f"{share * 100:.1f} %"])
    return header, body


def adbench_mean(runs, method, metric="auroc", setting="main") -> float:
    """Mean over tables of the mean over splits."""
    per_table = [
        np.mean([adbench_value(r, method, metric, setting) for r in rs])
        for rs in adbench_by_dataset(runs).values()
    ]
    return float(np.mean(per_table))


def adbench_paired(runs, pairs=ADBENCH_PAIRS, resamples: int = 10_000, seed: int = 0):
    """Paired AUROC differences over tables and splits with a bootstrap interval.

    The difference is the mean over tables of the mean over splits. The interval
    resamples the tables with replacement, then the splits inside each drawn table.
    """
    rng = np.random.default_rng(seed)
    by_dataset = adbench_by_dataset(runs)
    header = ["paired difference in AUROC", "mean", "95 % interval"]
    header += ["tables above 0", "splits above 0"]
    body = []
    for a, b, name in pairs:
        diffs = np.array(
            [
                [adbench_value(r, a) - adbench_value(r, b) for r in rs]
                for rs in by_dataset.values()
            ]
        )
        n_tables, n_splits = diffs.shape
        tables = rng.integers(n_tables, size=(resamples, n_tables, 1))
        splits = rng.integers(n_splits, size=(resamples, n_tables, n_splits))
        boot = diffs[tables, splits].mean(axis=(1, 2))
        low, high = np.percentile(boot, [2.5, 97.5])
        body.append(
            [
                name,
                f"{diffs.mean():+.3f}",
                f"[{low:+.3f}, {high:+.3f}]",
                f"{int((diffs.mean(axis=1) > 0).sum())} of {n_tables}",
                f"{int((diffs > 0).sum())} of {diffs.size}",
            ]
        )
    return header, body


def copies_cell(counts, table, split, field):
    """'found of tested' for normal and anomalous test rows, summed over the splits."""
    parts = []
    for label in ("0", "1"):
        rows = [
            c
            for c in counts
            if (c["table"], c["split"], c["label"]) == (table, split, label)
        ]
        tested = sum(int(c["test_rows"]) for c in rows)
        parts.append(f"{sum(int(c[field]) for c in rows):,} of {tested:,}")
    return " / ".join(parts)


def duplicates_table(counts, tables):
    """Exact copies of test rows, from duplicates.csv and duplicate_rows.csv."""
    header = ["table", "rows", "rows in a repeated vector"]
    header += ["plain split: copy in the training rows"]
    header += ["plain split: copy in a scored context", "grouped split: any copy"]
    body = []
    for table in tables:
        name = table["table"]
        grouped = sum(
            int(c[field])
            for c in counts
            if c["table"] == name and c["split"] == "grouped"
            for field in COPY_FIELDS
        )
        body.append(
            [
                name,
                f"{int(table['rows']):,}",
                table["rows_in_repeats"],
                copies_cell(counts, name, "plain", "copy_in_train"),
                copies_cell(counts, name, "plain", "copy_in_context"),
                str(grouped),
            ]
        )
    return header, body


def context_size_table(synthetic, adbench):
    header = [
        "k",
        *(name for _, name in CONTEXT_METHODS),
        "ADBench mean, context model",
    ]
    body = []
    for k in SIZES:
        cells = []
        for key, _ in CONTEXT_METHODS:
            if key != "context_net" and k == "0":
                cells.append("n/a")
                continue
            cells.append(fmt([r["context_size"][k][key]["auroc"] for r in synthetic]))
        per_table = [
            np.mean([r["context_size"][k]["context_net"]["auroc"] for r in rs])
            for rs in adbench_by_dataset(adbench).values()
        ]
        body.append([k, *cells, f"{np.mean(per_table):.3f}"])
    return header, body


def contamination_table(synthetic):
    header = ["context anomalies", *(name for _, name in CONTEXT_METHODS)]
    body = []
    for rate in RATES:
        cells = [
            fmt([r["contamination"][rate][key]["auroc"] for r in synthetic])
            for key, _ in CONTEXT_METHODS
        ]
        body.append([f"{float(rate) * 100:.0f} %", *cells])
    return header, body


def schema_table(synthetic, adbench):
    header = [
        "method",
        "synthetic, all columns",
        "synthetic, half",
        "ADBench, all",
        "ADBench, half",
    ]
    body = []
    for key, name in SYNTHETIC_ROWS:
        full = [synthetic_value(r, key, "auroc") for r in synthetic]
        half = [
            np.mean([s["methods"][key]["auroc"] for s in r["schema"]])
            for r in synthetic
        ]
        cells = [fmt(full), fmt(half)]
        for setting in ("main", "schema"):
            if key in adbench[0]["main"]:
                cells.append(f"{adbench_mean(adbench, key, 'auroc', setting):.3f}")
            else:
                cells.append("n/a")
        body.append([name, *cells])
    return header, body


def paired_differences(synthetic):
    pairs = [
        ("context_net", "context_net_k0", "context model k=32 minus k=0"),
        ("context_net", "masked_mlp", "context model k=32 minus masked MLP"),
        ("context_net", "iforest", "context model k=32 minus Isolation Forest"),
        (
            "mahalanobis_context",
            "context_net",
            "Mahalanobis to context minus context model k=32",
        ),
    ]
    header = ["difference in AUROC", "all", "contextual", "ordinary"]
    body = []
    for a, b, name in pairs:
        cells = []
        for metric in ("auroc", "auroc_contextual", "auroc_ordinary"):
            diffs = [
                synthetic_value(r, a, metric) - synthetic_value(r, b, metric)
                for r in synthetic
            ]
            cells.append(fmt(diffs))
        body.append([name, *cells])
    return header, body


def ablation_table(variants):
    """Rows for (label, runs): AUROC of the context-free and context models."""
    header = [
        "training rows and loss",
        "context model k=32",
        "context model k=32, contextual only",
        "context model k=0",
        "masked MLP",
    ]
    body = []
    for label, runs in variants:
        body.append(
            [
                label,
                fmt([synthetic_value(r, "context_net", "auroc") for r in runs]),
                fmt(
                    [
                        synthetic_value(r, "context_net", "auroc_contextual")
                        for r in runs
                    ]
                ),
                fmt([synthetic_value(r, "context_net_k0", "auroc") for r in runs]),
                fmt([synthetic_value(r, "masked_mlp", "auroc") for r in runs]),
            ]
        )
    return header, body


def compute_table(synthetic, adbench):
    """Training time and parameters, to put the cost next to the results."""
    header = ["run", "context model s", "masked MLP s", "parameters (context, MLP)"]
    body = []
    if synthetic:
        net = [r["train"]["net_seconds"] for r in synthetic]
        mlp = [r["train"]["mlp_seconds"] for r in synthetic]
        params = synthetic[0]["train"]["parameters"]
        body.append(
            [
                "synthetic, one seed",
                f"{np.mean(net):.0f}",
                f"{np.mean(mlp):.0f}",
                f"{params['context_net']:,}, {params['mlp']:,}",
            ]
        )
    if adbench:
        seconds = [r["train"]["seconds"] for r in adbench]
        body.append(
            ["ADBench, both models, one table", f"{np.mean(seconds):.0f}", "", ""]
        )
    return header, body


def markdown(header, body) -> str:
    rule = "|" + "|".join(["---"] * len(header)) + "|"
    lines = ["| " + " | ".join(header) + " |", rule]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)
