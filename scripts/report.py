"""Aggregate results/ into markdown tables and CSV files.

usage: python scripts/report.py [results_dir]
Prints the tables and writes results/tables.md and results/*.csv.
"""

import csv
import json
import sys
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
    ("masked_mlp", "Masked MLP"),
    ("context_net_k0", "Context model k=0"),
    ("context_net", "Context model k=32"),
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


def load(directory: Path, pattern: str):
    return [json.loads(p.read_text()) for p in sorted(directory.glob(pattern))]


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def fmt(values, digits=3):
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


def adbench_by_dataset(runs):
    out = {}
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
        means = []
        cells = []
        for dataset_runs in by_dataset.values():
            values = [adbench_value(r, key, metric) for r in dataset_runs]
            cells.append(fmt(values, 2))
            means.append(np.mean(values))
        body.append([name, *cells, f"{np.mean(means):.3f}"])
    return header, body


def context_size_table(synthetic, adbench):
    header = ["k", *(n for _, n in CONTEXT_METHODS), "ADBench mean, context model"]
    body = []
    for k in SIZES:
        cells = []
        for key, _ in CONTEXT_METHODS:
            if key != "context_net" and k == "0":
                cells.append("n/a")
                continue
            cells.append(fmt([r["context_size"][k][key]["auroc"] for r in synthetic]))
        by_dataset = adbench_by_dataset(adbench)
        per_dataset = [
            np.mean([r["context_size"][k]["context_net"]["auroc"] for r in rs])
            for rs in by_dataset.values()
        ]
        body.append([k, *cells, f"{np.mean(per_dataset):.3f}"])
    return header, body


def contamination_table(synthetic):
    header = ["context anomalies", *(n for _, n in CONTEXT_METHODS)]
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
    rows = [r for r in SYNTHETIC_ROWS if r[0] not in ("oracle",)]
    by_dataset = adbench_by_dataset(adbench)
    body = []
    for key, name in rows:
        full = [synthetic_value(r, key, "auroc") for r in synthetic]
        half = [
            np.mean([s["methods"][key]["auroc"] for s in r["schema"]])
            for r in synthetic
        ]
        cells = [fmt(full), fmt(half)]
        for setting in ("main", "schema"):
            if key not in adbench[0]["main"]:
                cells.append("n/a")
                continue
            per_dataset = [
                np.mean([adbench_value(r, key, "auroc", setting) for r in rs])
                for rs in by_dataset.values()
            ]
            cells.append(f"{np.mean(per_dataset):.3f}")
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
            "Mahalanobis to context minus context model",
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


def markdown(header, body):
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def write_csv(path: Path, header, body):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(body)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    synthetic = load(root / "synthetic", "seed*.json")
    adbench = load(root / "adbench", "*_seed*.json")
    sections = []
    if synthetic:
        n = len(synthetic)
        sections += [
            (
                f"Synthetic populations, held-out populations, {n} seeds",
                "synthetic_main",
                synthetic_table(synthetic, SYNTHETIC_ROWS),
            ),
            (
                "Synthetic, baselines fit on the rows of the test populations",
                "synthetic_testpool",
                synthetic_table(synthetic, TESTPOOL_ROWS),
            ),
            (
                "Synthetic, paired differences over seeds",
                "synthetic_paired",
                paired_differences(synthetic),
            ),
            (
                "Synthetic, contaminated context",
                "contamination",
                contamination_table(synthetic),
            ),
        ]
    if adbench:
        sections.append(
            ("ADBench tables, AUROC", "adbench_auroc", adbench_table(adbench))
        )
        sections.append(
            ("ADBench tables, AUPRC", "adbench_auprc", adbench_table(adbench, "auprc"))
        )
    if synthetic and adbench:
        sections.append(
            (
                "AUROC against context size",
                "context_size",
                context_size_table(synthetic, adbench),
            )
        )
        sections.append(
            (
                "Schema change, half of the columns",
                "schema",
                schema_table(synthetic, adbench),
            )
        )
    text = []
    for title, name, (header, body) in sections:
        write_csv(root / f"{name}.csv", header, body)
        text.append(f"### {title}\n\n{markdown(header, body)}\n")
    (root / "tables.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
