"""Draw the README figures from results/.

usage: python scripts/figures.py [results_dir]
Writes context_size.png, contamination.png and adbench.png next to the JSON files.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from contextad.report import (  # noqa: E402
    CONTEXT_METHODS,
    RATES,
    SIZES,
    adbench_by_dataset,
    load,
)

COLORS = {
    "context_net": "tab:red",
    "knn_context": "tab:blue",
    "mahalanobis_context": "tab:green",
}


def band(ax, x, values, label, color, style="-"):
    values = np.asarray(values)
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
    ax.plot(x, mean, style, color=color, label=label, marker="o", ms=4)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


def reference_line(ax, value, label, style):
    ax.axhline(value, color="gray", linestyle=style, linewidth=1.2, label=label)


def context_size_figure(synthetic, adbench, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    x = np.arange(len(SIZES))
    panels = [
        (axes[0], "synthetic, all anomalies", "auroc"),
        (axes[1], "synthetic, contextual anomalies", "auroc_contextual"),
    ]
    for ax, title, metric in panels:
        for key, name in CONTEXT_METHODS:
            start = 0 if key == "context_net" else 1
            curves = [
                [r["context_size"][k][key][metric] for k in SIZES[start:]]
                for r in synthetic
            ]
            band(ax, x[start:], curves, name, COLORS[key])
        for key, name, style in (
            ("iforest", "Isolation Forest", "--"),
            ("masked_mlp", "masked MLP", ":"),
        ):
            reference_line(
                ax, np.mean([r["main"][key][metric] for r in synthetic]), name, style
            )
        ax.set_title(title)
    by_dataset = adbench_by_dataset(adbench)
    ax = axes[2]
    for key, name in CONTEXT_METHODS:
        start = 0 if key == "context_net" else 1
        curves = []
        for seed in range(min(len(rs) for rs in by_dataset.values())):
            curves.append(
                [
                    np.mean(
                        [
                            rs[seed]["context_size"][k][key]["auroc"]
                            for rs in by_dataset.values()
                        ]
                    )
                    for k in SIZES[start:]
                ]
            )
        band(ax, x[start:], curves, name, COLORS[key])
    for key, name, style in (
        ("iforest", "Isolation Forest", "--"),
        ("masked_mlp", "masked MLP", ":"),
    ):
        value = np.mean(
            [
                np.mean([r["main"][key]["auroc"] for r in rs])
                for rs in by_dataset.values()
            ]
        )
        reference_line(ax, value, name, style)
    ax.set_title("ADBench, mean over the eight tables")
    for ax in axes:
        ax.set_xticks(x, SIZES)
        ax.set_xlabel("context rows k")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def contamination_figure(synthetic, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    x = [float(r) * 100 for r in RATES]
    for ax, title, metric in (
        (axes[0], "all anomalies", "auroc"),
        (axes[1], "contextual anomalies", "auroc_contextual"),
    ):
        for key, name in CONTEXT_METHODS:
            curves = [
                [r["contamination"][rate][key][metric] for rate in RATES]
                for r in synthetic
            ]
            band(ax, x, curves, name, COLORS[key])
        ax.set_title(title)
        ax.set_xlabel("anomalies in the context set (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUROC")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def adbench_figure(adbench, path):
    by_dataset = adbench_by_dataset(adbench)
    methods = [
        ("iforest", "Isolation Forest", "tab:gray"),
        ("masked_mlp", "masked MLP", "tab:orange"),
        ("context_net", "context model k=32", "tab:red"),
        ("mahalanobis_context", "Mahalanobis to context", "tab:green"),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    width = 0.2
    x = np.arange(len(by_dataset))
    for i, (key, name, color) in enumerate(methods):
        means = [
            np.mean([r["main"][key]["auroc"] for r in rs]) for rs in by_dataset.values()
        ]
        stds = [
            np.std([r["main"][key]["auroc"] for r in rs], ddof=1)
            for rs in by_dataset.values()
        ]
        ax.bar(
            x + (i - 1.5) * width,
            means,
            width,
            yerr=stds,
            label=name,
            color=color,
            capsize=2,
        )
    ax.set_xticks(x, list(by_dataset))
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc="lower center")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    synthetic = load(root / "synthetic", "seed*.json")
    adbench = load(root / "adbench", "*_seed*.json")
    if synthetic and adbench:
        context_size_figure(synthetic, adbench, root / "context_size.png")
        print(root / "context_size.png")
    if synthetic:
        contamination_figure(synthetic, root / "contamination.png")
        print(root / "contamination.png")
    if adbench:
        adbench_figure(adbench, root / "adbench.png")
        print(root / "adbench.png")


if __name__ == "__main__":
    main()
