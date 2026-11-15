"""Aggregate results/ into markdown tables and CSV files.

usage: python scripts/report.py [results_dir]
Prints the tables and writes results/tables.md and results/*.csv.
"""

import csv
import sys
from pathlib import Path

from contextad.report import (
    SYNTHETIC_ROWS,
    TESTPOOL_ROWS,
    ablation_table,
    adbench_table,
    compute_table,
    contamination_table,
    context_size_table,
    dataset_table,
    load,
    markdown,
    paired_differences,
    schema_table,
    synthetic_table,
)


def write_csv(path: Path, header, body):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(body)


def build_sections(root: Path):
    synthetic = load(root / "synthetic", "seed*.json")
    adbench = load(root / "adbench", "*_seed*.json")
    plain = load(root / "ablation_plain", "seed*.json")
    clean = load(root / "ablation_clean", "seed*.json")
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
                "Synthetic, detectors fit on the rows of the test populations",
                "synthetic_testpool",
                synthetic_table(synthetic, TESTPOOL_ROWS),
            ),
            (
                "Synthetic, paired differences over seeds",
                "synthetic_paired",
                paired_differences(synthetic),
            ),
            (
                "Synthetic, anomalies in the context set",
                "contamination",
                contamination_table(synthetic),
            ),
        ]
    if synthetic and plain and clean:
        variants = [
            ("2 to 5 % anomalies, trimmed loss (default)", synthetic),
            ("2 to 5 % anomalies, plain loss", plain),
            ("no anomalies, plain loss", clean),
        ]
        sections.append(
            ("Synthetic, training rows and loss", "ablation", ablation_table(variants))
        )
    if adbench:
        sections.append(
            ("ADBench tables used", "adbench_tables", dataset_table(adbench))
        )
        sections.append(("ADBench, AUROC", "adbench_auroc", adbench_table(adbench)))
        sections.append(
            ("ADBench, AUPRC", "adbench_auprc", adbench_table(adbench, "auprc"))
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
    if synthetic or adbench:
        sections.append(("Training cost", "compute", compute_table(synthetic, adbench)))
    return sections


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    text = []
    for title, name, (header, body) in build_sections(root):
        write_csv(root / f"{name}.csv", header, body)
        text.append(f"### {title}\n\n{markdown(header, body)}\n")
    (root / "tables.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
