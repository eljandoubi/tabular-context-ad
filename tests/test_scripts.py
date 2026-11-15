import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run(script, **options):
    args = [sys.executable, str(SCRIPTS / script)]
    for name, value in options.items():
        args += [f"--{name.replace('_', '-')}", str(value)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=170)
    assert result.returncode == 0, result.stderr[-2000:]
    return Path(result.stdout.strip().splitlines()[-1])


def test_synthetic_script_runs_end_to_end_on_a_tiny_setup(tmp_path):
    path = run(
        "run_synthetic.py",
        seed=0,
        steps=20,
        train_pops=12,
        train_rows=96,
        test_pops=3,
        test_rows=150,
        pool_rows=70,
        schema_repeats=1,
        out=tmp_path,
    )
    result = json.loads(path.read_text())
    assert {"main", "context_size", "contamination", "schema", "train"} <= set(result)
    assert set(result["context_size"]) == {"0", "4", "8", "16", "32", "64"}
    assert len(result["schema"][0]["columns"]) == 5
    for name in (
        "iforest",
        "masked_mlp",
        "context_net",
        "context_net_mse",
        "oracle",
        "knn_context",
    ):
        assert 0.0 <= result["main"][name]["auroc"] <= 1.0
    assert result["main"]["oracle"]["auroc"] > 0.9


def test_adbench_script_runs_end_to_end_on_a_tiny_table(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(240, 5))
    y = np.zeros(240, dtype=int)
    y[:20] = 1
    x[:20] += 4.0
    np.savez(tmp_path / "6_cardio.npz", X=x, y=y)
    path = run(
        "run_adbench.py",
        dataset="cardio",
        seed=0,
        steps=60,
        schema_repeats=1,
        root=tmp_path,
        out=tmp_path / "out",
    )
    result = json.loads(path.read_text())
    assert result["shape"] == {"train": [168, 5], "test": [72, 5]}
    assert result["main"]["iforest"]["auroc"] > 0.9
    assert set(result["main"]) >= {
        "masked_mlp",
        "context_net",
        "context_net_k0",
        "context_net_mse",
    }
    assert result["train"]["net_best_step"] == 50


def test_adbench_script_runs_on_a_table_with_repeated_rows(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(240, 5))
    y = np.zeros(240, dtype=int)
    y[:20] = 1
    x[:20] += 4.0
    x[120:200] = x[40:120]
    y[120:200] = y[40:120]
    np.savez(tmp_path / "6_cardio.npz", X=x, y=y)
    path = run(
        "run_adbench.py",
        dataset="cardio",
        seed=0,
        steps=60,
        schema_repeats=1,
        root=tmp_path,
        out=tmp_path / "out",
    )
    result = json.loads(path.read_text())
    assert result["shape"]["train"][0] + result["shape"]["test"][0] == 240
    assert result["main"]["iforest"]["auroc"] > 0.9


def test_duplicate_count_finds_copies_in_the_plain_split_only(tmp_path):
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 5, size=150)
    x = np.repeat(rng.normal(size=(150, 4)), counts, axis=0)
    y = np.repeat((np.arange(150) < 15).astype(int), counts)
    first = np.cumsum(counts) - counts
    y[first[np.argmax(counts > 1)] + 1] ^= 1
    np.savez(tmp_path / "6_cardio.npz", X=x, y=y)
    path = run(
        "count_duplicates.py",
        tables="cardio",
        seeds=2,
        root=tmp_path,
        out=tmp_path / "out",
    )
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2 * 2 * 2
    fields = ("copy_in_train", "copy_in_context", "copy_in_any_context")
    for split in ("plain", "grouped"):
        part = [r for r in rows if r["split"] == split]
        assert sum(int(r["test_rows"]) for r in part) > 100
        found = [sum(int(r[f]) for r in part) for f in fields]
        assert (found[0] > 0) == (split == "plain")
        assert (found[2] > 0) == (split == "plain")
        assert found[1] <= found[2]
        clipped = [sum(int(r["clipped_" + f]) for r in part) for f in fields]
        assert clipped == found
    table = list(csv.DictReader((path.parent / "duplicate_rows.csv").open()))
    assert table[0]["rows"] == str(len(x)) and table[0]["distinct_rows"] == "150"
    assert table[0]["mixed_label_vectors"] == "1"


def test_validation_count_applies_the_minimum_to_a_table_with_large_groups(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(240, 4))
    y = np.zeros(240, dtype=int)
    y[120:144] = 1
    for start in (0, 40, 80):
        x[start : start + 40] = x[start]
    np.savez(tmp_path / "6_cardio.npz", X=x, y=y)
    path = run(
        "count_validation.py",
        tables="cardio",
        seeds=5,
        root=tmp_path,
        out=tmp_path / "out",
    )
    reader = csv.DictReader(path.open())
    rows = [{k: int(v) for k, v in r.items() if k != "table"} for r in reader]
    assert [r["seed"] for r in rows] == [0, 1, 2, 3, 4]
    assert all(r["kept"] >= 16 and r["copy_in_bank"] == 0 for r in rows)
    assert any(r["grouped"] < 16 and r["kept"] > r["grouped"] for r in rows)
    assert all(r["kept"] == r["grouped"] for r in rows if r["grouped"] >= 16)
