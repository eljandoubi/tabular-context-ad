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
    for name in ("iforest", "masked_mlp", "context_net", "oracle", "knn_context"):
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
    assert set(result["main"]) >= {"masked_mlp", "context_net", "context_net_k0"}
    assert result["train"]["net_best_step"] == 50
