import pytest

from contextad.report import (
    RATES,
    SIZES,
    SYNTHETIC_ROWS,
    TESTPOOL_ROWS,
    ablation_table,
    adbench_by_dataset,
    adbench_mean,
    adbench_paired,
    adbench_table,
    compute_table,
    contamination_table,
    context_size_table,
    dataset_table,
    duplicates_table,
    fmt,
    markdown,
    mean_std,
    paired_differences,
    read_csv,
    schema_table,
    synthetic_table,
)

CONTEXT = ("context_net", "knn_context", "mahalanobis_context")


def block(auroc, contextual=None, ordinary=None, auprc=0.4):
    return {
        "auroc": auroc,
        "auprc": auprc,
        "auroc_contextual": auroc if contextual is None else contextual,
        "auroc_ordinary": auroc if ordinary is None else ordinary,
        "auroc_pooled": auroc,
    }


def synthetic_run(shift=0.0):
    names = [k for k, _ in SYNTHETIC_ROWS + TESTPOOL_ROWS if k != "context_net_k0"]
    main = {name: block(0.70 + shift) for name in names}
    main["context_net"] = block(0.90 + shift, 0.95 + shift, 0.85 + shift)
    names_k0 = [*main, "context_net_k0"]
    sizes = {
        k: {m: block(0.60 + 0.05 * i + shift) for m in CONTEXT}
        for i, k in enumerate(SIZES)
    }
    return {
        "config": {"seed": 0},
        "train": {
            "mlp_seconds": 10.0,
            "net_seconds": 30.0,
            "parameters": {"mlp": 1000, "context_net": 2000},
        },
        "main": main,
        "context_size": sizes,
        "contamination": {
            rate: {m: block(0.9 - float(rate) + shift) for m in CONTEXT}
            for rate in RATES
        },
        "schema": [
            {"columns": [0, 1], "methods": {n: block(0.6 + shift) for n in names_k0}},
            {"columns": [2, 3], "methods": {n: block(0.8 + shift) for n in names_k0}},
        ],
    }


def adbench_run(dataset, shift=0.0):
    methods = [
        "iforest",
        "lof",
        "knn",
        "pca",
        "masked_mlp",
        "masked_mlp_mse",
        "context_net_k0",
        "context_net",
        "context_net_mse",
        "knn_context",
        "mahalanobis_context",
    ]
    return {
        "config": {"dataset": dataset},
        "train": {"seconds": 50.0},
        "shape": {"train": [700, 9], "test": [300, 9]},
        "anomaly_rate": {"train": 0.1, "test": 0.2},
        "main": {m: block(0.8 + shift) for m in methods},
        "context_size": {
            k: {"context_net": block(0.7 + 0.02 * i + shift)}
            for i, k in enumerate(SIZES)
        },
        "schema": [
            {"columns": [0], "methods": {m: block(0.6 + shift) for m in methods}}
        ],
    }


def test_mean_std_uses_the_sample_standard_deviation():
    assert mean_std([0.5, 0.7]) == pytest.approx((0.6, 0.1414213562))
    assert mean_std([0.4]) == (0.4, 0.0)
    assert fmt([0.5, 0.7]) == "0.600 ± 0.141"
    assert fmt([0.5, 0.7], 2) == "0.60 ± 0.14"


def test_synthetic_table_reads_the_no_context_row_from_the_size_sweep():
    runs = [synthetic_run(0.0), synthetic_run(0.02)]
    header, body = synthetic_table(runs, SYNTHETIC_ROWS)
    assert header[:2] == ["method", "AUROC"]
    assert len(body) == len(SYNTHETIC_ROWS)
    rows = {row[0]: row for row in body}
    assert rows["Context model (k=32)"][1] == "0.910 ± 0.014"
    assert rows["Context model (k=32)"][2] == "0.960 ± 0.014"
    assert rows["Context model without context (k=0)"][1] == "0.610 ± 0.014"
    assert rows["Isolation Forest"][4] == "0.400 ± 0.000"


def test_paired_differences_are_taken_per_seed():
    runs = [synthetic_run(0.0), synthetic_run(0.1)]
    header, body = paired_differences(runs)
    row = body[0]
    assert row[0] == "context model k=32 minus k=0"
    assert row[1] == "0.300 ± 0.000"
    assert body[1][2] == "0.250 ± 0.000"


def test_context_size_table_has_no_reference_row_without_context():
    runs = [synthetic_run(), synthetic_run(0.02)]
    tables = [adbench_run("a"), adbench_run("b")]
    header, body = context_size_table(runs, tables)
    assert [row[0] for row in body] == SIZES
    assert body[0][2] == "n/a" and body[0][3] == "n/a"
    assert body[0][1] == "0.610 ± 0.014"
    assert body[0][4] == "0.700"
    assert body[-1][4] == "0.800"


def test_contamination_table_labels_and_values():
    header, body = contamination_table([synthetic_run()])
    assert [row[0] for row in body] == ["0 %", "5 %", "10 %", "20 %"]
    assert body[0][1] == "0.900 ± 0.000" and body[3][1] == "0.700 ± 0.000"


def test_adbench_tables_group_by_dataset_and_average_over_tables():
    runs = [
        adbench_run("a"),
        adbench_run("a", 0.1),
        adbench_run("b", 0.2),
        adbench_run("b", 0.2),
    ]
    assert list(adbench_by_dataset(runs)) == ["a", "b"]
    header, body = adbench_table(runs)
    assert header[:3] == ["method", "a", "b"] and header[-1] == "mean"
    assert body[0][1] == "0.85 ± 0.07" and body[0][2] == "1.00 ± 0.00"
    assert body[0][3] == "0.925"
    assert adbench_mean(runs, "iforest") == pytest.approx(0.925)
    assert adbench_mean(runs, "iforest", setting="schema") == pytest.approx(0.725)
    assert adbench_table(runs, "auprc")[1][0][3] == "0.400"


def test_schema_table_marks_methods_missing_on_adbench():
    header, body = schema_table([synthetic_run()], [adbench_run("a")])
    rows = {row[0]: row for row in body}
    assert rows["Oracle density of the true population"][3:] == ["n/a", "n/a"]
    assert rows["Isolation Forest"][2] == "0.700 ± 0.000"
    assert rows["Isolation Forest"][3:] == ["0.800", "0.600"]


def test_ablation_and_compute_tables():
    runs = [synthetic_run(), synthetic_run(0.02)]
    header, body = ablation_table([("default", runs), ("plain", runs[:1])])
    assert body[0][0] == "default" and body[1][1] == "0.900 ± 0.000"
    assert len(header) == len(body[0])
    header, body = compute_table(runs, [adbench_run("a")])
    assert body[0][1:] == ["30", "10", "2,000, 1,000"]
    assert body[1][1] == "50"


def test_dataset_table_reports_size_and_anomaly_share():
    header, body = dataset_table([adbench_run("b"), adbench_run("a")])
    assert header == ["table", "rows", "columns", "anomalies"]
    assert body == [["a", "1,000", "9", "13.0 %"], ["b", "1,000", "9", "13.0 %"]]


def test_markdown_layout():
    text = markdown(["a", "b"], [["1", "2"], ["3", "4"]])
    assert text.splitlines() == ["| a | b |", "|---|---|", "| 1 | 2 |", "| 3 | 4 |"]


def test_paired_differences_over_tables_and_splits():
    runs = []
    for dataset, gap in (("a", 0.1), ("b", 0.1), ("c", -0.1)):
        for split in range(4):
            run = adbench_run(dataset)
            run["main"]["context_net"] = block(0.8 + gap + 0.01 * split)
            runs.append(run)
    pairs = [
        ("context_net", "masked_mlp", "gap"),
        ("iforest", "lof", "none"),
    ]
    header, body = adbench_paired(runs, pairs, resamples=2000, seed=1)
    assert header[0] == "paired difference in AUROC" and len(header) == len(body[0])
    name, mean, interval, tables, splits = body[0]
    low, high = map(float, interval.strip("[]").split(", "))
    assert mean == "+0.048" and low < 0.048 < high
    assert (tables, splits) == ("2 of 3", "8 of 12")
    assert body[1] == ["none", "+0.000", "[+0.000, +0.000]", "0 of 3", "0 of 12"]
    assert adbench_paired(runs, pairs, resamples=2000, seed=1) == (header, body)


def test_duplicates_table_sums_the_splits_and_the_grouped_copies():
    fields = ["table", "seed", "split", "label", "test_rows", "copy_in_train"]
    fields += ["copy_in_context", "copy_in_any_context"]
    values = [
        ("a", "0", "plain", "0", "100", "30", "10", "20"),
        ("a", "1", "plain", "0", "100", "20", "5", "12"),
        ("a", "0", "plain", "1", "50", "2", "0", "0"),
        ("a", "0", "grouped", "0", "90", "0", "0", "1"),
        ("a", "0", "grouped", "1", "40", "0", "0", "0"),
        ("b", "0", "plain", "0", "10", "0", "0", "0"),
        ("b", "0", "plain", "1", "5", "0", "0", "0"),
    ]
    counts = [dict(zip(fields, row, strict=True)) for row in values]
    tables = [
        {"table": "a", "rows": "1234", "rows_in_repeats": "300"},
        {"table": "b", "rows": "15", "rows_in_repeats": "0"},
    ]
    header, body = duplicates_table(counts, tables)
    assert len(header) == len(body[0])
    assert body[0] == [
        "a",
        "1,234",
        "300",
        "50 of 200 / 2 of 50",
        "15 of 200 / 0 of 50",
        "1",
    ]
    assert body[1][3:] == ["0 of 10 / 0 of 5", "0 of 10 / 0 of 5", "0"]


def test_read_csv_returns_one_dict_per_row(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    assert read_csv(path) == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
