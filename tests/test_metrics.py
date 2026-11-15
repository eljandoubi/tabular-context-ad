import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from contextad.metrics import auprc, auroc, grouped_mean


@pytest.mark.parametrize("seed", range(5))
def test_metrics_match_sklearn_on_continuous_scores(seed):
    rng = np.random.default_rng(seed)
    y = rng.random(400) < 0.1
    score = rng.normal(size=400) + 1.5 * y
    assert np.isclose(auroc(y, score), roc_auc_score(y, score))
    assert np.isclose(auprc(y, score), average_precision_score(y, score))


@pytest.mark.parametrize("seed", range(5))
def test_metrics_match_sklearn_with_ties(seed):
    rng = np.random.default_rng(seed)
    y = rng.random(300) < 0.2
    score = np.round(rng.normal(size=300) + y, 0)
    assert np.isclose(auroc(y, score), roc_auc_score(y, score))
    assert np.isclose(auprc(y, score), average_precision_score(y, score))


def test_perfect_and_reversed_ranking():
    y = np.array([0, 0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == 1.0
    assert auroc(y, -np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == 0.0
    assert auprc(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == 1.0


def test_constant_score_is_chance():
    y = np.array([0, 1, 0, 1, 0, 0])
    assert auroc(y, np.zeros(6)) == 0.5
    assert np.isclose(auprc(y, np.zeros(6)), y.mean())


def test_single_class_gives_nan():
    assert np.isnan(auroc(np.zeros(5), np.arange(5)))
    assert np.isnan(auprc(np.zeros(5), np.arange(5)))
    assert np.isnan(auroc(np.ones(5), np.arange(5)))


def test_grouped_mean_averages_per_group_not_pooled():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    score = np.array([0.1, 0.9, 0.2, 0.8, 5.0, 6.0, 5.5, 4.0])
    group = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    per_group = [auroc(y[:4], score[:4]), auroc(y[4:], score[4:])]
    assert np.isclose(grouped_mean(auroc, y, score, group), np.mean(per_group))
    assert grouped_mean(auroc, y, score, group) != auroc(y, score)


def test_grouped_mean_skips_groups_without_both_classes():
    y = np.array([0, 0, 0, 1, 0, 1])
    score = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0])
    group = np.array([0, 0, 0, 1, 1, 1])
    assert grouped_mean(auroc, y, score, group) == 1.0
    assert np.isnan(grouped_mean(auroc, np.zeros(4), np.arange(4), np.zeros(4)))
