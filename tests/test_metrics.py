"""
Unit tests for src/metrics.py.

Run from repo root: pytest tests/ -v
"""
import pytest
import numpy as np
from src.metrics import compute_eer, compute_full_metrics


def test_eer_perfect_separation():
    genuine = np.full(50, 5.0)
    impostor = np.full(50, -5.0)
    eer, _ = compute_eer(genuine, impostor)
    assert eer == pytest.approx(0.0, abs=1e-6)


def test_eer_identical_distributions():
    rng = np.random.RandomState(1)
    genuine = rng.normal(0, 1, 2000)
    impostor = rng.normal(0, 1, 2000)
    eer, _ = compute_eer(genuine, impostor)
    assert 0.4 < eer < 0.6


def test_full_metrics_keys_present():
    genuine = np.array([1.0, 2.0, 3.0])
    impostor = np.array([-1.0, -2.0, -3.0])
    result = compute_full_metrics(genuine, impostor)
    for key in ["eer", "eer_threshold", "roc_auc", "pr_auc", "n_genuine", "n_impostor"]:
        assert key in result
    assert result["roc_auc"] == pytest.approx(1.0)
