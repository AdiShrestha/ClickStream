"""
Unit tests for Week 1 feature extraction.
Run from the repo root: pytest tests/ -v
Expected: 5 passed, 0 failed.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest
from feature_extraction import (
    extract_features_from_raw_events,
    load_cmu_features,
)


def test_extract_features_basic_dwell_and_flight():
    # Typing "hi": h down at 1.000, h up at 1.080, i down at 1.150, i up at 1.210
    events = [
        ("h", "down", 1.000),
        ("h", "up", 1.080),
        ("i", "down", 1.150),
        ("i", "up", 1.210),
    ]
    features = extract_features_from_raw_events(events)
    assert features["H.h"] == pytest.approx(0.080, abs=1e-6)
    assert features["H.i"] == pytest.approx(0.060, abs=1e-6)
    assert features["DD.h.i"] == pytest.approx(0.150, abs=1e-6)
    assert features["UD.h.i"] == pytest.approx(0.070, abs=1e-6)


def test_extract_features_handles_unmatched_keyup():
    # A stray keyup with no matching keydown must be ignored, not crash
    events = [("x", "up", 1.0), ("h", "down", 1.0), ("h", "up", 1.05)]
    features = extract_features_from_raw_events(events)
    assert features["H.h"] == pytest.approx(0.05, abs=1e-6)
    assert "H.x" not in features


def test_extract_features_empty_input():
    assert extract_features_from_raw_events([]) == {}


def test_extract_features_out_of_order_input():
    # Events deliberately shuffled -- function must sort internally
    events = [
        ("i", "up", 1.210),
        ("h", "down", 1.000),
        ("i", "down", 1.150),
        ("h", "up", 1.080),
    ]
    features = extract_features_from_raw_events(events)
    assert features["H.h"] == pytest.approx(0.080, abs=1e-6)
    assert features["DD.h.i"] == pytest.approx(0.150, abs=1e-6)


def test_cmu_loader_shapes_match(tmp_path):
    # A tiny fake CMU-format CSV so this test never depends on the
    # network or the real 20400-row file
    fake = pd.DataFrame({
        "subject": ["s001", "s001", "s002"],
        "sessionIndex": [1, 1, 1],
        "rep": [1, 2, 1],
        "H.a": [0.08, 0.09, 0.07],
        "DD.a.b": [0.15, 0.16, 0.14],
        "UD.a.b": [0.07, 0.07, 0.07],
    })
    fake_path = tmp_path / "fake_cmu.csv"
    fake.to_csv(fake_path, index=False)

    X, subjects, sessions, cols = load_cmu_features(str(fake_path))
    assert X.shape == (3, 3)
    assert list(subjects) == ["s001", "s001", "s002"]
    assert cols == ["H.a", "DD.a.b", "UD.a.b"]
