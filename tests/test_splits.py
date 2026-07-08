"""
Unit tests for src/splits.py.
The leakage test is the most important: it encodes feature value == subject
index by design so any leakage of a subject's own data into their own
impostor set is caught directly by the assertion.

Run from repo root: pytest tests/ -v
"""
import numpy as np
from src.splits import build_evaluation_splits


def test_split_sizes_correct():
    n_subjects, n_sessions, n_reps = 3, 8, 2
    rows, subjects, sessions = [], [], []
    for s in range(n_subjects):
        for sess in range(1, n_sessions + 1):
            for r in range(n_reps):
                rows.append([float(s), float(sess)])
                subjects.append(f"s{s}")
                sessions.append(sess)
    X = np.array(rows)
    subjects = np.array(subjects)
    sessions = np.array(sessions)

    splits = build_evaluation_splits(X, subjects, sessions)
    assert set(splits.keys()) == {"s0", "s1", "s2"}
    for subj, data in splits.items():
        assert data["enroll"].shape[0] == 4 * n_reps
        assert data["genuine_test"].shape[0] == 4 * n_reps
        assert data["impostor_test"].shape[0] == (n_subjects - 1) * 4 * n_reps


def test_split_no_subject_leakage_into_impostor():
    # Feature value == subject index by design so leakage is detectable directly
    n_subjects, n_sessions = 2, 8
    rows, subjects, sessions = [], [], []
    for s in range(n_subjects):
        for sess in range(1, n_sessions + 1):
            rows.append([float(s)])
            subjects.append(f"s{s}")
            sessions.append(sess)
    X = np.array(rows)
    subjects = np.array(subjects)
    sessions = np.array(sessions)

    splits = build_evaluation_splits(X, subjects, sessions)
    assert (splits["s0"]["impostor_test"] == 1.0).all(), (
        "s0's impostor set must contain ONLY s1's data"
    )
    assert (splits["s1"]["impostor_test"] == 0.0).all(), (
        "s1's impostor set must contain ONLY s0's data"
    )
