"""
Week 2: builds the enrollment/genuine-test/impostor-test split for
per-user continuous-authentication evaluation.

Design (resolved after Week 1 review):
  Enrollment: sessions 1-4 of a given subject.
  Genuine test: sessions 5-8 of the SAME subject.
  Impostor test: sessions 5-8 of EVERY OTHER subject.
This is a temporal split, not leave-one-subject-out. LOSO is the
wrong mental model for per-user anomaly models; it is for population
classifiers, which this system is not.

Assertions check internal consistency, not hardcoded CMU literal counts,
so this function stays reusable for Balabit and HMOG later.
"""
from typing import Dict
import numpy as np


def build_evaluation_splits(
    X: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    enroll_sessions=(1, 2, 3, 4),
    test_sessions=(5, 6, 7, 8),
    strict_size_check: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    enroll_mask = np.isin(sessions, enroll_sessions)
    test_mask = np.isin(sessions, test_sessions)

    splits = {}
    for subj in np.unique(subjects):
        subj_mask = subjects == subj
        enroll_X = X[subj_mask & enroll_mask]
        genuine_test_X = X[subj_mask & test_mask]
        impostor_test_X = X[(~subj_mask) & test_mask]

        assert enroll_X.shape[0] > 0, f"{subj}: no enrollment data found"
        assert genuine_test_X.shape[0] > 0, f"{subj}: no genuine test data found"
        assert impostor_test_X.shape[0] > 0, f"{subj}: no impostor test data found"

        splits[subj] = {
            "enroll": enroll_X,
            "genuine_test": genuine_test_X,
            "impostor_test": impostor_test_X,
        }

    enroll_sizes = {s: splits[s]["enroll"].shape[0] for s in splits}
    genuine_sizes = {s: splits[s]["genuine_test"].shape[0] for s in splits}
    if strict_size_check:
        assert len(set(enroll_sizes.values())) == 1, (
            f"Subjects have inconsistent enrollment set sizes: {enroll_sizes}"
        )
        assert len(set(genuine_sizes.values())) == 1, (
            f"Subjects have inconsistent genuine-test set sizes: {genuine_sizes}"
        )

    return splits
