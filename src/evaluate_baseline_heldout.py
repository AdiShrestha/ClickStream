"""
Recomputes Week 2s classical baseline EER restricted to ONLY this
weeks held-out subjects. Week 2s original 17.26 percent pooled EER was
computed across all 51 subjects. Comparing that number directly against
this weeks held-out-only encoder EER would NOT be a fair apples-to-apples
comparison. Reuses Week 2s build_evaluation_splits and PerUserModel
directly, just pre-filtered to the held-out subset.
"""
import numpy as np
from src.splits import build_evaluation_splits
from src.models import PerUserModel
from src.metrics import compute_full_metrics


def evaluate_baseline_on_subset(X, subjects, sessions, subject_subset, algorithm="isolation_forest"):
    subset_mask = np.isin(subjects, subject_subset)
    X_subset = X[subset_mask]
    subjects_subset = subjects[subset_mask]
    sessions_subset = sessions[subset_mask]

    splits = build_evaluation_splits(X_subset, subjects_subset, sessions_subset)

    per_subject_results = {}
    all_genuine, all_impostor = [], []
    for subj, data in splits.items():
        model = PerUserModel(algorithm=algorithm).fit(data["enroll"])
        genuine_scores = model.score(data["genuine_test"])
        impostor_scores = model.score(data["impostor_test"])
        per_subject_results[subj] = compute_full_metrics(genuine_scores, impostor_scores)
        all_genuine.append(genuine_scores)
        all_impostor.append(impostor_scores)

    pooled_genuine = np.concatenate(all_genuine)
    pooled_impostor = np.concatenate(all_impostor)
    pooled_metrics = compute_full_metrics(pooled_genuine, pooled_impostor)
    return per_subject_results, pooled_metrics
