"""
Week 2: EER and related metrics, matching the formulas in the
compendium Appendix D.

EER (Equal Error Rate): the operating threshold where FAR == FRR.
Lower EER is better. ROC-AUC and PR-AUC are additional metrics;
ROC-AUC near 0.5 means no discrimination, near 1.0 means perfect.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray):
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores)),
    ])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    return eer, eer_threshold


def compute_full_metrics(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> dict:
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores)),
    ])
    scores = np.concatenate([genuine_scores, impostor_scores])
    eer, eer_threshold = compute_eer(genuine_scores, impostor_scores)
    return {
        "eer": eer,
        "eer_threshold": float(eer_threshold),
        "roc_auc": roc_auc_score(labels, scores),
        "pr_auc": average_precision_score(labels, scores),
        "n_genuine": len(genuine_scores),
        "n_impostor": len(impostor_scores),
    }
