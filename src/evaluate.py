"""
Week 2: main evaluation script. Runs the full per-user pipeline across
all 51 CMU subjects, both algorithms, and reports pooled and per-subject
metrics. Saves full results to results/week2/ for reproducibility.

Run from repo root: python -m src.evaluate
"""
import json
from pathlib import Path
import numpy as np

from src.feature_extraction import load_cmu_features
from src.splits import build_evaluation_splits
from src.models import PerUserModel
from src.metrics import compute_full_metrics

RESULTS_DIR = Path("results/week2")


def evaluate_all_subjects(X, subjects, sessions, algorithm="isolation_forest", strict_size_check=True):
    splits = build_evaluation_splits(X, subjects, sessions, strict_size_check=strict_size_check)

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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, subjects, sessions, feature_cols = load_cmu_features()

    for algorithm in ["isolation_forest", "one_class_svm"]:
        print(f"\n=== Evaluating: {algorithm} ===")
        per_subject, pooled = evaluate_all_subjects(X, subjects, sessions, algorithm=algorithm)

        print(f"Pooled EER: {pooled['eer']*100:.2f}%")
        print(f"Pooled ROC-AUC: {pooled['roc_auc']:.4f}")
        print(f"Pooled PR-AUC: {pooled['pr_auc']:.4f}")

        per_subject_eers = {s: r["eer"] for s, r in per_subject.items()}
        eers = np.array(list(per_subject_eers.values()))
        best_subj = min(per_subject_eers, key=per_subject_eers.get)
        worst_subj = max(per_subject_eers, key=per_subject_eers.get)
        print(f"Per-subject EER: mean={eers.mean()*100:.2f}%, std={eers.std()*100:.2f}%, "
              f"best={eers.min()*100:.2f}% ({best_subj}), "
              f"worst={eers.max()*100:.2f}% ({worst_subj})")

        out = {"algorithm": algorithm, "pooled": pooled, "per_subject": per_subject}
        out_path = RESULTS_DIR / f"eval_{algorithm}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"Saved full results to {out_path}")


if __name__ == "__main__":
    main()
