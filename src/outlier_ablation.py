"""
Week 2: per-subject outlier ablation. Week 1 found subject s049 has the
2.0353s hold-time outlier (Appendix C: highest Std_H 43.50ms, source of
the 2035.3ms hold outlier). This checks whether that single repetition,
if it falls in s049's ENROLLMENT sessions (1-4), materially changes
s049's own model. Concrete evidence rather than a hunch.

Run from repo root: python -m src.outlier_ablation
"""
import numpy as np
from src.feature_extraction import load_cmu_features
from src.models import PerUserModel
from src.metrics import compute_full_metrics


def find_outlier_row_mask(X, feature_cols, hold_threshold_s=2.0, dd_threshold_s=25.0):
    hold_idx = [i for i, c in enumerate(feature_cols) if c.startswith("H.")]
    dd_idx = [i for i, c in enumerate(feature_cols) if c.startswith("DD.")]
    hold_outlier = np.any(X[:, hold_idx] > hold_threshold_s, axis=1)
    dd_outlier = np.any(X[:, dd_idx] > dd_threshold_s, axis=1)
    return hold_outlier | dd_outlier


def run_ablation_for_subject(target_subject: str, algorithm="isolation_forest"):
    X, subjects, sessions, feature_cols = load_cmu_features()
    outlier_mask = find_outlier_row_mask(X, feature_cols)
    subj_mask = subjects == target_subject
    enroll_mask = np.isin(sessions, (1, 2, 3, 4))
    test_mask = np.isin(sessions, (5, 6, 7, 8))

    n_outliers_in_enroll = int((subj_mask & enroll_mask & outlier_mask).sum())
    print(f"{target_subject}: {n_outliers_in_enroll} outlier repetition(s) "
          f"found in enrollment sessions.")
    if n_outliers_in_enroll == 0:
        print("No outliers in this subject's enrollment set -- ablation is a "
              "no-op here. Nothing to compare for this subject/session split.")
        return

    genuine_test = X[subj_mask & test_mask]
    impostor_test = X[(~subj_mask) & test_mask]

    enroll_with = X[subj_mask & enroll_mask]
    model_with = PerUserModel(algorithm=algorithm).fit(enroll_with)
    metrics_with = compute_full_metrics(
        model_with.score(genuine_test), model_with.score(impostor_test)
    )

    enroll_without = X[(subj_mask & enroll_mask) & (~outlier_mask)]
    model_without = PerUserModel(algorithm=algorithm).fit(enroll_without)
    metrics_without = compute_full_metrics(
        model_without.score(genuine_test), model_without.score(impostor_test)
    )

    print(f"\n{target_subject} ({algorithm}):")
    print(f"  WITH outlier    -- EER: {metrics_with['eer']*100:.2f}%, "
          f"enroll n={enroll_with.shape[0]}")
    print(f"  WITHOUT outlier -- EER: {metrics_without['eer']*100:.2f}%, "
          f"enroll n={enroll_without.shape[0]}")
    delta = (metrics_without['eer'] - metrics_with['eer']) * 100
    print(f"  Delta: {delta:+.2f} percentage points")
    print("  Interpretation: a small delta (roughly a point or two) means "
          "Isolation Forest is already handling this outlier reasonably on "
          "its own; a large delta means outlier handling is a real modeling "
          "decision worth discussing in the paper, not a footnote.")


if __name__ == "__main__":
    run_ablation_for_subject("s049", algorithm="isolation_forest")
    run_ablation_for_subject("s049", algorithm="one_class_svm")
