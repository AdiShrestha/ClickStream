"""
Week 2: two negative controls catching two DIFFERENT bug classes.
Run both. A failure in either means STOP and do not proceed to Week 3.

Control 1: metrics sanity on synthetic data -- if this fails, the bug
           is in the metrics code, independent of any model or real data.
Control 2: shuffled subject labels on the full pipeline -- if EER is
           good (under roughly 35%), there is a data leakage bug.

Run from repo root: python -m src.negative_control
"""
import numpy as np
from src.feature_extraction import load_cmu_features
from src.evaluate import evaluate_all_subjects
from src.metrics import compute_eer


def sanity_check_metrics_code():
    """
    Tests compute_eer in total isolation from any model or real data.
    If this fails, the bug is in the metrics code, not the model.
    """
    genuine = np.full(100, 1.0)
    impostor = np.full(100, -1.0)
    eer, _ = compute_eer(genuine, impostor)
    assert abs(eer - 0.0) < 1e-6, f"Perfect separation should give EER=0, got {eer}"

    rng = np.random.RandomState(0)
    genuine = rng.normal(0, 1, 1000)
    impostor = rng.normal(0, 1, 1000)
    eer, _ = compute_eer(genuine, impostor)
    assert 0.35 < eer < 0.65, (
        f"Identical distributions should give EER near 50%, got {eer*100:.1f}%"
    )
    print("Metrics sanity checks passed: perfect separation -> ~0%, "
          "identical distributions -> ~50%.")


def negative_control_shuffled_subjects(algorithm="isolation_forest", random_state=42):
    """
    Shuffles subject-identity labels (breaking the true person-to-behavior
    mapping) and re-runs the FULL pipeline. A correct pipeline must show
    EER collapse to approximately 50%. If it does not, there is a data
    leakage bug and Week 3 must not start until the cause is found.
    """
    X, subjects, sessions, feature_cols = load_cmu_features()
    rng = np.random.RandomState(random_state)
    shuffled_subjects = subjects.copy()
    rng.shuffle(shuffled_subjects)

    _, pooled = evaluate_all_subjects(X, shuffled_subjects, sessions, algorithm=algorithm, strict_size_check=False)
    print(f"Shuffled-subject negative control ({algorithm}): "
          f"pooled EER = {pooled['eer']*100:.2f}% (expect roughly 40-60%)")
    assert 0.35 < pooled["eer"] < 0.65, (
        f"Shuffled-subject EER should be near chance (~50%), got "
        f"{pooled['eer']*100:.1f}% -- this indicates a LEAKAGE BUG. "
        "Do not proceed to Week 3 until this is understood and fixed."
    )
    return pooled["eer"]


if __name__ == "__main__":
    sanity_check_metrics_code()
    negative_control_shuffled_subjects("isolation_forest")
    negative_control_shuffled_subjects("one_class_svm")
