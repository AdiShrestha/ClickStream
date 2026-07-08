# Week 2 — Classical Continuous-Authentication Baseline

**Read this entire file before writing or running anything.** This incorporates every decision resolved from the Week 1 review: the temporal enrollment/test split, RobustScaler over StandardScaler (given the confirmed outliers), the `python -m src.X` import convention replacing the sys.path hack, and the confirmed Balabit repository URL. If you are an AI executing this on Aditya's behalf: the split logic in Section 3 is the most important part of this entire week — get the masking exactly right, because a subtle leakage bug here (a subject's own data appearing in their own impostor set, or test-session data leaking into enrollment) would silently invalidate every result in Weeks 4 through 8, which all build on this evaluation harness.

**Environment for this entire week: local machine, no GPU, no Colab/Kaggle.** Isolation Forest and One-Class SVM train in seconds on CPU.

---

## 1. Objective

By the end of this week you have: (a) a per-user enrollment/scoring pipeline using Isolation Forest and One-Class SVM, evaluated with the temporal split resolved after Week 1 (enroll on sessions 1–4, test on sessions 5–8, cross-subject impostor trials), (b) full EER/ROC-AUC/PR-AUC metrics, both pooled across all 51 subjects and broken out per-subject, (c) two negative controls that must both pass before this baseline is trusted (a metrics-code sanity check, and a shuffled-subject-label leakage check), (d) a concrete outlier ablation on subject s049 answering the open question Week 1 raised about whether the 2.035s hold-time outlier actually matters, and (e) the Balabit mouse-dynamics repository cloned and its actual structure inspected (not parsed yet — that's explicitly deferred until the real format is confirmed by inspection, not assumed).

## 2. What changed from `week1.md`'s plan, and why

- **Enrollment/test split, now fully specified:** sessions 1–4 (200 reps) enroll each subject's own model; sessions 5–8 (200 reps) of that same subject are genuine test trials; sessions 5–8 of every other subject (50 × 200 = 10,000 reps) are impostor test trials against that subject's model. Repeated independently for all 51 subjects.
- **RobustScaler, not StandardScaler:** Week 1 confirmed genuine outliers up to 2.0353s (hold) and 25.9873s (DD). StandardScaler's mean/std are themselves distorted by such values; RobustScaler's median/IQR-based scaling is far less affected.
- **Import convention:** `python -m src.MODULE` from the repo root, `from src.X import Y` everywhere, no `sys.path.insert`.
- **Balabit acquisition is inspect-first, not parse-first:** unlike the CMU dataset (whose exact column format was independently verified before `week1.md` was written), Balabit's precise internal file structure has not been verified to the same level of confidence. This week clones the repo and prints its actual structure and a sample file's actual header — the parser gets written next week once the real format is confirmed, not assumed now.

## 3. `src/splits.py` — full code (the most important file this week)

```python
"""
Week 2: builds the enrollment/genuine-test/impostor-test split for
per-user continuous-authentication evaluation.

Design (resolved after Week 1 review):
  - Enrollment: sessions 1-4 of a given subject.
  - Genuine test: sessions 5-8 of the SAME subject.
  - Impostor test: sessions 5-8 of EVERY OTHER subject.
This is a temporal split, not leave-one-subject-out -- LOSO is the
wrong mental model for per-user anomaly models; it's for population
classifiers, which this system is not.

Assertions here deliberately do NOT hardcode CMU-specific literal
counts (e.g. "200" or "10000") so this function stays reusable for
Balabit/HMOG later. Instead they check INTERNAL CONSISTENCY: every
subject must have the same enroll/test set sizes as every other
subject, which Week 1's "no missingness" finding (Appendix E) means
should hold exactly for CMU.
"""
from typing import Dict
import numpy as np


def build_evaluation_splits(
    X: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    enroll_sessions=(1, 2, 3, 4),
    test_sessions=(5, 6, 7, 8),
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
    assert len(set(enroll_sizes.values())) == 1, (
        f"Subjects have inconsistent enrollment set sizes: {enroll_sizes}"
    )
    assert len(set(genuine_sizes.values())) == 1, (
        f"Subjects have inconsistent genuine-test set sizes: {genuine_sizes}"
    )

    return splits
```

## 4. `src/models.py` — full code

```python
"""
Week 2: per-user anomaly-detection model wrapper. One instance is
fit per enrolled subject, on that subject's own enrollment data only --
this is the verification (1:1) framing, never population-level.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler


class PerUserModel:
    """
    algorithm: "isolation_forest" or "one_class_svm"

    Uses RobustScaler deliberately, not StandardScaler: Week 1 confirmed
    real outliers up to 2.0353s (hold) and 25.9873s (DD) in this dataset.
    StandardScaler's mean/std are themselves distorted by such outliers;
    RobustScaler's median/IQR-based scaling is far less affected.
    """

    def __init__(self, algorithm="isolation_forest", contamination=0.05, random_state=42):
        self.algorithm = algorithm
        self.scaler = RobustScaler()
        if algorithm == "isolation_forest":
            self.model = IsolationForest(
                n_estimators=200, contamination=contamination, random_state=random_state
            )
        elif algorithm == "one_class_svm":
            self.model = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def fit(self, X_enroll: np.ndarray) -> "PerUserModel":
        X_scaled = self.scaler.fit_transform(X_enroll)
        self.model.fit(X_scaled)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher score = more consistent with the enrolled baseline."""
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
```

## 5. `src/metrics.py` — full code

```python
"""
Week 2: EER and related metrics, matching the formulas in the
compendium's Appendix D.
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
```

## 6. `src/evaluate.py` — full code, the main runner

```python
"""
Week 2: main evaluation script. Runs the full per-user pipeline across
all 51 CMU subjects, both algorithms, and reports pooled + per-subject
metrics. Saves full results to results/week2/ for reproducibility.
"""
import json
from pathlib import Path
import numpy as np

from src.feature_extraction import load_cmu_features
from src.splits import build_evaluation_splits
from src.models import PerUserModel
from src.metrics import compute_full_metrics

RESULTS_DIR = Path("results/week2")


def evaluate_all_subjects(X, subjects, sessions, algorithm="isolation_forest"):
    splits = build_evaluation_splits(X, subjects, sessions)

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
```

Run with: `python -m src.evaluate`

## 7. `src/negative_control.py` — full code (do not skip either check)

```python
"""
Week 2: two negative controls catching two DIFFERENT bug classes.
Run both. A failure in either means STOP -- do not proceed to Week 3.
"""
import numpy as np
from src.feature_extraction import load_cmu_features
from src.evaluate import evaluate_all_subjects
from src.metrics import compute_eer


def sanity_check_metrics_code():
    """
    Tests compute_eer() in total isolation from any model or real data.
    If THIS fails, the bug is in the metrics code, not the model.
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
    EER collapse to approximately 50% -- if it doesn't, there is a data
    leakage bug (e.g. test-set information reaching training, or subject
    identity leaking through an unintended channel).
    """
    X, subjects, sessions, feature_cols = load_cmu_features()
    rng = np.random.RandomState(random_state)
    shuffled_subjects = subjects.copy()
    rng.shuffle(shuffled_subjects)

    _, pooled = evaluate_all_subjects(X, shuffled_subjects, sessions, algorithm=algorithm)
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
```

Run with: `python -m src.negative_control`

## 8. `src/outlier_ablation.py` — full code (resolves the s049 open question concretely)

```python
"""
Week 2: per-subject outlier ablation. Week 1 found subject s049 has the
2.0353s hold-time outlier (Appendix C: "highest Std_H 43.50ms, source of
the 2035.3ms hold outlier"). This checks whether that single repetition,
if it falls in s049's ENROLLMENT sessions (1-4), materially changes
s049's own model -- concrete evidence rather than a hunch about whether
outlier handling matters here.
"""
import numpy as np
from src.feature_extraction import load_cmu_features
from src.models import PerUserModel
from src.metrics import compute_full_metrics


def find_outlier_row_mask(X, feature_cols, hold_threshold_s=0.5, dd_threshold_s=2.0):
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
```

Run with: `python -m src.outlier_ablation`

## 9. `src/balabit_acquisition.py` — full code (secondary task, inspect-first)

```python
"""
Week 2 (secondary task): clone the Balabit Mouse Dynamics Challenge
repository and inspect its ACTUAL structure before writing any parser.
Unlike CMU, this project has not independently verified Balabit's exact
internal file format to the same confidence level -- guessing a column
layout here would risk building on an unverified assumption. This script
only clones and prints; the parser is written next once the real format
is confirmed from this output.
"""
import subprocess
from pathlib import Path
import pandas as pd

BALABIT_REPO = "https://github.com/balabit/Mouse-Dynamics-Challenge.git"
BALABIT_DIR = Path("data/raw/balabit")


def clone_balabit_repo():
    if BALABIT_DIR.exists():
        print(f"Already cloned: {BALABIT_DIR}")
        return
    BALABIT_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {BALABIT_REPO} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", BALABIT_REPO, str(BALABIT_DIR)],
        check=True,
    )


def inspect_balabit_structure():
    print("\n=== Balabit repo structure (top 3 levels) ===")
    for path in sorted(BALABIT_DIR.rglob("*")):
        depth = len(path.relative_to(BALABIT_DIR).parts)
        if depth <= 3:
            print("  " * depth + path.name)

    csv_candidates = list(BALABIT_DIR.rglob("*.csv"))[:1]
    if csv_candidates:
        sample = pd.read_csv(csv_candidates[0], nrows=5)
        print(f"\n=== Sample file: {csv_candidates[0]} ===")
        print(f"Columns found: {list(sample.columns)}")
        print(sample)
    else:
        print("\nNo .csv found in the first pass -- inspect subdirectories "
              "manually (the challenge data may be nested differently, e.g. "
              "per-session folders) before writing a parser next week. "
              "Do not guess the format from memory.")


if __name__ == "__main__":
    clone_balabit_repo()
    inspect_balabit_structure()
```

Run with: `python -m src.balabit_acquisition`. **Send the printed structure and sample-file output back with your report** — the actual parser gets written from what this prints, not from an assumption.

## 10. `tests/test_splits.py` — full code

```python
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
    # A subject's own data must NEVER appear in their own impostor set --
    # this is the single most important correctness property in this file.
    n_subjects, n_sessions = 2, 8
    rows, subjects, sessions = [], [], []
    for s in range(n_subjects):
        for sess in range(1, n_sessions + 1):
            rows.append([float(s)])  # feature value == subject index, by design
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
```

## 11. `tests/test_metrics.py` — full code

```python
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
```

## 12. Exact command sequence

```bash
cd "Click Stream"   # or wherever the repo root actually is
source .venv/bin/activate

# -- create src/splits.py, src/models.py, src/metrics.py, src/evaluate.py,
#    src/negative_control.py, src/outlier_ablation.py, src/balabit_acquisition.py,
#    tests/test_splits.py, tests/test_metrics.py exactly as above --

pytest tests/ -v
python -m src.negative_control
python -m src.evaluate
python -m src.outlier_ablation
python -m src.balabit_acquisition

git add -A && git commit -m "Week 2: classical baseline, temporal split, negative controls, outlier ablation" && git tag week02
```

## 13. Verification checklist — all of these must pass before Week 3 starts

- [ ] `pytest tests/ -v` shows all tests passing, including the two new split tests (especially `test_split_no_subject_leakage_into_impostor`).
- [ ] `python -m src.negative_control` passes both checks: the metrics sanity check (perfect separation → ~0%, identical distributions → ~50%) and the shuffled-subject check (pooled EER lands between roughly 35–65%). **If the shuffled-subject EER is good (e.g., under 20%), stop — there is a leakage bug, and nothing past this point should be trusted until it's found.**
- [ ] `python -m src.evaluate` completes for both algorithms and prints a pooled EER in a plausible range — **roughly 5% to 20%** is a defensible, reportable band for classical methods on this exact temporal split; a result under ~2% or over ~35% needs investigation (likely leakage in the first case, or a broken model/scaler in the second) before being treated as final.
- [ ] The per-subject EER spread is reported, not just the pooled number — expect meaningful subject-to-subject variation (some subjects easier to model than others), which is itself worth a sentence in the eventual paper.
- [ ] `results/week2/eval_isolation_forest.json` and `eval_one_class_svm.json` exist and contain both pooled and per-subject metrics.
- [ ] `python -m src.outlier_ablation` runs for s049 and reports a concrete before/after EER delta — record the actual number, don't just note that it ran.
- [ ] `python -m src.balabit_acquisition` clones successfully and prints the repo's actual structure and a sample file's actual columns.

## 14. What to send back to Claude at the end of this week

Paste or attach: (1) the full console output of `pytest -v`, `negative_control.py`, `evaluate.py`, `outlier_ablation.py`, and `balabit_acquisition.py`; (2) the two saved JSON result files; (3) which subject had the best and worst per-subject EER for each algorithm, since that's worth a sentence of interpretation either way. I'll check the EER against the expected band, confirm both negative controls actually passed rather than were skipped, look at the outlier-ablation delta, and — based on what `balabit_acquisition.py` actually prints — write the real Balabit parser as part of Week 3's spec rather than guessing it now.