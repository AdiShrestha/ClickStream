"""
Week 1 EDA: prints sanity-check numbers and saves a distribution plot.
Run only after data_acquisition.py has completed successfully.

Run from the repo root:
    python src/eda.py

Or from inside src/:
    cd src && python eda.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extraction import load_cmu_features

OUTPUT_DIR = "data/processed"


def run_eda():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, subjects, sessions, feature_cols = load_cmu_features()
    hold_cols = [c for c in feature_cols if c.startswith("H.")]
    dd_cols = [c for c in feature_cols if c.startswith("DD.")]
    hold_idx = [feature_cols.index(c) for c in hold_cols]
    dd_idx = [feature_cols.index(c) for c in dd_cols]

    hold_values_ms = X[:, hold_idx].flatten() * 1000
    dd_values_ms = X[:, dd_idx].flatten() * 1000

    print("=== Week 1 EDA sanity report ===")
    print(f"Total repetitions loaded: {X.shape[0]}")
    print(f"Total feature columns: {X.shape[1]}")
    print(f"Unique subjects: {len(set(subjects))}")
    print(f"Hold time (ms): mean={hold_values_ms.mean():.1f}, "
          f"std={hold_values_ms.std():.1f}, "
          f"min={hold_values_ms.min():.1f}, max={hold_values_ms.max():.1f}")
    print(f"Digraph DD time (ms): mean={dd_values_ms.mean():.1f}, "
          f"std={dd_values_ms.std():.1f}, "
          f"min={dd_values_ms.min():.1f}, max={dd_values_ms.max():.1f}")
    print()
    print("SANITY BAR: published keystroke-dynamics literature reports")
    print("typical hold times around 60-150ms for average adult typists.")
    print("If your mean is far outside roughly 40-200ms, suspect a units")
    print("bug (seconds vs milliseconds) before doing anything else.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(hold_values_ms, bins=50, ax=axes[0])
    axes[0].set_title("Hold-time distribution, all subjects (ms)")
    axes[0].set_xlabel("Hold time (ms)")

    sns.histplot(dd_values_ms, bins=50, ax=axes[1])
    axes[1].set_title("Digraph DD time distribution, all subjects (ms)")
    axes[1].set_xlabel("DD time (ms)")

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/week1_timing_distributions.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved distribution plot to {out_path}")

    per_rep_mean_hold = X[:, hold_idx].mean(axis=1) * 1000
    per_subject_mean = pd.Series(per_rep_mean_hold).groupby(subjects).mean()
    print(f"\nPer-subject mean hold time range: "
          f"{per_subject_mean.min():.1f}ms to {per_subject_mean.max():.1f}ms "
          f"(spread: {per_subject_mean.max() - per_subject_mean.min():.1f}ms)")
    print("If this spread is very narrow (e.g. under 10ms across 51 people),")
    print("flag it honestly now -- it would mean this feature space alone")
    print("may not separate subjects well, which Week 2's EER will confirm")
    print("or refute quantitatively.")


if __name__ == "__main__":
    run_eda()
