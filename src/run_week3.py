"""
Main Week 3 script. Loads CMU data, reconstructs true key order, builds
sequence features, creates or loads the background and held-out split,
fits the scaler on background data only, trains the encoder, runs the
embedding sanity check, evaluates on held-out subjects, and recomputes
the Week 2 classical baseline on the SAME held-out subjects for a fair
comparison. Saves everything under results/week3/.
"""
import json
from pathlib import Path
import numpy as np
import torch

from src.feature_extraction import load_cmu_features
from src.key_sequence import reconstruct_key_sequence_order, build_sequence_features
from src.subject_split import create_or_load_subject_split
from src.sequence_scaler import fit_sequence_scaler, apply_sequence_scaler
from src.device import get_device
from src.train_encoder import train_encoder
from src.embedding_check import embedding_sanity_check
from src.evaluate_encoder import evaluate_held_out_subjects
from src.evaluate_baseline_heldout import evaluate_baseline_on_subset

RESULTS_DIR = Path("results/week3")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    X, subjects, sessions, feature_cols = load_cmu_features()

    ordered_keys = reconstruct_key_sequence_order(feature_cols)
    print(f"Reconstructed true key order: {ordered_keys}")
    sequence_data = build_sequence_features(X, feature_cols, ordered_keys)
    print(f"Sequence feature shape: {sequence_data.shape}")

    background_ids, held_out_ids = create_or_load_subject_split(subjects)
    print(f"Background subjects ({len(background_ids)}): {background_ids}")
    print(f"Held-out subjects ({len(held_out_ids)}): {held_out_ids}")

    enroll_mask = np.isin(sessions, (1, 2, 3, 4))
    background_enroll_mask = np.isin(subjects, background_ids) & enroll_mask

    scaler = fit_sequence_scaler(
        sequence_data[background_enroll_mask],
        subjects[background_enroll_mask],
        background_ids,
    )
    scaled_sequence_data = apply_sequence_scaler(sequence_data, scaler)

    print("\n=== Training encoder (background subjects, enrollment sessions only) ===")
    train_X = scaled_sequence_data[background_enroll_mask]
    train_subjects = subjects[background_enroll_mask]
    model, history = train_encoder(train_X, train_subjects, background_ids, device)

    with open(RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n=== Embedding sanity check (background subjects) ===")
    intra_mean, inter_mean, ratio = embedding_sanity_check(
        model, train_X, train_subjects, background_ids, device
    )

    print("\n=== Evaluating on held-out subjects ===")
    per_subject_encoder, pooled_encoder = evaluate_held_out_subjects(
        model, scaled_sequence_data, subjects, sessions, held_out_ids, device
    )
    print(f"Held-out encoder pooled EER: {pooled_encoder['eer']*100:.2f}%")
    print(f"Held-out encoder pooled ROC-AUC: {pooled_encoder['roc_auc']:.4f}")
    print(f"Held-out encoder pooled PR-AUC: {pooled_encoder['pr_auc']:.4f}")

    print("\n=== Recomputing Week 2 classical baseline on the SAME held-out subjects ===")
    per_subject_baseline, pooled_baseline = evaluate_baseline_on_subset(
        X, subjects, sessions, held_out_ids, algorithm="isolation_forest"
    )
    print(f"Held-out classical (Isolation Forest) pooled EER: {pooled_baseline['eer']*100:.2f}%")

    print("\n=== Fair comparison (same held-out subjects, both methods) ===")
    print(f"  Isolation Forest EER: {pooled_baseline['eer']*100:.2f}%")
    print(f"  Deep encoder EER:     {pooled_encoder['eer']*100:.2f}%")
    delta = (pooled_encoder['eer'] - pooled_baseline['eer']) * 100
    print(f"  Delta (encoder minus classical): {delta:+.2f} percentage points")
    print("  Report this honestly either way. A classical baseline "
          "beating the deep model on this small dataset is a legitimate "
          "reportable finding, not a failure to fix.")

    out = {
        "held_out_subjects": held_out_ids,
        "background_subjects": background_ids,
        "embedding_sanity": {
            "intra_mean": intra_mean,
            "inter_mean": inter_mean,
            "ratio": ratio,
        },
        "encoder": {"per_subject": per_subject_encoder, "pooled": pooled_encoder},
        "classical_baseline_heldout": {
            "per_subject": per_subject_baseline, "pooled": pooled_baseline,
        },
    }
    with open(RESULTS_DIR / "week3_full_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved full results to {RESULTS_DIR / 'week3_full_results.json'}")

    torch.save(model.state_dict(), RESULTS_DIR / "encoder_weights.pt")
    print(f"Saved model weights to {RESULTS_DIR / 'encoder_weights.pt'}")


if __name__ == "__main__":
    main()
