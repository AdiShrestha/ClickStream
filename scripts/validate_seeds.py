import json
import torch
import numpy as np
from src.feature_extraction import load_cmu_features
from src.sequence_scaler import fit_sequence_scaler, apply_sequence_scaler
from src.key_sequence import build_sequence_features, reconstruct_key_sequence_order
from src.device import get_device
from src.encoder_model import KeystrokeEncoder
import torch.nn.functional as F
from src.metrics import compute_full_metrics
from sklearn.ensemble import IsolationForest
import sys
from src.train_encoder import train_encoder
from src.evaluate_encoder import evaluate_held_out_subjects

def evaluate_if(X, subjects, sessions, held_out_subjects):
    enroll_mask = np.isin(sessions, (1, 2, 3, 4))
    test_mask = np.isin(sessions, (5, 6, 7, 8))
    
    genuine_scores_all = []
    impostor_scores_all = []
    
    for subj in held_out_subjects:
        enroll_idx = (subjects == subj) & enroll_mask
        X_enroll = X[enroll_idx]
        
        clf = IsolationForest(n_estimators=100, random_state=42)
        clf.fit(X_enroll)
        
        test_idx = (subjects == subj) & test_mask
        X_test_genuine = X[test_idx]
        genuine_scores = clf.decision_function(X_test_genuine)
        genuine_scores_all.extend(genuine_scores.tolist())
        
        impostor_idx = np.isin(subjects, held_out_subjects) & (subjects != subj) & test_mask
        X_test_impostor = X[impostor_idx]
        impostor_scores = clf.decision_function(X_test_impostor)
        impostor_scores_all.extend(impostor_scores.tolist())
        
    return compute_full_metrics(np.array(genuine_scores_all), np.array(impostor_scores_all))

def train_and_eval_encoder(X_seq, subjects, sessions, background_subjects, held_out_subjects, seed):
    device = get_device()
    enroll_mask = np.isin(sessions, (1, 2, 3, 4))
    bg_mask = np.isin(subjects, background_subjects) & enroll_mask
    X_train = X_seq[bg_mask]
    y_train = subjects[bg_mask]
    
    model, _ = train_encoder(X_train, y_train, background_subjects, device)
    _, pooled_metrics = evaluate_held_out_subjects(model, X_seq, subjects, sessions, held_out_subjects, device)
    
    return pooled_metrics

def main():
    print("Loading data...")
    X, subjects, sessions, feature_cols = load_cmu_features()
    ordered_keys = reconstruct_key_sequence_order(feature_cols)
    all_subjects = sorted(np.unique(subjects).tolist())
    
    seeds_to_test = list(range(10))
    
    improvements = []
    
    for seed in seeds_to_test:
        print(f"\n--- Running evaluation for seed {seed} ---")
        rng = np.random.RandomState(seed)
        shuffled = all_subjects.copy()
        rng.shuffle(shuffled)
        n_background = int(round(len(shuffled) * 0.70))
        bg_subjs = sorted(shuffled[:n_background])
        ho_subjs = sorted(shuffled[n_background:])
        
        # Isolation Forest baseline
        if_metrics = evaluate_if(X, subjects, sessions, ho_subjs)
        print(f"IF Pooled EER:      {if_metrics['eer']:.4f}")
        
        # Sequence preprocessing for Encoder
        X_seq_raw = build_sequence_features(X, feature_cols, ordered_keys)
        enroll_mask = np.isin(sessions, (1, 2, 3, 4))
        background_enroll_mask = np.isin(subjects, bg_subjs) & enroll_mask
        scaler = fit_sequence_scaler(
            X_seq_raw[background_enroll_mask],
            subjects[background_enroll_mask],
            bg_subjs
        )
        X_seq_scaled = apply_sequence_scaler(X_seq_raw, scaler)
        
        enc_metrics = train_and_eval_encoder(X_seq_scaled, subjects, sessions, bg_subjs, ho_subjs, seed)
        print(f"Encoder Pooled EER: {enc_metrics['eer']:.4f}")
        improvement = if_metrics['eer'] - enc_metrics['eer']
        improvements.append(improvement)
        print(f"Improvement:        {improvement:.4f} ({((improvement) / if_metrics['eer']) * 100:.1f}%)")

    print("\n=== Summary over 10 seeds ===")
    print(f"Mean improvement: {np.mean(improvements):.4f} (std: {np.std(improvements):.4f})")

if __name__ == "__main__":
    main()
