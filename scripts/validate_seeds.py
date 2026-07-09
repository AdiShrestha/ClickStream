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
    test_mask = np.isin(sessions, (5, 6, 7, 8))
    
    # Train
    bg_mask = np.isin(subjects, background_subjects) & enroll_mask
    X_train = X_seq[bg_mask]
    y_train = subjects[bg_mask]
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    
    unique_bg = np.unique(y_train)
    subj_to_idx = {s: np.where(y_train == s)[0] for s in unique_bg}
    
    model = KeystrokeEncoder(input_dim=3, hidden_dim=32, embedding_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    margin = 0.3
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    epochs = 50
    steps_per_epoch = 50
    batch_size = 32
    
    model.train()
    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            anchors, positives, negatives = [], [], []
            for _ in range(batch_size):
                anchor_subj = np.random.choice(unique_bg)
                anchor_idx, pos_idx = np.random.choice(subj_to_idx[anchor_subj], 2, replace=True)
                neg_subj = np.random.choice(unique_bg[unique_bg != anchor_subj])
                neg_idx = np.random.choice(subj_to_idx[neg_subj])
                
                anchors.append(anchor_idx)
                positives.append(pos_idx)
                negatives.append(neg_idx)
                
            a = X_train_tensor[anchors]
            p = X_train_tensor[positives]
            n = X_train_tensor[negatives]
            
            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)
            
            dist_ap = F.pairwise_distance(emb_a, emb_p, p=2)
            dist_an = F.pairwise_distance(emb_a, emb_n, p=2)
            
            loss = F.relu(dist_ap - dist_an + margin).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # Eval
    model.eval()
    with torch.no_grad():
        X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
        all_embeddings = model(X_seq_tensor).cpu().numpy()
        
    genuine_scores_all = []
    impostor_scores_all = []
    
    for subj in held_out_subjects:
        enroll_idx = (subjects == subj) & enroll_mask
        centroid = all_embeddings[enroll_idx].mean(axis=0)
        
        test_idx = (subjects == subj) & test_mask
        genuine_emb = all_embeddings[test_idx]
        genuine_scores = -np.linalg.norm(genuine_emb - centroid, axis=1)
        genuine_scores_all.extend(genuine_scores.tolist())
        
        impostor_idx = np.isin(subjects, held_out_subjects) & (subjects != subj) & test_mask
        impostor_emb = all_embeddings[impostor_idx]
        impostor_scores = -np.linalg.norm(impostor_emb - centroid, axis=1)
        impostor_scores_all.extend(impostor_scores.tolist())
        
    return compute_full_metrics(np.array(genuine_scores_all), np.array(impostor_scores_all))

def main():
    print("Loading data...")
    X, subjects, sessions, feature_cols = load_cmu_features()
    ordered_keys = reconstruct_key_sequence_order(feature_cols)
    all_subjects = sorted(np.unique(subjects).tolist())
    
    seeds_to_test = [0, 1, 2]
    
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
        scaler = fit_sequence_scaler(X_seq_raw, subjects, bg_subjs)
        X_seq_scaled = apply_sequence_scaler(X_seq_raw, scaler)
        
        enc_metrics = train_and_eval_encoder(X_seq_scaled, subjects, sessions, bg_subjs, ho_subjs, seed)
        print(f"Encoder Pooled EER: {enc_metrics['eer']:.4f}")
        print(f"Improvement:        {if_metrics['eer'] - enc_metrics['eer']:.4f} ({((if_metrics['eer'] - enc_metrics['eer']) / if_metrics['eer']) * 100:.1f}%)")

if __name__ == "__main__":
    main()
