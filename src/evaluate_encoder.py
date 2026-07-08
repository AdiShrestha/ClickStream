"""
Evaluates the trained encoder on HELD-OUT subjects only, using the
SAME temporal split logic as Week 2 and the SAME compute_full_metrics
function from src/metrics.py, reused directly so Week 2 and Week 3
numbers are computed identically, not reimplemented and subtly diverged.

Score convention: negative L2 distance from centroid. Higher score
means more similar to enrollment centroid, matching the higher equals
more genuine convention from Week 2 src/models.py and src/metrics.py.
"""
import numpy as np
import torch
from src.metrics import compute_full_metrics


def embed_all(model, sequence_data, device, batch_size=256):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sequence_data), batch_size):
            batch = torch.tensor(
                sequence_data[i:i + batch_size], dtype=torch.float32
            ).to(device)
            embeddings.append(model(batch).cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def evaluate_held_out_subjects(model, sequence_data, subjects, sessions, held_out_subject_ids, device):
    enroll_mask = np.isin(sessions, (1, 2, 3, 4))
    test_mask = np.isin(sessions, (5, 6, 7, 8))
    all_embeddings = embed_all(model, sequence_data, device)

    per_subject_results = {}
    all_genuine, all_impostor = [], []

    for subj in held_out_subject_ids:
        subj_mask = subjects == subj
        enroll_emb = all_embeddings[subj_mask & enroll_mask]
        genuine_test_emb = all_embeddings[subj_mask & test_mask]

        held_out_test_mask = np.isin(subjects, held_out_subject_ids) & test_mask
        impostor_mask = held_out_test_mask & (~subj_mask)
        impostor_test_emb = all_embeddings[impostor_mask]

        centroid = enroll_emb.mean(axis=0, keepdims=True)
        genuine_scores = -np.linalg.norm(genuine_test_emb - centroid, axis=1)
        impostor_scores = -np.linalg.norm(impostor_test_emb - centroid, axis=1)

        per_subject_results[subj] = compute_full_metrics(genuine_scores, impostor_scores)
        all_genuine.append(genuine_scores)
        all_impostor.append(impostor_scores)

    pooled_genuine = np.concatenate(all_genuine)
    pooled_impostor = np.concatenate(all_impostor)
    pooled_metrics = compute_full_metrics(pooled_genuine, pooled_impostor)
    return per_subject_results, pooled_metrics
