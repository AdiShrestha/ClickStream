"""
Confirms the trained encoder actually separates subjects in embedding
space. If the inter-to-intra distance ratio is near 1.0, the network
has not learned anything useful, and no downstream EER should be
trusted regardless of what number it produces.
"""
import numpy as np
import torch


def embedding_sanity_check(model, sequence_data, subjects, subject_ids, device):
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for s in subject_ids:
            mask = subjects == s
            x = torch.tensor(sequence_data[mask], dtype=torch.float32).to(device)
            embeddings[s] = model(x).cpu().numpy()

    intra_dists = []
    for s, emb in embeddings.items():
        if len(emb) > 1:
            d = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=-1)
            iu = np.triu_indices(len(emb), k=1)
            intra_dists.extend(d[iu].tolist())

    inter_dists = []
    subject_list = list(embeddings.keys())
    for i in range(len(subject_list)):
        for j in range(i + 1, len(subject_list)):
            emb_i = embeddings[subject_list[i]]
            emb_j = embeddings[subject_list[j]]
            d = np.linalg.norm(emb_i[:, None, :] - emb_j[None, :, :], axis=-1)
            inter_dists.extend(d.flatten().tolist())

    intra_mean = float(np.mean(intra_dists))
    inter_mean = float(np.mean(inter_dists))
    ratio = inter_mean / intra_mean if intra_mean > 0 else float("inf")

    print(f"Mean intra-subject embedding distance: {intra_mean:.4f}")
    print(f"Mean inter-subject embedding distance: {inter_mean:.4f}")
    print(f"Ratio (inter divided by intra): {ratio:.2f}x")
    print("Expect clearly above 1.0 (roughly 1.5x or higher is healthy); "
          "a ratio near 1.0 means the network has not separated subjects at all.")
    return intra_mean, inter_mean, ratio
