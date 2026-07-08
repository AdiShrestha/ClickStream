"""
Triplet sampling and training loop, using ONLY background subjects
enrollment sessions 1 to 4 data. An explicit assertion blocks held-out
data from ever reaching this function, rather than trusting the caller.
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.encoder_model import KeystrokeEncoder


def sample_triplet_batch(sequence_data, subjects, background_subject_ids, batch_size, rng):
    subject_to_indices = {
        s: np.where(subjects == s)[0] for s in background_subject_ids
    }
    for s, idxs in subject_to_indices.items():
        assert len(idxs) >= 2, (
            f"Background subject {s} has fewer than 2 samples, "
            "cannot form an anchor positive pair."
        )

    background_subject_ids = list(background_subject_ids)
    anchors, positives, negatives = [], [], []
    for _ in range(batch_size):
        pos_subject, neg_subject = rng.choice(
            background_subject_ids, size=2, replace=False
        )
        anchor_idx, positive_idx = rng.choice(
            subject_to_indices[pos_subject], size=2, replace=False
        )
        negative_idx = rng.choice(subject_to_indices[neg_subject])

        anchors.append(sequence_data[anchor_idx])
        positives.append(sequence_data[positive_idx])
        negatives.append(sequence_data[negative_idx])

    return (
        torch.tensor(np.stack(anchors), dtype=torch.float32),
        torch.tensor(np.stack(positives), dtype=torch.float32),
        torch.tensor(np.stack(negatives), dtype=torch.float32),
    )


def train_encoder(
    sequence_data, subjects, background_subject_ids, device,
    epochs=50, steps_per_epoch=50, batch_size=32, lr=1e-3, margin=0.3, seed=42,
):
    """
    sequence_data and subjects must ALREADY be restricted to background
    subjects enrollment sessions before calling this, enforced by
    assertion, not just documented as a caller responsibility.
    """
    assert set(np.unique(subjects)) == set(background_subject_ids), (
        "sequence_data and subjects passed to train_encoder contains subjects "
        "outside background_subject_ids. STOP held-out data must never "
        "reach the training loop in any form."
    )

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = KeystrokeEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"epoch": [], "train_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for _ in range(steps_per_epoch):
            anchor, positive, negative = sample_triplet_batch(
                sequence_data, subjects, background_subject_ids, batch_size, rng
            )
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)
            pos_dist = F.pairwise_distance(emb_a, emb_p)
            neg_dist = F.pairwise_distance(emb_a, emb_n)
            loss = F.relu(pos_dist - neg_dist + margin).mean()

            loss_value = loss.item()
            if not np.isfinite(loss_value):
                raise RuntimeError(
                    f"Loss became non-finite ({loss_value}) at epoch "
                    f"{epoch}. STOP do not continue or report results "
                    "from this run. Check the learning rate, confirm the "
                    "scaler was applied before this data reached the "
                    "model, and check for NaN in the raw input features."
                )

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss_value)

        mean_loss = float(np.mean(epoch_losses))
        history["epoch"].append(epoch)
        history["train_loss"].append(mean_loss)
        print(f"Epoch {epoch}/{epochs}: mean triplet loss = {mean_loss:.4f}")

    return model, history
