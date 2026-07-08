# Week 3 — Deep Sequence Model Baseline

**Read this entire file before writing or running anything. Do not skip sections because they look like prose — every section contains a specific correctness requirement that the code depends on.** This week has more ways to go subtly wrong than Weeks 1–2 combined, because a sequence model can produce a plausible-looking loss curve and a plausible-looking EER even when the underlying sequence order, data split, or normalization is wrong. Every design decision below exists specifically to make a wrong result *fail loudly* instead of *look fine*.

**If you are an AI executing this on Aditya's behalf:** this project has already had one round where a threshold was quietly changed until a problem looked resolved instead of being investigated. Do not do that here. If any assertion in this file fails, or any number looks implausible, stop and report the raw values — do not adjust a parameter to make the failure go away without explaining why the original value was wrong.

---

## 1. Objective

By the end of this week you have: (a) a Siamese/triplet-loss keystroke encoder trained via PyTorch, with a device-detection utility (MPS locally, CUDA on Colab, CPU fallback) established once and reused for the rest of the project, (b) a **background/held-out subject split** — a new, methodologically necessary split distinct from Week 2's session split, explained fully in Section 2 — used to test whether the learned embedding space actually generalizes to subjects never seen during training, (c) a programmatically-reconstructed true chronological key order (not alphabetical), verified by a dedicated unit test, (d) a trained encoder with monotonically-decreasing loss and no NaNs, (e) an embedding-space sanity check confirming intra-subject distances are meaningfully smaller than inter-subject distances, (f) an EER evaluation on held-out subjects using Week 2's *exact same* `compute_eer`/`compute_full_metrics` functions — not reimplemented, reused, for consistency, and (g) a fair, apples-to-apples comparison against a **recomputed** classical baseline restricted to the same held-out subjects, not against Week 2's full-51-subject pooled number.

## 2. Critical design decision: why this week needs a NEW subject split, not just Week 2's session split

This is worth understanding before touching any code, because it changes what "the test set" means this week.

Week 2's temporal split (enroll on sessions 1–4, test on sessions 5–8) answers: *"given a per-user model trained on this specific person's own data, does it recognize this same person's later sessions?"* That's the right question for Isolation Forest and One-Class SVM, which are fit fresh, per user, with no shared learned representation.

A Siamese network is different: its whole point is a **shared embedding space** learned once, that should generalize to *new* people it never saw during training — exactly how a real bank would need to onboard a new customer without retraining the entire network from scratch. If the encoder is trained using triplets drawn from all 51 subjects and then evaluated on those same 51 subjects' held-out sessions, that's a **closed-set** evaluation — the network already implicitly "knows" every identity it's being tested on. That would inflate the reported EER relative to what the network would actually achieve on a genuinely new customer, and a reviewer would correctly ask this exact question.

**The fix:** split the 51 subjects themselves — not their sessions — into a **background set** (used only to train the encoder's weights) and a **held-out set** (never seen during training; used only for evaluation, exactly mirroring Week 2's enrollment/test logic within this held-out group). This is standard practice in deep-metric-learning verification systems (face verification, speaker verification, and the TypeNet keystroke-embedding line of work referenced in the compendium all evaluate this way) and it is the only version of this experiment that actually tests generalization rather than memorization.

**Split size:** 70% background (36 subjects) / 30% held-out (15 subjects), fixed by a seeded random split, saved to disk so it never silently changes between runs.

## 3. Environment: Colab this week

This is the first week that needs GPU. Follow this exactly.

**3.1 — Get the repo into Colab.** If you haven't already, push your local repo to a GitHub remote:
```bash
# On your Mac, in the repo root, only if you haven't set up a remote yet:
git remote add origin <your-repo-url>
git push -u origin main
git push origin week01 week02   # push the tags too
```

**3.2 — New Colab notebook, first cell:**
```python
from google.colab import drive
drive.mount('/content/drive')

# Clone into a persistent Drive location so results survive session resets
import os
PROJECT_DIR = '/content/drive/MyDrive/clickstream'
if not os.path.exists(PROJECT_DIR):
    !git clone <your-repo-url> {PROJECT_DIR}
%cd {PROJECT_DIR}
!git pull
```

**3.3 — Second cell: confirm GPU and install deps.**
```python
!nvidia-smi
!pip install -r requirements.txt --quiet
```
Expected: `nvidia-smi` prints a Tesla T4 (or similar) — if it prints "command not found," go to Runtime → Change runtime type → GPU before continuing.

**3.4 — Every subsequent Week 3 script writes its outputs under `results/week3/` inside `PROJECT_DIR`**, which is on Drive, so nothing is lost when the Colab runtime disconnects. Pull those results back to your Mac with `git pull` once you commit them from Colab, or download `results/week3/` directly from the Drive folder.

## 4. `src/device.py` — full code (established once, reused everywhere from here on)

```python
"""
Device detection, established as a project-wide convention starting
Week 3: CUDA on Colab/Linux/Windows, MPS on the M3 locally, CPU as a
last-resort fallback. Every training/inference script in this project
imports get_device() from here rather than re-detecting it inline.
"""
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
```

## 5. `src/key_sequence.py` — full code (the most important file this week)

This reconstructs the TRUE chronological key order from the DD column names' encoded transitions, rather than trusting alphabetical sort order. Read the docstring before touching this file — this is the single place where a silent, hard-to-detect bug would do the most damage.

```python
"""
Week 3: reconstructs the true chronological key-press order for the
CMU password (".tie5Roanl" + Return, 11 key positions) directly from
the DD column names, which encode "from_key.to_key" transitions.

CRITICAL: do NOT use sorted(column_names) for this purpose. That gives
alphabetical order, which the classical baseline (Week 2) uses because
it doesn't care about order -- but it is NOT the order the password was
actually typed in, and using it here would silently feed the LSTM a
scrambled, meaningless sequence while still producing a plausible-
looking loss curve and EER. The bug would not announce itself.

This reconstruction is deliberately data-driven (parsed from whatever
column names actually exist), not hardcoded against a guessed password
spelling, specifically so it keeps working even if this project is later
pointed at a different fixed-password dataset.
"""
from typing import List, Tuple, Dict


def parse_dd_column(dd_col_name: str, valid_key_names: set) -> Tuple[str, str]:
    """
    dd_col_name: e.g. "DD.period.t" or "DD.five.Shift.r"
    valid_key_names: the set of key names parsed from H. column suffixes
                      (e.g. {"period", "t", "i", ..., "five", "Shift.r", ...})

    Splits robustly against the KNOWN valid key-name set rather than
    guessing based on dot positions -- this correctly handles key names
    that themselves contain a literal dot, like "Shift.r".
    """
    assert dd_col_name.startswith("DD."), f"Not a DD column: {dd_col_name}"
    suffix = dd_col_name[len("DD."):]
    for from_key in valid_key_names:
        prefix = from_key + "."
        if suffix.startswith(prefix):
            to_key = suffix[len(prefix):]
            if to_key in valid_key_names:
                return from_key, to_key
    raise ValueError(
        f"Could not parse '{dd_col_name}' against known key names "
        f"{sorted(valid_key_names)}. STOP -- do not guess a split; "
        "this means the column-naming assumption is wrong and needs "
        "to be understood before proceeding."
    )


def reconstruct_key_sequence_order(feature_cols: List[str]) -> List[str]:
    """
    Returns the list of key names in TRUE chronological typing order,
    reconstructed from the DD-column transition chain, not alphabetical
    sort. Raises immediately and loudly if the chain doesn't form a
    single clean sequence covering every key exactly once.
    """
    h_cols = [c for c in feature_cols if c.startswith("H.")]
    valid_key_names = set(c[len("H."):] for c in h_cols)
    dd_cols = [c for c in feature_cols if c.startswith("DD.")]

    assert len(dd_cols) == len(valid_key_names) - 1, (
        f"Expected {len(valid_key_names) - 1} DD columns (one fewer than "
        f"the {len(valid_key_names)} keys), found {len(dd_cols)}. "
        "STOP -- the chain-reconstruction logic below assumes exactly "
        "this relationship between key count and transition count."
    )

    transitions: Dict[str, str] = {}
    to_keys_seen = set()
    for dd_col in dd_cols:
        from_key, to_key = parse_dd_column(dd_col, valid_key_names)
        assert from_key not in transitions, (
            f"Key '{from_key}' appears as the source of more than one "
            "transition -- the chain is not linear. STOP and investigate "
            "rather than picking one arbitrarily."
        )
        transitions[from_key] = to_key
        to_keys_seen.add(to_key)

    first_key_candidates = valid_key_names - to_keys_seen
    assert len(first_key_candidates) == 1, (
        f"Expected exactly one starting key (a key that is never a "
        f"transition target), found {first_key_candidates}. STOP -- "
        "the DD-column chain does not form a single clean sequence."
    )
    first_key = next(iter(first_key_candidates))

    ordered_keys = [first_key]
    current = first_key
    while current in transitions:
        current = transitions[current]
        ordered_keys.append(current)

    assert len(ordered_keys) == len(valid_key_names), (
        f"Reconstructed chain has {len(ordered_keys)} keys but expected "
        f"{len(valid_key_names)} -- the chain is broken, has a cycle, or "
        "does not reach every key. STOP and investigate before proceeding; "
        "do not truncate or pad the sequence to force a match."
    )
    assert set(ordered_keys) == valid_key_names, (
        "Reconstructed chain does not cover every known key exactly "
        "once. STOP -- do not proceed with a partial or duplicated chain."
    )

    return ordered_keys


def build_sequence_features(X, feature_cols: List[str], ordered_keys: List[str]):
    """
    Reshapes the flat (n_samples, 31) feature array into
    (n_samples, n_keys, 3), where axis 1 is TRUE chronological key
    position and axis 2 is [hold_time, digraph_dd, digraph_ud],
    zero-filled for the final key position (no outgoing transition).
    """
    import numpy as np
    n_samples = X.shape[0]
    n_keys = len(ordered_keys)
    col_index = {c: i for i, c in enumerate(feature_cols)}

    sequence = np.zeros((n_samples, n_keys, 3), dtype=np.float32)
    for pos, key in enumerate(ordered_keys):
        h_col = f"H.{key}"
        assert h_col in col_index, f"Missing expected column: {h_col}"
        sequence[:, pos, 0] = X[:, col_index[h_col]]
        if pos < n_keys - 1:
            next_key = ordered_keys[pos + 1]
            dd_col = f"DD.{key}.{next_key}"
            ud_col = f"UD.{key}.{next_key}"
            assert dd_col in col_index, f"Missing expected column: {dd_col}"
            assert ud_col in col_index, f"Missing expected column: {ud_col}"
            sequence[:, pos, 1] = X[:, col_index[dd_col]]
            sequence[:, pos, 2] = X[:, col_index[ud_col]]
    return sequence
```

## 6. `tests/test_key_sequence.py` — full code (proves Section 5 is correct, do not skip this)

```python
"""
Verifies reconstruct_key_sequence_order and build_sequence_features
against a SYNTHETIC example with a KNOWN correct answer, deliberately
including a key name with an embedded dot ("Shift.r") to prove the
parser handles it -- this is exactly the case that would silently
break a naive split-on-dots implementation.
"""
import numpy as np
from src.key_sequence import (
    parse_dd_column,
    reconstruct_key_sequence_order,
    build_sequence_features,
)

# Synthetic 4-key password: a -> b -> Shift.r -> c
# (mirrors the real CMU case where "Shift.r" contains an embedded dot)
FAKE_COLUMNS = [
    "H.a", "H.b", "H.Shift.r", "H.c",
    "DD.a.b", "DD.b.Shift.r", "DD.Shift.r.c",
    "UD.a.b", "UD.b.Shift.r", "UD.Shift.r.c",
]


def test_parse_dd_column_handles_embedded_dot():
    valid_keys = {"a", "b", "Shift.r", "c"}
    assert parse_dd_column("DD.b.Shift.r", valid_keys) == ("b", "Shift.r")
    assert parse_dd_column("DD.Shift.r.c", valid_keys) == ("Shift.r", "c")
    assert parse_dd_column("DD.a.b", valid_keys) == ("a", "b")


def test_reconstruct_key_sequence_order_correct():
    order = reconstruct_key_sequence_order(FAKE_COLUMNS)
    assert order == ["a", "b", "Shift.r", "c"]


def test_reconstruct_raises_on_broken_chain():
    broken_columns = [
        "H.a", "H.b", "H.c", "H.d",
        "DD.a.b", "DD.c.d",  # two disconnected pairs, not a single chain
        "UD.a.b", "UD.c.d",
    ]
    try:
        reconstruct_key_sequence_order(broken_columns)
        assert False, "Expected an AssertionError for a broken/disconnected chain"
    except AssertionError:
        pass  # expected


def test_build_sequence_features_shape_and_values():
    order = ["a", "b", "Shift.r", "c"]
    col_index = {c: i for i, c in enumerate(FAKE_COLUMNS)}
    X = np.zeros((2, len(FAKE_COLUMNS)), dtype=np.float32)
    # Row 0: distinguishable values per column so we can check exact placement
    X[0, col_index["H.a"]] = 0.10
    X[0, col_index["DD.a.b"]] = 0.20
    X[0, col_index["UD.a.b"]] = 0.05
    X[0, col_index["H.b"]] = 0.11
    X[0, col_index["H.Shift.r"]] = 0.09
    X[0, col_index["H.c"]] = 0.08

    seq = build_sequence_features(X, FAKE_COLUMNS, order)
    assert seq.shape == (2, 4, 3)
    assert seq[0, 0, 0] == 0.10  # H.a at position 0
    assert seq[0, 0, 1] == 0.20  # DD.a.b at position 0
    assert seq[0, 0, 2] == 0.05  # UD.a.b at position 0
    assert seq[0, 1, 0] == 0.11  # H.b at position 1
    assert seq[0, 2, 0] == 0.09  # H.Shift.r at position 2
    assert seq[0, 3, 0] == 0.08  # H.c at position 3 (last key)
    assert seq[0, 3, 1] == 0.0   # no outgoing transition from last key
    assert seq[0, 3, 2] == 0.0
```

Run with: `pytest tests/test_key_sequence.py -v` — **all 4 tests must pass before you write or run any model code.** If `test_reconstruct_key_sequence_order_correct` fails against the real CMU columns when you run this against real data later, do not proceed to training; the sequence order feeding the LSTM would be meaningless.

## 7. `src/subject_split.py` — full code (background/held-out split, persisted)

```python
"""
Splits the 51 subjects into a background set (trains the encoder) and
a held-out set (evaluation only, never seen during training). Saved to
disk on first creation so the split is FIXED across every subsequent
run in this project -- regenerating a different random split between
the training script and the evaluation script would silently invalidate
the entire generalization claim this week is built around.
"""
import json
from pathlib import Path
import numpy as np

SPLIT_PATH = Path("results/week3/subject_split.json")
BACKGROUND_FRACTION = 0.70
SPLIT_SEED = 42


def create_or_load_subject_split(all_subjects):
    unique_subjects = sorted(np.unique(all_subjects).tolist())

    if SPLIT_PATH.exists():
        with open(SPLIT_PATH) as f:
            split = json.load(f)
        loaded_all = set(split["background"]) | set(split["held_out"])
        assert loaded_all == set(unique_subjects), (
            "Existing subject_split.json does not match the current "
            "dataset's subject list. STOP -- do not silently regenerate "
            "a new split; figure out why the subject list changed first."
        )
        print(f"Loaded existing split from {SPLIT_PATH}")
        return split["background"], split["held_out"]

    rng = np.random.RandomState(SPLIT_SEED)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)
    n_background = int(round(len(shuffled) * BACKGROUND_FRACTION))
    background = sorted(shuffled[:n_background])
    held_out = sorted(shuffled[n_background:])

    assert len(background) + len(held_out) == len(unique_subjects)
    assert len(set(background) & set(held_out)) == 0, "Overlap between splits"

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_PATH, "w") as f:
        json.dump({"background": background, "held_out": held_out}, f, indent=2)

    print(f"Created new subject split: {len(background)} background, "
          f"{len(held_out)} held-out")
    print(f"Saved to {SPLIT_PATH}")
    return background, held_out
```

## 8. `src/sequence_scaler.py` — full code

```python
"""
Fits a RobustScaler on BACKGROUND SUBJECTS' enrollment data ONLY,
consistent with Week 2's RobustScaler decision (Week 1 confirmed real
outliers that distort StandardScaler's mean/std). Fitting only on
background data -- never held-out data -- avoids leaking held-out
subjects' feature-scale statistics into anything the model touches.
"""
import numpy as np
from sklearn.preprocessing import RobustScaler


def fit_sequence_scaler(sequence_data, subjects, background_subject_ids):
    mask = np.isin(subjects, background_subject_ids)
    n_keys, n_feat = sequence_data.shape[1], sequence_data.shape[2]
    flat = sequence_data[mask].reshape(-1, n_keys * n_feat)
    scaler = RobustScaler()
    scaler.fit(flat)
    return scaler


def apply_sequence_scaler(sequence_data, scaler):
    n_samples, n_keys, n_feat = sequence_data.shape
    flat = sequence_data.reshape(n_samples, n_keys * n_feat)
    scaled_flat = scaler.transform(flat)
    return scaled_flat.reshape(n_samples, n_keys, n_feat)
```

## 9. `src/encoder_model.py` — full code

```python
"""
The Siamese keystroke encoder: a bidirectional LSTM over the 11-key
chronological sequence, producing a fixed-size L2-normalized embedding.
Trained via triplet loss (Section 10).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KeystrokeEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, embedding_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, x):
        # x: (batch, n_keys, 3) -- fixed length (11), no padding needed
        _, (h_n, _) = self.lstm(x)
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        embedding = F.normalize(self.fc(h_cat), p=2, dim=1)
        return embedding
```

## 10. `src/train_encoder.py` — full code

```python
"""
Triplet sampling and training loop, using ONLY background subjects'
enrollment (sessions 1-4) data. An explicit assertion blocks held-out
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
            f"Background subject {s} has fewer than 2 samples -- "
            "cannot form an anchor/positive pair."
        )

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
    sequence_data/subjects must ALREADY be restricted to background
    subjects' enrollment sessions before calling this -- enforced by
    assertion, not just documented as a caller responsibility.
    """
    assert set(np.unique(subjects)) == set(background_subject_ids), (
        "sequence_data/subjects passed to train_encoder contains subjects "
        "outside background_subject_ids. STOP -- held-out data must never "
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
            anchor, positive, negative = (
                anchor.to(device), positive.to(device), negative.to(device)
            )

            optimizer.zero_grad()
            emb_a, emb_p, emb_n = model(anchor), model(positive), model(negative)
            pos_dist = F.pairwise_distance(emb_a, emb_p)
            neg_dist = F.pairwise_distance(emb_a, emb_n)
            loss = F.relu(pos_dist - neg_dist + margin).mean()

            loss_value = loss.item()
            if not np.isfinite(loss_value):
                raise RuntimeError(
                    f"Loss became non-finite ({loss_value}) at epoch "
                    f"{epoch}. STOP -- do not continue or report results "
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
```

## 11. `src/embedding_check.py` — full code

```python
"""
Confirms the trained encoder actually separates subjects in embedding
space. If the inter/intra distance ratio is near 1.0, the network has
not learned anything useful, and no downstream EER should be trusted
regardless of what number it produces.
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
            emb_i, emb_j = embeddings[subject_list[i]], embeddings[subject_list[j]]
            d = np.linalg.norm(emb_i[:, None, :] - emb_j[None, :, :], axis=-1)
            inter_dists.extend(d.flatten().tolist())

    intra_mean = float(np.mean(intra_dists))
    inter_mean = float(np.mean(inter_dists))
    ratio = inter_mean / intra_mean if intra_mean > 0 else float("inf")

    print(f"Mean intra-subject embedding distance: {intra_mean:.4f}")
    print(f"Mean inter-subject embedding distance: {inter_mean:.4f}")
    print(f"Ratio (inter/intra): {ratio:.2f}x")
    print("Expect clearly > 1.0 (roughly 1.5x+ is healthy); a ratio near "
          "1.0 means the network has not separated subjects at all.")
    return intra_mean, inter_mean, ratio
```

## 12. `src/evaluate_encoder.py` — full code (reuses Week 2's `metrics.py` directly)

```python
"""
Evaluates the trained encoder on HELD-OUT subjects only, using the
SAME temporal split logic as Week 2 and the SAME compute_full_metrics
function from src/metrics.py -- reused directly so Week 2 and Week 3
numbers are computed identically, not reimplemented and subtly diverged.
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
        # Score = NEGATIVE distance to centroid: higher = more similar,
        # matching the "higher = more genuine" convention from Week 2's
        # src/models.py and src/metrics.py.
        genuine_scores = -np.linalg.norm(genuine_test_emb - centroid, axis=1)
        impostor_scores = -np.linalg.norm(impostor_test_emb - centroid, axis=1)

        per_subject_results[subj] = compute_full_metrics(genuine_scores, impostor_scores)
        all_genuine.append(genuine_scores)
        all_impostor.append(impostor_scores)

    pooled_genuine = np.concatenate(all_genuine)
    pooled_impostor = np.concatenate(all_impostor)
    pooled_metrics = compute_full_metrics(pooled_genuine, pooled_impostor)
    return per_subject_results, pooled_metrics
```

## 13. `src/evaluate_baseline_heldout.py` — full code (the fair-comparison recompute)

```python
"""
Recomputes Week 2's classical baseline EER restricted to ONLY this
week's held-out subjects. Week 2's original 17.26% pooled EER was
computed across all 51 subjects -- comparing that number directly
against this week's held-out-only encoder EER would NOT be a fair,
apples-to-apples comparison. Reuses Week 2's build_evaluation_splits
and PerUserModel directly, just pre-filtered to the held-out subset.
"""
import numpy as np
from src.splits import build_evaluation_splits
from src.models import PerUserModel
from src.metrics import compute_full_metrics


def evaluate_baseline_on_subset(X, subjects, sessions, subject_subset, algorithm="isolation_forest"):
    subset_mask = np.isin(subjects, subject_subset)
    X_subset = X[subset_mask]
    subjects_subset = subjects[subset_mask]
    sessions_subset = sessions[subset_mask]

    splits = build_evaluation_splits(X_subset, subjects_subset, sessions_subset)

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
```

## 14. `src/run_week3.py` — full code (main orchestration script, run this one)

```python
"""
Main Week 3 script. Loads CMU data, reconstructs true key order, builds
sequence features, creates/loads the background/held-out split, fits
the scaler on background data only, trains the encoder, runs the
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
    embedding_sanity_check(model, train_X, train_subjects, background_ids, device)

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

    print("\n=== Fair comparison (same 15 held-out subjects, both methods) ===")
    print(f"  Isolation Forest EER: {pooled_baseline['eer']*100:.2f}%")
    print(f"  Deep encoder EER:     {pooled_encoder['eer']*100:.2f}%")
    delta = (pooled_encoder['eer'] - pooled_baseline['eer']) * 100
    print(f"  Delta (encoder - classical): {delta:+.2f} percentage points")
    print("  Report this honestly either way -- a classical baseline "
          "beating the deep model on this small a dataset is a legitimate, "
          "reportable finding, not a failure to fix.")

    out = {
        "held_out_subjects": held_out_ids,
        "background_subjects": background_ids,
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
```

Run with: `python -m src.run_week3` (from the Colab cell, inside `PROJECT_DIR`).

## 15. `tests/test_subject_split.py` — full code

```python
import numpy as np
import pytest
from src import subject_split as ss


def test_split_creates_no_overlap_and_correct_proportions(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "SPLIT_PATH", tmp_path / "split.json")
    fake_subjects = np.array([f"s{i:03d}" for i in range(51) for _ in range(8)])
    background, held_out = ss.create_or_load_subject_split(fake_subjects)
    assert len(background) + len(held_out) == 51
    assert len(set(background) & set(held_out)) == 0
    assert len(background) == round(51 * 0.70)


def test_split_is_deterministic_across_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "SPLIT_PATH", tmp_path / "split.json")
    fake_subjects = np.array([f"s{i:03d}" for i in range(51)])
    bg1, ho1 = ss.create_or_load_subject_split(fake_subjects)
    bg2, ho2 = ss.create_or_load_subject_split(fake_subjects)  # loads from disk
    assert bg1 == bg2
    assert ho1 == ho2


def test_split_detects_stale_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "SPLIT_PATH", tmp_path / "split.json")
    original_subjects = np.array([f"s{i:03d}" for i in range(51)])
    ss.create_or_load_subject_split(original_subjects)
    different_subjects = np.array([f"s{i:03d}" for i in range(45)])
    with pytest.raises(AssertionError):
        ss.create_or_load_subject_split(different_subjects)
```

## 16. `tests/test_train_encoder_safety.py` — full code (proves the NaN safety net actually works)

```python
"""
Confirms train_encoder actually raises on non-finite loss rather than
silently continuing. This deliberately forces the failure condition --
it is testing the safety net itself, not trying to train a good model.
"""
import numpy as np
import pytest
from src.train_encoder import train_encoder


def test_train_encoder_raises_on_exploding_lr():
    rng = np.random.RandomState(0)
    n_keys = 4
    subjects = np.array([f"s{i}" for i in range(3) for _ in range(10)])
    sequence_data = rng.normal(size=(30, n_keys, 3)).astype(np.float32)
    background_ids = ["s0", "s1", "s2"]

    with pytest.raises(RuntimeError, match="non-finite"):
        train_encoder(
            sequence_data, subjects, background_ids,
            device="cpu", epochs=5, steps_per_epoch=5,
            batch_size=8, lr=1e6, seed=0,  # absurd LR reliably forces divergence
        )
```

## 17. Exact command sequence (run inside the Colab notebook, `PROJECT_DIR`)

```bash
pytest tests/test_key_sequence.py tests/test_subject_split.py tests/test_train_encoder_safety.py -v
python -m src.device
python -m src.run_week3
git add -A && git commit -m "Week 3: deep sequence encoder, background/held-out split, fair baseline comparison"
git tag week03
git push origin main --tags
```

## 18. Verification checklist — all of these must pass before Week 4 starts

- [ ] `pytest tests/test_key_sequence.py -v` — all 4 tests pass, **especially** `test_reconstruct_key_sequence_order_correct`. If this fails against real data when run, stop immediately; nothing downstream can be trusted.
- [ ] `python -m src.device` prints `cuda` on Colab (confirm it says `cuda`, not `cpu` — if it says `cpu`, the Colab runtime isn't set to GPU).
- [ ] Training loss in `training_history.json` decreases roughly monotonically over the 50 epochs, with no `NaN`/`Inf` values (if it had hit one, `train_encoder` would have raised `RuntimeError` and the script would have stopped — so a completed run already implies no NaN, but check the actual printed per-epoch values anyway).
- [ ] Embedding sanity check reports an inter/intra distance ratio **clearly above 1.0** — roughly 1.5x or higher is healthy. A ratio near 1.0 means stop and do not trust the EER below, regardless of what it says.
- [ ] The held-out encoder EER and the recomputed held-out-only classical baseline EER are both printed and saved — **report both **whichever is better**, don't just report the deep model's number in isolation.
- [ ] `results/week3/subject_split.json`, `training_history.json`, `week3_full_results.json`, and `encoder_weights.pt` all exist.
- [ ] The reconstructed key order printed by `run_week3.py` is a plausible reading of `.tie5Roanl` + Return (sanity-check it by eye against the known password before trusting anything downstream).

## 19. What to send back to Claude at the end of this week

Paste or attach: (1) the full console output of the three test files, `python -m src.device`, and `python -m src.run_week3`; (2) `training_history.json` (or just describe the loss trend if it's long); (3) the embedding sanity-check ratio; (4) the fair-comparison block specifically — both EER numbers and the delta, printed side by side; (5) the printed reconstructed key order. I'll check the key order against the known password by eye, confirm the loss curve is healthy, check the embedding ratio, and look at whether the deep model actually beat the held-out-only classical baseline or not — either answer is fine, but I want to see the honest number, not just a claim that it worked.
