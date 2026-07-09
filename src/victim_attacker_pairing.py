"""
Assigns each of the 51 subjects a victim role and a different,
randomly-selected subject as their attacker. Fixed seed, no
self-pairing, persisted to disk so the pairing cannot silently change
between runs -- same discipline as Week 3's subject_split.json.
"""
import json
from pathlib import Path
import numpy as np

PAIRING_PATH = Path("results/week4/victim_attacker_pairs.json")
PAIRING_SEED = 42


def create_or_load_pairing(all_subjects):
    unique_subjects = sorted(np.unique(all_subjects).tolist())

    if PAIRING_PATH.exists():
        with open(PAIRING_PATH) as f:
            pairs = json.load(f)
        assert set(pairs.keys()) == set(unique_subjects), (
            "Existing pairing file does not match the current subject "
            "list. STOP -- do not silently regenerate."
        )
        print(f"Loaded existing victim-attacker pairing from {PAIRING_PATH}")
        return pairs

    rng = np.random.RandomState(PAIRING_SEED)
    pairs = {}
    for victim in unique_subjects:
        candidates = [s for s in unique_subjects if s != victim]
        attacker = candidates[rng.randint(0, len(candidates))]
        pairs[victim] = attacker

    PAIRING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PAIRING_PATH, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Created new victim-attacker pairing, saved to {PAIRING_PATH}")
    return pairs
