"""
Splits the 51 subjects into a background set (trains the encoder) and
a held-out set (evaluation only, never seen during training). Saved to
disk on first creation so the split is FIXED across every subsequent
run in this project. Regenerating a different random split between
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
            "dataset's subject list. STOP do not silently regenerate "
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
