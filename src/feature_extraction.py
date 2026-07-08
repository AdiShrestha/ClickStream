"""
Week 1 feature extraction. Two responsibilities:
1. Reshape the CMU dataset's pre-computed H/DD/UD columns into a clean
   (X, subjects, sessions, feature_names) tuple.
2. Provide a raw keydown/keyup event-stream extractor using the IDENTICAL
   feature semantics (Hold, Down-Down, Up-Down), for use in Weeks 6-7 on
   self-collected data, so a model trained on CMU-derived features can
   score self-collected data on a like-for-like basis.

Feature naming convention (matches CMU column names exactly):
  H.<key>           hold time for a single key (keyup minus keydown)
  DD.<key1>.<key2>  down-down digraph time (next keydown minus current keydown)
  UD.<key1>.<key2>  up-down digraph time (next keydown minus current keyup)

All values in seconds throughout, matching the CMU dataset's units.
"""
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd


def load_cmu_features(
    path: str = "data/raw/DSL-StrongPasswordData.csv",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load and reshape the CMU Keystroke Dynamics Benchmark CSV into arrays.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features), dtype float64
    subjects : ndarray of shape (n_samples,), subject ID strings
    sessions : ndarray of shape (n_samples,), session index integers
    feature_cols : list of str, column names in the same order as X columns
    """
    df = pd.read_csv(path)
    hold_cols = sorted(c for c in df.columns if c.startswith("H."))
    dd_cols = sorted(c for c in df.columns if c.startswith("DD."))
    ud_cols = sorted(c for c in df.columns if c.startswith("UD."))
    feature_cols = hold_cols + dd_cols + ud_cols

    X = df[feature_cols].to_numpy(dtype=float)
    subjects = df["subject"].to_numpy()
    sessions = df["sessionIndex"].to_numpy()
    return X, subjects, sessions, feature_cols


def extract_features_from_raw_events(
    events: List[Tuple[str, str, float]],
) -> Dict[str, float]:
    """
    Extract H/DD/UD features from a raw keydown/keyup event stream.

    Parameters
    ----------
    events : list of (key_id, action, timestamp_seconds) tuples.
             action must be either 'down' or 'up'.
             Events need not be pre-sorted; internal sorting is applied.

    Returns
    -------
    dict mapping feature name to value in seconds, using the same H./DD./UD.
    naming convention as the CMU dataset columns. Returns empty dict for
    empty or fully-unpaired input.
    """
    features: Dict[str, float] = {}
    downs: Dict[str, float] = {}
    sequence: List[Tuple[str, float, float]] = []

    for key_id, action, t in sorted(events, key=lambda e: e[2]):
        if action == "down":
            downs[key_id] = t
        elif action == "up" and key_id in downs:
            sequence.append((key_id, downs.pop(key_id), t))

    sequence.sort(key=lambda x: x[1])

    for i, (key_id, down_t, up_t) in enumerate(sequence):
        features[f"H.{key_id}"] = up_t - down_t
        if i + 1 < len(sequence):
            next_key, next_down_t, _ = sequence[i + 1]
            features[f"DD.{key_id}.{next_key}"] = next_down_t - down_t
            features[f"UD.{key_id}.{next_key}"] = next_down_t - up_t

    return features
