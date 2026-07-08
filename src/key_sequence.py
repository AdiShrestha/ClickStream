"""
Week 3 reconstructs the true chronological key-press order for the
CMU password (.tie5Roanl plus Return, 11 key positions) directly from
the DD column names, which encode from_key.to_key transitions.

CRITICAL do NOT use sorted column names for this purpose. That gives
alphabetical order which the classical baseline Week 2 uses because it
does not care about order, but it is NOT the order the password was
actually typed in, and using it here would silently feed the LSTM a
scrambled meaningless sequence while still producing a plausible-looking
loss curve and EER. The bug would not announce itself.

This reconstruction is data-driven parsed from whatever column names
actually exist, not hardcoded against a guessed password spelling,
specifically so it keeps working even if this project is later pointed
at a different fixed-password dataset.
"""
from typing import List, Tuple, Dict


def parse_dd_column(dd_col_name: str, valid_key_names: set) -> Tuple[str, str]:
    """
    dd_col_name: e.g. DD.period.t or DD.five.Shift.r
    valid_key_names: the set of key names parsed from H. column suffixes
                     e.g. period, t, i, five, Shift.r

    Splits robustly against the known valid key-name set rather than
    guessing based on dot positions. This correctly handles key names
    that themselves contain a literal dot like Shift.r.
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
        f"{sorted(valid_key_names)}. STOP do not guess a split; "
        "this means the column-naming assumption is wrong and needs "
        "to be understood before proceeding."
    )


def reconstruct_key_sequence_order(feature_cols: List[str]) -> List[str]:
    """
    Returns the list of key names in TRUE chronological typing order,
    reconstructed from the DD-column transition chain, not alphabetical
    sort. Raises immediately and loudly if the chain does not form a
    single clean sequence covering every key exactly once.
    """
    h_cols = [c for c in feature_cols if c.startswith("H.")]
    valid_key_names = set(c[len("H."):] for c in h_cols)
    dd_cols = [c for c in feature_cols if c.startswith("DD.")]

    assert len(dd_cols) == len(valid_key_names) - 1, (
        f"Expected {len(valid_key_names) - 1} DD columns (one fewer than "
        f"the {len(valid_key_names)} keys), found {len(dd_cols)}. "
        "STOP the chain-reconstruction logic assumes exactly "
        "this relationship between key count and transition count."
    )

    transitions: Dict[str, str] = {}
    to_keys_seen = set()
    for dd_col in dd_cols:
        from_key, to_key = parse_dd_column(dd_col, valid_key_names)
        assert from_key not in transitions, (
            f"Key '{from_key}' appears as the source of more than one "
            "transition, the chain is not linear. STOP and investigate "
            "rather than picking one arbitrarily."
        )
        transitions[from_key] = to_key
        to_keys_seen.add(to_key)

    first_key_candidates = valid_key_names - to_keys_seen
    assert len(first_key_candidates) == 1, (
        f"Expected exactly one starting key (a key that is never a "
        f"transition target), found {first_key_candidates}. STOP "
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
        f"{len(valid_key_names)}, the chain is broken, has a cycle, or "
        "does not reach every key. STOP and investigate before proceeding; "
        "do not truncate or pad the sequence to force a match."
    )
    assert set(ordered_keys) == valid_key_names, (
        "Reconstructed chain does not cover every known key exactly "
        "once. STOP do not proceed with a partial or duplicated chain."
    )

    return ordered_keys


def build_sequence_features(X, feature_cols: List[str], ordered_keys: List[str]):
    """
    Reshapes the flat (n_samples, 31) feature array into
    (n_samples, n_keys, 3), where axis 1 is TRUE chronological key
    position and axis 2 is hold_time, digraph_dd, digraph_ud,
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
