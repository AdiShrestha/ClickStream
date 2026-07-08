"""
Verifies reconstruct_key_sequence_order and build_sequence_features
against a SYNTHETIC example with a KNOWN correct answer, deliberately
including a key name with an embedded dot Shift.r to prove the
parser handles it. This is exactly the case that would silently
break a naive split-on-dots implementation.
"""
import numpy as np
from src.key_sequence import (
    parse_dd_column,
    reconstruct_key_sequence_order,
    build_sequence_features,
)

# Synthetic 4-key password: a -> b -> Shift.r -> c
# mirrors the real CMU case where Shift.r contains an embedded dot
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
    # Row 0: distinguishable values per column to verify exact placement
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
