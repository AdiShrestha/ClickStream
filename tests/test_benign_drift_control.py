import numpy as np
from src.benign_drift_control import craft_benign_drift_sequence


def test_benign_sequence_only_uses_victims_own_samples():
    rng = np.random.RandomState(0)
    victim_later = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sequence = craft_benign_drift_sequence(victim_later, n_rounds=5, rng=rng)
    assert sequence.shape == (5, 2)
    for row in sequence:
        assert any(np.allclose(row, v) for v in victim_later), (
            "Benign control must only ever draw from the victim's own "
            "provided samples -- never fabricate or blend with anything else, "
            "which is exactly what would distinguish it from the attack."
        )


def test_benign_sequence_replace_true_branch():
    """
    RISK HIGH guard: exercises n_rounds > n_available to confirm the
    replace=True branch returns the correct shape and stays within
    victim samples. This branch is never triggered with N_ROUNDS=20 and
    200 later-session rows, but must be correct in case N_ROUNDS grows.
    """
    rng = np.random.RandomState(7)
    victim_later = np.array([[10.0, 20.0], [30.0, 40.0]])
    sequence = craft_benign_drift_sequence(victim_later, n_rounds=10, rng=rng)
    assert sequence.shape == (10, 2), "Shape must be (n_rounds, n_features) even with replacement"
    for row in sequence:
        assert any(np.allclose(row, v) for v in victim_later), (
            "Even with replacement, must only draw from victim_later"
        )
