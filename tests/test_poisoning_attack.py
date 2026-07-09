import numpy as np
from src.poisoning_attack import craft_poisoning_sequence


def test_alpha_increases_monotonically():
    rng = np.random.default_rng(0)
    victim = np.zeros((10, 3))
    attacker = np.ones((10, 3))
    _, alphas = craft_poisoning_sequence(victim, attacker, n_rounds=10, rng=rng)
    assert np.all(np.diff(alphas) > 0), "Alpha must strictly increase round over round"
    assert alphas[0] < 0.2, "First round should be mostly victim-like"
    assert alphas[-1] == 1.0, "Final round should be fully attacker-like"


def test_candidates_drift_from_victim_toward_attacker():
    rng = np.random.default_rng(0)
    victim = np.zeros((10, 3))
    attacker = np.full((10, 3), 10.0)
    candidates, _ = craft_poisoning_sequence(victim, attacker, n_rounds=10, rng=rng)

    dist_to_victim_round0 = np.linalg.norm(candidates[0] - 0.0)
    dist_to_attacker_round0 = np.linalg.norm(candidates[0] - 10.0)
    assert dist_to_victim_round0 < dist_to_attacker_round0, (
        "Round 0 candidate should be closer to victim than attacker"
    )

    dist_to_victim_last = np.linalg.norm(candidates[-1] - 0.0)
    dist_to_attacker_last = np.linalg.norm(candidates[-1] - 10.0)
    assert dist_to_attacker_last < dist_to_victim_last, (
        "Final candidate should be closer to attacker than victim"
    )


def test_craft_accepts_numpy_generator_not_random_state():
    """
    RISK HIGH guard. craft_poisoning_sequence uses rng.integers() which
    requires a numpy Generator, NOT a RandomState. This test explicitly
    confirms the function works with np.random.default_rng and fails
    with np.random.RandomState so Phase 2 does not accidentally revert
    run_poisoning_experiment.py to use RandomState for this function.
    """
    rng_generator = np.random.default_rng(42)
    victim = np.zeros((5, 3))
    attacker = np.ones((5, 3))
    candidates, alphas = craft_poisoning_sequence(victim, attacker, n_rounds=5, rng=rng_generator)
    assert candidates.shape == (5, 3)

    import pytest
    rng_state = np.random.RandomState(42)
    with pytest.raises(AttributeError):
        craft_poisoning_sequence(victim, attacker, n_rounds=5, rng=rng_state)
