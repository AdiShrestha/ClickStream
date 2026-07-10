import numpy as np
from src.victim_attacker_similarity import compute_pair_similarity


def test_identical_distributions_give_near_zero_distance():
    rng = np.random.RandomState(0)
    shared = rng.normal(0, 1, size=(100, 5))
    victim = shared[:50]
    attacker = shared[50:]
    distance = compute_pair_similarity(victim, attacker)
    assert distance < 0.5, "Two samples from the same distribution should be close"


def test_very_different_distributions_give_large_distance():
    rng = np.random.RandomState(1)
    victim = rng.normal(0, 1, size=(50, 5))
    attacker = rng.normal(20, 1, size=(50, 5))
    distance = compute_pair_similarity(victim, attacker)
    assert distance > 2.0, "Two well-separated distributions should be far apart"
