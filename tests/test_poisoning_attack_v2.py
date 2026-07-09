import numpy as np
import pytest
from src.poisoning_attack_v2 import craft_poisoning_sequence_meanshift


def test_v2_alpha_increases_monotonically():
    rng = np.random.default_rng(0)
    victim = np.zeros((20, 3))
    attacker = np.full((20, 3), 10.0)
    _, alphas = craft_poisoning_sequence_meanshift(victim, attacker, n_rounds=10, rng=rng)
    assert np.all(np.diff(alphas) > 0)
    assert alphas[-1] == 1.0


def test_v2_candidates_center_drifts_toward_attacker():
    rng = np.random.default_rng(0)
    victim = np.zeros((200, 3))  # zero std -- isolates the CENTER shift from noise
    attacker = np.full((200, 3), 10.0)
    candidates, _ = craft_poisoning_sequence_meanshift(victim, attacker, n_rounds=10, rng=rng)
    # With zero victim std, noise term is exactly zero (victim_std + 1e-8 is tiny),
    # so candidates land essentially on the interpolated mean at each round.
    assert np.allclose(candidates[0], 1.0, atol=1e-5)   # alpha=0.1 -> 0.1*10
    assert np.allclose(candidates[-1], 10.0, atol=1e-5)  # alpha=1.0 -> fully attacker


def test_v2_uses_victim_std_not_attacker_std():
    rng = np.random.default_rng(1)
    # victim has tiny std; attacker has large std (5.0)
    victim = np.random.default_rng(2).normal(0, 0.01, size=(100, 3))
    attacker = np.random.default_rng(3).normal(10, 5.0, size=(100, 3))
    # Use n_rounds=1 so there is only one candidate; then the spread we
    # see is ONLY the per-candidate noise, not the center-drift across rounds.
    # At round 1 of 1, alpha=1.0 -> center is exactly attacker_mean (~10).
    # The noise is N(0, victim_std) which is tiny (~0.01 per feature).
    single_candidate, _ = craft_poisoning_sequence_meanshift(
        victim, attacker, n_rounds=1, rng=rng
    )
    # single_candidate is (1, 3). Its values should be close to attacker_mean
    # (because alpha=1.0 -> shifted_center = attacker_mean) but with tiny
    # noise from victim_std, not from attacker_std.
    # The max absolute noise is single_candidate - attacker.mean(axis=0).
    noise_magnitude = np.abs(single_candidate[0] - attacker.mean(axis=0))
    assert np.all(noise_magnitude < 0.5), (
        "Noise on a single candidate should be governed by victim's tiny std (~0.01), "
        "not attacker's large std (~5.0). If this fails, victim_std was swapped for "
        "attacker_std in the noise term."
    )



def test_v2_requires_numpy_generator():
    """
    Confirms V2 expects a numpy Generator (np.random.default_rng), not a
    RandomState. V2's rng.normal() would work with either type, but the sweep
    (run_poisoning_sweep.py) passes the same rng to both craft_v2 and craft_v1.
    craft_v1 uses rng.integers() which is Generator-only. Both functions must
    share the Generator contract to prevent silent type mismatches in the sweep.
    This test verifies V2 works with a Generator (positive case) and documents
    that passing RandomState is the wrong interface (negative case left as
    a TypeError at the sweep boundary via craft_v1, not caught here, since
    rng.normal() would not raise on RandomState -- the contract is architectural).
    """
    victim = np.zeros((20, 3))
    attacker = np.full((20, 3), 5.0)

    gen_rng = np.random.default_rng(42)
    candidates, alphas = craft_poisoning_sequence_meanshift(victim, attacker, n_rounds=5, rng=gen_rng)
    assert candidates.shape == (5, 3), "Generator path must produce correct shape"
    assert alphas.shape == (5,), "Generator path must produce correct alpha shape"
    assert isinstance(gen_rng, np.random.Generator), (
        "Test setup error: gen_rng must be a numpy Generator, not RandomState. "
        "If this fails, the test itself is wrong."
    )
