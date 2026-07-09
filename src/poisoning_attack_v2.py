"""
Week 4 Extension: an alternative Frog-Boiling crafting method. V1
(src/poisoning_attack.py) interpolates directly between one specific
victim point and one specific attacker point, which can land in a
sparse, unrealistic region of feature space when the two people's
typing is genuinely different (this is the mechanism Section 5.2 of
report4.md identifies as the likely reason V1 showed a null effect).

V2 instead samples each candidate from a distribution centered on a
point gradually shifting from the victim's mean toward the attacker's
mean, using the VICTIM's OWN per-feature standard deviation as the
spread. Each individual candidate should look like a plausible draw
from "a slightly recentered version of the victim's own distribution,"
not a literal blend of two different people's specific samples.

Note on rng: requires a numpy Generator (np.random.default_rng),
NOT a numpy RandomState. rng.normal() exists on both, but the sweep
(run_poisoning_sweep.py) passes the same rng to both craft_v1 and
craft_v2, and craft_v1 uses rng.integers() which is Generator-only.
V2 must share the Generator contract to keep the sweep's single rng
usable for both methods without silent type mismatches.
"""
import numpy as np


def craft_poisoning_sequence_meanshift(victim_samples, attacker_samples, n_rounds, rng):
    """
    victim_samples: (n_victim, n_features) victim enrollment data,
                    mean and std computed from this.
    attacker_samples: (n_attacker, n_features) attacker enrollment,
                    only the mean is used; victim std governs spread.
    n_rounds: number of candidates to generate.
    rng: numpy Generator (np.random.default_rng), NOT RandomState.
    Returns (candidates: (n_rounds, n_features), alphas: (n_rounds,)).
    """
    victim_mean = victim_samples.mean(axis=0)
    victim_std = victim_samples.std(axis=0) + 1e-8  # avoid zero-std features
    attacker_mean = attacker_samples.mean(axis=0)

    candidates = []
    alphas = []
    for r in range(1, n_rounds + 1):
        alpha = r / n_rounds
        shifted_center = (1 - alpha) * victim_mean + alpha * attacker_mean
        noise = rng.normal(0, 1, size=victim_mean.shape) * victim_std
        candidate = shifted_center + noise
        candidates.append(candidate)
        alphas.append(alpha)
    return np.array(candidates), np.array(alphas)
