"""
The Frog-Boiling-style gradual poisoning attack: crafts a sequence of
candidates that linearly interpolate from victim-like to attacker-like
real feature vectors, with the blend weight increasing round by round.
Uses the attacker's real CMU data as the target -- Frog-Boiling's
actual mechanism is pacing, not fabrication.

Note on rng: this function requires a numpy Generator (np.random.default_rng),
NOT a numpy RandomState, because it uses rng.integers(). See history.md
Week 4 entry for the reason this differs from the RandomState used elsewhere.
"""
import numpy as np


def craft_poisoning_sequence(victim_samples, attacker_samples, n_rounds, rng):
    """
    victim_samples: (n_victim, n_features) victim's own enrollment,
                    used as interpolation anchors.
    attacker_samples: (n_attacker, n_features) attacker's real data,
                    used as the interpolation target.
    Returns (candidates: (n_rounds, n_features), alphas: (n_rounds,)),
    where alpha is the blend weight used at each round, returned
    explicitly for transparency and logging, not hidden inside the array.
    """
    candidates = []
    alphas = []
    for r in range(1, n_rounds + 1):
        alpha = r / n_rounds
        victim_anchor = victim_samples[rng.integers(0, len(victim_samples))]
        attacker_anchor = attacker_samples[rng.integers(0, len(attacker_samples))]
        candidate = (1 - alpha) * victim_anchor + alpha * attacker_anchor
        candidates.append(candidate)
        alphas.append(alpha)
    return np.array(candidates), np.array(alphas)
