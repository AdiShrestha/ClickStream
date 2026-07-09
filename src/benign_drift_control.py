"""
The non-adversarial control: candidates drawn from the victim's own
later sessions (5-8), not from an attacker. Real, naturally-occurring
variation from the same person -- including whatever heavy-tailed
outliers Week 1 confirmed exist -- rather than another person's
behavior. Run through the identical adaptation mechanism as the attack,
so the only thing that differs between scenarios is whose data it is.
"""
import numpy as np


def craft_benign_drift_sequence(victim_later_sessions, n_rounds, rng):
    n_available = len(victim_later_sessions)
    if n_rounds <= n_available:
        indices = rng.choice(n_available, size=n_rounds, replace=False)
    else:
        indices = rng.choice(n_available, size=n_rounds, replace=True)
    return victim_later_sessions[indices]
