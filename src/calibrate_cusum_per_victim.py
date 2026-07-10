"""
Week 5 Extension: calibrates a SEPARATE cusum_h per victim, from that
individual victim's OWN benign drift on the CALIBRATION split
(sessions 5-6) only -- never sessions 7-8, which are reserved for
evaluation. Replaces calibrate_cusum.py's single global threshold,
which was calibrated on the same pool later used to test it, and which
Section 3's earlier data (min=0.13, median=4.00, max=12.62 across
victims) already showed was a poor fit for most individuals regardless
of the leakage.
"""
import numpy as np
from src.feature_extraction import load_cmu_features
from src.benign_drift_control import craft_benign_drift_sequence
from src.victim_attacker_pairing import create_or_load_pairing
from src.run_poisoning_experiment import get_subject_sessions
from src.cusum_defense import DefendedAdaptiveBaseline

N_ROUNDS = 200
CALIBRATION_SEED = 789
N_CALIBRATION_TRIALS = 20  # repeated resampled trials on the small
                           # calibration split, to get a stabler
                           # per-victim distribution than a single pass
                           # over only 100 available reps would give


def calibrate_h_for_victim(victim_calibration_sessions, target_false_alarm_rate=0.05, k=0.0, seed=CALIBRATION_SEED):
    rng = np.random.default_rng(seed)
    max_values = []
    for trial in range(N_CALIBRATION_TRIALS):
        baseline = DefendedAdaptiveBaseline(cusum_k=k, cusum_h=1e9).initialize(
            victim_calibration_sessions.copy()
        )
        sequence = craft_benign_drift_sequence(victim_calibration_sessions, N_ROUNDS, rng)
        for i, candidate in enumerate(sequence):
            baseline.offer_candidate(candidate, round_index=i)
        max_values.append(max(baseline.cusum_history))

    percentile = 100 * (1 - target_false_alarm_rate)
    h = float(np.percentile(max_values, percentile))
    return h, np.array(max_values)


def calibrate_all_victims(target_false_alarm_rate=0.05):
    X, subjects, sessions, feature_cols = load_cmu_features()
    pairing = create_or_load_pairing(subjects)

    per_victim_h = {}
    for victim in pairing:
        calibration_sessions = get_subject_sessions(X, subjects, sessions, victim, (5, 6))
        h, max_values = calibrate_h_for_victim(calibration_sessions, target_false_alarm_rate)
        per_victim_h[victim] = h
        print(f"{victim}: calibration max-CUSUM min={max_values.min():.3f} "
              f"median={np.median(max_values):.3f} max={max_values.max():.3f} -> h={h:.3f}")

    return per_victim_h


if __name__ == "__main__":
    per_victim_h = calibrate_all_victims()
    print(f"\nPer-victim h range: min={min(per_victim_h.values()):.3f}, "
          f"max={max(per_victim_h.values()):.3f} "
          f"(compare against the single global h=9.7498 from Week 5's original run)")
