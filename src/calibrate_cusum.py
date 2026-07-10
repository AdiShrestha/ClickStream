"""
Week 5: calibrates cusum_h using ONLY the benign-drift control across
all 51 victims. The false-alarm rate on genuine drift must be
controlled using non-adversarial data alone -- calibrating against the
attack itself would be circular and would invalidate the defense
evaluation in Section 6.
"""
import numpy as np
from src.feature_extraction import load_cmu_features
from src.benign_drift_control import craft_benign_drift_sequence
from src.victim_attacker_pairing import create_or_load_pairing
from src.run_poisoning_experiment import get_subject_sessions
from src.cusum_defense import DefendedAdaptiveBaseline

N_ROUNDS = 200
CALIBRATION_SEED = 789


def get_max_cusum_under_benign(k=0.0, n_rounds=N_ROUNDS, seed=CALIBRATION_SEED):
    """
    Runs the benign-drift control for all 51 victims with h set
    effectively infinite (so nothing actually gets rejected during
    calibration -- we only want to observe how high the CUSUM statistic
    naturally climbs under pure, non-adversarial drift), and returns
    the maximum value each victim's trajectory reaches.
    """
    rng = np.random.default_rng(seed)
    X, subjects, sessions, feature_cols = load_cmu_features()
    pairing = create_or_load_pairing(subjects)

    max_cusum_values = []
    for victim in pairing:
        victim_enroll = get_subject_sessions(X, subjects, sessions, victim, (1, 2, 3, 4))
        victim_later = get_subject_sessions(X, subjects, sessions, victim, (5, 6, 7, 8))

        baseline = DefendedAdaptiveBaseline(cusum_k=k, cusum_h=1e9).initialize(victim_enroll.copy())
        benign_sequence = craft_benign_drift_sequence(victim_later, n_rounds, rng)
        for i, candidate in enumerate(benign_sequence):
            baseline.offer_candidate(candidate, round_index=i)

        max_cusum_values.append(max(baseline.cusum_history))

    return np.array(max_cusum_values)


def calibrate_h(target_false_alarm_rate=0.05, k=0.0):
    max_values = get_max_cusum_under_benign(k=k)
    percentile = 100 * (1 - target_false_alarm_rate)
    h = float(np.percentile(max_values, percentile))
    print(f"Max-CUSUM distribution under 51 victims' benign drift (k={k}):")
    print(f"  min={max_values.min():.4f}, median={np.median(max_values):.4f}, "
          f"max={max_values.max():.4f}")
    print(f"  Setting h = {percentile:.0f}th percentile = {h:.4f} "
          f"(targets ~{target_false_alarm_rate*100:.0f}% false-alarm rate on benign drift)")
    return h


if __name__ == "__main__":
    h = calibrate_h(target_false_alarm_rate=0.05)
    print(f"\nCalibrated h = {h:.4f}")
