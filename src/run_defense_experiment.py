"""
Week 5 main experiment: evaluates the calibrated CUSUM defense against
the CONFIRMED working attack (V1 point-interpolation, N_ROUNDS=200,
victim-specific EER thresholds -- exactly the configuration validated
across the Week 4 extension checkpoints) and against the benign-drift
control, for all 51 victims, defended vs undefended side by side.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

from src.feature_extraction import load_cmu_features
from src.metrics import compute_eer
from src.models import PerUserModel
from src.poisoning_attack import craft_poisoning_sequence
from src.benign_drift_control import craft_benign_drift_sequence
from src.victim_attacker_pairing import create_or_load_pairing
from src.run_poisoning_experiment import get_subject_sessions
from src.adaptive_baseline import AdaptiveBaseline
from src.cusum_defense import DefendedAdaptiveBaseline
from src.calibrate_cusum import calibrate_h

RESULTS_DIR = Path("results/week5")
N_ROUNDS = 200
EXPERIMENT_SEED = 123  # SAME seed as the confirmed Week 4-extension run


def compute_victim_eer_threshold(victim_enroll, victim_genuine_test, impostor_test, algorithm="isolation_forest"):
    model = PerUserModel(algorithm=algorithm).fit(victim_enroll)
    genuine_scores = model.score(victim_genuine_test)
    impostor_scores = model.score(impostor_test)
    _, eer_threshold = compute_eer(genuine_scores, impostor_scores)
    return eer_threshold


def run_one_scenario(baseline_cls, baseline_kwargs, initial_enrollment, sequence,
                      eer_threshold, attacker_genuine_samples):
    baseline = baseline_cls(**baseline_kwargs).initialize(initial_enrollment.copy())
    scores_before = baseline.score(attacker_genuine_samples)
    accept_before = float((scores_before > eer_threshold).mean())

    for i, candidate in enumerate(sequence):
        baseline.offer_candidate(candidate, round_index=i)

    scores_after = baseline.score(attacker_genuine_samples)
    accept_after = float((scores_after > eer_threshold).mean())

    n_triggers = len(getattr(baseline, "defense_triggered_rounds", []))
    return {
        "accept_before": accept_before,
        "accept_after": accept_after,
        "delta": accept_after - accept_before,
        "n_defense_triggers": n_triggers,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, subjects, sessions, feature_cols = load_cmu_features()
    pairing = create_or_load_pairing(subjects)

    print("=== Calibrating CUSUM threshold on benign-only data ===")
    cusum_h = calibrate_h(target_false_alarm_rate=0.05)
    cusum_k = 0.0

    results = {}
    for victim, attacker in pairing.items():
        rng = np.random.default_rng(EXPERIMENT_SEED)

        victim_enroll = get_subject_sessions(X, subjects, sessions, victim, (1, 2, 3, 4))
        victim_later = get_subject_sessions(X, subjects, sessions, victim, (5, 6, 7, 8))
        attacker_enroll = get_subject_sessions(X, subjects, sessions, attacker, (1, 2, 3, 4))
        impostor_pool = X[(subjects != victim) & np.isin(sessions, (5, 6, 7, 8))]

        eer_threshold = compute_victim_eer_threshold(victim_enroll, victim_later, impostor_pool)

        poison_sequence, _ = craft_poisoning_sequence(victim_enroll, attacker_enroll, N_ROUNDS, rng)
        benign_sequence = craft_benign_drift_sequence(victim_later, N_ROUNDS, rng)

        undefended_attack = run_one_scenario(
            AdaptiveBaseline, {}, victim_enroll, poison_sequence, eer_threshold, attacker_enroll
        )
        defended_attack = run_one_scenario(
            DefendedAdaptiveBaseline, {"cusum_k": cusum_k, "cusum_h": cusum_h},
            victim_enroll, poison_sequence, eer_threshold, attacker_enroll
        )
        undefended_benign = run_one_scenario(
            AdaptiveBaseline, {}, victim_enroll, benign_sequence, eer_threshold, attacker_enroll
        )
        defended_benign = run_one_scenario(
            DefendedAdaptiveBaseline, {"cusum_k": cusum_k, "cusum_h": cusum_h},
            victim_enroll, benign_sequence, eer_threshold, attacker_enroll
        )

        results[victim] = {
            "attacker": attacker,
            "eer_threshold": eer_threshold,
            "undefended_attack": undefended_attack,
            "defended_attack": defended_attack,
            "undefended_benign": undefended_benign,
            "defended_benign": defended_benign,
        }

        print(f"{victim}: undef_attack Δ={undefended_attack['delta']*100:+.2f}pp | "
              f"def_attack Δ={defended_attack['delta']*100:+.2f}pp "
              f"(triggers={defended_attack['n_defense_triggers']}/{N_ROUNDS}) | "
              f"def_benign Δ={defended_benign['delta']*100:+.2f}pp "
              f"(triggers={defended_benign['n_defense_triggers']}/{N_ROUNDS})")

    ua = [r["undefended_attack"]["delta"] for r in results.values()]
    da = [r["defended_attack"]["delta"] for r in results.values()]
    ub = [r["undefended_benign"]["delta"] for r in results.values()]
    db = [r["defended_benign"]["delta"] for r in results.values()]
    triggers_attack = [r["defended_attack"]["n_defense_triggers"] for r in results.values()]
    triggers_benign = [r["defended_benign"]["n_defense_triggers"] for r in results.values()]

    print("\n=== Aggregate across 51 victims (mean Δ attacker acceptance) ===")
    print(f"UNDEFENDED attack: {np.mean(ua)*100:+.2f}pp (std {np.std(ua)*100:.2f}pp)")
    print(f"DEFENDED   attack: {np.mean(da)*100:+.2f}pp (std {np.std(da)*100:.2f}pp)")
    print(f"UNDEFENDED benign: {np.mean(ub)*100:+.2f}pp (std {np.std(ub)*100:.2f}pp)")
    print(f"DEFENDED   benign: {np.mean(db)*100:+.2f}pp (std {np.std(db)*100:.2f}pp)")

    t_stat, p_value = stats.ttest_rel(ua, da)
    print(f"\nPaired t-test (undefended vs defended, ATTACK scenario, n=51): "
          f"t={t_stat:.3f}, p={p_value:.5f}")

    print(f"\nDefense trigger rate -- ATTACK scenario: mean {np.mean(triggers_attack):.1f}/{N_ROUNDS} rounds")
    print(f"Defense trigger rate -- BENIGN scenario: mean {np.mean(triggers_benign):.1f}/{N_ROUNDS} rounds "
          f"(this IS the empirical false-alarm rate on genuine drift -- compare "
          f"against the ~5% target used during calibration)")

    with open(RESULTS_DIR / "defense_results.json", "w") as f:
        json.dump(
            {"cusum_h": cusum_h, "cusum_k": cusum_k, "per_victim": results},
            f, indent=2, default=float,
        )
    print(f"\nSaved full results to {RESULTS_DIR / 'defense_results.json'}")


if __name__ == "__main__":
    main()
