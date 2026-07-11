"""
Week 4 Extension: sweeps both crafting methods (V1 point-interpolation,
V2 mean-shift) across N_ROUNDS in {20, 100, 200}, against the SAME
victim-attacker pairing and benign-drift control as Week 4, so every
row in the final comparison table is apples-to-apples with last week's
already-committed result.
"""
import json
from pathlib import Path
import numpy as np
import time

from src.feature_extraction import load_cmu_features
from src.adaptive_baseline import AdaptiveBaseline
from src.poisoning_attack import craft_poisoning_sequence as craft_v1
from src.poisoning_attack_v2 import craft_poisoning_sequence_meanshift as craft_v2
from src.benign_drift_control import craft_benign_drift_sequence
from src.victim_attacker_pairing import create_or_load_pairing
from src.run_poisoning_experiment import get_subject_sessions, run_scenario_for_victim, compute_victim_eer_threshold

RESULTS_DIR = Path("results/week4_extension")
ROUND_COUNTS = [20, 100, 200]
SWEEP_SEED = 456


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, subjects, sessions, feature_cols = load_cmu_features()
    pairing = create_or_load_pairing(subjects)  # reuses the SAME pairing as Week 4

    sweep_results = {}

    # Timing check for n_rounds=200
    print("Running performance timing check for n_rounds=200...")
    first_victim = list(pairing.keys())[0]
    first_attacker = pairing[first_victim]
    v_en = get_subject_sessions(X, subjects, sessions, first_victim, (1, 2, 3, 4))
    v_la = get_subject_sessions(X, subjects, sessions, first_victim, (5, 6, 7, 8))
    a_en = get_subject_sessions(X, subjects, sessions, first_attacker, (1, 2, 3, 4))
    impostor_test = X[(subjects != first_victim) & np.isin(sessions, (5, 6, 7, 8))]
    eer_threshold = compute_victim_eer_threshold(v_en, v_la, impostor_test)
    timing_rng = np.random.default_rng(999)
    p_seq, _ = craft_v1(v_en, a_en, 200, timing_rng)
    t0 = time.time()
    run_scenario_for_victim(v_en.copy(), p_seq, v_la, a_en, eer_threshold)
    t1 = time.time()
    print(f"Time for 1 victim at n_rounds=200 (attack): {t1 - t0:.3f}s")
    total_est = (t1 - t0) * 51 * 2 * 3 * 2 # Roughly 51 victims * 2 scenarios * 3 round counts * 2 methods
    print(f"Estimated total sweep time (very conservative): {total_est / 60:.2f} minutes")
    # Even if it's 600s, it's 10 mins. We proceed.

    for method_name, craft_fn in [("v1_interpolation", craft_v1), ("v2_meanshift", craft_v2)]:
        for n_rounds in ROUND_COUNTS:
            rng = np.random.default_rng(SWEEP_SEED)  # fresh, independent rng per configuration
            config_key = f"{method_name}_rounds{n_rounds}"
            print(f"\n=== {config_key} ===")

            attack_attacker_deltas, benign_attacker_deltas = [], []
            attack_victim_deltas = []

            for victim, attacker in pairing.items():
                victim_enroll = get_subject_sessions(X, subjects, sessions, victim, (1, 2, 3, 4))
                victim_later = get_subject_sessions(X, subjects, sessions, victim, (5, 6, 7, 8))
                attacker_craft_pool = get_subject_sessions(X, subjects, sessions, attacker, (1, 2))
                attacker_eval_pool = get_subject_sessions(X, subjects, sessions, attacker, (3, 4))

                impostor_test = X[(subjects != victim) & np.isin(sessions, (5, 6, 7, 8))]
                eer_threshold = compute_victim_eer_threshold(victim_enroll, victim_later, impostor_test)

                poison_sequence, _ = craft_fn(victim_enroll, attacker_craft_pool, n_rounds, rng)
                attack_result = run_scenario_for_victim(
                    victim_enroll.copy(), poison_sequence, victim_later, attacker_eval_pool, eer_threshold
                )

                benign_sequence = craft_benign_drift_sequence(victim_later, n_rounds, rng)
                benign_result = run_scenario_for_victim(
                    victim_enroll.copy(), benign_sequence, victim_later, attacker_eval_pool, eer_threshold
                )

                attack_attacker_deltas.append(attack_result["attacker_acceptance_delta"])
                benign_attacker_deltas.append(benign_result["attacker_acceptance_delta"])
                attack_victim_deltas.append(attack_result["victim_acceptance_delta"])

            mean_attack = float(np.mean(attack_attacker_deltas))
            mean_benign = float(np.mean(benign_attacker_deltas))
            gap = mean_attack - mean_benign
            mean_victim_delta = float(np.mean(attack_victim_deltas))

            print(f"  ATTACK mean Δattacker: {mean_attack*100:+.2f}pp "
                  f"(std {np.std(attack_attacker_deltas)*100:.2f}pp)")
            print(f"  BENIGN mean Δattacker: {mean_benign*100:+.2f}pp "
                  f"(std {np.std(benign_attacker_deltas)*100:.2f}pp)")
            print(f"  GAP (attack - benign): {gap*100:+.2f}pp")
            print(f"  ATTACK mean Δvictim's own acceptance: {mean_victim_delta*100:+.2f}pp")

            sweep_results[config_key] = {
                "method": method_name,
                "n_rounds": n_rounds,
                "mean_attack_attacker_delta": mean_attack,
                "mean_benign_attacker_delta": mean_benign,
                "gap": gap,
                "mean_attack_victim_delta": mean_victim_delta,
                "attack_attacker_deltas_raw": attack_attacker_deltas,
                "benign_attacker_deltas_raw": benign_attacker_deltas,
            }

    with open(RESULTS_DIR / "sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2, default=float)

    print("\n=== Summary across all 6 configurations ===")
    print(f"{'Config':<25} {'Attack Δ':>10} {'Benign Δ':>10} {'Gap':>8}")
    for key, r in sweep_results.items():
        print(f"{key:<25} {r['mean_attack_attacker_delta']*100:>+9.2f}pp "
              f"{r['mean_benign_attacker_delta']*100:>+9.2f}pp "
              f"{r['gap']*100:>+7.2f}pp")

    print(
        "\nDecision rule, apply honestly: if ANY configuration shows a gap "
        "clearly and consistently larger than V1/20-round's 0.44pp (say, "
        "materially above the ~8pp noise floor already observed in that "
        "run's std), that configuration is evidence the vulnerability is "
        "real and the crafting method/round-count matters. If NONE do, "
        "report that honestly too -- it means this specific adaptive-"
        "baseline design is more robust to gradual poisoning than the "
        "compendium's literature review would suggest, which is itself "
        "worth a paragraph in the paper, not a result to bury."
    )
    print(f"\nSaved full sweep results to {RESULTS_DIR / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
