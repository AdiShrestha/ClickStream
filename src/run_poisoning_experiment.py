"""
Main Week 4 experiment. For all 51 subjects as victims (fixed, persisted
victim-attacker pairing), runs BOTH the Frog-Boiling attack and the
benign-drift control against the SAME adaptive-baseline mechanism, and
reports both side by side for every victim -- never one without the
other, per Section 3's design requirement.
"""
import json
from pathlib import Path
import numpy as np

from src.feature_extraction import load_cmu_features
from src.adaptive_baseline import AdaptiveBaseline
from src.poisoning_attack import craft_poisoning_sequence
from src.benign_drift_control import craft_benign_drift_sequence
from src.victim_attacker_pairing import create_or_load_pairing
from src.metrics import compute_eer
from src.models import PerUserModel

RESULTS_DIR = Path("results/week4")
N_ROUNDS = 20
EXPERIMENT_SEED = 123


def get_subject_sessions(X, subjects, sessions, subject_id, session_ids):
    mask = (subjects == subject_id) & np.isin(sessions, session_ids)
    return X[mask]

def compute_victim_eer_threshold(victim_enroll, victim_genuine_test, impostor_test, algorithm="isolation_forest"):
    model = PerUserModel(algorithm=algorithm).fit(victim_enroll)
    genuine_scores = model.score(victim_genuine_test)
    impostor_scores = model.score(impostor_test)
    _, eer_threshold = compute_eer(genuine_scores, impostor_scores)
    return eer_threshold


def run_scenario_for_victim(initial_enrollment, poison_sequence, victim_genuine_test,
                             attacker_genuine_samples, eer_threshold, algorithm="isolation_forest"):
    baseline = AdaptiveBaseline(algorithm=algorithm).initialize(initial_enrollment)

    scores_before_victim = baseline.score(victim_genuine_test)
    scores_before_attacker = baseline.score(attacker_genuine_samples)
    accept_rate_before_victim = float((scores_before_victim > eer_threshold).mean())
    accept_rate_before_attacker = float((scores_before_attacker > eer_threshold).mean())

    n_absorbed = 0
    for i, candidate in enumerate(poison_sequence):
        result = baseline.offer_candidate(candidate, round_index=i)
        if result.absorbed:
            n_absorbed += 1

    scores_after_victim = baseline.score(victim_genuine_test)
    scores_after_attacker = baseline.score(attacker_genuine_samples)
    accept_rate_after_victim = float((scores_after_victim > eer_threshold).mean())
    accept_rate_after_attacker = float((scores_after_attacker > eer_threshold).mean())

    return {
        "n_rounds": len(poison_sequence),
        "n_absorbed": n_absorbed,
        "accept_rate_victim_before": accept_rate_before_victim,
        "accept_rate_victim_after": accept_rate_after_victim,
        "accept_rate_attacker_before": accept_rate_before_attacker,
        "accept_rate_attacker_after": accept_rate_after_attacker,
        "attacker_acceptance_delta": accept_rate_after_attacker - accept_rate_before_attacker,
        "victim_acceptance_delta": accept_rate_after_victim - accept_rate_before_victim,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(EXPERIMENT_SEED)

    X, subjects, sessions, feature_cols = load_cmu_features()
    pairing = create_or_load_pairing(subjects)

    all_results = {}
    for victim, attacker in pairing.items():
        victim_enroll = get_subject_sessions(X, subjects, sessions, victim, (1, 2, 3, 4))
        victim_later = get_subject_sessions(X, subjects, sessions, victim, (5, 6, 7, 8))
        attacker_enroll = get_subject_sessions(X, subjects, sessions, attacker, (1, 2, 3, 4))

        impostor_test = X[(subjects != victim) & np.isin(sessions, (5, 6, 7, 8))]
        eer_threshold = compute_victim_eer_threshold(victim_enroll, victim_later, impostor_test)

        poison_sequence, alphas = craft_poisoning_sequence(
            victim_enroll, attacker_enroll, N_ROUNDS, rng
        )
        attack_result = run_scenario_for_victim(
            victim_enroll.copy(), poison_sequence, victim_later, attacker_enroll, eer_threshold
        )

        benign_sequence = craft_benign_drift_sequence(victim_later, N_ROUNDS, rng)
        benign_result = run_scenario_for_victim(
            victim_enroll.copy(), benign_sequence, victim_later, attacker_enroll, eer_threshold
        )

        all_results[victim] = {
            "attacker": attacker,
            "attack": attack_result,
            "benign_control": benign_result,
        }
        print(f"{victim} (attacker={attacker}): "
              f"attack Δattacker={attack_result['attacker_acceptance_delta']:+.3f} "
              f"Δvictim={attack_result['victim_acceptance_delta']:+.3f} | "
              f"benign Δattacker={benign_result['attacker_acceptance_delta']:+.3f} "
              f"Δvictim={benign_result['victim_acceptance_delta']:+.3f}")

    attack_attacker_deltas = [r["attack"]["attacker_acceptance_delta"] for r in all_results.values()]
    benign_attacker_deltas = [r["benign_control"]["attacker_acceptance_delta"] for r in all_results.values()]
    attack_victim_deltas = [r["attack"]["victim_acceptance_delta"] for r in all_results.values()]
    benign_victim_deltas = [r["benign_control"]["victim_acceptance_delta"] for r in all_results.values()]

    print("\n=== Aggregate across all 51 victims ===")
    print(f"ATTACK -- mean Δ attacker acceptance: {np.mean(attack_attacker_deltas)*100:+.2f}pp "
          f"(std {np.std(attack_attacker_deltas)*100:.2f}pp)")
    print(f"BENIGN -- mean Δ attacker acceptance: {np.mean(benign_attacker_deltas)*100:+.2f}pp "
          f"(std {np.std(benign_attacker_deltas)*100:.2f}pp)")
    print(f"ATTACK -- mean Δ victim's own acceptance: {np.mean(attack_victim_deltas)*100:+.2f}pp")
    print(f"BENIGN -- mean Δ victim's own acceptance: {np.mean(benign_victim_deltas)*100:+.2f}pp")
    print(
        "\nThe finding this week rests on ATTACK's attacker-acceptance delta "
        "being MEANINGFULLY LARGER than BENIGN's. If they're similar, the "
        "system isn't specifically vulnerable to adversarial poisoning, it's "
        "just generally sensitive to any absorbed data -- a different, "
        "weaker finding. Report whichever is actually true, do not adjust "
        "N_ROUNDS or the absorption_percentile until the gap looks better."
    )

    with open(RESULTS_DIR / "poisoning_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved full results to {RESULTS_DIR / 'poisoning_results.json'}")


if __name__ == "__main__":
    main()
