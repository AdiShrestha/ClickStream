"""
Week 5 Extension main experiment. Uses sessions 7-8 (disjoint from the
5-6 calibration split) for both the EER threshold and the benign-drift
evaluation sequence. Uses each victim's OWN per-victim threshold
instead of one global h. Computes victim-attacker similarity for every
pair. Reports results STRATIFIED by undefended attack severity, not
just one pooled mean -- so the six-victim pattern from the original
Week 5 checkpoint is visible in the output, not averaged away.
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
from src.calibrate_cusum_per_victim import calibrate_all_victims
from src.victim_attacker_similarity import compute_pair_similarity

RESULTS_DIR = Path("results/week5_extension")
N_ROUNDS = 200
EXPERIMENT_SEED = 123  # unchanged, for comparability with prior checkpoints


def compute_victim_eer_threshold(victim_enroll, victim_eval_genuine, impostor_pool, algorithm="isolation_forest"):
    model = PerUserModel(algorithm=algorithm).fit(victim_enroll)
    genuine_scores = model.score(victim_eval_genuine)
    impostor_scores = model.score(impostor_pool)
    _, eer_threshold = compute_eer(genuine_scores, impostor_scores)
    return eer_threshold


def run_one_scenario(baseline_cls, baseline_kwargs, initial_enrollment, sequence, eer_threshold, attacker_genuine_samples):
    baseline = baseline_cls(**baseline_kwargs).initialize(initial_enrollment.copy())
    accept_before = float((baseline.score(attacker_genuine_samples) > eer_threshold).mean())
    for i, candidate in enumerate(sequence):
        baseline.offer_candidate(candidate, round_index=i)
    accept_after = float((baseline.score(attacker_genuine_samples) > eer_threshold).mean())
    return {
        "accept_before": accept_before,
        "accept_after": accept_after,
        "delta": accept_after - accept_before,
        "n_defense_triggers": len(getattr(baseline, "defense_triggered_rounds", [])),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, subjects, sessions, feature_cols = load_cmu_features()
    pairing = create_or_load_pairing(subjects)

    print("=== Calibrating PER-VICTIM CUSUM thresholds (sessions 5-6 only) ===")
    per_victim_h = calibrate_all_victims()

    results = {}
    for victim, attacker in pairing.items():
        rng = np.random.default_rng(EXPERIMENT_SEED)

        victim_enroll = get_subject_sessions(X, subjects, sessions, victim, (1, 2, 3, 4))
        victim_eval = get_subject_sessions(X, subjects, sessions, victim, (7, 8))  # EVAL split, disjoint from calibration
        attacker_enroll = get_subject_sessions(X, subjects, sessions, attacker, (1, 2, 3, 4))
        impostor_pool = X[(subjects != victim) & np.isin(sessions, (7, 8))]

        eer_threshold = compute_victim_eer_threshold(victim_enroll, victim_eval, impostor_pool)
        similarity = compute_pair_similarity(victim_enroll, attacker_enroll)
        h = per_victim_h[victim]

        poison_sequence, _ = craft_poisoning_sequence(victim_enroll, attacker_enroll, N_ROUNDS, rng)
        benign_sequence = craft_benign_drift_sequence(victim_eval, N_ROUNDS, rng)

        undefended_attack = run_one_scenario(AdaptiveBaseline, {}, victim_enroll, poison_sequence, eer_threshold, attacker_enroll)
        defended_attack = run_one_scenario(
            DefendedAdaptiveBaseline, {"cusum_k": 0.0, "cusum_h": h},
            victim_enroll, poison_sequence, eer_threshold, attacker_enroll
        )
        undefended_benign = run_one_scenario(AdaptiveBaseline, {}, victim_enroll, benign_sequence, eer_threshold, attacker_enroll)
        defended_benign = run_one_scenario(
            DefendedAdaptiveBaseline, {"cusum_k": 0.0, "cusum_h": h},
            victim_enroll, benign_sequence, eer_threshold, attacker_enroll
        )

        results[victim] = {
            "attacker": attacker, "similarity": similarity, "per_victim_h": h,
            "undefended_attack": undefended_attack, "defended_attack": defended_attack,
            "undefended_benign": undefended_benign, "defended_benign": defended_benign,
        }
        print(f"{victim} (sim={similarity:.3f}, h={h:.3f}): "
              f"undef_attack={undefended_attack['delta']*100:+.2f}pp "
              f"def_attack={defended_attack['delta']*100:+.2f}pp "
              f"(triggers={defended_attack['n_defense_triggers']}/{N_ROUNDS})")

    # --- Similarity vs severity correlation (tests the mechanistic hypothesis) ---
    similarities = [r["similarity"] for r in results.values()]
    severities = [abs(r["undefended_attack"]["delta"]) for r in results.values()]
    corr, p_corr = stats.pearsonr(similarities, severities)
    print(f"\n=== Similarity vs. attack severity correlation ===")
    print(f"Pearson r = {corr:.3f}, p = {p_corr:.5f}")
    print("Expect a NEGATIVE correlation if the hypothesis holds: lower "
          "similarity distance (more alike) should predict HIGHER severity.")

    # --- Stratified reporting: split victims by undefended attack severity ---
    sorted_victims = sorted(results.items(), key=lambda kv: abs(kv[1]["undefended_attack"]["delta"]), reverse=True)
    high_severity = dict(sorted_victims[:15])   # top 15 most-attacked victims
    low_severity = dict(sorted_victims[-15:])   # bottom 15 least-attacked victims

    for label, group in [("HIGH-SEVERITY (top 15)", high_severity), ("LOW-SEVERITY (bottom 15)", low_severity)]:
        ua = [r["undefended_attack"]["delta"] for r in group.values()]
        da = [r["defended_attack"]["delta"] for r in group.values()]
        triggers = [r["defended_attack"]["n_defense_triggers"] for r in group.values()]
        print(f"\n=== {label} ===")
        print(f"  Mean undefended attack Δ: {np.mean(ua)*100:+.2f}pp")
        print(f"  Mean defended attack Δ:   {np.mean(da)*100:+.2f}pp")
        print(f"  Mean defense triggers:    {np.mean(triggers):.1f}/{N_ROUNDS}")
        print(f"  Recovery: {(np.mean(ua) - np.mean(da))*100:+.2f}pp")

    with open(RESULTS_DIR / "defense_v2_results.json", "w") as f:
        json.dump({"per_victim_h": per_victim_h, "per_victim": results,
                    "similarity_correlation": {"r": corr, "p": p_corr}},
                   f, indent=2, default=float)
    print(f"\nSaved to {RESULTS_DIR / 'defense_v2_results.json'}")


if __name__ == "__main__":
    main()
