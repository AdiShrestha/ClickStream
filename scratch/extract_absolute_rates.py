import numpy as np
from src.feature_extraction import load_cmu_features
from src.poisoning_attack import craft_poisoning_sequence as craft_v1
from src.victim_attacker_pairing import create_or_load_pairing
from src.run_poisoning_experiment import get_subject_sessions, run_scenario_for_victim, compute_victim_eer_threshold

def main():
    X, subjects, sessions, feature_cols = load_cmu_features()
    pairing = create_or_load_pairing(subjects)
    
    rng = np.random.default_rng(456)
    n_rounds = 200
    
    before_rates = []
    after_rates = []
    
    for victim, attacker in pairing.items():
        victim_enroll = get_subject_sessions(X, subjects, sessions, victim, (1, 2, 3, 4))
        victim_later = get_subject_sessions(X, subjects, sessions, victim, (5, 6, 7, 8))
        attacker_enroll = get_subject_sessions(X, subjects, sessions, attacker, (1, 2, 3, 4))
        
        impostor_test = X[(subjects != victim) & np.isin(sessions, (5, 6, 7, 8))]
        eer_threshold = compute_victim_eer_threshold(victim_enroll, victim_later, impostor_test)
        
        poison_sequence, _ = craft_v1(victim_enroll, attacker_enroll, n_rounds, rng)
        attack_result = run_scenario_for_victim(
            victim_enroll.copy(), poison_sequence, victim_later, attacker_enroll, eer_threshold
        )
        before_rates.append(attack_result["accept_rate_attacker_before"])
        after_rates.append(attack_result["accept_rate_attacker_after"])
        
    print(f"Mean attacker acceptance BEFORE (round 0): {np.mean(before_rates)*100:.2f}%")
    print(f"Mean attacker acceptance AFTER (round 200): {np.mean(after_rates)*100:.2f}%")
    print(f"Absolute change: {(np.mean(after_rates) - np.mean(before_rates))*100:+.2f}pp")

if __name__ == "__main__":
    main()
