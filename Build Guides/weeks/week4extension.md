# Week 4 Extension — Isolating Why the Attack Showed a Null Result

**Read this before writing any code.** This is not a redo of Week 4 — everything from last week (the adaptive baseline, the test suite, the null-result data) stays exactly as committed. This extension adds two new variables to isolate *why* the attack didn't work, so that Week 5's defense evaluation has a real attack to defend against, rather than defending against nothing. If both variables tested here still produce a null result, that itself becomes a legitimate, citable finding — "naive gradual poisoning does not trivially work against a percentile-gated adaptive baseline" is a real contribution — but we should know that's the actual finding before building a paper section around it, not assume it.

**If you are an AI executing this on Aditya's behalf:** the s040/s046/s047 full-absorption-but-no-effect cases from last week are the single most important data point driving this extension. Do not lose track of them — re-check whether they behave differently under the new crafting method specifically.

---

## 1. Objective

By the end of this extension you have: (a) the existing Frog-Boiling attack (now called V1, point-interpolation) swept across N_ROUNDS ∈ {20, 100, 200} to check whether more gradual steps over a longer window change the result, (b) a second crafting method (V2, mean-shift) that samples candidates from a distribution centered on a gradually-shifting point between victim and attacker means, using the victim's own per-feature spread — designed specifically to avoid landing in the unrealistic "between-two-different-people" region that Section 5.2 of last week's report correctly identified as the likely failure mode of V1 — swept across the same three round counts, (c) both compared side by side against the same benign-drift control from last week, and (d) as a byproduct, `scripts/validate_seeds.py` (the orphaned script from last week's Section 7.2) repurposed to finally close out the Week 3 cross-seed question, since it already does most of what that needs.

## 2. Everything from Week 4 stays frozen

Do not modify `src/adaptive_baseline.py`, `src/poisoning_attack.py`, `src/benign_drift_control.py`, `src/victim_attacker_pairing.py`, or any of last week's tests. This extension **adds** a second crafting function and a parameterized sweep runner; it does not touch what's already committed and tagged `week04`.

## 3. `src/poisoning_attack_v2.py` — full code (the mean-shift crafting method)

```python
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
"""
import numpy as np


def craft_poisoning_sequence_meanshift(victim_samples, attacker_samples, n_rounds, rng):
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
```

## 4. `tests/test_poisoning_attack_v2.py` — full code

```python
import numpy as np
from src.poisoning_attack_v2 import craft_poisoning_sequence_meanshift


def test_v2_alpha_increases_monotonically():
    rng = np.random.default_rng(0)
    victim = np.zeros((20, 3))
    attacker = np.full((20, 3), 10.0)
    _, alphas = craft_poisoning_sequence_meanshift(victim, attacker, n_rounds=10, rng=rng)
    assert np.all(np.diff(alphas) > 0)
    assert alphas[-1] == 1.0


def test_v2_candidates_center_drifts_toward_attacker():
    rng = np.random.default_rng(0)
    victim = np.zeros((200, 3))  # zero std -- isolates the CENTER shift from noise
    attacker = np.full((200, 3), 10.0)
    candidates, _ = craft_poisoning_sequence_meanshift(victim, attacker, n_rounds=10, rng=rng)
    # With zero victim std, noise is exactly zero, so candidates should land
    # exactly on the interpolated mean at each round.
    assert np.allclose(candidates[0], 1.0)   # alpha=0.1 -> 0.1*10
    assert np.allclose(candidates[-1], 10.0)  # alpha=1.0 -> fully attacker


def test_v2_uses_victim_std_not_attacker_std():
    rng = np.random.default_rng(1)
    victim = np.random.default_rng(2).normal(0, 0.01, size=(100, 3))  # tiny victim spread
    attacker = np.random.default_rng(3).normal(10, 5.0, size=(100, 3))  # large attacker spread
    candidates, _ = craft_poisoning_sequence_meanshift(victim, attacker, n_rounds=50, rng=rng)
    # Candidate spread should be small (driven by victim's tiny std), not
    # large (attacker's std is never used for the noise term at all).
    assert candidates.std() < 1.0, (
        "Candidate spread should reflect the victim's tight distribution, "
        "not the attacker's -- if this fails, victim_std was swapped for "
        "attacker_std somewhere."
    )
```

## 5. `src/run_poisoning_sweep.py` — full code (parameterized sweep, both methods, three round counts)

```python
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

from src.feature_extraction import load_cmu_features
from src.adaptive_baseline import AdaptiveBaseline
from src.poisoning_attack import craft_poisoning_sequence as craft_v1
from src.poisoning_attack_v2 import craft_poisoning_sequence_meanshift as craft_v2
from src.benign_drift_control import craft_benign_drift_sequence
from src.victim_attacker_pairing import create_or_load_pairing
from src.run_poisoning_experiment import get_subject_sessions, run_scenario_for_victim

RESULTS_DIR = Path("results/week4_extension")
ROUND_COUNTS = [20, 100, 200]
SWEEP_SEED = 456


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, subjects, sessions, feature_cols = load_cmu_features()
    pairing = create_or_load_pairing(subjects)  # reuses the SAME pairing as Week 4

    sweep_results = {}

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
                attacker_enroll = get_subject_sessions(X, subjects, sessions, attacker, (1, 2, 3, 4))

                poison_sequence, _ = craft_fn(victim_enroll, attacker_enroll, n_rounds, rng)
                attack_result = run_scenario_for_victim(
                    victim_enroll.copy(), poison_sequence, victim_later, attacker_enroll
                )

                benign_sequence = craft_benign_drift_sequence(victim_later, n_rounds, rng)
                benign_result = run_scenario_for_victim(
                    victim_enroll.copy(), benign_sequence, victim_later, attacker_enroll
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
```

Run with: `python -m src.run_poisoning_sweep`

## 6. Resolving `scripts/validate_seeds.py` — repurpose it, don't just decide its fate in the abstract

Rather than deciding whether to delete or keep an unused script, use it for the thing it was apparently already built for: closing out the Week 3 cross-seed question (mean ~3.95pp advantage, std ~3.02pp across only 4 seeds — too few to trust the magnitude). Inspect what it currently does:

```bash
cat scripts/validate_seeds.py
```

If it already re-runs the Week 3 background/held-out split, trains the encoder, and evaluates against the recomputed classical baseline for a given seed, extend it to loop over 10 total seeds (0 through 9) and report the mean and standard deviation of the encoder-vs-classical gap across all 10 — exactly what I asked for after last week's Week 3 checkpoint. If it does something different or incomplete, say so plainly rather than forcing it to fit; this is a bonus task, not a blocker for the poisoning-sweep decision above.

## 7. Exact command sequence

```bash
pytest tests/test_poisoning_attack_v2.py -v
python -m src.run_poisoning_sweep
cat scripts/validate_seeds.py   # inspect before deciding how to extend it
git add -A && git commit -m "Week 4 extension: crafting-method and round-count sweep, isolating the null result's cause"
git tag week04-extension
git push origin main --tags
```

## 8. What to send back to Claude

Paste or attach: (1) the full console output of the test file and `run_poisoning_sweep.py`, especially the final 6-row summary table; (2) `sweep_results.json`; (3) whichever configuration(s) show the largest attack-vs-benign gap, with the same per-victim breakdown detail Section 5.3/5.4 gave last week for those specific configurations — I want to see if s040/s046/s047 (full absorption, no effect under V1/20-rounds) behave differently under V2 specifically; (4) whatever `validate_seeds.py` turns out to actually do, and the 10-seed result if you extend it.

Depending on what comes back, one of two things happens next: if a configuration shows a real, non-noise-level gap, "true Week 5" becomes building the residue-feature defense against *that* configuration specifically. If nothing does even at 200 rounds with mean-shift crafting, we talk about reframing RQ1 around that honestly-earned negative result before deciding what Week 5 becomes instead.