# Week 4 — Poisoning Attack Implementation

**Read this entire file before writing or running anything.** This week is conceptually the most complex so far — it introduces a mechanism that has never existed in this codebase before (a continuously-*retraining* per-user model), then attacks it, then controls for the attack being genuinely adversarial rather than just "any new data breaks this." Get the control wrong and the whole week's finding is meaningless, even if the code runs cleanly and produces a chart that looks compelling.

**If you are an AI executing this on Aditya's behalf:** the single most important discipline this week is Section 3's distinction between the attack and the benign-drift control. If you find yourself tempted to skip the control because the attack result "looks interesting enough on its own," don't — an attack result without a matching control is not evidence of an attack, it's evidence that the system is fragile in general, which is a much weaker and less publishable claim.

---

## 1. Objective

By the end of this week you have: (a) a genuinely new component — a continuously-adapting per-user baseline that absorbs new sessions into enrollment and periodically retrains, which did not exist before this week (Weeks 1–3 only built *static* enrollment models, fit once and never updated), (b) a Frog-Boiling-style gradual poisoning attack against that adaptation mechanism, using one real subject's genuine typing as the "attacker" gradually smuggled into another real subject's ("victim's") baseline via interpolated samples, (c) a benign-drift control using the *victim's own* later sessions (not another subject's data) as the non-adversarial comparison case, explicitly including the kind of real heavy-tailed variation Week 1 confirmed exists in this dataset, and (d) a full-population experiment (all 51 subjects as victims, not a handful, given the n=1 lesson from Week 2) measuring both FAR degradation under attack and correct absorption under the benign control.

## 2. Critical design decision: which baseline gets attacked, and why

Week 2 built a *static* per-user Isolation Forest/One-Class SVM: fit once on sessions 1–4, never updated. Week 3 built a *shared, fixed* embedding space with per-user centroids computed once. **Neither of these has a continuous-adaptation mechanism yet** — Section 6.7/7.5 of the compendium described the *concept* of absorbing low-risk anomalies into a baseline over time, but nothing in this codebase actually does that yet. This week builds that mechanism for the first time, on top of Week 2's classical Isolation Forest specifically (not Week 3's encoder), because the per-user retrain-on-new-data pattern maps directly onto Isolation Forest's existing per-user fit/refit interface from `src/models.py`. Extending this to the Week 3 encoder (via centroid-drift rather than full retraining) is a natural follow-on, not required for this week's core result — noted as a Week 5-or-later extension, not attempted here, to keep this week's scope achievable.

## 3. Critical design decision: the attack versus the benign-drift control, precisely defined

**The attack scenario:** subject A ("victim") has an enrolled Isolation Forest baseline. Subject B ("attacker," a different real subject, not synthetic data) wants to gradually shift A's baseline toward B's own typing, so that B's genuine behavior eventually scores as "genuine A" against the drifted model. This uses B's **real** CMU data as the target — Frog-Boiling's actual innovation isn't inventing fake data, it's *pacing* the introduction of real, dissimilar behavior slowly enough that each individual step looks unremarkable, even though the cumulative drift is large.

**The benign-drift control:** subject A's baseline absorbs new sessions that are **also from subject A** (their own later, naturally varying sessions), not from an attacker. This directly tests Section 7.3's "broken wrist" scenario and Week 1's confirmed heavy-tailed outliers: does the adaptation mechanism correctly tolerate a real person's real variation without being fooled, while — separately — being vulnerable to the adversarial version above? **Both scenarios use the identical adaptation mechanism and identical absorption threshold.** The only thing that differs is whose data is being fed in. If the mechanism can't tell these apart, that is itself the finding — but you cannot know that without running both.

## 4. Critical design decision: full-population, not a handful of subjects

Week 2's original s049-only ablation was correctly identified as an n=1 overclaim risk two weeks ago. This week does not repeat that mistake: **all 51 subjects serve as victims**, each paired with one attacker (a different randomly-selected subject, fixed seed, no self-pairing), so the eventual claim is "across all 51 subjects" rather than "for one subject we picked." Isolation Forest refits in milliseconds on this data volume, so running 51 victims × ~20 adaptation rounds × 2 scenarios (attack + control) is computationally trivial — there is no reason to scope this down to a handful of subjects the way the outlier ablation initially (incorrectly) did.

## 5. `src/adaptive_baseline.py` — full code (the new mechanism this week introduces)

```python
"""
Week 4: a continuously-adapting per-user baseline. This did not exist
before this week -- Weeks 1-3 only fit static models once. This module
implements the "absorb low-risk anomalies into the baseline" mechanism
described conceptually in the compendium (Sections 6.7/7.5), for the
first time as actual, testable code.

Design: at each adaptation round, a candidate sample is scored against
the CURRENT model. If its score is above an absorption threshold
(computed from the model's own enrollment score distribution, not a
fixed constant -- this makes it adaptive to each individual subject's
own natural score spread), it is folded into the enrollment set and the
model is refit. If not, it is rejected and the enrollment set is
unchanged for that round.
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np

from src.models import PerUserModel


@dataclass
class AdaptationRoundResult:
    round_index: int
    candidate_score: float
    absorption_threshold: float
    absorbed: bool


@dataclass
class AdaptiveBaseline:
    """
    Wraps a PerUserModel with a continuously-updating enrollment set.
    """
    algorithm: str = "isolation_forest"
    absorption_percentile: float = 10.0  # candidate must score above this
                                          # percentile of the model's OWN
                                          # current enrollment scores
    enrollment: np.ndarray = field(default=None)
    model: PerUserModel = field(default=None)
    history: List[AdaptationRoundResult] = field(default_factory=list)

    def initialize(self, initial_enrollment: np.ndarray):
        self.enrollment = initial_enrollment.copy()
        self.model = PerUserModel(algorithm=self.algorithm).fit(self.enrollment)
        return self

    def _current_absorption_threshold(self) -> float:
        own_scores = self.model.score(self.enrollment)
        return float(np.percentile(own_scores, self.absorption_percentile))

    def offer_candidate(self, candidate: np.ndarray, round_index: int) -> AdaptationRoundResult:
        """
        candidate: shape (1, n_features) or (n_features,) -- a single
        repetition being offered for absorption into the baseline.
        """
        candidate = candidate.reshape(1, -1)
        threshold = self._current_absorption_threshold()
        score = float(self.model.score(candidate)[0])
        absorbed = score > threshold

        if absorbed:
            self.enrollment = np.vstack([self.enrollment, candidate])
            self.model = PerUserModel(algorithm=self.algorithm).fit(self.enrollment)

        result = AdaptationRoundResult(
            round_index=round_index,
            candidate_score=score,
            absorption_threshold=threshold,
            absorbed=absorbed,
        )
        self.history.append(result)
        return result

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.score(X)
```

## 6. `tests/test_adaptive_baseline.py` — full code

```python
"""
Verifies the adaptation mechanism itself, independent of any attack --
this must be correct before an attack result built on top of it means
anything.
"""
import numpy as np
from src.adaptive_baseline import AdaptiveBaseline


def test_initialize_fits_on_given_enrollment():
    rng = np.random.RandomState(0)
    enrollment = rng.normal(0, 1, size=(50, 5))
    baseline = AdaptiveBaseline().initialize(enrollment)
    assert baseline.enrollment.shape == (50, 5)


def test_offer_candidate_similar_to_enrollment_is_absorbed():
    rng = np.random.RandomState(1)
    enrollment = rng.normal(0, 1, size=(100, 5))
    baseline = AdaptiveBaseline(absorption_percentile=10.0).initialize(enrollment)

    # A candidate drawn from the SAME distribution should usually be absorbed
    similar_candidate = rng.normal(0, 1, size=(5,))
    result = baseline.offer_candidate(similar_candidate, round_index=0)
    assert result.absorbed, (
        "A same-distribution candidate should typically clear the "
        "absorption threshold -- if this fails often, the threshold "
        "percentile may be miscalibrated."
    )
    assert baseline.enrollment.shape[0] == 101


def test_offer_candidate_wildly_different_is_rejected():
    rng = np.random.RandomState(2)
    enrollment = rng.normal(0, 1, size=(100, 5))
    baseline = AdaptiveBaseline(absorption_percentile=10.0).initialize(enrollment)

    extreme_candidate = np.full(5, 100.0)  # wildly outside the enrolled distribution
    result = baseline.offer_candidate(extreme_candidate, round_index=0)
    assert not result.absorbed, "An extreme outlier must not be absorbed"
    assert baseline.enrollment.shape[0] == 100, (
        "Enrollment set must be UNCHANGED when a candidate is rejected"
    )


def test_rejected_candidate_does_not_change_model():
    rng = np.random.RandomState(3)
    enrollment = rng.normal(0, 1, size=(100, 5))
    baseline = AdaptiveBaseline(absorption_percentile=10.0).initialize(enrollment)
    scores_before = baseline.score(enrollment[:5])

    extreme_candidate = np.full(5, 100.0)
    baseline.offer_candidate(extreme_candidate, round_index=0)
    scores_after = baseline.score(enrollment[:5])

    np.testing.assert_array_almost_equal(scores_before, scores_after, decimal=10)
```

## 7. `src/poisoning_attack.py` — full code (the Frog-Boiling crafter)

```python
"""
The Frog-Boiling-style gradual poisoning attack: crafts a sequence of
candidates that linearly interpolate from victim-like to attacker-like
real feature vectors, with the blend weight increasing round by round.
Uses the ATTACKER's real CMU data as the target -- Frog-Boiling's
actual mechanism is pacing, not fabrication.
"""
import numpy as np


def craft_poisoning_sequence(victim_samples, attacker_samples, n_rounds, rng):
    """
    victim_samples: (n_victim, n_features) -- victim's own enrollment,
                     used as interpolation anchors.
    attacker_samples: (n_attacker, n_features) -- attacker's real data,
                     used as the interpolation target.
    Returns (candidates: (n_rounds, n_features), alphas: (n_rounds,)),
    where alpha is the blend weight used at each round, returned
    explicitly for transparency/logging, not hidden inside the array.
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
```

## 8. `src/benign_drift_control.py` — full code (the required control, not optional)

```python
"""
The non-adversarial control: candidates drawn from the VICTIM's OWN
later sessions (5-8), not from an attacker. Real, naturally-occurring
variation from the same person -- including whatever heavy-tailed
outliers Week 1 confirmed exist -- rather than another person's
behavior. Run through the IDENTICAL adaptation mechanism as the attack,
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
```

## 9. `src/victim_attacker_pairing.py` — full code

```python
"""
Assigns each of the 51 subjects a victim role and a different,
randomly-selected subject as their attacker. Fixed seed, no
self-pairing, persisted to disk so the pairing cannot silently change
between runs -- same discipline as Week 3's subject_split.json.
"""
import json
from pathlib import Path
import numpy as np

PAIRING_PATH = Path("results/week4/victim_attacker_pairs.json")
PAIRING_SEED = 42


def create_or_load_pairing(all_subjects):
    unique_subjects = sorted(np.unique(all_subjects).tolist())

    if PAIRING_PATH.exists():
        with open(PAIRING_PATH) as f:
            pairs = json.load(f)
        assert set(pairs.keys()) == set(unique_subjects), (
            "Existing pairing file does not match the current subject "
            "list. STOP -- do not silently regenerate."
        )
        print(f"Loaded existing victim-attacker pairing from {PAIRING_PATH}")
        return pairs

    rng = np.random.RandomState(PAIRING_SEED)
    pairs = {}
    for victim in unique_subjects:
        candidates = [s for s in unique_subjects if s != victim]
        attacker = candidates[rng.randint(0, len(candidates))]
        pairs[victim] = attacker

    PAIRING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PAIRING_PATH, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Created new victim-attacker pairing, saved to {PAIRING_PATH}")
    return pairs
```

## 10. `src/run_poisoning_experiment.py` — full code (main script, all 51 victims)

```python
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

RESULTS_DIR = Path("results/week4")
N_ROUNDS = 20
EXPERIMENT_SEED = 123


def get_subject_sessions(X, subjects, sessions, subject_id, session_ids):
    mask = (subjects == subject_id) & np.isin(sessions, session_ids)
    return X[mask]


def run_scenario_for_victim(initial_enrollment, poison_sequence, victim_genuine_test,
                             attacker_genuine_samples, algorithm="isolation_forest"):
    baseline = AdaptiveBaseline(algorithm=algorithm).initialize(initial_enrollment)

    scores_before_victim = baseline.score(victim_genuine_test)
    scores_before_attacker = baseline.score(attacker_genuine_samples)
    # "Accept" = decision_function > 0, IsolationForest's own inlier
    # convention -- matches the scoring convention already used
    # throughout this project (higher = more normal).
    accept_rate_before_victim = float((scores_before_victim > 0).mean())
    accept_rate_before_attacker = float((scores_before_attacker > 0).mean())

    n_absorbed = 0
    for i, candidate in enumerate(poison_sequence):
        result = baseline.offer_candidate(candidate, round_index=i)
        if result.absorbed:
            n_absorbed += 1

    scores_after_victim = baseline.score(victim_genuine_test)
    scores_after_attacker = baseline.score(attacker_genuine_samples)
    accept_rate_after_victim = float((scores_after_victim > 0).mean())
    accept_rate_after_attacker = float((scores_after_attacker > 0).mean())

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
    rng = np.random.RandomState(EXPERIMENT_SEED)

    X, subjects, sessions, feature_cols = load_cmu_features()
    pairing = create_or_load_pairing(subjects)

    all_results = {}
    for victim, attacker in pairing.items():
        victim_enroll = get_subject_sessions(X, subjects, sessions, victim, (1, 2, 3, 4))
        victim_later = get_subject_sessions(X, subjects, sessions, victim, (5, 6, 7, 8))
        attacker_enroll = get_subject_sessions(X, subjects, sessions, attacker, (1, 2, 3, 4))

        poison_sequence, alphas = craft_poisoning_sequence(
            victim_enroll, attacker_enroll, N_ROUNDS, rng
        )
        attack_result = run_scenario_for_victim(
            victim_enroll.copy(), poison_sequence, victim_later, attacker_enroll
        )

        benign_sequence = craft_benign_drift_sequence(victim_later, N_ROUNDS, rng)
        benign_result = run_scenario_for_victim(
            victim_enroll.copy(), benign_sequence, victim_later, attacker_enroll
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
```

Run with: `python -m src.run_poisoning_experiment`

## 11. `tests/test_poisoning_attack.py` — full code

```python
import numpy as np
from src.poisoning_attack import craft_poisoning_sequence


def test_alpha_increases_monotonically():
    rng = np.random.RandomState(0)
    victim = np.zeros((10, 3))
    attacker = np.ones((10, 3))
    _, alphas = craft_poisoning_sequence(victim, attacker, n_rounds=10, rng=rng)
    assert np.all(np.diff(alphas) > 0), "Alpha must strictly increase round over round"
    assert alphas[0] < 0.2, "First round should be mostly victim-like"
    assert alphas[-1] == 1.0, "Final round should be fully attacker-like"


def test_candidates_drift_from_victim_toward_attacker():
    rng = np.random.RandomState(0)
    victim = np.zeros((10, 3))
    attacker = np.full((10, 3), 10.0)
    candidates, _ = craft_poisoning_sequence(victim, attacker, n_rounds=10, rng=rng)

    dist_to_victim_round0 = np.linalg.norm(candidates[0] - 0.0)
    dist_to_attacker_round0 = np.linalg.norm(candidates[0] - 10.0)
    assert dist_to_victim_round0 < dist_to_attacker_round0, (
        "Round 0 candidate should be closer to victim than attacker"
    )

    dist_to_victim_last = np.linalg.norm(candidates[-1] - 0.0)
    dist_to_attacker_last = np.linalg.norm(candidates[-1] - 10.0)
    assert dist_to_attacker_last < dist_to_victim_last, (
        "Final candidate should be closer to attacker than victim"
    )
```

## 12. `tests/test_benign_drift_control.py` — full code

```python
import numpy as np
from src.benign_drift_control import craft_benign_drift_sequence


def test_benign_sequence_only_uses_victims_own_samples():
    rng = np.random.RandomState(0)
    victim_later = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sequence = craft_benign_drift_sequence(victim_later, n_rounds=5, rng=rng)
    assert sequence.shape == (5, 2)
    for row in sequence:
        assert any(np.allclose(row, v) for v in victim_later), (
            "Benign control must only ever draw from the victim's own "
            "provided samples -- never fabricate or blend with anything else, "
            "which is exactly what would distinguish it from the attack."
        )
```

## 13. `tests/test_victim_attacker_pairing.py` — full code

```python
import numpy as np
from src import victim_attacker_pairing as vap


def test_pairing_has_no_self_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(vap, "PAIRING_PATH", tmp_path / "pairs.json")
    fake_subjects = np.array([f"s{i:03d}" for i in range(10)])
    pairs = vap.create_or_load_pairing(fake_subjects)
    for victim, attacker in pairs.items():
        assert victim != attacker, f"{victim} is paired with themselves"
    assert set(pairs.keys()) == set(fake_subjects.tolist())


def test_pairing_is_deterministic_across_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(vap, "PAIRING_PATH", tmp_path / "pairs.json")
    fake_subjects = np.array([f"s{i:03d}" for i in range(10)])
    pairs1 = vap.create_or_load_pairing(fake_subjects)
    pairs2 = vap.create_or_load_pairing(fake_subjects)
    assert pairs1 == pairs2
```

## 14. Exact command sequence

```bash
pytest tests/test_adaptive_baseline.py tests/test_poisoning_attack.py \
       tests/test_benign_drift_control.py tests/test_victim_attacker_pairing.py -v
python -m src.run_poisoning_experiment
git add -A && git commit -m "Week 4: adaptive baseline, Frog-Boiling attack, benign-drift control, all 51 victims"
git tag week04
git push origin main --tags
```

## 15. Verification checklist — all of these must pass before Week 5 starts

- [ ] All 9 new unit tests pass (4 in `test_adaptive_baseline.py`, 2 in `test_poisoning_attack.py`, 1 in `test_benign_drift_control.py`, 2 in `test_victim_attacker_pairing.py`).
- [ ] `results/week4/victim_attacker_pairs.json` covers all 51 subjects with zero self-pairs.
- [ ] `python -m src.run_poisoning_experiment` completes for all 51 victims without error.
- [ ] **The core comparison**: mean attacker-acceptance delta under ATTACK is reported alongside the mean under BENIGN control — not one without the other. The attack finding is only real if attack's delta is meaningfully larger than benign's; if they're close, that is itself the honest result to report, not a bug to fix by adjusting parameters.
- [ ] The victim's-own-acceptance delta under attack is checked, not just ignored — if it collapses toward the same degradation as the attacker's acceptance improves, that's a "the whole model breaks" finding rather than a "the model gets confused specifically about identity" finding, and changes how this should be described.
- [ ] `n_absorbed` counts are sane (not 0 for every victim, which would mean no drift occurred at all; not 20/20 for every victim with no rejections at any point, which would suggest the absorption threshold isn't discriminating anything).
- [ ] `results/week4/poisoning_results.json` exists with per-victim detail for all 51 subjects, both scenarios.

## 16. What to send back to Claude at the end of this week

Paste or attach: (1) full console output of the four new test files and `run_poisoning_experiment.py`, including the per-victim print lines, not just the final aggregate; (2) the aggregate ATTACK vs BENIGN comparison specifically, both attacker-acceptance and victim-acceptance deltas; (3) `poisoning_results.json`; (4) which victims (if any) showed the *opposite* of the expected pattern — attack delta smaller than benign delta — since a handful of exceptions is fine to report honestly, but worth knowing about before writing anything up. I'll check whether the attack is doing something specifically adversarial rather than just generally destabilizing, and whether the victim's own access held up reasonably well throughout.
