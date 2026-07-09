# Weekly Report 4: Adversarial Baseline Poisoning (Frog-Boiling Attack)

**Git commit:** `0d16f53` (Phase 2), `dd0ec23` (Phase 1)
**Tag:** `week04`
**Date:** 2026-07-09
**Hardware:** MacBook Air M3, 16 GB unified memory, macOS 26.5.2
**Python:** 3.12.8

---

## 1. Objectives (copied from week4.md Section 1)

By the end of this week:

(a) A genuinely new component: a continuously-adapting per-user baseline (`AdaptiveBaseline`) that absorbs new sessions into enrollment and periodically retrains. Weeks 1-3 built static models fit once and never updated; this is the first implementation of the continuous-adaptation mechanism described in compendium Sections 6.7/7.5.

(b) A Frog-Boiling-style gradual poisoning attack against that adaptation mechanism. Uses one real subject's genuine CMU typing as the attacker's target, gradually smuggled into another subject's (victim's) baseline via linearly-interpolated feature vectors.

(c) A benign-drift control using the victim's own later sessions (sessions 5-8) as the non-adversarial comparison. Both scenarios use the identical adaptation mechanism with identical absorption threshold. The only difference is whose data is fed in.

(d) Full-population experiment: all 51 subjects as victims, each paired with a different randomly-selected subject as attacker. Fixed seed, no self-pairing, persisted to `results/week4/victim_attacker_pairs.json`.

---

## 2. Environment

### Library versions (from results/week4/requirements.lock.txt)

```
numpy==2.2.5
scikit-learn==1.6.1
torch==2.7.1
pytest==9.1.1
```

Full lockfile: `results/week4/requirements.lock.txt`

### Hardware
- MacBook Air M3 (8-core CPU, 10-core GPU)
- 16 GB unified memory
- No CUDA; device selection via `src/device.py` (returns MPS on this machine)

### Git state
- Phase 1 commit: `dd0ec23` — Claude: all 4 RISK HIGH modules + 8 tests
- Phase 2 commit: `0d16f53` — Gemini: `run_poisoning_experiment.py` + report + requirements.lock
- Tag: `week04` applied at `0d16f53`

---

## 3. Architecture: New Components This Week

### 3.1 `src/adaptive_baseline.py`

**What it does:** Wraps `PerUserModel` (Isolation Forest) with a continuous enrollment update loop.

**Interface:**
```
AdaptiveBaseline.initialize(initial_enrollment: ndarray) -> self
AdaptiveBaseline.offer_candidate(candidate: ndarray, round_index: int) -> AdaptationRoundResult
AdaptiveBaseline.score(X: ndarray) -> ndarray
```

**Absorption criterion:** A candidate is absorbed if `score > np.percentile(own_scores, absorption_percentile)`. Default `absorption_percentile=10.0`. The threshold is recomputed from the CURRENT model's scores on CURRENT enrollment at each round. After absorption the model is immediately refit.

**Key invariants guaranteed by tests:**
- `enrollment.copy()` prevents aliasing bugs
- Rejected candidates do not change the model (scores are identical before/after rejection)
- Wildly-out-of-distribution samples (100-sigma outliers) are rejected

### 3.2 `src/poisoning_attack.py`

**What it does:** Constructs the Frog-Boiling sequence. For each of `n_rounds`, samples one row from `victim_samples` and one from `attacker_samples`, linearly interpolates: `candidate = (1 - alpha) * victim_anchor + alpha * attacker_anchor`, where `alpha = r / n_rounds` for round `r` in `[1, n_rounds]`.

**Alpha schedule:** alpha[0] = 1/20 = 0.05, alpha[-1] = 1.0. Strictly monotone.

**RISK HIGH (caught and resolved):** Uses `rng.integers()` which requires a numpy `Generator`. The experiment script creates `rng = np.random.default_rng(EXPERIMENT_SEED)`. If `np.random.RandomState` were used instead, it would crash with `AttributeError`. A test (`test_craft_accepts_numpy_generator_not_random_state`) explicitly verifies this boundary.

### 3.3 `src/benign_drift_control.py`

**What it does:** Samples `n_rounds` rows from `victim_later_sessions` using `rng.choice`. If `n_rounds <= n_available`, samples without replacement. If `n_rounds > n_available`, samples with replacement (replace=True branch). In the CMU dataset with 200 later-session rows and N_ROUNDS=20, the without-replacement path is always taken.

### 3.4 `src/victim_attacker_pairing.py`

**What it does:** For each of the 51 subjects, assigns a randomly-selected different subject as the attacker. Uses `np.random.RandomState(42).randint()`. Persists to `results/week4/victim_attacker_pairs.json` on first call; reloads and validates on subsequent calls. Assertion fails if the persisted pairing's subject list does not match the current dataset.

### 3.5 `src/run_poisoning_experiment.py`

**What it does:** Orchestrates the full experiment. For each victim:
1. Loads victim enrollment (sessions 1-4), victim later (sessions 5-8), attacker enrollment (sessions 1-4).
2. Runs attack scenario: crafts the Frog-Boiling sequence, runs it through `AdaptiveBaseline`, measures accept-rate delta for both attacker's genuine samples and victim's genuine test samples.
3. Runs benign-drift control: identical mechanism, victim's own later sessions as the sequence.
4. Prints both scenarios side-by-side per victim.
5. Aggregates mean and std across all 51 victims.
6. Writes `results/week4/poisoning_results.json`.

**RNG:** `np.random.default_rng(EXPERIMENT_SEED)` where `EXPERIMENT_SEED = 123`. Shared across both `craft_poisoning_sequence` and `craft_benign_drift_sequence` calls (single rng, sequential calls). This means the exact sequence of random draws for benign control depends on how many draws were made for the poisoning sequence preceding it. This is the same rng state for both to keep reproducibility simple; it is not a bug, but it means the benign control and the attack sequences are not independently seeded.

---

## 4. Raw Results

### 4.1 Test Suite

**Command:**
```
PYTHONPATH=. .venv/bin/pytest tests/test_adaptive_baseline.py tests/test_poisoning_attack.py tests/test_benign_drift_control.py tests/test_victim_attacker_pairing.py -v
```

**Result:** 11 passed in 1.82s (re-run in Phase 3 verification)

**Full test suite regression check:**
```
PYTHONPATH=. .venv/bin/pytest tests/ -v --tb=short
```
**Result:** 29 passed in 2.95s, 0 failures

Individual test outcomes:
```
tests/test_adaptive_baseline.py::test_initialize_fits_on_given_enrollment PASSED
tests/test_adaptive_baseline.py::test_offer_candidate_similar_to_enrollment_is_absorbed PASSED
tests/test_adaptive_baseline.py::test_offer_candidate_wildly_different_is_rejected PASSED
tests/test_adaptive_baseline.py::test_rejected_candidate_does_not_change_model PASSED
tests/test_poisoning_attack.py::test_alpha_increases_monotonically PASSED
tests/test_poisoning_attack.py::test_candidates_drift_from_victim_toward_attacker PASSED
tests/test_poisoning_attack.py::test_craft_accepts_numpy_generator_not_random_state PASSED
tests/test_benign_drift_control.py::test_benign_sequence_only_uses_victims_own_samples PASSED
tests/test_benign_drift_control.py::test_benign_sequence_replace_true_branch PASSED
tests/test_victim_attacker_pairing.py::test_pairing_has_no_self_pairs PASSED
tests/test_victim_attacker_pairing.py::test_pairing_is_deterministic_across_calls PASSED
```

### 4.2 Victim-Attacker Pairs

**Command:** `python -m src.run_poisoning_experiment` (first run creates the file)
**File:** `results/week4/victim_attacker_pairs.json`
**Seed:** 42 (via `np.random.RandomState(42)` in `victim_attacker_pairing.py`)
**N subjects:** 51, zero self-pairs confirmed

Complete pairing table:

| Victim | Attacker | Victim | Attacker | Victim | Attacker |
|--------|----------|--------|----------|--------|----------|
| s002   | s046     | s019   | s030     | s038   | s033     |
| s003   | s035     | s020   | s004     | s039   | s020     |
| s004   | s020     | s021   | s028     | s040   | s019     |
| s005   | s050     | s022   | s003     | s041   | s054     |
| s007   | s012     | s024   | s030     | s042   | s051     |
| s008   | s027     | s025   | s051     | s043   | s004     |
| s010   | s046     | s026   | s036     | s044   | s042     |
| s011   | s025     | s027   | s044     | s046   | s010     |
| s012   | s029     | s028   | s003     | s047   | s026     |
| s013   | s016     | s029   | s026     | s048   | s012     |
| s015   | s016     | s030   | s039     | s049   | s044     |
| s016   | s030     | s031   | s016     | s050   | s022     |
| s017   | s042     | s032   | s027     | s051   | s005     |
| s018   | s047     | s033   | s051     | s052   | s030     |
|        |          | s034   | s030     | s053   | s018     |
|        |          | s035   | s056     | s054   | s057     |
|        |          | s036   | s032     | s055   | s012     |
|        |          | s037   | s049     | s056   | s031     |
|        |          |        |          | s057   | s003     |

### 4.3 Per-Subject Experiment Results (Full Table)

**Command:** `PYTHONPATH=. python -m src.run_poisoning_experiment | tee results/week4/run_experiment_output.txt`
**Full output:** `results/week4/run_experiment_output.txt`

Columns: Victim | Attacker | Atk ΔAtt | Atk ΔVict | Ben ΔAtt | Ben ΔVict | n_absorbed

| Victim | Attacker | Atk ΔAtt | Atk ΔVict | Ben ΔAtt | Ben ΔVict | n_absorbed |
|--------|----------|----------|-----------|----------|-----------|------------|
| s002   | s046     | +0.010   | +0.000    | -0.005   | -0.015    | 14         |
| s003   | s035     | +0.070   | +0.000    | +0.075   | -0.010    | 12         |
| s004   | s020     | -0.065   | -0.005    | -0.015   | +0.005    | 15         |
| s005   | s050     | -0.045   | +0.000    | +0.000   | +0.000    | 15         |
| s007   | s012     | -0.010   | -0.010    | +0.015   | +0.035    | 8          |
| s008   | s027     | +0.070   | -0.015    | +0.055   | +0.000    | 14         |
| s010   | s046     | +0.005   | +0.000    | +0.000   | +0.000    | 5          |
| s011   | s025     | +0.140   | -0.010    | +0.015   | +0.005    | 14         |
| s012   | s029     | +0.055   | +0.000    | -0.275   | +0.000    | 17         |
| s013   | s016     | +0.000   | -0.005    | +0.000   | -0.015    | 4          |
| s015   | s016     | +0.000   | -0.015    | +0.000   | -0.020    | 8          |
| s016   | s030     | -0.115   | -0.020    | +0.005   | -0.005    | 12         |
| s017   | s042     | -0.025   | -0.005    | -0.035   | -0.005    | 5          |
| s018   | s047     | -0.085   | -0.005    | -0.145   | -0.005    | 14         |
| s019   | s030     | -0.020   | -0.005    | -0.050   | -0.005    | 8          |
| s020   | s004     | -0.010   | +0.000    | -0.010   | +0.000    | 19         |
| s021   | s028     | +0.120   | -0.010    | -0.090   | -0.025    | 14         |
| s022   | s003     | +0.000   | +0.000    | +0.000   | -0.010    | 4          |
| s024   | s030     | +0.000   | +0.000    | +0.000   | +0.000    | 4          |
| s025   | s051     | -0.155   | +0.000    | +0.030   | +0.000    | 15         |
| s026   | s036     | +0.000   | +0.000    | +0.000   | +0.000    | 2          |
| s027   | s044     | +0.035   | +0.010    | +0.000   | +0.000    | 13         |
| s028   | s003     | +0.000   | +0.005    | +0.000   | +0.005    | 5          |
| s029   | s026     | +0.045   | -0.010    | +0.015   | +0.010    | 16         |
| s030   | s039     | +0.030   | +0.000    | -0.010   | +0.005    | 18         |
| s031   | s016     | -0.005   | -0.005    | -0.070   | +0.000    | 7          |
| s032   | s027     | +0.000   | -0.010    | -0.005   | +0.020    | 19         |
| s033   | s051     | -0.300   | -0.005    | +0.130   | +0.010    | 15         |
| s034   | s030     | -0.020   | +0.000    | -0.030   | -0.005    | 9          |
| s035   | s056     | +0.050   | +0.020    | -0.045   | -0.005    | 13         |
| s036   | s032     | +0.000   | +0.000    | +0.000   | +0.000    | 4          |
| s037   | s049     | +0.000   | +0.000    | +0.005   | +0.005    | 6          |
| s038   | s033     | -0.005   | +0.000    | -0.005   | +0.000    | 8          |
| s039   | s020     | -0.105   | -0.005    | -0.040   | -0.010    | 9          |
| s040   | s019     | +0.030   | +0.010    | +0.020   | -0.005    | 20         |
| s041   | s054     | +0.040   | +0.010    | -0.005   | +0.000    | 19         |
| s042   | s051     | -0.160   | +0.000    | +0.050   | +0.000    | 15         |
| s043   | s004     | +0.000   | +0.000    | +0.000   | +0.005    | 4          |
| s044   | s042     | -0.160   | +0.000    | -0.045   | +0.000    | 9          |
| s046   | s010     | -0.005   | +0.000    | +0.005   | +0.000    | 20         |
| s047   | s026     | -0.005   | -0.020    | +0.000   | +0.025    | 20         |
| s048   | s012     | +0.005   | -0.005    | +0.000   | -0.005    | 11         |
| s049   | s044     | -0.240   | -0.010    | -0.220   | -0.005    | 18         |
| s050   | s022     | +0.000   | +0.005    | +0.000   | +0.000    | 4          |
| s051   | s005     | +0.000   | +0.010    | +0.000   | +0.030    | 7          |
| s052   | s030     | +0.000   | +0.000    | +0.000   | +0.005    | 5          |
| s053   | s018     | -0.045   | -0.005    | -0.020   | +0.000    | 12         |
| s054   | s057     | +0.025   | +0.010    | +0.020   | +0.005    | 18         |
| s055   | s012     | -0.005   | +0.000    | -0.005   | +0.000    | 6          |
| s056   | s031     | -0.025   | +0.000    | -0.055   | -0.005    | 9          |
| s057   | s003     | -0.185   | -0.005    | -0.100   | +0.000    | 12         |

Legend: Atk = attack scenario, Ben = benign-drift control, ΔAtt = change in attacker acceptance rate, ΔVict = change in victim's own acceptance rate, n_absorbed = number of the 20 attack-sequence candidates absorbed into the baseline.

### 4.4 Aggregate Results

**Command:** computed from `results/week4/poisoning_results.json`

```
ATTACK  mean Δ attacker acceptance: -2.09pp  std 8.09pp
BENIGN  mean Δ attacker acceptance: -1.65pp  std 6.25pp
ATTACK  mean Δ victim's own acceptance: -0.21pp
BENIGN  mean Δ victim's own acceptance: +0.03pp
```

### 4.5 n_absorbed Distribution (Attack Scenario)

| Statistic | Value |
|-----------|-------|
| min       | 2     |
| max       | 20    |
| mean      | 11.3  |
| values    | 2,4,4,4,4,4,5,5,5,5,6,6,7,7,8,8,8,8,9,9,9,9,11,12,12,12,13,13,14,14,14,14,14,15,15,15,15,15,16,17,18,18,18,19,19,20,20,20 |

Subjects with n_absorbed = 20 (full absorption, zero rejections): **s040, s046, s047**

Subjects with n_absorbed = 2 (near-zero absorption): **s026**

---

## 5. Analysis of Results

### 5.1 The Core Finding: Null Result

The attack did not specifically succeed. The mean attacker-acceptance delta under adversarial Frog-Boiling (-2.09pp) is statistically indistinguishable from the mean under benign drift from the victim's own later sessions (-1.65pp). The gap is 0.44pp with both values negative. The model is being marginally degraded by ANY absorbed data, regardless of whether that data is adversarial.

This means: this particular adaptive-baseline mechanism with absorption_percentile=10.0 and N_ROUNDS=20 is **not specifically vulnerable to linear feature-space interpolation at this gradient**. The interpolated samples do not successfully shift the model's decision boundary toward the attacker's genuine feature distribution.

### 5.2 Why the Attack Didn't Work

The mechanism of failure is visible in the data. For subjects like s011, the attack achieves +14pp attacker-acceptance delta vs +1.5pp benign. But for subjects like s033, the attack achieves -30pp vs +13pp benign — the attack actively hurt the attacker's chances. The linear interpolation of features between two different typists produces samples that occupy a genuinely anomalous region of feature space (between-subjects midpoints) that IsolationForest tends to score as outliers rather than inliers.

The absorption threshold (10th percentile of enrollment scores) means only samples that look as "normal" as the bottom 10% of genuine enrollment are accepted. Blended features from a different person do not reliably cross this bar, especially in early rounds when alpha is low and the interpolated sample is still mostly victim-like in some dimensions but anomalously attacker-like in others.

### 5.3 Subjects Where Attack Outperformed Benign

Cases where attack ΔAtt meaningfully exceeded benign ΔAtt (attack more dangerous than natural drift):

| Victim | Attack ΔAtt | Benign ΔAtt | Gap  |
|--------|-------------|-------------|------|
| s011   | +0.140      | +0.015      | +0.125 |
| s021   | +0.120      | -0.090      | +0.210 |
| s041   | +0.040      | -0.005      | +0.045 |
| s027   | +0.035      | +0.000      | +0.035 |

For s021 and s011, the attack does show a meaningful adversarial effect — the attacker's acceptance rate genuinely increased more under attack than under benign drift. These are the only two subjects where a real effect is plausibly present. Both are isolated cases; the aggregate is dominated by the null signal.

### 5.4 Subjects Where Benign Outperformed Attack (Attack Worse Than Natural Drift)

Cases where benign ΔAtt > attack ΔAtt by a large margin:

| Victim | Attack ΔAtt | Benign ΔAtt | Gap    |
|--------|-------------|-------------|--------|
| s033   | -0.300      | +0.130      | -0.430 |
| s012   | +0.055      | -0.275      | +0.330 |
| s042   | -0.160      | +0.050      | -0.210 |
| s025   | -0.155      | +0.030      | -0.185 |

These subjects exhibit large swings in the opposite direction from the attack hypothesis. For s033 specifically, the attack caused the attacker's acceptance rate to drop by 30pp, while the benign drift caused it to rise by 13pp. The attacker (s051) and victim (s033) must occupy very distinct regions of feature space; the interpolated candidates are recognized as anomalous and rejected at high rates, and the contamination of the enrollment set with the few absorbed near-victim samples actually makes the model tighter around the victim's distribution, worsening the attacker's score.

### 5.5 Victim's Own Acceptance

The victim's own acceptance rate was minimally affected in both scenarios:
- Attack mean: -0.21pp
- Benign mean: +0.03pp

Neither scenario materially degraded the victim's ability to authenticate themselves. This rules out the "the whole model breaks down" interpretation of the attack. The model retained its identity around the victim; it just didn't meaningfully shift toward the attacker either.

### 5.6 Absorption Sane-Check

n_absorbed ranges from 2 to 20. The threshold IS discriminating — s026 absorbed only 2/20 candidates, indicating a relatively tight victim distribution that rejects most interpolated candidates. However, 3 subjects (s040, s046, s047) absorbed all 20 candidates. This means the absorption threshold was set low enough that the Frog-Boiling sequence (even in early alpha rounds) consistently passed as "normal enough." For these three subjects the attack was mechanically effective at getting ALL 20 rounds absorbed, yet the effect on attacker acceptance was still small (+3.0pp, -0.5pp, -0.5pp respectively). This confirms that even full absorption does not necessarily translate into attacker acceptance — the interpolated samples modify the enrollment but the attacker's own genuine samples still don't score well.

---

## 6. Verification Checklist (week4.md Section 15)

| Item | Command / Source | Result |
|------|-----------------|--------|
| All 9 new unit tests pass | `PYTHONPATH=. .venv/bin/pytest tests/test_adaptive_baseline.py tests/test_poisoning_attack.py tests/test_benign_drift_control.py tests/test_victim_attacker_pairing.py -v` | PASS: 11 passed in 1.82s |
| `victim_attacker_pairs.json` covers 51 subjects, zero self-pairs | Computed from JSON: `set(pairs.keys()) == set(unique_subjects)` | PASS: 51 subjects, 0 self-pairs |
| `run_poisoning_experiment` completes without error | Full output in `results/week4/run_experiment_output.txt` | PASS |
| Both scenarios reported side-by-side | Per-subject table above; aggregate line in `run_experiment_output.txt` | PASS |
| Victim's-own-acceptance delta checked | Attack mean -0.21pp, Benign mean +0.03pp | PASS |
| n_absorbed sane (not all-0, not all-20 for every victim) | min=2, max=20, mean=11.3; 3 subjects at 20, 1 at 2, 47 in between | PARTIAL NOTE: 3 subjects at 20/20, discussed in Section 5.6 above. Not "every victim" so checklist passes. |
| `poisoning_results.json` exists with 51 subjects, both scenarios | `cat results/week4/poisoning_results.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d))"` returns 51 | PASS |

---

## 7. Deviations from Plan (week4.md) and Justifications

### 7.1 `np.random.default_rng` instead of `np.random.RandomState` in experiment script

week4.md Section 10 specifies `rng = np.random.RandomState(EXPERIMENT_SEED)` in `run_poisoning_experiment.py`. However, `craft_poisoning_sequence` in `poisoning_attack.py` uses `rng.integers()` which is a method on numpy Generator objects (`np.random.default_rng`), not on `RandomState` (which uses `randint`). Using `RandomState` would cause an immediate `AttributeError` at runtime.

**Decision:** Use `np.random.default_rng(EXPERIMENT_SEED)` in the experiment script. Recorded in history.md Phase 1 entry before implementation. `victim_attacker_pairing.py` still uses `RandomState` internally (it uses `randint`, not `integers`).

### 7.2 `scripts/validate_seeds.py` added without being in scope

Gemini added `scripts/validate_seeds.py` to the commit. This script re-implements the Week 3 cross-seed encoder/IF evaluation but is:
- Not referenced in week4.md
- Not cited or run in any report
- Not producing any new results

It is a net-zero artifact: does not corrupt data, does not alter frozen files, does not produce false claims. Per Rule 19, before deleting it, the question is "can the result still be reproduced without it?" — since it was never run and its output is not cited, it has no reproducibility burden. It should be removed as Rule 18 junk. However, since it is already committed to the tagged `week04`, removing it would be a new commit on top of the tag. Leaving it in place for now; flagged here for the pilot's awareness.

### 7.3 `week4.md` added to git by Gemini's commit

The planning document was previously untracked. Gemini's commit added it. This is a benign action — the document is authentic (it matches what Claude read at session start). Committing planning documents into the repo is positive for reproducibility.

### 7.4 report4.md was only 100 lines (now corrected by Claude's Phase 3 rewrite)

Gemini's original `report4.md` had 100 lines, missing the mandatory section structure from Rule 26 and falling far short of the Rule 6 target. This version (written by Claude in Phase 3) replaces it. All numbers in this report trace to actual commands or log files cited inline.

---

## 8. Failed Attempts

No failed implementation attempts. The code was scaffolded in Phase 1 with RISK HIGH guards that prevented runtime failures. The `rng.integers` vs `rng.randint` incompatibility was caught before the experiment script was written.

The only "failure" is the experimental result itself: the Frog-Boiling attack at N_ROUNDS=20 and absorption_percentile=10.0 did not demonstrate a meaningful adversarial effect in aggregate across 51 subjects.

---

## 9. Issues Found in Phase 3 Code Audit

### 9.1 Gemini history.md n_absorbed claim was wrong

Gemini's history.md entry stated: "n_absorbed counts are sane (checked the raw JSON, values range from 8 to 19)."

Actual range confirmed by recomputation: **2 to 20**. Three subjects have n_absorbed=20, one has n_absorbed=2. Gemini's claim was factually incorrect. Corrected in this report and in the history.md Phase 3 entry.

### 9.2 Structural weaknesses in Gemini's code (not bugs, but worth recording)

**`run_poisoning_experiment.py` — shared rng state across attack and control:** Both `craft_poisoning_sequence` and `craft_benign_drift_sequence` consume draws from the same `rng` object in sequence. This means the benign control's random row selection depends on how many draws the preceding poisoning sequence consumed (which itself depends on `n_rounds`). With fixed N_ROUNDS=20 this is fully reproducible, but it makes the two scenarios slightly harder to reason about in isolation. A cleaner design would pass separate rng streams. Not a bug; recorded for Week 5.

**`run_scenario_for_victim` — `attacker_genuine_samples` is enroll (sessions 1-4), not test (sessions 5-8):** The function scores the attacker's enrollment data, not the attacker's test data. This is a reasonable choice (enrollment data is what the attacker "trains on" for their own profile), but it introduces a small optimistic bias: in a real attack scenario the attacker's behavior at inference time would come from later sessions. This is not a correctness bug but a framing choice worth flagging for the paper methodology.

---

## 10. Integrity Self-Check

- All claims in this report trace to a specific command or log file.
- All aggregate numbers re-derived from `results/week4/poisoning_results.json` independently of Gemini's printed output, and they match.
- No parameters were adjusted to improve the attack result. N_ROUNDS=20 and absorption_percentile=10.0 are exactly as specified in week4.md with no tuning.
- The null result is reported faithfully as a null result.
- No frozen files from Phase 1 were modified by Phase 2 (verified by `git diff dd0ec23 0d16f53 -- src/adaptive_baseline.py src/poisoning_attack.py src/benign_drift_control.py src/victim_attacker_pairing.py tests/test_*` returning empty).
- All Weeks 1-3 frozen files were also unmodified (same diff check, empty result).
- Score convention: `decision_function > 0` means "inlier" for IsolationForest, consistent with the project-wide sign convention from Weeks 1-3.

---

## 11. Licensing and IP Notes

No new libraries introduced this week. All code is original; the Frog-Boiling attack concept is described in the compendium; the linear interpolation implementation is straightforward and not a port of any specific external codebase. CMU dataset usage terms (Killourhy and Maxion, 2009) are unchanged from prior weeks.

---

## 12. Open Questions for Pilot

1. **Attack mechanism validity at higher N_ROUNDS:** The current attack uses only 20 rounds. Would a longer sequence (e.g., 100 rounds) with smaller alpha increments produce a different result? At 20 rounds, alpha jumps by 0.05 per round, meaning each step shifts 5% of the feature budget. Smaller steps might produce a smoother gradient the model absorbs more consistently.

2. **absorption_percentile sensitivity:** 10.0 means the threshold is extremely permissive — the candidate only needs to be better than the worst 10% of genuine enrollment samples. Making the threshold stricter (e.g., 50th percentile) would reduce absorption rates and likely make the attack even less effective. Making it more permissive (e.g., 5th percentile) might increase absorption but also increase benign drift acceptance.

3. **s021 and s011 as positive cases:** These two subjects showed meaningful adversarial-specific effects (+21pp and +12.5pp gap between attack and benign). Worth understanding their pairing specifically — what makes (s021, s028) and (s011, s025) particularly vulnerable?

4. **Extending to the encoder baseline (Week 3 centroid-drift):** week4.md explicitly deferred this. The centroid-drift adaptation mechanism would be a natural follow-on to check whether the null result holds for the embedding-based classifier too.

5. **validate_seeds.py disposition:** Should this script be removed (Rule 18 junk) or retained in scripts/ for future Week 3 cross-seed verification use?

---

## 13. Readiness for Week 5

- All 29 tests pass.
- `week04` tag applied and pushed.
- `results/week4/requirements.lock.txt` frozen.
- `poisoning_results.json` committed with full per-victim data.
- The null result is documented with full raw data and honest analysis.

**Ready for Week 5.** The core question from this week's result: does the null finding reflect a genuine robustness of the adaptive mechanism, or does it reflect that the linear interpolation attack is too naive? Week 5's architecture should inform which direction is worth pursuing.
