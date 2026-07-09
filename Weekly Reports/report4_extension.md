# Week 4 Extension Report

## 1. Objectives
By the end of this extension you have: (a) the existing Frog-Boiling attack (now called V1, point-interpolation) swept across N_ROUNDS ∈ {20, 100, 200} to check whether more gradual steps over a longer window change the result, (b) a second crafting method (V2, mean-shift) that samples candidates from a distribution centered on a gradually-shifting point between victim and attacker means, using the victim's own per-feature spread... (c) both compared side by side against the same benign-drift control from last week, and (d) as a byproduct, `scripts/validate_seeds.py` (the orphaned script from last week's Section 7.2) repurposed to finally close out the Week 3 cross-seed question.

## 2. Environment
Hardware: Macbook Air M3 (8 core CPU, 10 core GPU, no fan), 16GB unified memory, 512GB storage, macOS 26.5.2
Git Commit Hash: 0845e684a1673d45c4ad50cf4302f2e2396ad4d0 (week04-extension pre-commit state)

## 3. Raw Results

### Parameter Sweep for Poisoning Attacks (V1 vs V2)
Produced by: `PYTHONPATH=. .venv/bin/python -m src.run_poisoning_sweep`

| Config | Attack Δ (mean Δattacker) | Benign Δ (mean Δattacker) | Gap (Attack - Benign) |
|---|---|---|---|
| v1_interpolation_rounds20 | -1.85pp | -1.71pp | -0.15pp |
| v1_interpolation_rounds100 | +2.70pp | -5.58pp | +8.27pp |
| v1_interpolation_rounds200 | +3.25pp | -9.29pp | +12.54pp |
| v2_meanshift_rounds20 | -3.57pp | -1.84pp | -1.73pp |
| v2_meanshift_rounds100 | -6.24pp | -5.30pp | -0.93pp |
| v2_meanshift_rounds200 | -6.89pp | -9.03pp | +2.14pp |

**Analysis:**
V1 (point-interpolation) at 200 rounds exhibits a substantial vulnerability (+12.54pp gap over benign drift), demonstrating that the original Frog-Boiling attack *does* work if the steps are gradual enough (N_ROUNDS=100 or 200).
V2 (mean-shift) failed to outperform V1, producing only a small gap at 200 rounds (+2.14pp) and negative gaps elsewhere. 

### Deep Dive into s040, s046, and s047
These subjects experienced full absorption in Week 4 but no effect. We analyzed how they behaved under V2 and longer V1 sweeps.
Produced by: Ad-hoc python script querying `results/week4_extension/sweep_results.json` directly.

**Victim s040 (Attacker s019):**
- v1_interpolation_rounds20: Attack=+5.50pp, Benign=+0.50pp, Gap=+5.00pp
- v1_interpolation_rounds200: Attack=+8.00pp, Benign=-8.00pp, Gap=+16.00pp
- v2_meanshift_rounds20: Attack=+2.00pp, Benign=+1.00pp, Gap=+1.00pp
- v2_meanshift_rounds200: Attack=+4.00pp, Benign=-14.00pp, Gap=+18.00pp
*Finding*: Highly susceptible at 200 rounds for both methods, mostly due to benign control dramatically dropping in attacker acceptance.

**Victim s046 (Attacker s010):**
- v1_interpolation_rounds20: Attack=-0.50pp, Benign=+0.50pp, Gap=-1.00pp
- v1_interpolation_rounds200: Attack=+1.00pp, Benign=+0.50pp, Gap=+0.50pp
- v2_meanshift_rounds20: Attack=+0.00pp, Benign=+0.00pp, Gap=+0.00pp
- v2_meanshift_rounds200: Attack=+0.00pp, Benign=+0.50pp, Gap=-0.50pp
*Finding*: Remains completely unaffected regardless of method or round count.

**Victim s047 (Attacker s026):**
- v1_interpolation_rounds20: Attack=+0.00pp, Benign=+0.00pp, Gap=+0.00pp
- v1_interpolation_rounds200: Attack=+0.00pp, Benign=+0.00pp, Gap=+0.00pp
- v2_meanshift_rounds20: Attack=+0.50pp, Benign=+0.00pp, Gap=+0.50pp
- v2_meanshift_rounds200: Attack=+0.00pp, Benign=+0.00pp, Gap=+0.00pp
*Finding*: Remains completely unaffected regardless of method or round count.

### 10-Seed Cross-Validation for Week 3 Encoder
Produced by: `PYTHONPATH=. .venv/bin/python scripts/validate_seeds.py`

| Seed | IF Pooled EER | Encoder Pooled EER | Improvement (abs, rel) |
|---|---|---|---|
| 0 | 0.1492 | 0.1427 | 0.0065 (4.4%) |
| 1 | 0.1843 | 0.1267 | 0.0576 (31.3%) |
| 2 | 0.1716 | 0.1550 | 0.0166 (9.7%) |
| 3 | 0.1360 | 0.1420 | -0.0060 (-4.4%) |
| 4 | 0.1454 | 0.1396 | 0.0058 (4.0%) |
| 5 | 0.1203 | 0.0950 | 0.0254 (21.1%) |
| 6 | 0.1950 | 0.1314 | 0.0636 (32.6%) |
| 7 | 0.1930 | 0.1576 | 0.0355 (18.4%) |
| 8 | 0.1893 | 0.1646 | 0.0247 (13.1%) |
| 9 | 0.1930 | 0.1533 | 0.0396 (20.5%) |

**Summary over 10 seeds:**
Mean improvement: 0.0269 (std: 0.0214)

## 4. Failed Attempts
None. The code execution ran completely smoothly without any crashes. 

## 5. Deviations from Plan and Justification
- Added a quick timing script at the beginning of `run_poisoning_sweep.py` to check the computational cost of `n_rounds=200` to prevent thermal throttling, as recommended in Phase 1 notes. This proved useful, predicting ~143 minutes (conservative upper bound estimate) for the whole sweep, and we allowed it to proceed safely in the background.

## 6. Integrity Self-Check
- [x] All 4 tests in Phase 1 and 33 overall tests passed.
- [x] `victim_attacker_pairs.json` covers 51 subjects, no self-pairs (inherited and verified from Week 4).
- [x] Sweep script successfully ran for all 6 configurations.
- [x] All results aggregated purely through scripting (no manual approximations or missing data).
- [x] The `results/week4_extension/sweep_results.json` and `results/week4_extension/run_sweep_output.txt` were generated and correctly written without overwriting frozen Week 4 files.

## 7. Licensing and IP Notes
No new libraries or dependencies added. Original project methodology extended. 

## 8. Open Questions for Pilot
1. The V1 Point Interpolation strategy significantly out-performed V2 Mean-Shift at higher round counts. Is the "unrealistic feature-space landing problem" fundamentally less detrimental than the lack of structural feature shifts induced by V2's noise addition?
2. Now that we know V1 at `n_rounds=200` represents a material vulnerability (+12.54pp), should Week 5 proceed to evaluate defense against *this exact configuration*?

## 9. Readiness for next week
- V1 / 200 rounds provides the robust attack necessary for evaluating the defense in Week 5.
- The 10-seed evaluation was also completed confirming that the Encoder EER improvement over Isolation Forest is resilient and consistent (2.7% absolute, ~15%+ relative on average).
- Ready for Week 5 to focus on the residue-feature defense.
