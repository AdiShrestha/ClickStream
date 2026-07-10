# Weekly Report 5: Poisoning Defense (Residue-Feature / CUSUM Validation Gate)

## 1. Objectives
- Implement a residue-feature validation gate as `DefendedAdaptiveBaseline` extending Week 4's `AdaptiveBaseline`.
- Use a CUSUM (cumulative sum) change-detection statistic to catch persistent, directional drift across successive candidates.
- Calibrate the CUSUM threshold (`cusum_h`) exclusively on benign-drift data to target a specific false-alarm rate (~5%).
- Perform a full before/after comparison (undefended vs. defended) for both the confirmed attack (V1 point-interpolation, N_ROUNDS=200) and the benign control across all 51 victims.
- Execute a paired statistical significance test (t-test) on the difference in attacker acceptance deltas.
- Assess the defense's efficacy, its false-alarm rate, and explicitly state its limitations regarding genuinely permanent behavioral changes.

## 2. Environment
- **Device:** MacBook Air M3, 16GB unified memory, no NVIDIA GPU, no fan.
- **OS:** macOS 26.5.2
- **Python Libraries:** `scikit-learn>=1.5`, `numpy`, `scipy`
- **Git Branch/Tag State:** Executed from `week04-extension` tag base, prior to `week05` tag.
- **Algorithm:** `IsolationForest` (100 estimators) within a `PerUserModel` wrapper.
- **Random Seeds:** `EXPERIMENT_SEED = 123` (shared with Week 4 extension), `CALIBRATION_SEED = 789`

## 3. Raw results
### 3.1 CUSUM Threshold Calibration
Command: `python -m src.calibrate_cusum`
```text
Max-CUSUM distribution under 51 victims' benign drift (k=0.0):
  min=0.1323, median=3.9951, max=12.6180
  Setting h = 95th percentile = 9.7498 (targets ~5% false-alarm rate on benign drift)
```
Note: The spread from min (0.13) to max (12.61) is massive (~100x). The 95th percentile (9.74) is interpolating between the 2nd and 3rd highest max-CUSUM values, heavily influenced by the most erratic typists in the benign set. This confirms Pre-Mortem Risk #2 (Threshold instability from small n).

### 3.2 Full Defense Experiment Results
Command: `PYTHONPATH=. .venv/bin/python -m src.run_defense_experiment | tee results/week5/run_defense_experiment_output.txt`

#### Aggregate Statistics (51 Victims)
```text
UNDEFENDED attack: +8.20pp (std 18.11pp)
DEFENDED   attack: +8.07pp (std 18.15pp)
UNDEFENDED benign: -3.49pp (std 9.26pp)
DEFENDED   benign: -3.32pp (std 9.11pp)

Paired t-test (undefended vs defended, ATTACK scenario, n=51): t=0.876, p=0.38497

Defense trigger rate -- ATTACK scenario: mean 37.0/200 rounds
Defense trigger rate -- BENIGN scenario: mean 1.9/200 rounds
```

#### Detailed Per-Victim Data (Sorted by Subject ID)
| Victim | Attacker | EER Threshold | Undef Attack Δ | Def Attack Δ | Attack Triggers | Undef Benign Δ | Def Benign Δ | Benign Triggers |
|---|---|---|---|---|---|---|---|---|
| s002 | s046 | 0.0529 | -2.00pp | -8.00pp | 34/200 | -8.00pp | -3.00pp | 43/200 |
| s003 | s035 | 0.0356 | +18.50pp | +18.50pp | 3/200 | +1.00pp | +1.00pp | 0/200 |
| s004 | s020 | 0.0403 | -0.50pp | -2.00pp | 14/200 | -8.00pp | -8.00pp | 0/200 |
| s005 | s050 | 0.0566 | +19.50pp | +19.50pp | 0/200 | -8.00pp | -8.00pp | 0/200 |
| s007 | s012 | 0.0339 | +0.00pp | +0.00pp | 73/200 | +0.50pp | +0.50pp | 0/200 |
| s008 | s027 | 0.0508 | +9.00pp | +9.00pp | 0/200 | -0.50pp | -0.50pp | 0/200 |
| s010 | s046 | 0.0401 | +0.00pp | +0.00pp | 76/200 | +0.00pp | +0.00pp | 0/200 |
| s011 | s025 | 0.0538 | +17.50pp | +17.50pp | 0/200 | +0.00pp | +0.00pp | 0/200 |
| s012 | s029 | 0.0861 | +10.00pp | +10.00pp | 0/200 | +0.00pp | +0.00pp | 0/200 |
| s013 | s016 | 0.0310 | +0.00pp | +0.00pp | 105/200 | +0.00pp | +0.00pp | 0/200 |
| s015 | s016 | 0.0463 | +0.00pp | +0.00pp | 67/200 | +0.00pp | +0.00pp | 0/200 |
| s016 | s030 | -0.0023 | +19.50pp | +19.50pp | 0/200 | +4.00pp | +4.00pp | 0/200 |
| s017 | s042 | 0.0070 | -3.50pp | -3.50pp | 78/200 | -3.50pp | -3.50pp | 0/200 |
| s018 | s047 | 0.0932 | -0.50pp | -0.50pp | 37/200 | -0.50pp | -0.50pp | 0/200 |
| s019 | s030 | 0.0448 | +0.00pp | +0.00pp | 67/200 | -2.00pp | -2.00pp | 0/200 |
| s020 | s004 | 0.1047 | +6.00pp | +6.00pp | 0/200 | -25.00pp | -25.00pp | 0/200 |
| s021 | s028 | 0.0426 | +16.00pp | +16.00pp | 0/200 | -0.50pp | -0.50pp | 0/200 |
| s022 | s003 | 0.0349 | +0.00pp | +0.00pp | 93/200 | +0.00pp | +0.00pp | 0/200 |
| s024 | s030 | 0.0323 | +0.00pp | +0.00pp | 84/200 | +0.00pp | +0.00pp | 0/200 |
| s025 | s051 | 0.0286 | -20.50pp | -20.50pp | 0/200 | -26.50pp | -26.50pp | 0/200 |
| s026 | s036 | 0.0350 | +0.00pp | +0.00pp | 111/200 | +0.00pp | +0.00pp | 0/200 |
| s027 | s044 | 0.0339 | +0.00pp | +0.50pp | 50/200 | +0.00pp | +0.00pp | 0/200 |
| s028 | s003 | -0.0019 | +0.00pp | +0.00pp | 105/200 | +0.00pp | +0.00pp | 0/200 |
| s029 | s026 | 0.0166 | +14.00pp | +14.00pp | 0/200 | -5.50pp | -5.50pp | 0/200 |
| s030 | s039 | 0.0503 | +39.00pp | +39.00pp | 0/200 | -36.00pp | -36.00pp | 0/200 |
| s031 | s016 | 0.0508 | +3.50pp | +3.00pp | 12/200 | +0.00pp | +0.00pp | 0/200 |
| s032 | s027 | 0.0494 | +10.00pp | +10.00pp | 0/200 | -23.00pp | -19.50pp | 45/200 |
| s033 | s051 | 0.0461 | +66.00pp | +66.00pp | 0/200 | +0.50pp | +0.50pp | 0/200 |
| s034 | s030 | 0.0820 | +0.00pp | -0.50pp | 63/200 | -0.50pp | -0.50pp | 9/200 |
| s035 | s056 | 0.0245 | +13.00pp | +13.00pp | 0/200 | +1.50pp | +1.50pp | 0/200 |
| s036 | s032 | -0.0032 | +0.00pp | +0.00pp | 95/200 | +0.00pp | +0.00pp | 0/200 |
| s037 | s049 | 0.0528 | +0.00pp | +0.00pp | 102/200 | +0.00pp | +0.00pp | 0/200 |
| s038 | s033 | 0.0844 | +0.00pp | +0.00pp | 60/200 | +0.00pp | +0.00pp | 0/200 |
| s039 | s020 | 0.0436 | +0.00pp | +0.00pp | 55/200 | +0.00pp | +0.00pp | 0/200 |
| s040 | s019 | 0.0490 | +77.00pp | +77.00pp | 0/200 | -4.50pp | -4.50pp | 0/200 |
| s041 | s054 | 0.0504 | +59.00pp | +59.00pp | 0/200 | -13.00pp | -13.00pp | 0/200 |
| s042 | s051 | 0.0448 | -0.50pp | +0.00pp | 9/200 | -12.50pp | -12.50pp | 0/200 |
| s043 | s004 | 0.0267 | +0.00pp | +0.00pp | 58/200 | +0.00pp | +0.00pp | 0/200 |
| s044 | s042 | 0.0708 | -15.00pp | -11.50pp | 29/200 | -19.00pp | -19.00pp | 0/200 |
| s046 | s010 | 0.1022 | +34.00pp | +34.00pp | 0/200 | +27.50pp | +27.50pp | 0/200 |
| s047 | s026 | 0.0490 | +0.50pp | +0.50pp | 0/200 | -2.50pp | -2.50pp | 0/200 |
| s048 | s012 | 0.0500 | +0.00pp | +0.00pp | 35/200 | +0.00pp | +0.00pp | 0/200 |
| s049 | s044 | 0.0680 | +26.00pp | +26.00pp | 0/200 | +1.00pp | +1.00pp | 0/200 |
| s050 | s022 | 0.0513 | +0.00pp | +0.00pp | 87/200 | +0.00pp | +0.00pp | 0/200 |
| s051 | s005 | 0.0332 | +0.00pp | +0.00pp | 46/200 | +0.00pp | +0.00pp | 0/200 |
| s052 | s030 | -0.0057 | +0.00pp | +0.00pp | 83/200 | +0.00pp | +0.00pp | 0/200 |
| s053 | s018 | 0.0489 | +2.50pp | +1.00pp | 21/200 | -1.00pp | -1.00pp | 0/200 |
| s054 | s057 | 0.0673 | +0.00pp | +0.00pp | 0/200 | -10.50pp | -10.50pp | 0/200 |
| s055 | s012 | 0.0336 | +0.00pp | +0.00pp | 79/200 | +0.00pp | +0.00pp | 0/200 |
| s056 | s031 | 0.0746 | -0.50pp | -1.50pp | 40/200 | -1.50pp | -1.50pp | 0/200 |
| s057 | s003 | 0.0440 | +0.50pp | +0.50pp | 15/200 | -2.00pp | -2.00pp | 0/200 |

## 4. Failed attempts
- **Global Threshold Failure:** The defense completely failed to mitigate the attack. The undefended attack yielded +8.20pp, while the defended attack yielded +8.07pp. The p-value of 0.385 means this difference is statistically indistinguishable from zero. The failure is directly tied to the global `cusum_h` threshold. Because it was calibrated on the 95th percentile of all subjects, it accommodated the noisiest subjects, leaving the stable subjects (who are actually vulnerable to poisoning) completely exposed. For example, `s040` (the most vulnerable subject, +77.00pp) had **0 triggers** out of 200 rounds. The CUSUM state never breached the massively inflated 9.7498 global threshold.
- **Performance/Throttling:** The script executed over 51,000 fits of `IsolationForest`. On the M3 Air (fanless), this took roughly an hour due to severe thermal throttling, whereas the non-throttled time would be <10 minutes. The lack of standard output during the run was misidentified initially as a hang, but was simply Python buffering the output block-by-block through the `tee` pipe.

## 5. Deviations from plan
- No code deviations from the explicit `week5.md` specification. The defense algorithm, calibration protocol, and test scenarios were executed strictly as provided.
- In Phase 1, the `run_defense_experiment.py` terminal runner was deferred to Phase 2 to mirror the safe architecture of Week 4. To compensate and ensure safety, a `scratch/test_calibrate_interface.py` script was written to validate that the interface boundary between the Phase 1 functions and the Phase 2 script was completely sound.

## 6. Integrity self-check
- **Are there 9 sections?** Yes.
- **Are all results backed by raw commands?** Yes, both the calibration script and the main experiment script are listed with their exact execution commands.
- **Pre-Mortem Mitigation Checklist:**
  - *Shared Mutable State:* Validated. `DefendedAdaptiveBaseline` utilizes `default_factory=list` correctly, and the experiment script calls `.initialize(victim_enroll.copy())` separately for every single scenario (undefended/defended, attack/benign).
  - *Threshold Instability:* Validated. The calibration min/median/max spread was printed and documented. It confirms the extreme variance (0.13 to 12.61).
  - *Data Reuse/Leakage:* Validated. `victim_later` is strictly used as the benign sequence pool and for `eer_threshold` computation. The acceptance metric is strictly calculated against `attacker_genuine_samples` (sessions 1-4).
  - *Silent Success at Extremes:* Validated. The CUSUM state correctly triggered for some subjects (e.g., `s013` triggered 105 times), proving the threshold logic works mechanically, even though it failed statistically on aggregate.
- **Range/Distribution from partial sample:** All numbers in Section 3 are aggregates derived from full `defense_results.json` parsing, recomputed independently in `scratch/recompute_results.py`.
- **Unreviewed file commits:** Only `src/run_defense_experiment.py` and `Weekly Reports/report5.md` are being staged in Phase 2. No `git add .` was used.

## 7. Licensing and IP notes
- The CUSUM algorithm implementation is standard change-detection mathematics and does not infringe on specific proprietary architectures.
- All libraries (`scikit-learn`, `numpy`, `scipy`) are open source (BSD/NumPy licenses) and fully compatible with IEEE publication constraints.

## 8. Open questions for pilot
- **Defense Efficacy:** The global CUSUM threshold (`h = 9.7498`) completely neutralized the defense for the subjects who needed it most. Stable subjects were poisoned without their residuals ever exceeding the threshold set by erratic subjects. Should we explore a **per-victim CUSUM threshold** in Week 6 (where `h` is calibrated strictly on the victim's own variance), or pivot to a fundamentally different defense mechanism?
- **Dataset Size:** The 95th percentile of n=51 is highly unstable. If we stick to CUSUM, should we inject synthetic benign drift to stabilize the calibration pool?

## 9. Readiness for next week
- **Status:** READY. Week 5's null result is statistically robust, cleanly documented, and the mechanistic failure point (global threshold on highly variable population) is fully understood. No further code changes are needed to validate this failure.
- **Known Limitations (Honest Accounting):** As stated in the Week 5 plan, the `DefendedAdaptiveBaseline` anchors its residual to a *fixed reference* computed from the original enrollment. This implies that any genuinely permanent behavioral shift (e.g., typing one-handed due to injury) will eventually trigger the CUSUM alarm, as it presents as a sustained directional drift identical to poisoning. The benign-drift control (sessions 5-8) tests day-to-day variance, not permanent injury, so our false-alarm rate of ~1% does *not* cover permanent behavioral shifts.

---
## Appendix: Raw JSON Dump (`results/week5/defense_results.json`)
```json
{
  "cusum_h": 9.749783316830136,
  "cusum_k": 0.0,
  "per_victim": {
    "s002": {
      "attacker": "s046",
      "eer_threshold": 0.05288012028838718,
      "undefended_attack": {
        "accept_before": 0.28,
        "accept_after": 0.26,
        "delta": -0.020000000000000018,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.28,
        "accept_after": 0.2,
        "delta": -0.08000000000000002,
        "n_defense_triggers": 34
      },
      "undefended_benign": {
        "accept_before": 0.28,
        "accept_after": 0.2,
        "delta": -0.08000000000000002,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.28,
        "accept_after": 0.25,
        "delta": -0.030000000000000027,
        "n_defense_triggers": 43
      }
    },
    "s003": {
      "attacker": "s035",
      "eer_threshold": 0.03558706046157328,
      "undefended_attack": {
        "accept_before": 0.075,
        "accept_after": 0.26,
        "delta": 0.185,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.075,
        "accept_after": 0.26,
        "delta": 0.185,
        "n_defense_triggers": 3
      },
      "undefended_benign": {
        "accept_before": 0.075,
        "accept_after": 0.085,
        "delta": 0.010000000000000009,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.075,
        "accept_after": 0.085,
        "delta": 0.010000000000000009,
        "n_defense_triggers": 0
      }
    },
    "s004": {
      "attacker": "s020",
      "eer_threshold": 0.04033349589361851,
      "undefended_attack": {
        "accept_before": 0.095,
        "accept_after": 0.09,
        "delta": -0.0050000000000000044,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.095,
        "accept_after": 0.075,
        "delta": -0.020000000000000004,
        "n_defense_triggers": 14
      },
      "undefended_benign": {
        "accept_before": 0.095,
        "accept_after": 0.015,
        "delta": -0.08,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.095,
        "accept_after": 0.015,
        "delta": -0.08,
        "n_defense_triggers": 0
      }
    },
    "s005": {
      "attacker": "s050",
      "eer_threshold": 0.05657029715622791,
      "undefended_attack": {
        "accept_before": 0.16,
        "accept_after": 0.355,
        "delta": 0.19499999999999998,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.16,
        "accept_after": 0.355,
        "delta": 0.19499999999999998,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.16,
        "accept_after": 0.08,
        "delta": -0.08,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.16,
        "accept_after": 0.08,
        "delta": -0.08,
        "n_defense_triggers": 0
      }
    },
    "s007": {
      "attacker": "s012",
      "eer_threshold": 0.03390514263161559,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 73
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.005,
        "delta": 0.005,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.005,
        "delta": 0.005,
        "n_defense_triggers": 0
      }
    },
    "s008": {
      "attacker": "s027",
      "eer_threshold": 0.050763254313963,
      "undefended_attack": {
        "accept_before": 0.005,
        "accept_after": 0.095,
        "delta": 0.09,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.005,
        "accept_after": 0.095,
        "delta": 0.09,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.005,
        "accept_after": 0.0,
        "delta": -0.005,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.005,
        "accept_after": 0.0,
        "delta": -0.005,
        "n_defense_triggers": 0
      }
    },
    "s010": {
      "attacker": "s046",
      "eer_threshold": 0.04009456775231196,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 76
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s011": {
      "attacker": "s025",
      "eer_threshold": 0.05382316241423918,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.175,
        "delta": 0.175,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.175,
        "delta": 0.175,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s012": {
      "attacker": "s029",
      "eer_threshold": 0.08607051155821277,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.1,
        "delta": 0.1,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.1,
        "delta": 0.1,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s013": {
      "attacker": "s016",
      "eer_threshold": 0.030986219813059157,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 105
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s015": {
      "attacker": "s016",
      "eer_threshold": 0.04634054064647092,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 67
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s016": {
      "attacker": "s030",
      "eer_threshold": -0.0022824866699501234,
      "undefended_attack": {
        "accept_before": 0.305,
        "accept_after": 0.5,
        "delta": 0.195,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.305,
        "accept_after": 0.5,
        "delta": 0.195,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.305,
        "accept_after": 0.345,
        "delta": 0.03999999999999998,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.305,
        "accept_after": 0.345,
        "delta": 0.03999999999999998,
        "n_defense_triggers": 0
      }
    },
    "s017": {
      "attacker": "s042",
      "eer_threshold": 0.007021574815890108,
      "undefended_attack": {
        "accept_before": 0.035,
        "accept_after": 0.0,
        "delta": -0.035,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.035,
        "accept_after": 0.0,
        "delta": -0.035,
        "n_defense_triggers": 78
      },
      "undefended_benign": {
        "accept_before": 0.035,
        "accept_after": 0.0,
        "delta": -0.035,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.035,
        "accept_after": 0.0,
        "delta": -0.035,
        "n_defense_triggers": 0
      }
    },
    "s018": {
      "attacker": "s047",
      "eer_threshold": 0.09320921254138526,
      "undefended_attack": {
        "accept_before": 0.005,
        "accept_after": 0.0,
        "delta": -0.005,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.005,
        "accept_after": 0.0,
        "delta": -0.005,
        "n_defense_triggers": 37
      },
      "undefended_benign": {
        "accept_before": 0.005,
        "accept_after": 0.0,
        "delta": -0.005,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.005,
        "accept_after": 0.0,
        "delta": -0.005,
        "n_defense_triggers": 0
      }
    },
    "s019": {
      "attacker": "s030",
      "eer_threshold": 0.04480103735847696,
      "undefended_attack": {
        "accept_before": 0.02,
        "accept_after": 0.02,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.02,
        "accept_after": 0.02,
        "delta": 0.0,
        "n_defense_triggers": 67
      },
      "undefended_benign": {
        "accept_before": 0.02,
        "accept_after": 0.0,
        "delta": -0.02,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.02,
        "accept_after": 0.0,
        "delta": -0.02,
        "n_defense_triggers": 0
      }
    },
    "s020": {
      "attacker": "s004",
      "eer_threshold": 0.10474260506257915,
      "undefended_attack": {
        "accept_before": 0.455,
        "accept_after": 0.515,
        "delta": 0.06,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.455,
        "accept_after": 0.515,
        "delta": 0.06,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.455,
        "accept_after": 0.205,
        "delta": -0.25,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.455,
        "accept_after": 0.205,
        "delta": -0.25,
        "n_defense_triggers": 0
      }
    },
    "s021": {
      "attacker": "s028",
      "eer_threshold": 0.0426095816998916,
      "undefended_attack": {
        "accept_before": 0.005,
        "accept_after": 0.165,
        "delta": 0.16,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.005,
        "accept_after": 0.165,
        "delta": 0.16,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.005,
        "accept_after": 0.0,
        "delta": -0.005,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.005,
        "accept_after": 0.0,
        "delta": -0.005,
        "n_defense_triggers": 0
      }
    },
    "s022": {
      "attacker": "s003",
      "eer_threshold": 0.03485734520372824,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 93
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s024": {
      "attacker": "s030",
      "eer_threshold": 0.03225660655401741,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 84
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s025": {
      "attacker": "s051",
      "eer_threshold": 0.028581943480408656,
      "undefended_attack": {
        "accept_before": 0.42,
        "accept_after": 0.215,
        "delta": -0.205,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.42,
        "accept_after": 0.215,
        "delta": -0.205,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.42,
        "accept_after": 0.155,
        "delta": -0.265,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.42,
        "accept_after": 0.155,
        "delta": -0.265,
        "n_defense_triggers": 0
      }
    },
    "s026": {
      "attacker": "s036",
      "eer_threshold": 0.035024758364353314,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 111
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s027": {
      "attacker": "s044",
      "eer_threshold": 0.03389312197322336,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.005,
        "delta": 0.005,
        "n_defense_triggers": 50
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s028": {
      "attacker": "s003",
      "eer_threshold": -0.0018881057281283642,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 105
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s029": {
      "attacker": "s026",
      "eer_threshold": 0.016557810070012335,
      "undefended_attack": {
        "accept_before": 0.37,
        "accept_after": 0.51,
        "delta": 0.14,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.37,
        "accept_after": 0.51,
        "delta": 0.14,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.37,
        "accept_after": 0.315,
        "delta": -0.05499999999999999,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.37,
        "accept_after": 0.315,
        "delta": -0.05499999999999999,
        "n_defense_triggers": 0
      }
    },
    "s030": {
      "attacker": "s039",
      "eer_threshold": 0.050346496617441006,
      "undefended_attack": {
        "accept_before": 0.385,
        "accept_after": 0.775,
        "delta": 0.39,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.385,
        "accept_after": 0.775,
        "delta": 0.39,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.385,
        "accept_after": 0.025,
        "delta": -0.36,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.385,
        "accept_after": 0.025,
        "delta": -0.36,
        "n_defense_triggers": 0
      }
    },
    "s031": {
      "attacker": "s016",
      "eer_threshold": 0.05076749175089729,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.035,
        "delta": 0.035,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.03,
        "delta": 0.03,
        "n_defense_triggers": 12
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s032": {
      "attacker": "s027",
      "eer_threshold": 0.049438084059238385,
      "undefended_attack": {
        "accept_before": 0.32,
        "accept_after": 0.42,
        "delta": 0.09999999999999998,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.32,
        "accept_after": 0.42,
        "delta": 0.09999999999999998,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.32,
        "accept_after": 0.09,
        "delta": -0.23,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.32,
        "accept_after": 0.125,
        "delta": -0.195,
        "n_defense_triggers": 45
      }
    },
    "s033": {
      "attacker": "s051",
      "eer_threshold": 0.0461262225959852,
      "undefended_attack": {
        "accept_before": 0.01,
        "accept_after": 0.67,
        "delta": 0.66,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.01,
        "accept_after": 0.67,
        "delta": 0.66,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.01,
        "accept_after": 0.015,
        "delta": 0.004999999999999999,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.01,
        "accept_after": 0.015,
        "delta": 0.004999999999999999,
        "n_defense_triggers": 0
      }
    },
    "s034": {
      "attacker": "s030",
      "eer_threshold": 0.08200495930555024,
      "undefended_attack": {
        "accept_before": 0.005,
        "accept_after": 0.005,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.005,
        "accept_after": 0.0,
        "delta": -0.005,
        "n_defense_triggers": 63
      },
      "undefended_benign": {
        "accept_before": 0.005,
        "accept_after": 0.0,
        "delta": -0.005,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.005,
        "accept_after": 0.0,
        "delta": -0.005,
        "n_defense_triggers": 9
      }
    },
    "s035": {
      "attacker": "s056",
      "eer_threshold": 0.02451837367525478,
      "undefended_attack": {
        "accept_before": 0.095,
        "accept_after": 0.225,
        "delta": 0.13,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.095,
        "accept_after": 0.225,
        "delta": 0.13,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.095,
        "accept_after": 0.11,
        "delta": 0.015,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.095,
        "accept_after": 0.11,
        "delta": 0.015,
        "n_defense_triggers": 0
      }
    },
    "s036": {
      "attacker": "s032",
      "eer_threshold": -0.0032128258590731207,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 95
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s037": {
      "attacker": "s049",
      "eer_threshold": 0.052770883787937206,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 102
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s038": {
      "attacker": "s033",
      "eer_threshold": 0.08443136204973112,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 60
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s039": {
      "attacker": "s020",
      "eer_threshold": 0.04360350586325484,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 55
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s040": {
      "attacker": "s019",
      "eer_threshold": 0.04899384876063917,
      "undefended_attack": {
        "accept_before": 0.07,
        "accept_after": 0.84,
        "delta": 0.77,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.07,
        "accept_after": 0.84,
        "delta": 0.77,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.07,
        "accept_after": 0.025,
        "delta": -0.045000000000000005,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.07,
        "accept_after": 0.025,
        "delta": -0.045000000000000005,
        "n_defense_triggers": 0
      }
    },
    "s041": {
      "attacker": "s054",
      "eer_threshold": 0.05040867904795909,
      "undefended_attack": {
        "accept_before": 0.16,
        "accept_after": 0.75,
        "delta": 0.59,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.16,
        "accept_after": 0.75,
        "delta": 0.59,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.16,
        "accept_after": 0.03,
        "delta": -0.13,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.16,
        "accept_after": 0.03,
        "delta": -0.13,
        "n_defense_triggers": 0
      }
    },
    "s042": {
      "attacker": "s051",
      "eer_threshold": 0.044848829326551765,
      "undefended_attack": {
        "accept_before": 0.185,
        "accept_after": 0.18,
        "delta": -0.0050000000000000044,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.185,
        "accept_after": 0.185,
        "delta": 0.0,
        "n_defense_triggers": 9
      },
      "undefended_benign": {
        "accept_before": 0.185,
        "accept_after": 0.06,
        "delta": -0.125,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.185,
        "accept_after": 0.06,
        "delta": -0.125,
        "n_defense_triggers": 0
      }
    },
    "s043": {
      "attacker": "s004",
      "eer_threshold": 0.02668724574120379,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 58
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s044": {
      "attacker": "s042",
      "eer_threshold": 0.0707658274665316,
      "undefended_attack": {
        "accept_before": 0.195,
        "accept_after": 0.045,
        "delta": -0.15000000000000002,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.195,
        "accept_after": 0.08,
        "delta": -0.115,
        "n_defense_triggers": 29
      },
      "undefended_benign": {
        "accept_before": 0.195,
        "accept_after": 0.005,
        "delta": -0.19,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.195,
        "accept_after": 0.005,
        "delta": -0.19,
        "n_defense_triggers": 0
      }
    },
    "s046": {
      "attacker": "s010",
      "eer_threshold": 0.10218362173390971,
      "undefended_attack": {
        "accept_before": 0.11,
        "accept_after": 0.45,
        "delta": 0.34,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.11,
        "accept_after": 0.45,
        "delta": 0.34,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.11,
        "accept_after": 0.385,
        "delta": 0.275,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.11,
        "accept_after": 0.385,
        "delta": 0.275,
        "n_defense_triggers": 0
      }
    },
    "s047": {
      "attacker": "s026",
      "eer_threshold": 0.04896779380661487,
      "undefended_attack": {
        "accept_before": 0.925,
        "accept_after": 0.93,
        "delta": 0.0050000000000000044,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.925,
        "accept_after": 0.93,
        "delta": 0.0050000000000000044,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.925,
        "accept_after": 0.9,
        "delta": -0.025000000000000022,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.925,
        "accept_after": 0.9,
        "delta": -0.025000000000000022,
        "n_defense_triggers": 0
      }
    },
    "s048": {
      "attacker": "s012",
      "eer_threshold": 0.0500287429382138,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 35
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s049": {
      "attacker": "s044",
      "eer_threshold": 0.0679535666841653,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.26,
        "delta": 0.26,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.26,
        "delta": 0.26,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.01,
        "delta": 0.01,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.01,
        "delta": 0.01,
        "n_defense_triggers": 0
      }
    },
    "s050": {
      "attacker": "s022",
      "eer_threshold": 0.051309658360954136,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 87
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s051": {
      "attacker": "s005",
      "eer_threshold": 0.033176958214303665,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 46
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s052": {
      "attacker": "s030",
      "eer_threshold": -0.005740074094168657,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 83
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s053": {
      "attacker": "s018",
      "eer_threshold": 0.04885024121491255,
      "undefended_attack": {
        "accept_before": 0.01,
        "accept_after": 0.035,
        "delta": 0.025,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.01,
        "accept_after": 0.02,
        "delta": 0.01,
        "n_defense_triggers": 21
      },
      "undefended_benign": {
        "accept_before": 0.01,
        "accept_after": 0.0,
        "delta": -0.01,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.01,
        "accept_after": 0.0,
        "delta": -0.01,
        "n_defense_triggers": 0
      }
    },
    "s054": {
      "attacker": "s057",
      "eer_threshold": 0.06727286461712428,
      "undefended_attack": {
        "accept_before": 0.405,
        "accept_after": 0.405,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.405,
        "accept_after": 0.405,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "undefended_benign": {
        "accept_before": 0.405,
        "accept_after": 0.3,
        "delta": -0.10500000000000004,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.405,
        "accept_after": 0.3,
        "delta": -0.10500000000000004,
        "n_defense_triggers": 0
      }
    },
    "s055": {
      "attacker": "s012",
      "eer_threshold": 0.03364084684355134,
      "undefended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 79
      },
      "undefended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.0,
        "accept_after": 0.0,
        "delta": 0.0,
        "n_defense_triggers": 0
      }
    },
    "s056": {
      "attacker": "s031",
      "eer_threshold": 0.07456736001034353,
      "undefended_attack": {
        "accept_before": 0.015,
        "accept_after": 0.01,
        "delta": -0.004999999999999999,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.015,
        "accept_after": 0.0,
        "delta": -0.015,
        "n_defense_triggers": 40
      },
      "undefended_benign": {
        "accept_before": 0.015,
        "accept_after": 0.0,
        "delta": -0.015,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.015,
        "accept_after": 0.0,
        "delta": -0.015,
        "n_defense_triggers": 0
      }
    },
    "s057": {
      "attacker": "s003",
      "eer_threshold": 0.043995499689908846,
      "undefended_attack": {
        "accept_before": 0.02,
        "accept_after": 0.025,
        "delta": 0.005000000000000001,
        "n_defense_triggers": 0
      },
      "defended_attack": {
        "accept_before": 0.02,
        "accept_after": 0.025,
        "delta": 0.005000000000000001,
        "n_defense_triggers": 15
      },
      "undefended_benign": {
        "accept_before": 0.02,
        "accept_after": 0.0,
        "delta": -0.02,
        "n_defense_triggers": 0
      },
      "defended_benign": {
        "accept_before": 0.02,
        "accept_after": 0.0,
        "delta": -0.02,
        "n_defense_triggers": 0
      }
    }
  }
}
```