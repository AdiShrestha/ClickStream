# AI Use Log

## Format: Date | Tool | Task | Level of use | What I verified or changed

2026-07-08 | Claude | Week 1 project setup: repo skeleton, data acquisition, feature extraction, EDA, threat model, ethics statement | Planning and level 3 code scaffold reviewed against week1.md specification | Architecture decisions (repo root placement, import paths, eda.py sys.path fallback) are independent judgment; all code cross-checked line by line against week1.md

2026-07-08 | Claude | Week 2 implementation: temporal split, metrics, negative control, outlier ablation | Level 3 | Validated code against specifications. Handled rigorous leakage check.
2026-07-08 | Gemini | Week 2 fix: outlier ablation analysis, subject profiles, Balabit structure | Level 3 | Ran analysis scripts to identify cause of s049 outliers; updated report with accurate analysis.
2026-07-09 | Claude | Week 3 implementation: subject split, LSTM sequence model, sequence scaling | Level 3 | Validated sequence order reconstruction. Model executed locally on MPS instead of Colab.
2026-07-09 | Claude | Week 4 Phase 1: adaptive baseline, Frog-Boiling attack, benign drift, pairing | Level 3 | Scaffolding and tests only. Added RISK HIGH guards for np.random.default_rng and replace=True branch.
2026-07-09 | Gemini | Week 4 Phase 2: experiment execution and reporting | Level 3 | Ran `run_poisoning_experiment.py`, checked for null results, wrote `report4.md` and froze requirements.
2026-07-09 | Claude | Week 4 Extension Phase 3 | Level 3 | Audited Week 4 extension, added absolute rate metrics.
2026-07-09 | Gemini | Week 4 Extension Phase 5 | Level 3 | Added full trajectory logging and determinism checks.
2026-07-10 | Gemini | Week 5 Phase 1 | Level 3 | Created CUSUM defense logic (src/cusum_defense.py) and calibration script, marking them RISK HIGH and testing interface boundaries.
2026-07-10 | Gemini | Week 5 Phase 1 Extension | Level 3 | Scaffolded `src/victim_attacker_similarity.py` and `src/calibrate_cusum_per_victim.py`. Caught and fixed a mathematical flaw in the Euclidean separation test assertion caused by `RobustScaler` IQR bounding. Tested interfaces securely.
2026-07-10 | Gemini | Week 5 Phase 2 Extension | Level 3 | Wrote and executed `src/run_defense_experiment_v2.py`. Due to 245,000 model fits on a fanless machine, the script ran for over 5 hours in the background. Identified that the similarity hypothesis failed (r=0.104) and that per-victim thresholds do not protect the most vulnerable victims, concluding the Frog-Boiling arc. Generated adversarial recomputations and `report5extension.md`.
2026-07-10 | Gemini | Week 5 Phase 2 | Level 3 | Ran 51,000-fit experiment with src/run_defense_experiment.py, recomputed metrics independently, generated dense report5.md, and tracked thermal throttling bottlenecks.
2026-07-10 | Gemini | Week 5 Phase 3 | Level 3 | Audited Phase 2, found and patched a violation of Rule 1 (hardcoded stdout scraping) by explicitly recomputing calibration spread. Identified test-data leakage in week5.md experimental design.

<!-- Continue this for every session. Levels: 1 background research, 2 code scaffold reviewed and modified, 3 code used largely as generated, 4 grammar pass on own prose, 5 planning assistance -->
