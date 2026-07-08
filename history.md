# Clickstream Project History

## claude : Week 1 Session Start (2026-07-08)

### Architecture and design decisions

This is a continuous behavioral biometrics research project targeting IEEE T-BIOM publication. The core thesis evaluates two adversarial threats against a keystroke and mouse authentication system: gradual baseline poisoning (Frog-Boiling style) and one-shot video-based injection spoofing. The project spans 10 weeks with federated learning as a stretch goal in week 8.

**Repository root decision:** The week1.md plan specifies a `clickstream/` subdirectory, but the git repo is already rooted at `Click Stream/`. I am placing all code directly under `Click Stream/` (i.e., `src/`, `data/`, `docs/`, `tests/` live at the repo root), not inside a nested `clickstream/` subfolder. This avoids a double-nesting situation and keeps the repo root clean. The week1.md directory structure is followed exactly otherwise.

**Device context:** MacBook Air M3, 16GB unified memory, no NVIDIA GPU. PyTorch device detection must be dynamic: MPS on this machine, CUDA on Linux/Windows clones, CPU fallback. This is enforced from the very first week even though week 1 has no PyTorch code, so the pattern is established early.

**Dataset:** CMU Keystroke Dynamics Benchmark (Killourhy and Maxion, 2009). 51 subjects, 400 repetitions each, 20400 rows total. Password typed: `.tie5Roanl`. All timing values are in seconds in the original CSV. The feature naming convention H. (hold time), DD. (down-down digraph time), UD. (up-down digraph time) must remain consistent between the CMU reshaper and the raw-event extractor because week 6 and 7 self-collected data will be scored against CMU-trained models.

**Feature extraction design:** Two separate responsibilities in one module. First: reshape the CMU precomputed columns into a clean numpy array. Second: extract identical features from raw keydown/keyup event streams (for weeks 6 and 7). Same naming convention, same units (seconds) throughout. This was a deliberate choice to avoid a week 6 unit mismatch that would silently invalidate the injection attack results.

**Testing strategy:** 5 unit tests covering: basic dwell and flight time correctness, stray keyup handling, empty input, out-of-order event handling, and the CMU loader shape. The fake CSV test uses a tmp_path fixture so the test suite never requires the real 20400-row file or network access.

**EDA approach:** Prints sanity numbers and saves one distribution plot. The critical check is per-subject mean hold time spread: if this is very narrow across 51 subjects, the feature space may not separate subjects well, which week 2 EER will confirm. This is flagged honestly in the output rather than assumed fine.

**AI use log:** Started this session. All code in week 1 is level 3 AI scaffold reviewed against the week1.md specification. Architecture decisions are my own interpretation of the roadmap constraints.

**Files created this session:**
- `.gitignore`
- `requirements.txt`
- `README.md`
- `AI_USE_LOG.md`
- `src/__init__.py`
- `src/data_acquisition.py`
- `src/feature_extraction.py`
- `src/eda.py`
- `tests/test_feature_extraction.py`
- `docs/threat_model.md`
- `docs/ethics_statement.md`
- `SetupGuide.md`
- `history.md` (this file)

**Note on eda.py import:** The week1.md code uses `from feature_extraction import load_cmu_features` which works when run as `cd src && python eda.py`. I kept this as specified in the plan but added a sys.path fallback for robustness when running from the repo root.

**Open question for pilot:** The ethics statement has a placeholder for KU DoCSE institutional review. Adi needs to actually check with his department and fill that in before week 6. Flagged in ethics_statement.md.

**Commit hash for week 1:** 3aa9657561be0b7d066001851307b62a66b09f84 tagged week01

**Deviation from week1.md plan:** week1.md specified hold time assertion < 2.0s and DD assertion < 5.0s. The actual CMU dataset contains 1 hold value at 2.0353s and 13 DD values up to 25.987s. These are genuine outliers in the real dataset, confirmed by inspecting the values. Thresholds raised to 3.0s and 30.0s. The outlier counts are printed in validation output and documented in the weekly report. This is NOT a data quality issue; the 51 subjects and 20400 rows are exactly as expected.

**Lockfile:** results/week1/requirements.lock.txt committed with exact versions used this session.


---

## claude : Week 2 Session Start (2026-07-08)

### Key decisions carried over from Week 1

Reading history.md before touching code as per rule 17. No disagreements with Week 1 decisions. Continuing.

### Architecture and design decisions for Week 2

**Import convention change:** Week1 used sys.path.insert hacks to allow `python src/eda.py`. Week2 drops this entirely. All modules now use `from src.X import Y` and are run as `python -m src.MODULE` from the repo root. This is cleaner and standard. The Week 1 eda.py sys.path trick is left in place for now since it works and there is no reason to touch working Week 1 code.

**RobustScaler over StandardScaler:** Confirmed decision by pilot. Week 1 found real outliers (H.a max 2.035s, DD.i.e max 25.987s). StandardScaler's mean and std are pulled by these; RobustScaler uses median/IQR and is far less distorted. This is applied in PerUserModel for both IF and OC-SVM.

**Temporal split design:** Enroll on sessions 1-4 (200 reps per subject), test on sessions 5-8. Genuine test: same subject sessions 5-8. Impostor test: ALL other subjects sessions 5-8. This is the right framing for verification (1:1), not LOSO which is for population classifiers. Internal consistency assertions (not hardcoded counts) make splits.py reusable for Balabit later.

**Split correctness is the critical property this week:** The leakage test in test_splits.py uses a synthetic dataset where feature value == subject index, so if subject s0's data ever appears in s0's impostor set, the assertion catches it directly. This is more reliable than trusting the logic by inspection.

**Two separate negative controls for two different bug classes:** (1) metrics sanity on synthetic data catches bugs IN the metrics code independently of any model. (2) shuffled subject labels on real data catches leakage in the pipeline. Both must pass before any real result is trusted.

**Balabit: inspect-first, not parse-first.** Agreed with pilot. Clone the repo, print the actual structure and sample CSV headers. Do not write a parser this week. Parser is written in Week 3 once real format is confirmed.

**Outlier ablation for s049:** Week 1 raised whether the 2.035s outlier in s049 enrollment sessions actually affects the model. This week runs a concrete before/after comparison. The result goes in the weekly report as raw numbers, not a hunch.

**No new pip dependencies.** All packages already in requirements.txt from Week 1: scikit-learn>=1.5 covers IsolationForest, OneClassSVM, RobustScaler, roc_auc_score, average_precision_score, roc_curve.

**Files to create this session:**
- src/splits.py
- src/models.py
- src/metrics.py
- src/evaluate.py
- src/negative_control.py
- src/outlier_ablation.py
- src/balabit_acquisition.py
- tests/test_splits.py
- tests/test_metrics.py
- results/week2/ (directory, created by evaluate.py at runtime)

