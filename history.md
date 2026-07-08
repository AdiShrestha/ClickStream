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

