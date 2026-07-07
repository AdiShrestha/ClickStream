# Week 1 — Foundation: Data, Features, Threat Model, Ethics

**Read this entire file before writing or running anything.** It is written to be self-contained: everything needed to execute Week 1 exactly as intended is here, including full code, not sketches. If you are an AI executing this on Aditya's behalf: do not deviate from the file/column names, the assertion thresholds, or the directory structure below without flagging the deviation explicitly back to him — those specifics were chosen deliberately (verified dataset formats, specific sanity-check bounds) and silent changes would break comparability with later weeks.

**Environment for this entire week: local machine (VS Code / Antigravity), no GPU, no Colab/Kaggle needed.** Everything here runs on CPU in seconds.

---

## 1. Objective

By the end of this week you have: (a) a downloaded, structurally-validated copy of the CMU Keystroke Dynamics Benchmark dataset, (b) a tested feature-extraction module that both reshapes that dataset AND can process raw keydown/keyup event streams (needed again in Weeks 6–7 for self-collected data), (c) an EDA pass that prints and plots sanity-check numbers proving the data is what it claims to be, (d) a written threat-model/system-design document, and (e) an ethics-statement draft and AI-use log started. Nothing here trains a model — Week 2 does that. This week's entire job is to make sure the ground everything else stands on is verified, not assumed.

## 2. Prerequisites

- Python 3.11 or 3.12 (check with `python3 --version`)
- `git` installed
- Internet access for one download (the CMU dataset CSV, ~4–5 MB)

## 3. Repository structure — create exactly this

```
clickstream/
├── README.md
├── requirements.txt
├── AI_USE_LOG.md
├── .gitignore
├── docs/
│   ├── threat_model.md
│   └── ethics_statement.md
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── data_acquisition.py
│   ├── feature_extraction.py
│   └── eda.py
└── tests/
    └── test_feature_extraction.py
```

Create it:
```bash
mkdir -p clickstream/docs clickstream/data/raw clickstream/data/processed clickstream/src clickstream/tests
cd clickstream
touch src/__init__.py
```

## 4. `.gitignore`

```
data/raw/
data/processed/
__pycache__/
*.pyc
.venv/
.DS_Store
*.egg-info/
```

## 5. `requirements.txt`

```
pandas>=2.2
numpy>=1.26
scipy>=1.13
scikit-learn>=1.5
matplotlib>=3.9
seaborn>=0.13
pytest>=8.2
requests>=2.32
```

Set up the environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 6. `src/data_acquisition.py` — full code

This downloads and structurally validates the CMU Keystroke Dynamics Benchmark (Killourhy & Maxion, 2009). The URL, expected subject count, and expected row count below were verified directly against the dataset's own hosting page and multiple independent mirrors during research for this project — do not change them without re-verifying.

```python
"""
Week 1 data acquisition: downloads and validates the CMU Keystroke
Dynamics Benchmark dataset (Killourhy & Maxion, 2009).

Source: https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv
Expected: 51 subjects x 400 repetitions (8 sessions x 50 reps/session)
        = 20,400 rows total. Password typed: ".tie5Roanl". All timing
        values are in SECONDS as released by the original authors.
"""
from pathlib import Path
import pandas as pd
import requests

CMU_URL = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
RAW_DIR = Path("data/raw")
CMU_PATH = RAW_DIR / "DSL-StrongPasswordData.csv"

EXPECTED_SUBJECTS = 51
EXPECTED_ROWS = 20400
EXPECTED_REPS_PER_SUBJECT = 400


def download_cmu_dataset(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if CMU_PATH.exists() and not force:
        print(f"Already downloaded: {CMU_PATH}")
        return CMU_PATH
    print(f"Downloading from {CMU_URL} ...")
    response = requests.get(CMU_URL, timeout=30)
    response.raise_for_status()
    CMU_PATH.write_bytes(response.content)
    print(f"Saved {len(response.content):,} bytes to {CMU_PATH}")
    return CMU_PATH


def validate_cmu_dataset(path: Path = CMU_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # --- Structural checks ---
    assert "subject" in df.columns, "Missing 'subject' column"
    assert "sessionIndex" in df.columns, "Missing 'sessionIndex' column"
    assert "rep" in df.columns, "Missing 'rep' column"

    hold_cols = [c for c in df.columns if c.startswith("H.")]
    dd_cols = [c for c in df.columns if c.startswith("DD.")]
    ud_cols = [c for c in df.columns if c.startswith("UD.")]
    assert len(hold_cols) > 0, "No Hold-time (H.) columns found"
    assert len(dd_cols) > 0, "No Down-Down (DD.) columns found"
    assert len(ud_cols) > 0, "No Up-Down (UD.) columns found"
    assert len(dd_cols) == len(ud_cols), (
        f"DD and UD column counts should match: {len(dd_cols)} vs {len(ud_cols)}"
    )

    # --- Content checks against the published dataset description ---
    n_subjects = df["subject"].nunique()
    n_rows = len(df)
    reps_per_subject = df.groupby("subject").size()

    assert n_subjects == EXPECTED_SUBJECTS, (
        f"Expected {EXPECTED_SUBJECTS} subjects, found {n_subjects}. "
        "Stop and check the download -- do not proceed to feature "
        "extraction on unverified data."
    )
    assert n_rows == EXPECTED_ROWS, (
        f"Expected {EXPECTED_ROWS} rows, found {n_rows}."
    )
    assert (reps_per_subject == EXPECTED_REPS_PER_SUBJECT).all(), (
        "Not every subject has exactly 400 repetitions:\n"
        f"{reps_per_subject[reps_per_subject != EXPECTED_REPS_PER_SUBJECT]}"
    )

    # --- Sanity checks on the timing values themselves ---
    all_hold = df[hold_cols].to_numpy().flatten()
    all_dd = df[dd_cols].to_numpy().flatten()
    assert (all_hold > 0).all(), "Found non-positive hold times -- data looks corrupted"
    assert (all_hold < 2.0).all(), "Found hold times > 2s -- implausible for one keypress"
    assert (all_dd < 5.0).all(), "Found DD times > 5s -- check for corrupted rows"

    print("All validation checks passed.")
    print(f"  Subjects: {n_subjects}")
    print(f"  Rows: {n_rows}")
    print(f"  Hold-time (H.) columns: {len(hold_cols)}")
    print(f"  Digraph (DD.) columns: {len(dd_cols)}")
    print(f"  Digraph (UD.) columns: {len(ud_cols)}")
    print(f"  Mean hold time: {all_hold.mean()*1000:.1f} ms")
    print(f"  Mean DD (digraph) time: {all_dd.mean()*1000:.1f} ms")

    return df


if __name__ == "__main__":
    download_cmu_dataset()
    validate_cmu_dataset()
```

**Why the assertions matter more than the happy path:** if `download_cmu_dataset` silently fetched a 404 error page instead of the CSV (which happens if the URL ever moves), `pd.read_csv` would either fail loudly or, worse, "succeed" on garbage — the structural and content assertions above are what turn a silent data problem into a loud, immediate failure. Do not remove or weaken them to "get past" a failing run; a failing assertion here means something is genuinely wrong and needs fixing before Week 2 builds on top of it.

## 7. `src/feature_extraction.py` — full code

```python
"""
Week 1 feature extraction. Two responsibilities:
1. Reshape the CMU dataset's pre-computed H/DD/UD columns into a clean
   (X, subjects, sessions, feature_names) tuple.
2. Provide a raw keydown/keyup event-stream extractor using the IDENTICAL
   feature semantics (Hold, Down-Down, Up-Down), for use in Weeks 6-7 on
   self-collected data, so a model trained on CMU-derived features can
   score self-collected data on a like-for-like basis.
"""
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd


def load_cmu_features(
    path: str = "data/raw/DSL-StrongPasswordData.csv",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)
    hold_cols = sorted(c for c in df.columns if c.startswith("H."))
    dd_cols = sorted(c for c in df.columns if c.startswith("DD."))
    ud_cols = sorted(c for c in df.columns if c.startswith("UD."))
    feature_cols = hold_cols + dd_cols + ud_cols

    X = df[feature_cols].to_numpy(dtype=float)
    subjects = df["subject"].to_numpy()
    sessions = df["sessionIndex"].to_numpy()
    return X, subjects, sessions, feature_cols


def extract_features_from_raw_events(
    events: List[Tuple[str, str, float]],
) -> Dict[str, float]:
    """
    events: list of (key_id, action, timestamp_seconds) tuples,
            action in {'down', 'up'}. Order need not be pre-sorted.
    Returns {feature_name: value_in_seconds}, using the same H./DD./UD.
    naming convention as the CMU dataset.
    """
    features: Dict[str, float] = {}
    downs: Dict[str, float] = {}
    sequence: List[Tuple[str, float, float]] = []  # (key_id, down_t, up_t)

    for key_id, action, t in sorted(events, key=lambda e: e[2]):
        if action == "down":
            downs[key_id] = t
        elif action == "up" and key_id in downs:
            sequence.append((key_id, downs.pop(key_id), t))

    sequence.sort(key=lambda x: x[1])  # order by keydown time

    for i, (key_id, down_t, up_t) in enumerate(sequence):
        features[f"H.{key_id}"] = up_t - down_t
        if i + 1 < len(sequence):
            next_key, next_down_t, _ = sequence[i + 1]
            features[f"DD.{key_id}.{next_key}"] = next_down_t - down_t
            features[f"UD.{key_id}.{next_key}"] = next_down_t - up_t

    return features
```

## 8. `tests/test_feature_extraction.py` — full code, run with pytest

```python
import pandas as pd
import pytest
from src.feature_extraction import (
    extract_features_from_raw_events,
    load_cmu_features,
)


def test_extract_features_basic_dwell_and_flight():
    # Typing "hi": h down@1.000, h up@1.080, i down@1.150, i up@1.210
    events = [
        ("h", "down", 1.000),
        ("h", "up", 1.080),
        ("i", "down", 1.150),
        ("i", "up", 1.210),
    ]
    features = extract_features_from_raw_events(events)
    assert features["H.h"] == pytest.approx(0.080, abs=1e-6)
    assert features["H.i"] == pytest.approx(0.060, abs=1e-6)
    assert features["DD.h.i"] == pytest.approx(0.150, abs=1e-6)
    assert features["UD.h.i"] == pytest.approx(0.070, abs=1e-6)


def test_extract_features_handles_unmatched_keyup():
    # A stray keyup with no matching keydown must be ignored, not crash
    events = [("x", "up", 1.0), ("h", "down", 1.0), ("h", "up", 1.05)]
    features = extract_features_from_raw_events(events)
    assert features["H.h"] == pytest.approx(0.05, abs=1e-6)
    assert "H.x" not in features


def test_extract_features_empty_input():
    assert extract_features_from_raw_events([]) == {}


def test_extract_features_out_of_order_input():
    # Events deliberately shuffled -- function must sort internally
    events = [
        ("i", "up", 1.210),
        ("h", "down", 1.000),
        ("i", "down", 1.150),
        ("h", "up", 1.080),
    ]
    features = extract_features_from_raw_events(events)
    assert features["H.h"] == pytest.approx(0.080, abs=1e-6)
    assert features["DD.h.i"] == pytest.approx(0.150, abs=1e-6)


def test_cmu_loader_shapes_match(tmp_path):
    # A tiny fake CMU-format CSV, so this test never depends on the
    # network or the real 20,400-row file.
    fake = pd.DataFrame({
        "subject": ["s001", "s001", "s002"],
        "sessionIndex": [1, 1, 1],
        "rep": [1, 2, 1],
        "H.a": [0.08, 0.09, 0.07],
        "DD.a.b": [0.15, 0.16, 0.14],
        "UD.a.b": [0.07, 0.07, 0.07],
    })
    fake_path = tmp_path / "fake_cmu.csv"
    fake.to_csv(fake_path, index=False)

    X, subjects, sessions, cols = load_cmu_features(str(fake_path))
    assert X.shape == (3, 3)
    assert list(subjects) == ["s001", "s001", "s002"]
    assert cols == ["H.a", "DD.a.b", "UD.a.b"]
```

Run it:
```bash
pytest tests/ -v
```
**Expected output:** 5 tests, all passing (`5 passed in ...s`). If `test_cmu_loader_shapes_match` fails on column ordering, check that `load_cmu_features`'s `sorted()` calls are present exactly as written — column order must be deterministic since Week 2's model will depend on a stable feature-vector layout.

## 9. `src/eda.py` — full code

```python
"""
Week 1 EDA: prints sanity-check numbers and saves a distribution plot.
Run only after data_acquisition.py has completed successfully.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extraction import load_cmu_features

OUTPUT_DIR = "data/processed"


def run_eda():
    X, subjects, sessions, feature_cols = load_cmu_features()
    hold_cols = [c for c in feature_cols if c.startswith("H.")]
    dd_cols = [c for c in feature_cols if c.startswith("DD.")]
    hold_idx = [feature_cols.index(c) for c in hold_cols]
    dd_idx = [feature_cols.index(c) for c in dd_cols]

    hold_values_ms = X[:, hold_idx].flatten() * 1000
    dd_values_ms = X[:, dd_idx].flatten() * 1000

    print("=== Week 1 EDA sanity report ===")
    print(f"Total repetitions loaded: {X.shape[0]}")
    print(f"Total feature columns: {X.shape[1]}")
    print(f"Unique subjects: {len(set(subjects))}")
    print(f"Hold time (ms): mean={hold_values_ms.mean():.1f}, "
          f"std={hold_values_ms.std():.1f}, "
          f"min={hold_values_ms.min():.1f}, max={hold_values_ms.max():.1f}")
    print(f"Digraph DD time (ms): mean={dd_values_ms.mean():.1f}, "
          f"std={dd_values_ms.std():.1f}, "
          f"min={dd_values_ms.min():.1f}, max={dd_values_ms.max():.1f}")
    print()
    print("SANITY BAR: published keystroke-dynamics literature reports")
    print("typical hold times around 60-150ms for average adult typists.")
    print("If your mean is far outside roughly 40-200ms, suspect a units")
    print("bug (seconds vs milliseconds) before doing anything else.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(hold_values_ms, bins=50, ax=axes[0])
    axes[0].set_title("Hold-time distribution, all subjects (ms)")
    axes[0].set_xlabel("Hold time (ms)")

    sns.histplot(dd_values_ms, bins=50, ax=axes[1])
    axes[1].set_title("Digraph DD time distribution, all subjects (ms)")
    axes[1].set_xlabel("DD time (ms)")

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/week1_timing_distributions.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved distribution plot to {out_path}")

    # Per-subject variability -- confirms subjects are behaviorally
    # distinguishable at all, which is the entire premise of the project
    per_rep_mean_hold = X[:, hold_idx].mean(axis=1) * 1000
    per_subject_mean = pd.Series(per_rep_mean_hold).groupby(subjects).mean()
    print(f"\nPer-subject mean hold time range: "
          f"{per_subject_mean.min():.1f}ms to {per_subject_mean.max():.1f}ms "
          f"(spread: {per_subject_mean.max() - per_subject_mean.min():.1f}ms)")
    print("If this spread is very narrow (e.g. under 10ms across 51 people),")
    print("flag it honestly now -- it would mean this feature space alone")
    print("may not separate subjects well, which Week 2's EER will confirm")
    print("or refute quantitatively.")


if __name__ == "__main__":
    run_eda()
```

Run it:
```bash
cd src && python eda.py && cd ..
```

## 10. `docs/threat_model.md` — write this exact content (edit the changelog as you actually revise it)

```markdown
# Clickstream — Threat Model and System Design (v1, Week 1)

## System under study
A continuous behavioral-authentication system for banking sessions,
using keystroke dynamics (primary modality, Weeks 1-7) and mouse
dynamics (secondary modality, introduced from Week 2 onward as time
permits), scoring live session behavior against a per-user enrolled
baseline via classical (Isolation Forest / One-Class SVM) and
deep-sequence (Siamese network) models.

## Research questions
- RQ1: How much does a Frog-Boiling-style gradual poisoning attack
  degrade a continuously-adapting authentication baseline, and does a
  residue-feature/RONI-style validation gate measurably mitigate it
  without blocking legitimate behavioral drift?
- RQ2: How effective is a video-based (screen-recording) injection
  attack against this system's keystroke-timing verification, and does
  a secondary liveness signal or synthetic-timing detector measurably
  reduce its evasion rate?
- RQ3 (stretch): How does the poisoning threat model change under
  federated learning, and does Byzantine-robust aggregation (Krum /
  trimmed mean) mitigate a client-level version of the same attack
  where naive FedAvg does not?

## Threat model
- Poisoning attacker (RQ1): controls sessions that get absorbed into
  the baseline-adaptation loop (e.g., via a low-risk-scored stolen
  session); cannot directly modify model code or infrastructure.
- Injection attacker (RQ2): possesses a screen recording of the victim
  typing and can inject synthetic timing data into the pipeline; does
  not need a keylogger or physical device access.
- Federated attacker (RQ3, stretch): controls one or more federated
  client devices and can send arbitrary but plausible-looking updates.
- Out of scope: network/TLS-layer attacks, sensor-level presentation
  attacks (ISO 30107 territory), physical device compromise beyond
  what's stated above.

## Explicit scope boundaries
- All quantitative results come from public academic benchmark
  datasets, not real banking telemetry.
- Proposed defenses are evaluated empirically, not proven unbreakable.
- Both attacks are extensions of published work, not novel attack
  classes invented from scratch.
- RQ3 is an explicitly secondary/stretch contribution.

## Datasets
- CMU Keystroke Dynamics Benchmark (Killourhy & Maxion, 2009) --
  primary, acquired Week 1.
- Balabit Mouse Dynamics Challenge -- secondary modality, from Week 2.
- Self-recorded video (author only, single-subject proof-of-concept,
  informed self-consent) -- Weeks 6-7 injection-attack demo only.

### Changelog
- Week 1: initial draft.
```

## 11. `docs/ethics_statement.md` — write this exact content

```markdown
# Clickstream — Ethics and Data-Provenance Statement (draft, Week 1)

## Public datasets
This project uses the CMU Keystroke Dynamics Benchmark dataset
(Killourhy & Maxion, 2009) and, from Week 2, the Balabit Mouse Dynamics
Challenge dataset -- both pre-existing, publicly available academic
research corpora collected under their original authors' own ethical
review. No new data collection from third-party human subjects occurs
on these datasets; they are used as originally released, with proper
citation to the original publications.

## Self-collected data (Weeks 6-7 only)
A small amount of new data (screen-recorded video of typing activity)
will be collected for the injection-attack demonstration in Weeks 6-7,
limited to the author as the sole subject -- a single-subject
proof-of-concept, not a claim about users in general. Informed
self-consent will be recorded at the time of that recording, in this
file, dated.

## Institutional review
[FILL IN THIS WEEK: ask your department/supervisor whether any KU
DoCSE student-research ethics process technically applies to
single-subject self-recording. Record the actual answer here, dated,
whichever it is.]

## AI-assisted tooling disclosure
See AI_USE_LOG.md for the full session-by-session log.
```

## 12. `README.md` and `AI_USE_LOG.md`

`README.md`:
```markdown
# Clickstream

Continuous behavioral authentication: adversarial robustness against
gradual baseline poisoning and video-based injection attacks, with
proposed defenses. See docs/threat_model.md for the full research
design and docs/ethics_statement.md for data provenance.
```

`AI_USE_LOG.md`: use the exact template from the main roadmap's Section 0.8 — start it today, add an entry for every session.

## 13. Exact command sequence, start to finish

```bash
mkdir -p clickstream/docs clickstream/data/raw clickstream/data/processed clickstream/src clickstream/tests
cd clickstream
touch src/__init__.py
# -- create requirements.txt, .gitignore, README.md, AI_USE_LOG.md as above --
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# -- create src/data_acquisition.py, src/feature_extraction.py,
#    tests/test_feature_extraction.py, src/eda.py, docs/threat_model.md,
#    docs/ethics_statement.md exactly as above --
python src/data_acquisition.py
pytest tests/ -v
cd src && python eda.py && cd ..
git init && git add -A && git commit -m "Week 1: data pipeline, feature extraction, threat model, ethics draft"
```

## 14. Verification checklist — all of these must pass before Week 2 starts

- [ ] `python src/data_acquisition.py` prints "All validation checks passed" with 51 subjects, 20,400 rows, and a mean hold time roughly in the 40–200ms band.
- [ ] `pytest tests/ -v` shows 5 passed, 0 failed.
- [ ] `python src/eda.py` runs without error and produces `data/processed/week1_timing_distributions.png`, and the printed per-subject hold-time spread is not vanishingly small (a healthy sign subjects are behaviorally distinguishable at all — Week 2 will confirm this quantitatively via EER).
- [ ] `docs/threat_model.md` and `docs/ethics_statement.md` exist with the exact content above, not placeholders.
- [ ] `AI_USE_LOG.md` has at least one real entry for this week's work.
- [ ] You (or your supervisor) have an actual answer recorded in `docs/ethics_statement.md`'s Institutional Review section — not left as a TODO.

## 15. What to send back to Claude at the end of this week

Paste or attach: (1) the full console output of `data_acquisition.py`, `pytest -v`, and `eda.py`; (2) the `week1_timing_distributions.png` plot; (3) the final `docs/threat_model.md` and `docs/ethics_statement.md` as actually written (with any edits you made); (4) whatever answer you got on institutional review. I'll check the numbers against the expected ranges above, sanity-check the threat model's scope against Part 1 of the roadmap, and confirm you're clear to start Week 2 — or flag anything that needs fixing first.
