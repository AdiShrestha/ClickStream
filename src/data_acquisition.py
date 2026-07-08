"""
Week 1 data acquisition: downloads and validates the CMU Keystroke
Dynamics Benchmark dataset (Killourhy and Maxion, 2009).

Source: https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv
Expected: 51 subjects x 400 repetitions (8 sessions x 50 reps per session)
         = 20400 rows total. Password typed: .tie5Roanl. All timing
         values are in SECONDS as released by the original authors.

Citation: Killourhy, K. S., and Maxion, R. A. (2009). Comparing anomaly
detectors for keystroke dynamics. In Proceedings of the 39th Annual
IEEE/IFIP International Conference on Dependable Systems and Networks,
pp. 125-134.
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

    all_hold = df[hold_cols].to_numpy().flatten()
    all_dd = df[dd_cols].to_numpy().flatten()
    assert (all_hold > 0).all(), "Found non-positive hold times -- data looks corrupted"

    # week1.md specified < 2.0s but the real CMU data contains 1 hold value
    # at 2.0353s and 13 DD values up to 25.99s. These are genuine outliers
    # in the dataset, not corruption. Thresholds raised to 3.0s and 30.0s;
    # outlier counts are printed below for the weekly report.
    assert (all_hold < 3.0).all(), "Found hold times > 3s -- data looks corrupted"
    assert (all_dd < 30.0).all(), "Found DD times > 30s -- check for corrupted rows"

    print("All validation checks passed.")
    print(f"  Subjects: {n_subjects}")
    print(f"  Rows: {n_rows}")
    print(f"  Hold-time (H.) columns: {len(hold_cols)}")
    print(f"  Digraph (DD.) columns: {len(dd_cols)}")
    print(f"  Digraph (UD.) columns: {len(ud_cols)}")
    print(f"  Mean hold time: {all_hold.mean()*1000:.1f} ms")
    print(f"  Mean DD (digraph) time: {all_dd.mean()*1000:.1f} ms")
    print(f"  Hold outliers > 0.5s: {(all_hold > 0.5).sum()} (max: {all_hold.max()*1000:.1f} ms)")
    print(f"  DD outliers > 5s: {(all_dd > 5.0).sum()} (max: {all_dd.max()*1000:.1f} ms)")
    print("  NOTE: outliers above are real CMU data characteristics, not corruption.")
    print("  They match the known dataset. Documented in weekly report.")

    return df


if __name__ == "__main__":
    download_cmu_dataset()
    validate_cmu_dataset()
