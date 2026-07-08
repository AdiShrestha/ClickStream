"""
Week 2 (secondary task): clone the Balabit Mouse Dynamics Challenge
repository and inspect its ACTUAL structure before writing any parser.
Unlike CMU, the exact internal file format has not been independently
verified. This script only clones and prints; the parser is written in
Week 3 once the real format is confirmed from this output, not assumed.

Run from repo root: python -m src.balabit_acquisition
Send the full printed output back with the weekly report.
"""
import subprocess
from pathlib import Path
import pandas as pd

BALABIT_REPO = "https://github.com/balabit/Mouse-Dynamics-Challenge.git"
BALABIT_DIR = Path("data/raw/balabit")


def clone_balabit_repo():
    if BALABIT_DIR.exists():
        print(f"Already cloned: {BALABIT_DIR}")
        return
    BALABIT_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {BALABIT_REPO} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", BALABIT_REPO, str(BALABIT_DIR)],
        check=True,
    )


def inspect_balabit_structure():
    print("\n=== Balabit repo structure (top 3 levels) ===")
    for path in sorted(BALABIT_DIR.rglob("*")):
        depth = len(path.relative_to(BALABIT_DIR).parts)
        if depth <= 3:
            print("  " * depth + path.name)

    csv_candidates = list(BALABIT_DIR.rglob("*.csv"))[:1]
    if csv_candidates:
        sample = pd.read_csv(csv_candidates[0], nrows=5)
        print(f"\n=== Sample file: {csv_candidates[0]} ===")
        print(f"Columns found: {list(sample.columns)}")
        print(sample)
    else:
        print("\nNo .csv found in the first pass -- inspect subdirectories "
              "manually (the challenge data may be nested differently, e.g. "
              "per-session folders) before writing a parser next week. "
              "Do not guess the format from memory.")


if __name__ == "__main__":
    clone_balabit_repo()
    inspect_balabit_structure()
