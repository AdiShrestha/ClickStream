# Clickstream Setup Guide

This guide is written for someone who has freshly cloned this repository
and wants to run the project from scratch.

## Requirements

- Python 3.11 or 3.12 (check with `python3 --version`)
- git
- Internet access for the first run (dataset download, about 4-5 MB)

## Step 1: Clone the repository

```bash
git clone <your-repo-url>
cd "Click Stream"
```

## Step 2: Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # on macOS/Linux
# .venv\Scripts\activate         # on Windows
pip install -r requirements.txt
```

*(Note: Starting Week 3, PyTorch is required. It is listed in `requirements.txt`, but if you need a specific GPU build (e.g., CUDA), install it directly via the [PyTorch website](https://pytorch.org/get-started/locally/) instructions before running the other requirements).*


## Step 3: Download and validate the CMU dataset

```bash
python src/data_acquisition.py
```

Expected output: "All validation checks passed." with 51 subjects and 20400 rows.
The file is saved to data/raw/DSL-StrongPasswordData.csv (gitignored, not committed).

## Step 4: Run the unit tests

```bash
pytest tests/ -v
```

Expected output: 5 passed, 0 failed. These tests do not require the downloaded
dataset or internet access.

## Step 5: Run the EDA

```bash
python src/eda.py
```

This prints sanity-check statistics and saves a distribution plot to
data/processed/week1_timing_distributions.png.

## Week-by-week additions

This guide will be updated as the project progresses through each week.

| Week | Main additions |
|------|---------------|
| 1    | Data pipeline, feature extraction, EDA |
| 2    | Classical baseline (Isolation Forest, One-Class SVM) |
| 3    | Deep sequence model (Siamese network, Colab T4) |
| 4    | Poisoning attack implementation |
| 5    | Poisoning defense (RONI-style gate) |
| 6    | Injection attack (screen-recording demo) |
| 7    | Injection defense |
| 8    | Federated learning extension (Colab/Kaggle) |
| 9    | Results consolidation, paper scaffold |
| 10   | Red-team review, submission prep |

## Notes on hardware

- macOS M-series: device is detected dynamically (MPS for GPU, CPU fallback).
  No CUDA required; the same code runs unchanged on Windows/Linux with CUDA.
- 16GB unified memory: data loaders are used instead of loading full datasets
  into memory. Flagged explicitly in weekly reports when any step gets close
  to the limit.
- Checkpoints are written for any run longer than about 20 minutes to protect
  against thermal throttling on fanless machines.
