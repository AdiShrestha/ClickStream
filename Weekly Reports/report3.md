# Week 3 Report: Deep Sequence Model Baseline

## 1. Objectives

Copied verbatim from week3.md section 1:

By the end of this week you have: (a) a Siamese/triplet-loss keystroke encoder trained via PyTorch, with a device-detection utility (MPS locally, CUDA on Colab, CPU fallback) established once and reused for the rest of the project, (b) a background/held-out subject split — a new, methodologically necessary split distinct from Week 2's session split — used to test whether the learned embedding space actually generalizes to subjects never seen during training, (c) a programmatically-reconstructed true chronological key order (not alphabetical), verified by a dedicated unit test, (d) a trained encoder with monotonically-decreasing loss and no NaNs, (e) an embedding-space sanity check confirming intra-subject distances are meaningfully smaller than inter-subject distances, (f) an EER evaluation on held-out subjects using Week 2's exact same `compute_eer`/`compute_full_metrics` functions — not reimplemented, reused, for consistency, and (g) a fair, apples-to-apples comparison against a recomputed classical baseline restricted to the same held-out subjects, not against Week 2's full-51-subject pooled number.

**Additional objective carried from Pilot AI Week 2 feedback:** Resolve the n=1 overclaim on Isolation Forest vs OCSVM robustness by running the moderate-threshold outlier ablation on multiple high-variance subjects (s032, s003, s011, s007) in addition to the original s049 result.

---

## 2. Environment

- **Hardware:** MacBook Air M3 (8-core CPU, 10-core GPU/MPS, no fan), 16 GB unified memory, 512 GB storage, macOS 26.5.2
- **Device used:** `mps` (Apple Metal Performance Shaders). CUDA selected automatically on Colab/Linux/Windows via `src/device.py` — same code, no changes required.
- **Python:** 3.12.8
- **PyTorch:** 2.13.0
- **numpy:** 2.5.1
- **scikit-learn:** 1.9.0
- **scipy:** 1.18.0
- **pandas:** 3.0.3
- **matplotlib:** 3.11.0
- **seaborn:** 0.13.2
- **Full environment lockfile:** `results/week3/requirements.lock.txt`
- **Git commit hash (pre-Week 3 work):** `fed2479c2cf2549ffb82591bb83ff810018b6c99`

**Deviation from plan (Colab vs local):** week3.md section 3 prescribes Colab for GPU access. This session ran locally instead, using MPS. The CMU dataset is small (20,400 rows, 51 subjects, 11 keys x 3 features); training 50 epochs x 50 steps x batch 32 on a bidirectional LSTM with hidden_dim=32 completed in under 5 minutes on MPS, well within thermal budget for the fanless M3. The `get_device()` utility in `src/device.py` returns `mps` locally and `cuda` on Colab/Linux automatically. No code change is required to run on Colab. This deviation is recorded in `history.md` (claude: Week 3 Session Start entry).

---

## 3. Raw Results

### 3.1 Device Detection

Command: `python -m src.device`

```
Selected device: mps
```

Executed at session start before any training. Confirms MPS is available and selected. On a CUDA-capable machine, this would print `cuda` and the GPU name.

---

### 3.2 Unit Test Results (run before any model code)

Command: `pytest tests/test_key_sequence.py tests/test_subject_split.py tests/test_train_encoder_safety.py -v`

```
============================= test session starts ==============================
platform darwin -- Python 3.12.8, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/adi/Desktop/Click Stream

collected 8 items

tests/test_key_sequence.py::test_parse_dd_column_handles_embedded_dot PASSED [12%]
tests/test_key_sequence.py::test_reconstruct_key_sequence_order_correct PASSED [25%]
tests/test_key_sequence.py::test_reconstruct_raises_on_broken_chain PASSED [37%]
tests/test_key_sequence.py::test_build_sequence_features_shape_and_values PASSED [50%]
tests/test_subject_split.py::test_split_creates_no_overlap_and_correct_proportions PASSED [62%]
tests/test_subject_split.py::test_split_is_deterministic_across_calls PASSED [75%]
tests/test_subject_split.py::test_split_detects_stale_file PASSED [87%]
tests/test_train_encoder_safety.py::test_train_encoder_raises_on_nan_input PASSED [100%]

============================== 8 passed in 1.24s ==============================
```

All 8 tests pass. This was confirmed before `python -m src.run_week3` was executed, per week3.md section 17.

**Test descriptions:**
- `test_parse_dd_column_handles_embedded_dot`: Confirms the DD-column parser correctly identifies `Shift.r` as a key name containing an embedded dot, not splitting it at the dot. Tests all three types: single-dot from-key, embedded-dot to-key, embedded-dot from-key.
- `test_reconstruct_key_sequence_order_correct`: Confirms the full reconstruction returns `['a', 'b', 'Shift.r', 'c']` for the synthetic 4-key test case.
- `test_reconstruct_raises_on_broken_chain`: Confirms the reconstruction raises `AssertionError` on a disconnected graph (two separate pairs with no linking transition), rather than silently returning a partial chain.
- `test_build_sequence_features_shape_and_values`: Confirms the 3D reshaping produces the correct output shape `(2, 4, 3)` and places feature values at the correct positions — H.a at `[0,0,0]`, DD.a.b at `[0,0,1]`, UD.a.b at `[0,0,2]`, and zeros at `[0,3,1]` and `[0,3,2]` (no outgoing transition from the last key).
- `test_split_creates_no_overlap_and_correct_proportions`: 51 synthetic subjects, confirms 36 background + 15 held-out, zero overlap.
- `test_split_is_deterministic_across_calls`: Creates split, calls again, confirms same result loaded from disk.
- `test_split_detects_stale_file`: Creates split for 51 subjects, then tries to load it with a 45-subject dataset, confirms AssertionError fires.
- `test_train_encoder_raises_on_nan_input`: Injects NaN into all input sequences, confirms `RuntimeError` with message matching `non-finite` is raised within the training loop.

---

### 3.3 Main Script Full Console Output

Command: `python -m src.run_week3`

```
Using device: mps
Reconstructed true key order: ['period', 't', 'i', 'e', 'five', 'Shift.r', 'o', 'a', 'n', 'l', 'Return']
Sequence feature shape: (20400, 11, 3)
Created new subject split: 36 background, 15 held-out
Saved to results/week3/subject_split.json
Background subjects (36): ['s002', 's003', 's005', 's007', 's008', 's010', 's012', 's013', 's016', 's017', 's018', 's020', 's021', 's022', 's025', 's027', 's030', 's031', 's032', 's033', 's035', 's036', 's037', 's038', 's039', 's040', 's042', 's043', 's047', 's050', 's051', 's052', 's053', 's054', 's055', 's056']
Held-out subjects (15): ['s004', 's011', 's015', 's019', 's024', 's026', 's028', 's029', 's034', 's041', 's044', 's046', 's048', 's049', 's057']

=== Training encoder (background subjects, enrollment sessions only) ===
Epoch 1/50: mean triplet loss = 0.1230
Epoch 2/50: mean triplet loss = 0.1035
Epoch 3/50: mean triplet loss = 0.0892
Epoch 4/50: mean triplet loss = 0.0956
Epoch 5/50: mean triplet loss = 0.0817
Epoch 6/50: mean triplet loss = 0.0834
Epoch 7/50: mean triplet loss = 0.0786
Epoch 8/50: mean triplet loss = 0.0750
Epoch 9/50: mean triplet loss = 0.0634
Epoch 10/50: mean triplet loss = 0.0699
Epoch 11/50: mean triplet loss = 0.0688
Epoch 12/50: mean triplet loss = 0.0651
Epoch 13/50: mean triplet loss = 0.0680
Epoch 14/50: mean triplet loss = 0.0701
Epoch 15/50: mean triplet loss = 0.0630
Epoch 16/50: mean triplet loss = 0.0645
Epoch 17/50: mean triplet loss = 0.0600
Epoch 18/50: mean triplet loss = 0.0600
Epoch 19/50: mean triplet loss = 0.0634
Epoch 20/50: mean triplet loss = 0.0501
Epoch 21/50: mean triplet loss = 0.0516
Epoch 22/50: mean triplet loss = 0.0518
Epoch 23/50: mean triplet loss = 0.0544
Epoch 24/50: mean triplet loss = 0.0530
Epoch 25/50: mean triplet loss = 0.0518
Epoch 26/50: mean triplet loss = 0.0539
Epoch 27/50: mean triplet loss = 0.0556
Epoch 28/50: mean triplet loss = 0.0484
Epoch 29/50: mean triplet loss = 0.0500
Epoch 30/50: mean triplet loss = 0.0502
Epoch 31/50: mean triplet loss = 0.0416
Epoch 32/50: mean triplet loss = 0.0463
Epoch 33/50: mean triplet loss = 0.0388
Epoch 34/50: mean triplet loss = 0.0494
Epoch 35/50: mean triplet loss = 0.0473
Epoch 36/50: mean triplet loss = 0.0468
Epoch 37/50: mean triplet loss = 0.0412
Epoch 38/50: mean triplet loss = 0.0423
Epoch 39/50: mean triplet loss = 0.0431
Epoch 40/50: mean triplet loss = 0.0402
Epoch 41/50: mean triplet loss = 0.0474
Epoch 42/50: mean triplet loss = 0.0392
Epoch 43/50: mean triplet loss = 0.0471
Epoch 44/50: mean triplet loss = 0.0362
Epoch 45/50: mean triplet loss = 0.0391
Epoch 46/50: mean triplet loss = 0.0470
Epoch 47/50: mean triplet loss = 0.0382
Epoch 48/50: mean triplet loss = 0.0406
Epoch 49/50: mean triplet loss = 0.0459
Epoch 50/50: mean triplet loss = 0.0381

=== Embedding sanity check (background subjects) ===
Mean intra-subject embedding distance: 0.6396
Mean inter-subject embedding distance: 1.3034
Ratio (inter divided by intra): 2.04x
Expect clearly above 1.0 (roughly 1.5x or higher is healthy); a ratio near 1.0 means the network has not separated subjects at all.

=== Evaluating on held-out subjects ===
Held-out encoder pooled EER: 9.59%
Held-out encoder pooled ROC-AUC: 0.9666
Held-out encoder pooled PR-AUC: 0.7331

=== Recomputing Week 2 classical baseline on the SAME held-out subjects ===
Held-out classical (Isolation Forest) pooled EER: 16.86%

=== Fair comparison (same held-out subjects, both methods) ===
  Isolation Forest EER: 16.86%
  Deep encoder EER:     9.59%
  Delta (encoder minus classical): -7.27 percentage points
  Report this honestly either way. A classical baseline beating the deep model on this small dataset is a legitimate reportable finding, not a failure to fix.

Saved full results to results/week3/week3_full_results.json
Saved model weights to results/week3/encoder_weights.pt
```

---

### 3.4 Reconstructed Key Order — Eye Check

Reconstructed: `['period', 't', 'i', 'e', 'five', 'Shift.r', 'o', 'a', 'n', 'l', 'Return']`

Known CMU password: `.tie5Roanl` + Enter

| Position | Reconstructed name | Character typed | Matches? |
|----------|--------------------|-----------------|---------|
| 0 | period | `.` | Yes |
| 1 | t | `t` | Yes |
| 2 | i | `i` | Yes |
| 3 | e | `e` | Yes |
| 4 | five | `5` | Yes |
| 5 | Shift.r | `R` (uppercase R via right Shift) | Yes |
| 6 | o | `o` | Yes |
| 7 | a | `a` | Yes |
| 8 | n | `n` | Yes |
| 9 | l | `l` | Yes |
| 10 | Return | Enter | Yes |

All 11 positions match. The reconstruction is correct. The LSTM is being fed keys in the correct chronological typing order.

---

### 3.5 Sequence Feature Shape Verification

```
Sequence feature shape: (20400, 11, 3)
```

- Axis 0: 20,400 = 51 subjects × 400 repetitions. Correct.
- Axis 1: 11 = the 11 keys in `.tie5Roanl` + Return. Correct.
- Axis 2: 3 = [hold_time (H.), down-down digraph (DD.), up-down digraph (UD.)]. Correct. The last key position (Return) has DD and UD set to 0.0 (no outgoing transition).

---

### 3.6 Subject Split Full Details

File: `results/week3/subject_split.json`

```json
{
  "background": [
    "s002", "s003", "s005", "s007", "s008", "s010", "s012", "s013",
    "s016", "s017", "s018", "s020", "s021", "s022", "s025", "s027",
    "s030", "s031", "s032", "s033", "s035", "s036", "s037", "s038",
    "s039", "s040", "s042", "s043", "s047", "s050", "s051", "s052",
    "s053", "s054", "s055", "s056"
  ],
  "held_out": [
    "s004", "s011", "s015", "s019", "s024", "s026", "s028", "s029",
    "s034", "s041", "s044", "s046", "s048", "s049", "s057"
  ]
}
```

- Background: 36 subjects (70.6% of 51)
- Held-out: 15 subjects (29.4% of 51)
- Overlap: 0 (verified by `assert len(set(background) & set(held_out)) == 0` in `src/subject_split.py`)
- Seed: 42, fixed. This file is committed to disk and will be loaded identically on any subsequent run.
- Note: s032 (Week 2's worst subject, 44.40% IF EER) is in the background set. s049 (Week 2's ablation subject) is in the held-out set.

---

### 3.7 Training Loss History (Full Precision)

File: `results/week3/training_history.json`

| Epoch | Mean Triplet Loss (full precision) |
|-------|------------------------------------|
| 1 | 0.1229603625833988 |
| 2 | 0.1035417997092009 |
| 3 | 0.0892329305410385 |
| 4 | 0.0955267145484686 |
| 5 | 0.0817102618515491 |
| 6 | 0.0833826476335526 |
| 7 | 0.0786250563338399 |
| 8 | 0.0749885505065322 |
| 9 | 0.0634078539535403 |
| 10 | 0.0699197497963905 |
| 11 | 0.0687523447349668 |
| 12 | 0.0650539955496788 |
| 13 | 0.0680356936901808 |
| 14 | 0.0700933900475502 |
| 15 | 0.0630049157515168 |
| 16 | 0.0645167481899261 |
| 17 | 0.0599738331511617 |
| 18 | 0.0600066994875670 |
| 19 | 0.0634172181412578 |
| 20 | 0.0500763041898608 |
| 21 | 0.0516160503774881 |
| 22 | 0.0517714163847268 |
| 23 | 0.0544434922561049 |
| 24 | 0.0530436858162284 |
| 25 | 0.0517935403063893 |
| 26 | 0.0539267424307764 |
| 27 | 0.0556105889379978 |
| 28 | 0.0483859445899725 |
| 29 | 0.0499932847544551 |
| 30 | 0.0501746555417776 |
| 31 | 0.0415848951973021 |
| 32 | 0.0462742990627885 |
| 33 | 0.0387602600082755 |
| 34 | 0.0494280767813325 |
| 35 | 0.0472941513359547 |
| 36 | 0.0468389324285099 |
| 37 | 0.0411827672645450 |
| 38 | 0.0423290007747710 |
| 39 | 0.0431178285926580 |
| 40 | 0.0401757524907589 |
| 41 | 0.0473776232451201 |
| 42 | 0.0391533296555281 |
| 43 | 0.0470721993595362 |
| 44 | 0.0362316015735269 |
| 45 | 0.0391309395432472 |
| 46 | 0.0470458431541920 |
| 47 | 0.0382313317246735 |
| 48 | 0.0405858816951513 |
| 49 | 0.0459289680793881 |
| 50 | 0.0381142323836684 |

**Summary:**
- First epoch loss: 0.12296036
- Final epoch loss: 0.03811423
- Total reduction: 69.0% from epoch 1 to epoch 50
- NaN/Inf count: 0 (verified by script: `sum(1 for x in losses if not np.isfinite(x))` = 0)
- Trend: Generally decreasing with stochastic noise per epoch (expected with random triplet sampling). No plateaus longer than 2-3 epochs. No divergence. The variance in later epochs (e.g., epochs 43-50 oscillating 0.036-0.047) is normal for triplet sampling without hard-negative mining — the margin-saturated triplets contribute zero loss, so each step's gradient comes from a random subset of the batch that still has a non-zero loss contribution.

---

### 3.8 Embedding Sanity Check (Full Precision)

```
Mean intra-subject embedding distance:  0.6396186657838725
Mean inter-subject embedding distance:  1.3033519698079374
Ratio (inter divided by intra):         2.0377015861640610x
```

**Threshold from week3.md:** "roughly 1.5x or higher is healthy"

**Result: 2.04x** — substantially above threshold. The encoder has learned a meaningful embedding space where samples from the same subject cluster together and are separated from other subjects. This is a prerequisite for trusting the downstream EER numbers.

**Interpretation:** The inter/intra ratio of 2.04x means that, on average, two samples from the same subject are roughly half as far apart in the 16-dimensional embedding space as two samples from different subjects. For L2-normalized embeddings (all embeddings lie on the unit hypersphere), this is a healthy margin and consistent with the triplet margin parameter of 0.3 used during training.

---

### 3.9 Encoder Evaluation on Held-Out Subjects (Full Precision)

**Pooled metrics:**

| Metric | Value (full precision) | Value (rounded) |
|--------|----------------------|-----------------|
| EER | 0.0959166666666667 | 9.59% |
| EER threshold | -0.7629245519638062 | -0.7629 |
| ROC-AUC | 0.9666312261904761 | 0.9666 |
| PR-AUC | 0.7331030597285804 | 0.7331 |
| Genuine trials | 3000 | — |
| Impostor trials | 42000 | — |

Trial counts: 15 held-out subjects × 200 test repetitions = 3,000 genuine trials. 15 subjects × 14 impostors × 200 test reps = 42,000 impostor trials. Correct.

**Per-subject encoder results (full precision):**

| Subject | EER | EER Threshold | ROC-AUC | PR-AUC | Genuine | Impostor |
|---------|-----|---------------|---------|--------|---------|----------|
| s004 | 0.107321428571 (10.73%) | -0.7756063938 | 0.9442857143 | 0.4339078435 | 200 | 2800 |
| s011 | 0.043750000000 (4.38%) | -0.6415105462 | 0.9909785714 | 0.9315055417 | 200 | 2800 |
| s015 | 0.168571428571 (16.86%) | -0.7714717984 | 0.9095375000 | 0.4523047008 | 200 | 2800 |
| s019 | 0.078928571429 (7.89%) | -0.7870711684 | 0.9789169643 | 0.8216655836 | 200 | 2800 |
| s024 | 0.050714285714 (5.07%) | -0.7833036780 | 0.9816982143 | 0.7054692868 | 200 | 2800 |
| s026 | 0.080714285714 (8.07%) | -0.7036495805 | 0.9784392857 | 0.8251605032 | 200 | 2800 |
| s028 | 0.043035714286 (4.30%) | -0.8383997083 | 0.9921142857 | 0.9358172383 | 200 | 2800 |
| s029 | 0.089642857143 (8.96%) | -0.6992541552 | 0.9669160714 | 0.8030035205 | 200 | 2800 |
| s034 | 0.104821428571 (10.48%) | -0.7603265047 | 0.9635392857 | 0.7285978745 | 200 | 2800 |
| s041 | 0.024285714286 (2.43%) | -0.7411074042 | 0.9960732143 | 0.9681583628 | 200 | 2800 |
| s044 | 0.114642857143 (11.46%) | -0.8443386555 | 0.9459375000 | 0.5243712956 | 200 | 2800 |
| s046 | 0.135178571429 (13.52%) | -0.8503215909 | 0.9478303571 | 0.7219015685 | 200 | 2800 |
| s048 | 0.079642857143 (7.96%) | -0.6480703354 | 0.9656535714 | 0.6606749377 | 200 | 2800 |
| s049 | 0.004285714286 (0.43%) | -0.7555384040 | 0.9995250000 | 0.9912541557 | 200 | 2800 |
| s057 | 0.160000000000 (16.00%) | -0.7563230395 | 0.9206071429 | 0.4622522725 | 200 | 2800 |

---

### 3.10 Recomputed Classical Baseline on Held-Out Subjects (Full Precision)

Note: Week 2's original pooled EER was 17.26% across all 51 subjects. This section recomputes IF restricted to the same 15 held-out subjects, giving a fair apples-to-apples comparison.

**Pooled metrics (held-out only):**

| Metric | Value (full precision) | Value (rounded) |
|--------|----------------------|-----------------|
| EER | 0.1686309523809524 | 16.86% |
| EER threshold | 0.0543913468443655 | 0.0544 |
| ROC-AUC | 0.9083600634920634 | 0.9084 |
| PR-AUC | 0.5225503342359550 | 0.5226 |
| Genuine trials | 3000 | — |
| Impostor trials | 42000 | — |

**Per-subject classical baseline results (full precision):**

| Subject | EER | EER Threshold | ROC-AUC | PR-AUC | Genuine | Impostor |
|---------|-----|---------------|---------|--------|---------|----------|
| s004 | 0.151785714286 (15.18%) | 0.0387386487 | 0.9273517857 | 0.6877209918 | 200 | 2800 |
| s011 | 0.285357142857 (28.54%) | 0.0557672791 | 0.8322125000 | 0.3432847062 | 200 | 2800 |
| s015 | 0.256785714286 (25.68%) | 0.0492666961 | 0.8245267857 | 0.2793145248 | 200 | 2800 |
| s019 | 0.040714285714 (4.07%) | 0.0447627355 | 0.9888250000 | 0.9656440739 | 200 | 2800 |
| s024 | 0.060535714286 (6.05%) | 0.0408922343 | 0.9913000000 | 0.9295364385 | 200 | 2800 |
| s026 | 0.099821428571 (9.98%) | 0.0350370840 | 0.9619678571 | 0.8481987209 | 200 | 2800 |
| s028 | 0.074642857143 (7.46%) | 0.0090008750 | 0.9766035714 | 0.7899642733 | 200 | 2800 |
| s029 | 0.142857142857 (14.29%) | 0.0167478776 | 0.9449339286 | 0.7883126908 | 200 | 2800 |
| s034 | 0.290535714286 (29.05%) | 0.0822720075 | 0.7807285714 | 0.2777407795 | 200 | 2800 |
| s041 | 0.153928571429 (15.39%) | 0.0459190567 | 0.9217732143 | 0.7836590600 | 200 | 2800 |
| s044 | 0.103750000000 (10.38%) | 0.0790065923 | 0.9582982143 | 0.7279117354 | 200 | 2800 |
| s046 | 0.148035714286 (14.80%) | 0.1003569859 | 0.9289750000 | 0.7451517713 | 200 | 2800 |
| s048 | 0.115178571429 (11.52%) | 0.0508919766 | 0.9490285714 | 0.6724550436 | 200 | 2800 |
| s049 | 0.046250000000 (4.63%) | 0.0679512823 | 0.9769821429 | 0.9528389074 | 200 | 2800 |
| s057 | 0.254464285714 (25.45%) | 0.0440096669 | 0.8342535714 | 0.2351198098 | 200 | 2800 |

---

### 3.11 Fair Comparison: Same 15 Held-Out Subjects

```
  Isolation Forest EER: 16.86%
  Deep encoder EER:     9.59%
  Delta (encoder minus classical): -7.27 percentage points
```

**Full per-subject side-by-side:**

| Subject | IF EER | Encoder EER | Delta (encoder - IF) | Encoder wins? |
|---------|--------|-------------|----------------------|---------------|
| s004 | 15.18% | 10.73% | -4.45pp | Yes |
| s011 | 28.54% | 4.38% | -24.16pp | Yes |
| s015 | 25.68% | 16.86% | -8.82pp | Yes |
| s019 | 4.07% | 7.89% | +3.82pp | No |
| s024 | 6.05% | 5.07% | -0.98pp | Yes |
| s026 | 9.98% | 8.07% | -1.91pp | Yes |
| s028 | 7.46% | 4.30% | -3.16pp | Yes |
| s029 | 14.29% | 8.96% | -5.33pp | Yes |
| s034 | 29.05% | 10.48% | -18.57pp | Yes |
| s041 | 15.39% | 2.43% | -12.96pp | Yes |
| s044 | 10.38% | 11.46% | +1.08pp | No |
| s046 | 14.80% | 13.52% | -1.28pp | Yes |
| s048 | 11.52% | 7.96% | -3.56pp | Yes |
| s049 | 4.63% | 0.43% | -4.20pp | Yes |
| s057 | 25.45% | 16.00% | -9.45pp | Yes |

**Encoder wins on 13 out of 15 subjects.** The two cases where IF outperforms the encoder (s019: +3.82pp, s044: +1.08pp) are small margins. The cases where the encoder dominates are large (s011: -24.16pp, s034: -18.57pp, s041: -12.96pp). The overall pooled gain of -7.27pp is driven by the encoder's strong generalization on the harder subjects that IF handles poorly.

**Comparison table — pooled metrics:**

| Method | Pooled EER | ROC-AUC | PR-AUC | Subjects |
|--------|------------|---------|--------|---------|
| Isolation Forest (held-out) | 16.86% | 0.9084 | 0.5226 | 15 held-out |
| Deep Encoder (held-out) | 9.59% | 0.9666 | 0.7331 | 15 held-out |
| Isolation Forest (Week 2 full) | 17.26% | 0.9078 | 0.3059 | 51 subjects |

The encoder improves on all three pooled metrics. The PR-AUC improvement (+0.2105) is particularly notable since PR-AUC reflects performance on the genuine class in the imbalanced setting (200 genuine vs 2800 impostor per subject), meaning the encoder is better at avoiding false accepts while maintaining a low miss rate.

---

### 3.12 Multi-Subject Outlier Ablation (Resolving Week 2 n=1 Overclaim)

**Background:** Pilot AI's Week 2 feedback identified that the claim "Isolation Forest is the more robust classical baseline against imperfect enrollment sets" was based on a single subject (s049). This ablation ran the same moderate-threshold (0.5s hold, 2.0s DD) analysis on the four next-highest-variance subjects recommended by Pilot AI: s032, s003, s011, and s007.

**Command:** inline `python -c` script using `src.feature_extraction`, `src.models`, `src.metrics`

**Full results:**

| Subject | Outlier rows flagged | Total enrollment rows | IF EER WITH | IF EER WITHOUT | IF Delta | OCSVM EER WITH | OCSVM EER WITHOUT | OCSVM Delta |
|---------|---------------------|----------------------|-------------|----------------|----------|----------------|-------------------|-------------|
| s049 | 49 | 200 | 5.02% | 4.80% | -0.22pp | 4.83% | 8.89% | +4.06pp |
| s032 | 1 | 200 | 44.40% | 44.05% | -0.35pp | 42.55% | 42.11% | -0.44pp |
| s003 | 0 | 200 | 32.48% | N/A (no-op) | N/A | 39.91% | N/A | N/A |
| s011 | 0 | 200 | 26.06% | N/A (no-op) | N/A | 36.98% | N/A | N/A |
| s007 | 0 | 200 | 28.00% | N/A (no-op) | N/A | 35.47% | N/A | N/A |

**Interpretation:**

- s032 has only 1 outlier row, so the ablation is functionally a no-op (removing 1/200 rows produces delta < 0.5pp for both models, entirely within noise).
- s003, s011, and s007 have zero flagged rows under the 0.5s/2.0s threshold — their high EERs come from within-distribution variance that is uniformly distributed across the feature space, not from specific extreme outlier rows.
- The s049 result (IF stable at -0.22pp, OCSVM degraded by +4.06pp) remains the only case where outlier removal caused a divergent and meaningful response between the two algorithms.

**Corrected claim for paper:** "For subject s049, removing moderate-variance enrollment rows (n=49, tripping the 0.5s hold or 2.0s DD threshold on at least one column, predominantly concentrated on the `five->Shift.r` and `e->five` transitions) caused One-Class SVM EER to increase by 4.06 percentage points while Isolation Forest EER changed by only -0.22 percentage points. The three other high-variance subjects tested (s032, s003, s011, s007) showed zero or negligible response, as they had at most 1 flagged row. This observation is specific to s049 and cannot be generalized to a categorical statement about model robustness without a broader ablation study."

---

## 4. Failed Attempts and Why

### 4.1 NaN Safety Net Test (Wrong Trigger Method)

The first version of `tests/test_train_encoder_safety.py` used `lr=1e6` to force loss divergence. The test failed: the encoder trained 5 epochs without NaN (loss stayed 0.28-0.35).

**Root cause:** `KeystrokeEncoder.forward` applies `F.normalize(self.fc(h_cat), p=2, dim=1)` as the final layer. L2-normalization clamps the embedding vector to unit length regardless of input magnitude. Even with an absurd learning rate, the gradient-updated weights produce large pre-normalization values that are then divided by their own L2 norm, bounding the loss. The triplet margin loss `relu(pos_dist - neg_dist + margin)` on L2-normalized embeddings has a natural ceiling at `2*margin + 2 = 2.6` (since pairwise L2 distances on the unit hypersphere are bounded by 2), preventing NaN.

**Fix:** Inject `np.nan` directly into all input sequence data before training. NaN propagates deterministically through any linear operation in the LSTM and through the loss computation, triggering the `not np.isfinite(loss_value)` check on the first step. This is also a more realistic test: it covers the actual failure mode that could occur in production (corrupted CSV row producing NaN feature values), not just a theoretical learning-rate explosion that the architecture's normalization layer defends against anyway.

**Test renamed:** `test_train_encoder_raises_on_exploding_lr` → `test_train_encoder_raises_on_nan_input` to accurately describe what is being tested.

---

## 5. Deviations from Plan and Justification

1. **Colab vs local execution:** Already described in Section 2. MPS used locally; code is identical for Colab with CUDA. No architecture change.

2. **NaN safety net trigger method changed:** Described in Section 4.1. The requirement (confirm the safety net raises on non-finite loss) is satisfied; only the mechanism changed from LR explosion to NaN input injection.

3. **Multi-subject ablation added:** Not in week3.md. Added to resolve the pending Pilot AI concern from Week 2. Results are in Section 3.12. The corrected claim is now accurate and appropriately scoped.

4. **Root directory cleanup:** Scripts from Week 2 fix iterations (`analyze_s049_outliers.py`, `generate_report2.py`, `generate_report2_v2.py`, `generate_report2_v3.py`) moved to `scripts/`. This is per Rule 18/19: they produced results cited in report2.md, so they were moved rather than deleted. A new script `scripts/generate_report3.py` was placed in `scripts/` directly rather than root.

5. **torch added to requirements.txt:** `torch>=2.0` added. This is a new dependency starting Week 3.

---

## 6. Integrity Self-Check

Checklist from week3.md section 18:

- [x] `pytest tests/test_key_sequence.py -v` — all 4 tests pass, including `test_reconstruct_key_sequence_order_correct`. Full output in Section 3.2.
- [x] `python -m src.device` prints `mps` (GPU equivalent on this machine). Full output in Section 3.1.
- [x] Training loss in `training_history.json` decreases from 0.1230 to 0.0381 across 50 epochs, with no NaN or Inf values. Full table in Section 3.7.
- [x] Embedding sanity check reports inter/intra ratio of **2.04x**, well above the 1.5x threshold. Full output in Section 3.8.
- [x] Held-out encoder EER (9.59%) and held-out classical baseline EER (16.86%) are both printed and saved side by side. Section 3.11.
- [x] `results/week3/subject_split.json` exists. Section 3.6.
- [x] `results/week3/training_history.json` exists. Section 3.7.
- [x] `results/week3/week3_full_results.json` exists. Sections 3.9, 3.10.
- [x] `results/week3/encoder_weights.pt` exists. Confirmed by `ls results/week3/`.
- [x] Reconstructed key order `['period', 't', 'i', 'e', 'five', 'Shift.r', 'o', 'a', 'n', 'l', 'Return']` verified against password `.tie5Roanl` + Return. Section 3.4.
- [x] Both EER results reported honestly. Encoder beats classical. The script explicitly prints the delta and states a classical-beats-deep result is equally valid to report.
- [x] Environment frozen in `results/week3/requirements.lock.txt`.
- [x] No result was adjusted, fabricated, or selectively reported.

---

## 7. Licensing and IP Notes

- **PyTorch:** BSD-style license (The PyTorch Authors). No IP issues.
- **Siamese/triplet-loss architecture:** The general deep metric learning paradigm (bidirectional LSTM + L2-normalized projection, triplet margin loss) follows established practice. The specific application to keystroke dynamics is directly inspired by the TypeNet line of work:
  - Acien et al., "TypeNet: Deep Learning Keystroke Biometrics," IEEE Transactions on Biometrics, Behavior, and Identity Science, 2022.
  - A citation comment must be added to `src/encoder_model.py` before paper submission per Rule 10.
- **Triplet loss (F.relu(pos_dist - neg_dist + margin).mean()):** Standard formulation from Schroff et al., FaceNet (CVPR 2015). No code directly copied; standard PyTorch idiom. Citation required in paper methods section.
- **CMU dataset:** Unchanged from Weeks 1-2. Public research data, Killourhy and Maxion (2009).
- **Balabit dataset:** Unchanged from Weeks 1-2. Terms still TBD.
- **scikit-learn (RobustScaler, Isolation Forest, One-Class SVM):** BSD license. No issues.
- **Doddington biometric taxonomy (Sheep, Goats, Lambs, Wolves):** Referenced in Week 2 report for s032. If used in the paper, cite: Doddington et al., "Sheep, Goats, Lambs and Wolves: A Statistical Analysis of Speaker Performance in the NIST 1998 Speaker Recognition Evaluation," ICSLP 1998.

---

## 8. Open Questions for Pilot

1. **Encoder beats classical by 7.27pp pooled EER (9.59% vs 16.86% on the same 15 held-out subjects).** This is a clean, well-controlled finding: the background/held-out split is the methodologically correct evaluation for a shared-embedding model. The inter/intra distance ratio (2.04x) confirms the embedding space has genuine structure. Is this gap expected at this scale? Should the split seed be varied (e.g., try seeds 0, 1, 2 in addition to 42) to check whether the 15 held-out subjects happened to be unusually favorable or unfavorable for either model? The current result is reproducible but relies on one fixed split.

2. **s011 shows the most dramatic improvement: IF 28.54% → encoder 4.38% (-24.16pp).** s034 is similar: IF 29.05% → encoder 10.48% (-18.57pp). Both were subjects IF genuinely struggled with. What makes them hard for IF but easy for the encoder? The likely explanation is that these subjects have smooth, consistent typing rhythms that form compact clusters in embedding space but have high absolute variance (in seconds) that throws off per-user IF models. If Week 1's EDA data has mean/std per subject, it would be worth checking whether s011 and s034 have particular characteristics.

3. **s019 and s044 are the two subjects where IF outperforms the encoder.** s019: IF 4.07% vs encoder 7.89% (+3.82pp for IF). s024: IF 6.05% vs encoder 5.07% (-0.98pp, essentially tied). s044: IF 10.38% vs encoder 11.46% (+1.08pp for IF). The encoder underperformance on s019 is the most notable outlier. No prior context on s019 from Weeks 1-2 (it was not in the EDA's notable-subjects list). Should these be flagged and investigated, or is a 2/15 failure rate acceptable for the paper?

4. **s049 EER: 0.43% (encoder) vs 4.63% (IF).** Consistent with Weeks 1-2 findings that s049 has a highly distinctive profile. The encoder found it nearly perfectly separable. No concern, just confirming cross-method consistency.

5. **Balabit parser status:** Still not implemented. The directory structure was confirmed in Week 2 (`training_files/` and `test_files/` both use `user*/session_*` layout). Is the Balabit parser planned for Week 4, or is it intended to remain deferred? If Week 4 includes Balabit evaluation, the parser needs to be written and validated before Week 4 model work begins.

6. **Encoder weights format:** Saved as `results/week3/encoder_weights.pt` (PyTorch state dict). If Week 4 builds on these weights (e.g., fine-tuning or using the encoder as a feature extractor), this is the file to load. If Week 4 trains fresh, this file is still needed as the Week 3 reproducibility artifact.

---

## 9. Readiness for Next Week

Week 3 is complete:

- All 8 unit tests pass (command: `pytest tests/test_key_sequence.py tests/test_subject_split.py tests/test_train_encoder_safety.py -v`)
- Device detection: `mps` (command: `python -m src.device`)
- Encoder trained: 50 epochs, loss 0.1230 → 0.0381, no NaN (command: `python -m src.run_week3`)
- Embedding sanity: 2.04x inter/intra ratio (above 1.5x threshold)
- Held-out results: encoder 9.59% EER, classical IF 16.86% EER (same 15 subjects, fair comparison)
- All four required result files exist: `subject_split.json`, `training_history.json`, `week3_full_results.json`, `encoder_weights.pt`
- Week 2 n=1 overclaim resolved with multi-subject ablation
- Environment frozen: `results/week3/requirements.lock.txt`
- `torch>=2.0` added to `requirements.txt`
- `history.md` updated with Week 3 session entry
- `SetupGuide.md` update needed (should mention PyTorch installation) — flagged as pending

Ready for Week 4.

---

## Appendix A: Full Environment Lockfile

```
certifi==2026.6.17
charset-normalizer==3.4.9
contourpy==1.3.3
cycler==0.12.1
filelock==3.29.7
fonttools==4.63.0
fsspec==2026.6.0
idna==3.18
iniconfig==2.3.0
Jinja2==3.1.6
joblib==1.5.3
kiwisolver==1.5.0
MarkupSafe==3.0.3
matplotlib==3.11.0
mpmath==1.3.0
narwhals==2.23.0
networkx==3.6.1
numpy==2.5.1
packaging==26.2
pandas==3.0.3
pillow==12.3.0
pluggy==1.6.0
Pygments==2.20.0
pyparsing==3.3.2
pytest==9.1.1
python-dateutil==2.9.0.post0
requests==2.34.2
scikit-learn==1.9.0
scipy==1.18.0
seaborn==0.13.2
setuptools==83.0.0
six==1.17.0
sympy==1.14.0
threadpoolctl==3.6.0
torch==2.13.0
typing_extensions==4.16.0
urllib3==2.7.0
```

---

## Appendix B: Full Raw Results JSON (week3_full_results.json)

```json
{
  "held_out_subjects": [
    "s004",
    "s011",
    "s015",
    "s019",
    "s024",
    "s026",
    "s028",
    "s029",
    "s034",
    "s041",
    "s044",
    "s046",
    "s048",
    "s049",
    "s057"
  ],
  "background_subjects": [
    "s002",
    "s003",
    "s005",
    "s007",
    "s008",
    "s010",
    "s012",
    "s013",
    "s016",
    "s017",
    "s018",
    "s020",
    "s021",
    "s022",
    "s025",
    "s027",
    "s030",
    "s031",
    "s032",
    "s033",
    "s035",
    "s036",
    "s037",
    "s038",
    "s039",
    "s040",
    "s042",
    "s043",
    "s047",
    "s050",
    "s051",
    "s052",
    "s053",
    "s054",
    "s055",
    "s056"
  ],
  "embedding_sanity": {
    "intra_mean": 0.6396186657838725,
    "inter_mean": 1.3033519698079374,
    "ratio": 2.037701586164061
  },
  "encoder": {
    "per_subject": {
      "s004": {
        "eer": 0.10732142857142857,
        "eer_threshold": -0.7756063938140869,
        "roc_auc": 0.9442857142857144,
        "pr_auc": 0.4339078435017144,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s011": {
        "eer": 0.043750000000000025,
        "eer_threshold": -0.641510546207428,
        "roc_auc": 0.9909785714285714,
        "pr_auc": 0.9315055416670743,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s015": {
        "eer": 0.1685714285714286,
        "eer_threshold": -0.7714717984199524,
        "roc_auc": 0.9095374999999999,
        "pr_auc": 0.4523047007957693,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s019": {
        "eer": 0.0789285714285714,
        "eer_threshold": -0.787071168422699,
        "roc_auc": 0.9789169642857142,
        "pr_auc": 0.8216655836071696,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s024": {
        "eer": 0.05071428571428574,
        "eer_threshold": -0.7833036780357361,
        "roc_auc": 0.9816982142857144,
        "pr_auc": 0.7054692867919137,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s026": {
        "eer": 0.0807142857142857,
        "eer_threshold": -0.7036495804786682,
        "roc_auc": 0.9784392857142858,
        "pr_auc": 0.8251605032449996,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s028": {
        "eer": 0.0430357142857143,
        "eer_threshold": -0.8383997082710266,
        "roc_auc": 0.9921142857142857,
        "pr_auc": 0.9358172382694184,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s029": {
        "eer": 0.08964285714285714,
        "eer_threshold": -0.6992541551589966,
        "roc_auc": 0.9669160714285714,
        "pr_auc": 0.8030035205497846,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s034": {
        "eer": 0.10482142857142857,
        "eer_threshold": -0.7603265047073364,
        "roc_auc": 0.9635392857142857,
        "pr_auc": 0.7285978744526745,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s041": {
        "eer": 0.0242857142857143,
        "eer_threshold": -0.7411074042320251,
        "roc_auc": 0.9960732142857143,
        "pr_auc": 0.9681583628489021,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s044": {
        "eer": 0.11464285714285713,
        "eer_threshold": -0.8443386554718018,
        "roc_auc": 0.9459375,
        "pr_auc": 0.5243712956038964,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s046": {
        "eer": 0.13517857142857143,
        "eer_threshold": -0.8503215909004211,
        "roc_auc": 0.9478303571428572,
        "pr_auc": 0.7219015685221853,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s048": {
        "eer": 0.07964285714285713,
        "eer_threshold": -0.6480703353881836,
        "roc_auc": 0.9656535714285713,
        "pr_auc": 0.6606749376968322,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s049": {
        "eer": 0.004285714285714288,
        "eer_threshold": -0.7555384039878845,
        "roc_auc": 0.999525,
        "pr_auc": 0.9912541556875204,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s057": {
        "eer": 0.16000000000000003,
        "eer_threshold": -0.7563230395317078,
        "roc_auc": 0.9206071428571428,
        "pr_auc": 0.46225227245783906,
        "n_genuine": 200,
        "n_impostor": 2800
      }
    },
    "pooled": {
      "eer": 0.09591666666666665,
      "eer_threshold": -0.7629245519638062,
      "roc_auc": 0.9666312261904761,
      "pr_auc": 0.7331030597285804,
      "n_genuine": 3000,
      "n_impostor": 42000
    }
  },
  "classical_baseline_heldout": {
    "per_subject": {
      "s004": {
        "eer": 0.1517857142857143,
        "eer_threshold": 0.038738648660526676,
        "roc_auc": 0.9273517857142856,
        "pr_auc": 0.6877209918176658,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s011": {
        "eer": 0.28535714285714286,
        "eer_threshold": 0.05576727906870049,
        "roc_auc": 0.8322125,
        "pr_auc": 0.34328470621799184,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s015": {
        "eer": 0.2567857142857143,
        "eer_threshold": 0.04926669612542761,
        "roc_auc": 0.8245267857142857,
        "pr_auc": 0.27931452481622404,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s019": {
        "eer": 0.04071428571428573,
        "eer_threshold": 0.044762735483020455,
        "roc_auc": 0.988825,
        "pr_auc": 0.96564407392778,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s024": {
        "eer": 0.06053571428571431,
        "eer_threshold": 0.04089223428004585,
        "roc_auc": 0.9913000000000001,
        "pr_auc": 0.9295364384701742,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s026": {
        "eer": 0.09982142857142856,
        "eer_threshold": 0.035037083957465076,
        "roc_auc": 0.9619678571428572,
        "pr_auc": 0.8481987208754512,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s028": {
        "eer": 0.07464285714285712,
        "eer_threshold": 0.009000875002676989,
        "roc_auc": 0.9766035714285715,
        "pr_auc": 0.7899642732858126,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s029": {
        "eer": 0.14285714285714285,
        "eer_threshold": 0.016747877636324515,
        "roc_auc": 0.9449339285714285,
        "pr_auc": 0.7883126908009754,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s034": {
        "eer": 0.29053571428571434,
        "eer_threshold": 0.08227200754555963,
        "roc_auc": 0.7807285714285714,
        "pr_auc": 0.277740779531683,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s041": {
        "eer": 0.15392857142857144,
        "eer_threshold": 0.045919056720172824,
        "roc_auc": 0.9217732142857142,
        "pr_auc": 0.7836590600210536,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s044": {
        "eer": 0.10374999999999998,
        "eer_threshold": 0.0790065922739468,
        "roc_auc": 0.9582982142857143,
        "pr_auc": 0.7279117354257677,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s046": {
        "eer": 0.1480357142857143,
        "eer_threshold": 0.10035698592393832,
        "roc_auc": 0.928975,
        "pr_auc": 0.7451517712781062,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s048": {
        "eer": 0.11517857142857142,
        "eer_threshold": 0.050891976603825495,
        "roc_auc": 0.9490285714285714,
        "pr_auc": 0.6724550436468653,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s049": {
        "eer": 0.04625000000000002,
        "eer_threshold": 0.06795128233997699,
        "roc_auc": 0.9769821428571429,
        "pr_auc": 0.9528389074428818,
        "n_genuine": 200,
        "n_impostor": 2800
      },
      "s057": {
        "eer": 0.2544642857142857,
        "eer_threshold": 0.04400966686246671,
        "roc_auc": 0.8342535714285714,
        "pr_auc": 0.2351198097895734,
        "n_genuine": 200,
        "n_impostor": 2800
      }
    },
    "pooled": {
      "eer": 0.16863095238095238,
      "eer_threshold": 0.05439134684436553,
      "roc_auc": 0.9083600634920634,
      "pr_auc": 0.522550334235955,
      "n_genuine": 3000,
      "n_impostor": 42000
    }
  }
}```

---

## Appendix C: Full Training History JSON (training_history.json)

```json
{
  "epoch": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50
  ],
  "train_loss": [
    0.12296036258339882,
    0.10354179970920085,
    0.08923293054103851,
    0.09552671454846859,
    0.08171026185154914,
    0.08338264763355255,
    0.0786250563338399,
    0.07498855050653219,
    0.06340785395354033,
    0.06991974979639054,
    0.06875234473496676,
    0.0650539955496788,
    0.06803569369018078,
    0.0700933900475502,
    0.06300491575151682,
    0.06451674818992614,
    0.05997383315116167,
    0.06000669948756695,
    0.06341721814125777,
    0.050076304189860824,
    0.051616050377488135,
    0.051771416384726765,
    0.054443492256104946,
    0.05304368581622839,
    0.05179354030638933,
    0.05392674243077636,
    0.05561058893799782,
    0.048385944589972495,
    0.04999328475445509,
    0.05017465554177761,
    0.0415848951973021,
    0.04627429906278849,
    0.03876026000827551,
    0.04942807678133249,
    0.047294151335954666,
    0.046838932428509,
    0.041182767264544964,
    0.042329000774770975,
    0.043117828592658045,
    0.040175752490758894,
    0.04737762324512005,
    0.03915332965552807,
    0.047072199359536174,
    0.03623160157352686,
    0.03913093954324722,
    0.04704584315419197,
    0.03823133172467351,
    0.04058588169515133,
    0.04592896807938814,
    0.03811423238366842
  ]
}```

---

## Appendix D: Per-Subject Feature Statistics (Held-Out Subjects, Enrollment Sessions 1-4)

All timing values in seconds. Statistics computed across all feature columns of each type and all 200 enrollment rows.

| Subject | N | Mean H | Std H | Mean DD | Std DD | Mean UD | Std UD | Max H | Max DD |
|---------|---|--------|-------|---------|--------|---------|--------|-------|--------|
| s004 | 200 | 0.096263 | 0.022796 | 0.230024 | 0.168647 | 0.131586 | 0.175436 | 0.215700 | 1.351800 |
| s011 | 200 | 0.108293 | 0.027139 | 0.188593 | 0.124314 | 0.081985 | 0.125296 | 0.207400 | 1.379200 |
| s015 | 200 | 0.075081 | 0.016322 | 0.197877 | 0.153439 | 0.122295 | 0.153465 | 0.146000 | 1.539000 |
| s019 | 200 | 0.084782 | 0.021427 | 0.309731 | 0.199659 | 0.226684 | 0.200772 | 0.188200 | 2.529400 |
| s024 | 200 | 0.058698 | 0.014021 | 0.286663 | 0.218232 | 0.229054 | 0.218733 | 0.111900 | 2.300700 |
| s026 | 200 | 0.084778 | 0.012953 | 0.221661 | 0.136605 | 0.138025 | 0.135654 | 0.132500 | 1.922800 |
| s028 | 200 | 0.058688 | 0.009095 | 0.229101 | 0.156530 | 0.170107 | 0.158096 | 0.111100 | 1.270600 |
| s029 | 200 | 0.077562 | 0.017519 | 0.196247 | 0.128332 | 0.118030 | 0.129689 | 0.150200 | 1.380800 |
| s034 | 200 | 0.089731 | 0.024613 | 0.221176 | 0.192249 | 0.130447 | 0.196678 | 0.242500 | 1.551300 |
| s041 | 200 | 0.139749 | 0.040043 | 0.286809 | 0.228633 | 0.146917 | 0.237187 | 0.687200 | 3.645000 |
| s044 | 200 | 0.064354 | 0.017688 | 0.301428 | 0.246961 | 0.237614 | 0.251781 | 0.148000 | 2.595100 |
| s046 | 200 | 0.097474 | 0.023796 | 0.314893 | 0.227170 | 0.216359 | 0.228149 | 0.200200 | 2.632500 |
| s048 | 200 | 0.090026 | 0.017047 | 0.211763 | 0.146292 | 0.122590 | 0.146472 | 0.176800 | 1.636600 |
| s049 | 200 | 0.102351 | 0.054855 | 0.636856 | 0.841053 | 0.532117 | 0.840858 | 2.035300 | 25.987300 |
| s057 | 200 | 0.084824 | 0.022861 | 0.192163 | 0.123973 | 0.110984 | 0.124098 | 0.205500 | 1.465200 |

Notes on held-out subject profiles:
- s049: Dominant outlier. Mean DD 0.637s vs typical 0.2s. Std DD 0.841s vs typical 0.12 to 0.25s. Max DD 25.987s. Encoder 0.43% EER because extreme timing variance is a unique fingerprint.
- s028: Lowest Std H of all held-out subjects (0.009s). Highly consistent hold behavior. Encoder 4.30% EER.
- s041: Highest Mean H (0.140s) and large Max H (0.687s). IF 15.39% EER but encoder 2.43% EER, 12.96pp improvement.
- s034: High Std DD (0.192s) relative to Mean DD (0.221s). IF 29.05% EER but encoder 10.48% EER, 18.57pp improvement.
- s011: Low Std DD (0.124s). IF 28.54% EER but encoder 4.38% EER, 24.16pp improvement. Encoder learns sequence pattern even when per-feature variance is low and absolute values overlap with other subjects.

---

## Appendix E: Per-Subject Feature Statistics (Background Subjects, Enrollment Sessions 1-4)

| Subject | N | Mean H | Std H | Mean DD | Std DD | Max H | Max DD |
|---------|---|--------|-------|---------|--------|-------|--------|
| s002 | 200 | 0.091625 | 0.018411 | 0.263028 | 0.201839 | 0.168200 | 1.736000 |
| s003 | 200 | 0.115311 | 0.024397 | 0.219417 | 0.146979 | 0.246500 | 1.498800 |
| s005 | 200 | 0.102417 | 0.023920 | 0.320253 | 0.173540 | 0.187400 | 2.731400 |
| s007 | 200 | 0.085027 | 0.019517 | 0.192366 | 0.119583 | 0.164200 | 1.046400 |
| s008 | 200 | 0.094298 | 0.018861 | 0.198565 | 0.131487 | 0.160200 | 1.728000 |
| s010 | 200 | 0.072478 | 0.014151 | 0.186333 | 0.095752 | 0.124300 | 1.117600 |
| s012 | 200 | 0.133389 | 0.027549 | 0.228286 | 0.155716 | 0.297400 | 1.604900 |
| s013 | 200 | 0.075282 | 0.015764 | 0.169911 | 0.122445 | 0.127300 | 1.419200 |
| s016 | 200 | 0.090888 | 0.021293 | 0.468623 | 0.279695 | 0.171600 | 2.714900 |
| s017 | 200 | 0.059067 | 0.010549 | 0.193266 | 0.127020 | 0.101100 | 1.357200 |
| s018 | 200 | 0.096326 | 0.026114 | 0.227705 | 0.174451 | 0.234900 | 2.911300 |
| s020 | 200 | 0.108968 | 0.025132 | 0.240009 | 0.203325 | 0.254900 | 1.995400 |
| s021 | 200 | 0.082500 | 0.018739 | 0.240352 | 0.133238 | 0.164400 | 1.594200 |
| s022 | 200 | 0.057601 | 0.018148 | 0.525732 | 0.363756 | 0.144900 | 4.025200 |
| s025 | 200 | 0.078678 | 0.016038 | 0.285778 | 0.235855 | 0.174500 | 1.790900 |
| s027 | 200 | 0.100653 | 0.022372 | 0.283879 | 0.197971 | 0.181100 | 1.722800 |
| s030 | 200 | 0.105741 | 0.026942 | 0.327349 | 0.228210 | 0.246800 | 4.010500 |
| s031 | 200 | 0.083201 | 0.024694 | 0.289982 | 0.207487 | 0.223900 | 2.612700 |
| s032 | 200 | 0.084401 | 0.017599 | 0.216239 | 0.169732 | 0.149000 | 2.042800 |
| s033 | 200 | 0.126331 | 0.046180 | 0.417341 | 0.258966 | 0.722100 | 2.458200 |
| s035 | 200 | 0.070901 | 0.020061 | 0.249905 | 0.174409 | 0.161200 | 1.773200 |
| s036 | 200 | 0.044625 | 0.008271 | 0.603943 | 0.462935 | 0.075500 | 5.883600 |
| s037 | 200 | 0.095544 | 0.021341 | 0.223977 | 0.154891 | 0.184900 | 1.892800 |
| s038 | 200 | 0.080464 | 0.021582 | 0.293010 | 0.251284 | 0.158300 | 2.669300 |
| s039 | 200 | 0.092508 | 0.022369 | 0.244844 | 0.177604 | 0.185900 | 2.376400 |
| s040 | 200 | 0.118832 | 0.034500 | 0.355747 | 0.272111 | 0.259400 | 2.546200 |
| s042 | 200 | 0.089861 | 0.032714 | 0.242982 | 0.133183 | 0.225500 | 1.295300 |
| s043 | 200 | 0.064842 | 0.019696 | 0.385200 | 0.366606 | 0.157500 | 12.506100 |
| s047 | 200 | 0.083207 | 0.021492 | 0.363264 | 0.295093 | 0.160600 | 5.302700 |
| s050 | 200 | 0.087042 | 0.019619 | 0.236070 | 0.142328 | 0.204500 | 1.436800 |
| s051 | 200 | 0.078195 | 0.018424 | 0.187646 | 0.104396 | 0.180100 | 1.475100 |
| s052 | 200 | 0.085374 | 0.024873 | 0.291166 | 0.222608 | 0.204100 | 1.656000 |
| s053 | 200 | 0.087195 | 0.027807 | 0.177608 | 0.130227 | 0.182000 | 1.314300 |
| s054 | 200 | 0.092103 | 0.021086 | 0.215769 | 0.143470 | 0.203400 | 1.565100 |
| s055 | 200 | 0.094416 | 0.021061 | 0.149960 | 0.088193 | 0.226300 | 0.965900 |
| s056 | 200 | 0.093111 | 0.022058 | 0.196304 | 0.130327 | 0.187500 | 1.341400 |

Notes:
- s036: Lowest Mean H (0.045s, fastest hold). Highest Mean DD (0.604s) of any background subject. Max DD 5.884s. Week 2 best classical subject (0.98% IF EER). In background set, contributed to encoder training.
- s043: Max DD 12.506s, Std DD 0.367s. The encoder was trained on this extreme DD distribution without collapsing.
- s032: Unremarkable mean feature values (Mean H 0.084s, Mean DD 0.216s). Week 2 Goat (44.40% IF EER). Difficulty is from within-subject variance overlapping with other subjects, not from extreme absolute values.

---

## Appendix F: Outlier Ablation Multi-Subject Raw Data

Threshold: flag row if any H column exceeds 0.5s OR any DD column exceeds 2.0s

s049 (from Week 2 report, included for reference):
  Flagged rows: 49 out of 200
  IF EER WITH: 5.02%  IF EER WITHOUT (n=151): 4.80%  Delta: -0.22pp
  OCSVM EER WITH: 4.83%  OCSVM EER WITHOUT (n=151): 8.89%  Delta: +4.06pp

s032 (new this week):
  Flagged rows: 1 out of 200
  IF EER WITH: 44.40%  IF EER WITHOUT (n=199): 44.05%  Delta: -0.35pp
  OCSVM EER WITH: 42.55%  OCSVM EER WITHOUT (n=199): 42.11%  Delta: -0.44pp

s003 (new this week):
  Flagged rows: 0 out of 200
  Ablation is a no-op
  IF EER: 32.48%  OCSVM EER: 39.91%

s011 (new this week):
  Flagged rows: 0 out of 200
  Ablation is a no-op
  IF EER: 26.06%  OCSVM EER: 36.98%

s007 (new this week):
  Flagged rows: 0 out of 200
  Ablation is a no-op
  IF EER: 28.00%  OCSVM EER: 35.47%

---

## Appendix G: CMU Dataset Feature Columns

```
H.period, H.t, H.i, H.e, H.five, H.Shift.r, H.o, H.a, H.n, H.l, H.Return
DD.period.t, DD.t.i, DD.i.e, DD.e.five, DD.five.Shift.r, DD.Shift.r.o, DD.o.a, DD.a.n, DD.n.l, DD.l.Return
UD.period.t, UD.t.i, UD.i.e, UD.e.five, UD.five.Shift.r, UD.Shift.r.o, UD.o.a, UD.a.n, UD.n.l, UD.l.Return
```

11 H columns + 10 DD columns + 10 UD columns = 31 total feature columns per repetition row.
The DD and UD column names encode consecutive key transitions in the chronological typing order of the password.

---

## Appendix H: Cross-Week Results Summary

| Week | Method | Subjects | Pooled EER | ROC-AUC | PR-AUC |
|------|--------|---------|------------|---------|--------|
| 2 | Isolation Forest | All 51 | 17.26% | 0.9078 | 0.3059 |
| 2 | One-Class SVM | All 51 | 18.06% | 0.8994 | 0.2769 |
| 3 | Isolation Forest (held-out 15 only) | 15 held-out | 16.86% | 0.9084 | 0.5226 |
| 3 | Deep Encoder BiLSTM Triplet | 15 held-out | 9.59% | 0.9666 | 0.7331 |

Fair comparison for paper: Week 3 IF vs Week 3 Encoder on same 15 subjects.
Encoder wins: 7.27pp EER, 0.058 ROC-AUC, 0.211 PR-AUC.

---

## Appendix I: Per-Column Per-Subject Enrollment Statistics (Held-Out Subjects)

Mean and standard deviation per individual feature column, for each held-out subject's 200 enrollment rows.

### s004 (n=200 enrollment rows)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.074512 | 0.012066 | 0.038600 | 0.114300 |
| H.Shift.r | 0.129547 | 0.021199 | 0.075300 | 0.215700 |
| H.a | 0.124615 | 0.017077 | 0.058600 | 0.158100 |
| H.e | 0.078728 | 0.011451 | 0.008200 | 0.126400 |
| H.five | 0.077433 | 0.019112 | 0.042300 | 0.131200 |
| H.i | 0.102124 | 0.010591 | 0.073200 | 0.145200 |
| H.l | 0.094832 | 0.012120 | 0.044600 | 0.125900 |
| H.n | 0.085179 | 0.011465 | 0.051500 | 0.128600 |
| H.o | 0.094485 | 0.010991 | 0.062000 | 0.130100 |
| H.period | 0.103418 | 0.017117 | 0.046200 | 0.142500 |
| H.t | 0.094020 | 0.016232 | 0.051000 | 0.140700 |
| DD.Shift.r.o | 0.232015 | 0.080087 | 0.131800 | 0.637900 |
| DD.a.n | 0.125580 | 0.035155 | 0.026500 | 0.365000 |
| DD.e.five | 0.476127 | 0.208664 | 0.162100 | 1.335000 |
| DD.five.Shift.r | 0.454758 | 0.173226 | 0.284100 | 1.332000 |
| DD.i.e | 0.126046 | 0.125099 | 0.014000 | 1.349200 |
| DD.l.Return | 0.242325 | 0.078427 | 0.174400 | 1.021500 |
| DD.n.l | 0.155474 | 0.063018 | 0.095600 | 0.857700 |
| DD.o.a | 0.119852 | 0.055890 | 0.018000 | 0.514400 |
| DD.period.t | 0.217446 | 0.120451 | 0.102100 | 1.351800 |
| DD.t.i | 0.150619 | 0.052060 | 0.063100 | 0.551500 |
| UD.Shift.r.o | 0.102467 | 0.079729 | 0.008500 | 0.527800 |
| UD.a.n | 0.000964 | 0.041652 | -0.108800 | 0.306400 |
| UD.e.five | 0.397399 | 0.207805 | 0.087600 | 1.255300 |
| UD.five.Shift.r | 0.377325 | 0.170402 | 0.218900 | 1.249600 |
| UD.i.e | 0.023921 | 0.124282 | -0.085300 | 1.238900 |
| UD.l.Return | 0.147493 | 0.078565 | 0.085700 | 0.931700 |
| UD.n.l | 0.070295 | 0.060628 | 0.016600 | 0.787700 |
| UD.o.a | 0.025367 | 0.055319 | -0.060400 | 0.439200 |
| UD.period.t | 0.114029 | 0.118317 | 0.001300 | 1.234900 |
| UD.t.i | 0.056600 | 0.057821 | -0.043000 | 0.466800 |

### s011 (n=200 enrollment rows)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.125146 | 0.028476 | 0.024800 | 0.188400 |
| H.Shift.r | 0.128495 | 0.014495 | 0.078400 | 0.178200 |
| H.a | 0.130000 | 0.021568 | 0.075000 | 0.180500 |
| H.e | 0.114312 | 0.025957 | 0.039900 | 0.207400 |
| H.five | 0.101612 | 0.023348 | 0.006100 | 0.175000 |
| H.i | 0.093779 | 0.018229 | 0.029100 | 0.137500 |
| H.l | 0.102862 | 0.032272 | 0.012500 | 0.197500 |
| H.n | 0.104851 | 0.027665 | 0.040100 | 0.188800 |
| H.o | 0.089798 | 0.014828 | 0.029900 | 0.124100 |
| H.period | 0.086821 | 0.018919 | 0.037200 | 0.144400 |
| H.t | 0.113549 | 0.018792 | 0.052800 | 0.170500 |
| DD.Shift.r.o | 0.207412 | 0.048569 | 0.132900 | 0.487000 |
| DD.a.n | 0.110263 | 0.036689 | 0.005600 | 0.286200 |
| DD.e.five | 0.298042 | 0.177881 | 0.091800 | 1.303500 |
| DD.five.Shift.r | 0.357131 | 0.105328 | 0.228900 | 0.806800 |
| DD.i.e | 0.143578 | 0.095896 | 0.011400 | 0.989900 |
| DD.l.Return | 0.188952 | 0.156520 | 0.008300 | 1.379200 |
| DD.n.l | 0.105140 | 0.049326 | 0.035900 | 0.371200 |
| DD.o.a | 0.133791 | 0.049965 | 0.064500 | 0.453500 |
| DD.period.t | 0.185278 | 0.070245 | 0.099400 | 0.564100 |
| DD.t.i | 0.156340 | 0.067975 | 0.058900 | 0.523000 |
| UD.Shift.r.o | 0.078917 | 0.048726 | 0.002400 | 0.408600 |
| UD.a.n | -0.019738 | 0.031284 | -0.097900 | 0.134400 |
| UD.e.five | 0.183730 | 0.179692 | -0.044100 | 1.198200 |
| UD.five.Shift.r | 0.255519 | 0.105650 | 0.127100 | 0.693800 |
| UD.i.e | 0.049800 | 0.096738 | -0.039900 | 0.894100 |
| UD.l.Return | 0.086089 | 0.154592 | -0.036100 | 1.264900 |
| UD.n.l | 0.000289 | 0.047896 | -0.080000 | 0.304300 |
| UD.o.a | 0.043993 | 0.051174 | -0.031900 | 0.398600 |
| UD.period.t | 0.098456 | 0.071676 | 0.011000 | 0.503400 |
| UD.t.i | 0.042791 | 0.071259 | -0.032200 | 0.435700 |

### s015 (n=200 enrollment rows)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.070074 | 0.011671 | 0.027000 | 0.110600 |
| H.Shift.r | 0.068640 | 0.010431 | 0.016400 | 0.095600 |
| H.a | 0.080995 | 0.016835 | 0.023800 | 0.122000 |
| H.e | 0.081510 | 0.015218 | 0.032000 | 0.127000 |
| H.five | 0.060046 | 0.012866 | 0.006100 | 0.085000 |
| H.i | 0.066356 | 0.015047 | 0.013800 | 0.109000 |
| H.l | 0.076820 | 0.013295 | 0.023000 | 0.110900 |
| H.n | 0.076347 | 0.015703 | 0.036200 | 0.124600 |
| H.o | 0.088177 | 0.011865 | 0.052000 | 0.146000 |
| H.period | 0.083097 | 0.017794 | 0.024600 | 0.128500 |
| H.t | 0.073833 | 0.014522 | 0.018000 | 0.114600 |
| DD.Shift.r.o | 0.196927 | 0.087752 | 0.117400 | 0.658500 |
| DD.a.n | 0.108469 | 0.068409 | 0.021400 | 0.648200 |
| DD.e.five | 0.460801 | 0.208260 | 0.198500 | 1.377000 |
| DD.five.Shift.r | 0.308933 | 0.139583 | 0.196700 | 1.539000 |
| DD.i.e | 0.082434 | 0.123134 | 0.004500 | 1.416900 |
| DD.l.Return | 0.213813 | 0.073181 | 0.124000 | 0.855200 |
| DD.n.l | 0.149888 | 0.080440 | 0.062800 | 0.736300 |
| DD.o.a | 0.142085 | 0.062430 | 0.052100 | 0.717600 |
| DD.period.t | 0.202163 | 0.101611 | 0.048200 | 0.837000 |
| DD.t.i | 0.113261 | 0.057951 | 0.046000 | 0.568700 |
| UD.Shift.r.o | 0.128286 | 0.086835 | 0.038600 | 0.577700 |
| UD.a.n | 0.027474 | 0.072280 | -0.049300 | 0.624400 |
| UD.e.five | 0.379291 | 0.207189 | 0.116600 | 1.291200 |
| UD.five.Shift.r | 0.248887 | 0.139420 | 0.117200 | 1.480400 |
| UD.i.e | 0.016078 | 0.123621 | -0.051000 | 1.368800 |
| UD.l.Return | 0.136992 | 0.071095 | 0.086200 | 0.757500 |
| UD.n.l | 0.073542 | 0.079016 | 0.001300 | 0.635700 |
| UD.o.a | 0.053908 | 0.061652 | -0.014200 | 0.640800 |
| UD.period.t | 0.119066 | 0.099482 | -0.001200 | 0.748300 |
| UD.t.i | 0.039428 | 0.059816 | -0.026700 | 0.477900 |

### s019 (n=200 enrollment rows)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.102133 | 0.020947 | 0.045200 | 0.181600 |
| H.Shift.r | 0.109280 | 0.015903 | 0.065000 | 0.188200 |
| H.a | 0.112774 | 0.010873 | 0.085000 | 0.154100 |
| H.e | 0.095593 | 0.014143 | 0.010100 | 0.149900 |
| H.five | 0.073219 | 0.011499 | 0.042300 | 0.109800 |
| H.i | 0.059318 | 0.007548 | 0.015100 | 0.085500 |
| H.l | 0.071164 | 0.010415 | 0.041500 | 0.107400 |
| H.n | 0.070010 | 0.012703 | 0.034600 | 0.113200 |
| H.o | 0.070887 | 0.011250 | 0.050400 | 0.113200 |
| H.period | 0.078340 | 0.009100 | 0.051200 | 0.099000 |
| H.t | 0.089883 | 0.011137 | 0.059100 | 0.126200 |
| DD.Shift.r.o | 0.281075 | 0.109699 | 0.195700 | 1.582600 |
| DD.a.n | 0.183692 | 0.047606 | 0.108500 | 0.537900 |
| DD.e.five | 0.405575 | 0.153396 | 0.259200 | 1.715800 |
| DD.five.Shift.r | 0.505471 | 0.225700 | 0.315000 | 1.919500 |
| DD.i.e | 0.107108 | 0.064382 | 0.053100 | 0.600400 |
| DD.l.Return | 0.584938 | 0.211843 | 0.278400 | 2.529400 |
| DD.n.l | 0.303710 | 0.137663 | 0.200000 | 1.210000 |
| DD.o.a | 0.116946 | 0.053431 | 0.058900 | 0.639600 |
| DD.period.t | 0.330584 | 0.121501 | 0.192300 | 1.029300 |
| DD.t.i | 0.278211 | 0.068125 | 0.158400 | 0.593000 |
| UD.Shift.r.o | 0.171795 | 0.107837 | 0.067600 | 1.453500 |
| UD.a.n | 0.070918 | 0.045156 | -0.004200 | 0.391700 |
| UD.e.five | 0.309982 | 0.156187 | 0.166000 | 1.705700 |
| UD.five.Shift.r | 0.432252 | 0.222613 | 0.251600 | 1.873800 |
| UD.i.e | 0.047789 | 0.062797 | -0.010300 | 0.532000 |
| UD.l.Return | 0.513775 | 0.210406 | 0.210000 | 2.451300 |
| UD.n.l | 0.233700 | 0.137278 | 0.131200 | 1.145000 |
| UD.o.a | 0.046060 | 0.054179 | -0.027900 | 0.574600 |
| UD.period.t | 0.252244 | 0.124103 | 0.099900 | 0.943000 |
| UD.t.i | 0.188328 | 0.066199 | 0.079200 | 0.497200 |

### s024 (n=200 enrollment rows)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.069581 | 0.014173 | 0.008500 | 0.108000 |
| H.Shift.r | 0.052506 | 0.005426 | 0.038600 | 0.070200 |
| H.a | 0.078057 | 0.007925 | 0.047300 | 0.098200 |
| H.e | 0.057781 | 0.009100 | 0.002100 | 0.085000 |
| H.five | 0.048766 | 0.007004 | 0.031400 | 0.085500 |
| H.i | 0.046545 | 0.008725 | 0.008700 | 0.067300 |
| H.l | 0.069178 | 0.010031 | 0.043800 | 0.108200 |
| H.n | 0.056092 | 0.007265 | 0.038600 | 0.085300 |
| H.o | 0.044364 | 0.007448 | 0.019300 | 0.069700 |
| H.period | 0.069962 | 0.011892 | 0.030100 | 0.111900 |
| H.t | 0.052843 | 0.007012 | 0.034300 | 0.091600 |
| DD.Shift.r.o | 0.284978 | 0.082580 | 0.197700 | 1.161200 |
| DD.a.n | 0.131860 | 0.097341 | 0.055700 | 1.191500 |
| DD.e.five | 0.658353 | 0.211269 | 0.363900 | 1.591500 |
| DD.five.Shift.r | 0.484978 | 0.168225 | 0.333200 | 2.014600 |
| DD.i.e | 0.140823 | 0.175555 | 0.069200 | 1.895700 |
| DD.l.Return | 0.347458 | 0.245865 | 0.185400 | 2.300700 |
| DD.n.l | 0.262663 | 0.095589 | 0.184600 | 0.935800 |
| DD.o.a | 0.146339 | 0.049834 | 0.076200 | 0.435500 |
| DD.period.t | 0.236186 | 0.119321 | 0.106800 | 1.047800 |
| DD.t.i | 0.172995 | 0.088001 | 0.094300 | 0.862400 |
| UD.Shift.r.o | 0.232472 | 0.082334 | 0.154400 | 1.115500 |
| UD.a.n | 0.053803 | 0.097997 | -0.015000 | 1.130500 |
| UD.e.five | 0.600572 | 0.212972 | 0.300000 | 1.534700 |
| UD.five.Shift.r | 0.436212 | 0.168180 | 0.286400 | 1.979500 |
| UD.i.e | 0.094278 | 0.175842 | 0.021100 | 1.861400 |
| UD.l.Return | 0.278280 | 0.244903 | 0.117300 | 2.225500 |
| UD.n.l | 0.206572 | 0.095800 | 0.121400 | 0.878200 |
| UD.o.a | 0.101975 | 0.047541 | 0.038400 | 0.385600 |
| UD.period.t | 0.166224 | 0.117273 | 0.037400 | 0.959400 |
| UD.t.i | 0.120152 | 0.087253 | 0.048600 | 0.808000 |

### s026 (n=200 enrollment rows)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.096189 | 0.011337 | 0.031200 | 0.124900 |
| H.Shift.r | 0.082099 | 0.008203 | 0.057300 | 0.114000 |
| H.a | 0.088109 | 0.007249 | 0.065500 | 0.110300 |
| H.e | 0.084792 | 0.008073 | 0.065000 | 0.108800 |
| H.five | 0.072260 | 0.006613 | 0.039600 | 0.089200 |
| H.i | 0.080184 | 0.009351 | 0.046700 | 0.115400 |
| H.l | 0.106801 | 0.010441 | 0.079700 | 0.132500 |
| H.n | 0.074331 | 0.011222 | 0.041500 | 0.106400 |
| H.o | 0.086220 | 0.007741 | 0.068600 | 0.108800 |
| H.period | 0.083087 | 0.009158 | 0.066800 | 0.132000 |
| H.t | 0.078483 | 0.007558 | 0.061300 | 0.114600 |
| DD.Shift.r.o | 0.206097 | 0.061746 | 0.127300 | 0.672400 |
| DD.a.n | 0.119109 | 0.041766 | 0.062600 | 0.284600 |
| DD.e.five | 0.255454 | 0.111763 | 0.152600 | 0.714400 |
| DD.five.Shift.r | 0.381480 | 0.108974 | 0.270300 | 0.897300 |
| DD.i.e | 0.160921 | 0.078145 | 0.091200 | 0.727000 |
| DD.l.Return | 0.320754 | 0.175204 | 0.188100 | 1.659800 |
| DD.n.l | 0.182528 | 0.069237 | 0.117000 | 0.694800 |
| DD.o.a | 0.167413 | 0.067728 | 0.099700 | 0.671800 |
| DD.period.t | 0.283344 | 0.201975 | 0.127700 | 1.922800 |
| DD.t.i | 0.139509 | 0.061062 | 0.080500 | 0.693300 |
| UD.Shift.r.o | 0.123999 | 0.062855 | 0.054700 | 0.615100 |
| UD.a.n | 0.031001 | 0.043628 | -0.035100 | 0.206700 |
| UD.e.five | 0.170662 | 0.111650 | 0.074200 | 0.630700 |
| UD.five.Shift.r | 0.309221 | 0.110730 | 0.200100 | 0.836600 |
| UD.i.e | 0.080738 | 0.077269 | 0.013200 | 0.643600 |
| UD.l.Return | 0.213953 | 0.172517 | 0.094900 | 1.538600 |
| UD.n.l | 0.108197 | 0.069157 | 0.023000 | 0.633800 |
| UD.o.a | 0.081193 | 0.064796 | 0.014600 | 0.569600 |
| UD.period.t | 0.200257 | 0.199836 | 0.054800 | 1.816700 |
| UD.t.i | 0.061026 | 0.061642 | -0.001600 | 0.616200 |

### s028 (n=200 enrollment rows)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.055628 | 0.006364 | 0.033000 | 0.073400 |
| H.Shift.r | 0.061729 | 0.006712 | 0.045200 | 0.086300 |
| H.a | 0.066927 | 0.008075 | 0.043600 | 0.101400 |
| H.e | 0.057329 | 0.007876 | 0.043300 | 0.089300 |
| H.five | 0.051686 | 0.004829 | 0.038000 | 0.064200 |
| H.i | 0.060596 | 0.007491 | 0.043000 | 0.087200 |
| H.l | 0.054932 | 0.007116 | 0.032000 | 0.080500 |
| H.n | 0.061751 | 0.004205 | 0.050700 | 0.074800 |
| H.o | 0.055747 | 0.007353 | 0.018800 | 0.074400 |
| H.period | 0.069195 | 0.009403 | 0.049600 | 0.111100 |
| H.t | 0.050045 | 0.006968 | 0.030600 | 0.075800 |
| DD.Shift.r.o | 0.289513 | 0.099652 | 0.116400 | 0.692200 |
| DD.a.n | 0.123327 | 0.041549 | 0.052300 | 0.284300 |
| DD.e.five | 0.503145 | 0.132688 | 0.340500 | 1.270600 |
| DD.five.Shift.r | 0.418168 | 0.064704 | 0.328300 | 0.695000 |
| DD.i.e | 0.094377 | 0.101553 | 0.043800 | 0.815500 |
| DD.l.Return | 0.260444 | 0.074074 | 0.165400 | 0.689700 |
| DD.n.l | 0.221458 | 0.054767 | 0.139500 | 0.619900 |
| DD.o.a | 0.091659 | 0.079929 | 0.039900 | 0.604700 |
| DD.period.t | 0.170181 | 0.058718 | 0.096000 | 0.542700 |
| DD.t.i | 0.118740 | 0.059288 | 0.066800 | 0.738200 |
| UD.Shift.r.o | 0.227785 | 0.099434 | 0.053300 | 0.633100 |
| UD.a.n | 0.056400 | 0.040585 | -0.011400 | 0.218000 |
| UD.e.five | 0.445816 | 0.132705 | 0.280800 | 1.222800 |
| UD.five.Shift.r | 0.366482 | 0.064232 | 0.281700 | 0.649000 |
| UD.i.e | 0.033780 | 0.103672 | -0.021900 | 0.767700 |
| UD.l.Return | 0.205512 | 0.072892 | 0.114400 | 0.636400 |
| UD.n.l | 0.159707 | 0.055487 | 0.072600 | 0.557100 |
| UD.o.a | 0.035912 | 0.080825 | -0.015000 | 0.538200 |
| UD.period.t | 0.100985 | 0.061113 | 0.027400 | 0.484900 |
| UD.t.i | 0.068695 | 0.058423 | 0.010600 | 0.679300 |

### s029 (n=200 enrollment rows)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.071012 | 0.012076 | 0.008200 | 0.108800 |
| H.Shift.r | 0.079047 | 0.013830 | 0.045200 | 0.150200 |
| H.a | 0.079167 | 0.012106 | 0.028000 | 0.147000 |
| H.e | 0.070900 | 0.010654 | 0.044600 | 0.102400 |
| H.five | 0.072754 | 0.013175 | 0.011900 | 0.113500 |
| H.i | 0.054684 | 0.007906 | 0.029300 | 0.092100 |
| H.l | 0.093043 | 0.011033 | 0.065500 | 0.136700 |
| H.n | 0.103523 | 0.018010 | 0.052500 | 0.142600 |
| H.o | 0.077819 | 0.010533 | 0.046500 | 0.105600 |
| H.period | 0.086934 | 0.010604 | 0.052300 | 0.116100 |
| H.t | 0.064302 | 0.007571 | 0.044400 | 0.086600 |
| DD.Shift.r.o | 0.191612 | 0.072414 | 0.110300 | 0.665300 |
| DD.a.n | 0.111435 | 0.045482 | 0.051800 | 0.657400 |
| DD.e.five | 0.375641 | 0.164265 | 0.190300 | 1.130800 |
| DD.five.Shift.r | 0.358483 | 0.102661 | 0.240200 | 1.078100 |
| DD.i.e | 0.088617 | 0.043946 | 0.031700 | 0.446800 |
| DD.l.Return | 0.251917 | 0.054912 | 0.178300 | 0.728000 |
| DD.n.l | 0.100281 | 0.041613 | 0.049300 | 0.465300 |
| DD.o.a | 0.155815 | 0.108187 | 0.080600 | 1.380800 |
| DD.period.t | 0.180158 | 0.054247 | 0.108700 | 0.515600 |
| DD.t.i | 0.148511 | 0.068899 | 0.095900 | 0.593300 |
| UD.Shift.r.o | 0.112565 | 0.069990 | 0.036200 | 0.587200 |
| UD.a.n | 0.032268 | 0.043720 | -0.017900 | 0.575300 |
| UD.e.five | 0.304741 | 0.167349 | 0.116400 | 1.076100 |
| UD.five.Shift.r | 0.285729 | 0.102232 | 0.178200 | 1.012900 |
| UD.i.e | 0.033933 | 0.043007 | -0.029100 | 0.391900 |
| UD.l.Return | 0.158874 | 0.053808 | 0.095400 | 0.609500 |
| UD.n.l | -0.003242 | 0.044695 | -0.053400 | 0.353900 |
| UD.o.a | 0.077996 | 0.106609 | -0.002300 | 1.314500 |
| UD.period.t | 0.093225 | 0.054762 | 0.020000 | 0.431100 |
| UD.t.i | 0.084209 | 0.068124 | 0.034900 | 0.531200 |

### s034 (n=200 enrollment rows)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.079759 | 0.016876 | 0.006400 | 0.130000 |
| H.Shift.r | 0.110268 | 0.025275 | 0.049300 | 0.231800 |
| H.a | 0.101181 | 0.016292 | 0.010900 | 0.180800 |
| H.e | 0.070433 | 0.011289 | 0.031500 | 0.111600 |
| H.five | 0.092095 | 0.023236 | 0.033300 | 0.179600 |
| H.i | 0.059507 | 0.010677 | 0.035400 | 0.099700 |
| H.l | 0.092415 | 0.016125 | 0.025900 | 0.134000 |
| H.n | 0.108525 | 0.025116 | 0.010800 | 0.193900 |
| H.o | 0.100124 | 0.031194 | 0.046700 | 0.242500 |
| H.period | 0.087597 | 0.014941 | 0.050100 | 0.144800 |
| H.t | 0.085138 | 0.012969 | 0.057800 | 0.127100 |
| DD.Shift.r.o | 0.157815 | 0.095135 | 0.053400 | 0.955900 |
| DD.a.n | 0.109762 | 0.069652 | 0.008500 | 0.639100 |
| DD.e.five | 0.488598 | 0.224336 | 0.191200 | 1.300900 |
| DD.five.Shift.r | 0.477966 | 0.136998 | 0.320000 | 1.124100 |
| DD.i.e | 0.104549 | 0.069750 | 0.058300 | 0.819400 |
| DD.l.Return | 0.296623 | 0.201963 | 0.154200 | 1.551300 |
| DD.n.l | 0.171954 | 0.113955 | 0.077800 | 0.850400 |
| DD.o.a | 0.094654 | 0.030990 | 0.005100 | 0.239300 |
| DD.period.t | 0.191014 | 0.133282 | 0.092300 | 1.172400 |
| DD.t.i | 0.118821 | 0.078994 | 0.055700 | 1.040900 |
| UD.Shift.r.o | 0.047546 | 0.092278 | -0.033500 | 0.855700 |
| UD.a.n | 0.008581 | 0.072328 | -0.087300 | 0.574100 |
| UD.e.five | 0.418165 | 0.227033 | 0.118400 | 1.229100 |
| UD.five.Shift.r | 0.385871 | 0.140044 | 0.231100 | 1.038000 |
| UD.i.e | 0.045042 | 0.069281 | -0.002700 | 0.762700 |
| UD.l.Return | 0.204207 | 0.202723 | 0.081700 | 1.449200 |
| UD.n.l | 0.063429 | 0.120163 | -0.033300 | 0.756000 |
| UD.o.a | -0.005470 | 0.048102 | -0.228700 | 0.160200 |
| UD.period.t | 0.103417 | 0.132636 | 0.001300 | 1.097200 |
| UD.t.i | 0.033683 | 0.078335 | -0.042500 | 0.973400 |

### s041 (n=200 enrollment rows)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.138321 | 0.041162 | 0.013700 | 0.265100 |
| H.Shift.r | 0.161087 | 0.033406 | 0.038000 | 0.257500 |
| H.a | 0.173299 | 0.026838 | 0.107100 | 0.250800 |
| H.e | 0.146633 | 0.024815 | 0.073600 | 0.217100 |
| H.five | 0.096268 | 0.016834 | 0.035900 | 0.146100 |
| H.i | 0.116467 | 0.030630 | 0.045900 | 0.229700 |
| H.l | 0.145738 | 0.035362 | 0.029600 | 0.340700 |
| H.n | 0.123521 | 0.039916 | 0.055900 | 0.357700 |
| H.o | 0.136180 | 0.049386 | 0.024600 | 0.687200 |
| H.period | 0.173901 | 0.029391 | 0.101800 | 0.275900 |
| H.t | 0.125822 | 0.019705 | 0.043000 | 0.202600 |
| DD.Shift.r.o | 0.258763 | 0.136537 | 0.095000 | 0.835800 |
| DD.a.n | 0.165780 | 0.092866 | 0.001100 | 0.783300 |
| DD.e.five | 0.371772 | 0.264297 | 0.227900 | 3.425800 |
| DD.five.Shift.r | 0.644618 | 0.370969 | 0.388600 | 3.645000 |
| DD.i.e | 0.175556 | 0.100944 | 0.074900 | 1.022200 |
| DD.l.Return | 0.340855 | 0.105209 | 0.220400 | 1.120700 |
| DD.n.l | 0.211122 | 0.072584 | 0.111300 | 0.483700 |
| DD.o.a | 0.207686 | 0.181681 | 0.088000 | 1.727400 |
| DD.period.t | 0.289890 | 0.179205 | 0.131300 | 1.380700 |
| DD.t.i | 0.202050 | 0.105754 | 0.105300 | 1.228900 |
| UD.Shift.r.o | 0.097675 | 0.149817 | -0.086500 | 0.739200 |
| UD.a.n | -0.007519 | 0.100238 | -0.169600 | 0.640900 |
| UD.e.five | 0.225138 | 0.266127 | 0.063800 | 3.291800 |
| UD.five.Shift.r | 0.548350 | 0.372471 | 0.293100 | 3.547900 |
| UD.i.e | 0.059088 | 0.104817 | -0.058600 | 0.909800 |
| UD.l.Return | 0.195116 | 0.111579 | 0.070300 | 1.005700 |
| UD.n.l | 0.087601 | 0.065474 | -0.158300 | 0.358600 |
| UD.o.a | 0.071506 | 0.172469 | -0.110200 | 1.586800 |
| UD.period.t | 0.115989 | 0.182517 | -0.040000 | 1.234600 |
| UD.t.i | 0.076228 | 0.104920 | -0.030600 | 1.063000 |

### s044 (n=200 enrollment rows)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.069746 | 0.015626 | 0.005300 | 0.119800 |
| H.Shift.r | 0.049017 | 0.007228 | 0.007400 | 0.075200 |
| H.a | 0.083844 | 0.018540 | 0.016100 | 0.148000 |
| H.e | 0.051625 | 0.005768 | 0.034600 | 0.067300 |
| H.five | 0.047328 | 0.006327 | 0.030600 | 0.072800 |
| H.i | 0.052910 | 0.007546 | 0.016400 | 0.079100 |
| H.l | 0.087968 | 0.011844 | 0.048300 | 0.117600 |
| H.n | 0.063880 | 0.011178 | 0.015300 | 0.091000 |
| H.o | 0.066278 | 0.008600 | 0.045100 | 0.109700 |
| H.period | 0.077381 | 0.015842 | 0.044900 | 0.138000 |
| H.t | 0.057912 | 0.009011 | 0.030100 | 0.092900 |
| DD.Shift.r.o | 0.494391 | 0.349820 | 0.205100 | 2.187800 |
| DD.a.n | 0.181886 | 0.070254 | 0.080300 | 0.778100 |
| DD.e.five | 0.440444 | 0.331040 | 0.223900 | 2.595100 |
| DD.five.Shift.r | 0.521420 | 0.206603 | 0.327400 | 2.231600 |
| DD.i.e | 0.211687 | 0.129153 | 0.125600 | 1.021700 |
| DD.l.Return | 0.304434 | 0.157597 | 0.186100 | 1.581600 |
| DD.n.l | 0.229426 | 0.108302 | 0.162500 | 1.124400 |
| DD.o.a | 0.131097 | 0.115488 | 0.014600 | 1.161400 |
| DD.period.t | 0.353766 | 0.266185 | 0.182400 | 1.982000 |
| DD.t.i | 0.145731 | 0.066250 | 0.086900 | 0.764100 |
| UD.Shift.r.o | 0.445373 | 0.351371 | 0.151400 | 2.139200 |
| UD.a.n | 0.098042 | 0.071672 | 0.025600 | 0.712900 |
| UD.e.five | 0.388820 | 0.331775 | 0.175600 | 2.540500 |
| UD.five.Shift.r | 0.474091 | 0.205963 | 0.286200 | 2.182000 |
| UD.i.e | 0.158777 | 0.129558 | 0.072600 | 0.971800 |
| UD.l.Return | 0.216466 | 0.161771 | 0.083000 | 1.523300 |
| UD.n.l | 0.165545 | 0.110139 | 0.095200 | 1.109100 |
| UD.o.a | 0.064819 | 0.115624 | -0.036900 | 1.091000 |
| UD.period.t | 0.276385 | 0.269596 | 0.089600 | 1.916300 |
| UD.t.i | 0.087818 | 0.067960 | 0.024300 | 0.709700 |

### s046 (n=200 enrollment rows)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.086873 | 0.012821 | 0.040900 | 0.132200 |
| H.Shift.r | 0.115180 | 0.025320 | 0.067800 | 0.194100 |
| H.a | 0.110873 | 0.027548 | 0.008500 | 0.200200 |
| H.e | 0.086536 | 0.016920 | 0.036200 | 0.139300 |
| H.five | 0.084678 | 0.015711 | 0.046400 | 0.136400 |
| H.i | 0.072669 | 0.011986 | 0.019800 | 0.115500 |
| H.l | 0.112565 | 0.021416 | 0.039300 | 0.190200 |
| H.n | 0.105934 | 0.018237 | 0.033300 | 0.161700 |
| H.o | 0.090544 | 0.016547 | 0.055400 | 0.161700 |
| H.period | 0.115246 | 0.018871 | 0.064600 | 0.165600 |
| H.t | 0.091111 | 0.018935 | 0.050400 | 0.157500 |
| DD.Shift.r.o | 0.234471 | 0.175543 | 0.109700 | 2.100800 |
| DD.a.n | 0.175587 | 0.115089 | 0.001700 | 1.040700 |
| DD.e.five | 0.317621 | 0.167917 | 0.173800 | 1.298000 |
| DD.five.Shift.r | 0.602677 | 0.191048 | 0.392000 | 1.416800 |
| DD.i.e | 0.314121 | 0.235302 | 0.108200 | 1.436000 |
| DD.l.Return | 0.303647 | 0.117106 | 0.197800 | 0.913100 |
| DD.n.l | 0.290517 | 0.195598 | 0.141700 | 1.417700 |
| DD.o.a | 0.216619 | 0.140243 | 0.074500 | 0.953100 |
| DD.period.t | 0.481222 | 0.311007 | 0.185900 | 2.632500 |
| DD.t.i | 0.212449 | 0.162276 | 0.120900 | 1.640800 |
| UD.Shift.r.o | 0.119291 | 0.173311 | 0.017400 | 1.969900 |
| UD.a.n | 0.064714 | 0.114069 | -0.043300 | 0.918500 |
| UD.e.five | 0.231084 | 0.167335 | 0.087000 | 1.178000 |
| UD.five.Shift.r | 0.517998 | 0.185862 | 0.315000 | 1.310200 |
| UD.i.e | 0.241452 | 0.233782 | 0.053800 | 1.358400 |
| UD.l.Return | 0.191082 | 0.120202 | 0.090900 | 0.785700 |
| UD.n.l | 0.184583 | 0.194847 | 0.048800 | 1.330600 |
| UD.o.a | 0.126074 | 0.137679 | -0.010500 | 0.865300 |
| UD.period.t | 0.365976 | 0.311852 | 0.072700 | 2.501100 |
| UD.t.i | 0.121338 | 0.161819 | 0.042200 | 1.535800 |

### s048 (n=200 enrollment rows)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.098562 | 0.018942 | 0.019500 | 0.161700 |
| H.Shift.r | 0.090658 | 0.015430 | 0.037200 | 0.142700 |
| H.a | 0.098181 | 0.019374 | 0.045400 | 0.176800 |
| H.e | 0.099312 | 0.016110 | 0.062300 | 0.140600 |
| H.five | 0.082585 | 0.011797 | 0.046000 | 0.113700 |
| H.i | 0.073076 | 0.011704 | 0.034300 | 0.099200 |
| H.l | 0.089822 | 0.014005 | 0.023200 | 0.161200 |
| H.n | 0.082612 | 0.014470 | 0.040400 | 0.126900 |
| H.o | 0.084178 | 0.012823 | 0.037200 | 0.110300 |
| H.period | 0.102489 | 0.014685 | 0.050400 | 0.137700 |
| H.t | 0.088818 | 0.009412 | 0.056500 | 0.124000 |
| DD.Shift.r.o | 0.158888 | 0.059890 | 0.085200 | 0.522800 |
| DD.a.n | 0.124414 | 0.074546 | 0.044600 | 0.528500 |
| DD.e.five | 0.389003 | 0.182293 | 0.191800 | 1.636600 |
| DD.five.Shift.r | 0.442366 | 0.117082 | 0.291200 | 1.252000 |
| DD.i.e | 0.118438 | 0.054318 | 0.060400 | 0.603400 |
| DD.l.Return | 0.272502 | 0.127162 | 0.164700 | 0.866400 |
| DD.n.l | 0.163756 | 0.060365 | 0.091800 | 0.592300 |
| DD.o.a | 0.150381 | 0.039590 | 0.096400 | 0.427400 |
| DD.period.t | 0.198431 | 0.075487 | 0.114100 | 0.936400 |
| DD.t.i | 0.099448 | 0.040252 | 0.056700 | 0.519300 |
| UD.Shift.r.o | 0.068231 | 0.057278 | -0.004500 | 0.433300 |
| UD.a.n | 0.026233 | 0.076403 | -0.067200 | 0.437000 |
| UD.e.five | 0.289691 | 0.186505 | 0.072000 | 1.535600 |
| UD.five.Shift.r | 0.359782 | 0.118363 | 0.225800 | 1.157800 |
| UD.i.e | 0.045362 | 0.054580 | -0.006600 | 0.504200 |
| UD.l.Return | 0.182680 | 0.129205 | 0.084300 | 0.785700 |
| UD.n.l | 0.081144 | 0.061954 | -0.013800 | 0.551400 |
| UD.o.a | 0.066203 | 0.042376 | 0.012500 | 0.345600 |
| UD.period.t | 0.095942 | 0.075129 | 0.013900 | 0.826900 |
| UD.t.i | 0.010631 | 0.039207 | -0.026700 | 0.435700 |

### s049 (n=200 enrollment rows)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.078471 | 0.012071 | 0.042000 | 0.122100 |
| H.Shift.r | 0.126094 | 0.024688 | 0.067300 | 0.239200 |
| H.a | 0.167830 | 0.146069 | 0.088600 | 2.035300 |
| H.e | 0.129286 | 0.016129 | 0.091500 | 0.220800 |
| H.five | 0.096282 | 0.016638 | 0.057300 | 0.144300 |
| H.i | 0.077838 | 0.018299 | 0.030100 | 0.183000 |
| H.l | 0.075743 | 0.009919 | 0.053000 | 0.109200 |
| H.n | 0.074863 | 0.010038 | 0.049300 | 0.115300 |
| H.o | 0.092385 | 0.014148 | 0.054600 | 0.158500 |
| H.period | 0.080981 | 0.011844 | 0.054900 | 0.122900 |
| H.t | 0.126083 | 0.015148 | 0.047200 | 0.165900 |
| DD.Shift.r.o | 0.668494 | 0.483824 | 0.200200 | 2.660400 |
| DD.a.n | 0.366080 | 0.384738 | 0.178600 | 3.327800 |
| DD.e.five | 0.891316 | 0.748801 | 0.215500 | 4.715900 |
| DD.five.Shift.r | 1.239225 | 1.093500 | 0.457600 | 8.370200 |
| DD.i.e | 0.494633 | 1.820745 | 0.121100 | 25.987300 |
| DD.l.Return | 0.932052 | 0.484761 | 0.450300 | 4.073600 |
| DD.n.l | 0.510835 | 0.248405 | 0.261100 | 1.951400 |
| DD.o.a | 0.313816 | 0.302097 | 0.134400 | 2.108200 |
| DD.period.t | 0.657500 | 0.503783 | 0.259500 | 3.576900 |
| DD.t.i | 0.294603 | 0.362199 | 0.148300 | 4.919700 |
| UD.Shift.r.o | 0.542400 | 0.489992 | 0.068600 | 2.577800 |
| UD.a.n | 0.198249 | 0.301689 | 0.035900 | 2.524200 |
| UD.e.five | 0.762031 | 0.746278 | 0.088600 | 4.575000 |
| UD.five.Shift.r | 1.142944 | 1.095500 | 0.380600 | 8.290800 |
| UD.i.e | 0.416795 | 1.820606 | 0.042500 | 25.915800 |
| UD.l.Return | 0.856310 | 0.484992 | 0.368800 | 3.981800 |
| UD.n.l | 0.435973 | 0.244368 | 0.182500 | 1.848300 |
| UD.o.a | 0.221432 | 0.301848 | 0.044700 | 2.011400 |
| UD.period.t | 0.576518 | 0.505256 | 0.168200 | 3.515900 |
| UD.t.i | 0.168521 | 0.361167 | 0.032500 | 4.799900 |

### s057 (n=200 enrollment rows)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.121268 | 0.020782 | 0.050700 | 0.205500 |
| H.Shift.r | 0.069537 | 0.008457 | 0.040700 | 0.095000 |
| H.a | 0.084319 | 0.018373 | 0.008700 | 0.121100 |
| H.e | 0.064409 | 0.017017 | 0.021900 | 0.140300 |
| H.five | 0.078453 | 0.012859 | 0.031200 | 0.114500 |
| H.i | 0.079158 | 0.012026 | 0.023500 | 0.112600 |
| H.l | 0.104673 | 0.021387 | 0.035600 | 0.170900 |
| H.n | 0.070110 | 0.013595 | 0.023800 | 0.103700 |
| H.o | 0.084951 | 0.015953 | 0.040100 | 0.143800 |
| H.period | 0.098937 | 0.018452 | 0.025100 | 0.151700 |
| H.t | 0.077244 | 0.012359 | 0.014000 | 0.103700 |
| DD.Shift.r.o | 0.191383 | 0.078755 | 0.116300 | 0.620300 |
| DD.a.n | 0.113336 | 0.110770 | 0.021200 | 1.465200 |
| DD.e.five | 0.224896 | 0.115900 | 0.088600 | 0.875900 |
| DD.five.Shift.r | 0.362573 | 0.119602 | 0.220600 | 1.303400 |
| DD.i.e | 0.105603 | 0.083173 | 0.025100 | 0.718100 |
| DD.l.Return | 0.295120 | 0.106949 | 0.199800 | 0.935600 |
| DD.n.l | 0.200060 | 0.122763 | 0.040100 | 0.864800 |
| DD.o.a | 0.146073 | 0.053873 | 0.068500 | 0.446300 |
| DD.period.t | 0.166855 | 0.073731 | 0.075900 | 0.485900 |
| DD.t.i | 0.115730 | 0.051290 | 0.003500 | 0.507700 |
| UD.Shift.r.o | 0.121846 | 0.078336 | 0.050900 | 0.555400 |
| UD.a.n | 0.029017 | 0.109763 | -0.049100 | 1.350500 |
| UD.e.five | 0.160488 | 0.116582 | -0.018000 | 0.790400 |
| UD.five.Shift.r | 0.284120 | 0.118659 | 0.178100 | 1.222100 |
| UD.i.e | 0.026444 | 0.085234 | -0.049300 | 0.647700 |
| UD.l.Return | 0.190447 | 0.110653 | 0.093500 | 0.839000 |
| UD.n.l | 0.129949 | 0.122461 | -0.013800 | 0.763200 |
| UD.o.a | 0.061121 | 0.051937 | 0.001200 | 0.369800 |
| UD.period.t | 0.067919 | 0.072240 | -0.026200 | 0.394400 |
| UD.t.i | 0.038486 | 0.055173 | -0.050200 | 0.430700 |


---

## Appendix J: Per-Column Per-Subject Enrollment Statistics (Background Subjects)

Mean and standard deviation per individual feature column, for each background subject's 200 enrollment rows. These subjects contributed to encoder training.

### s002 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.079532 | 0.013816 | 0.031800 | 0.130100 |
| H.Shift.r | 0.109533 | 0.013769 | 0.071800 | 0.168100 |
| H.a | 0.116713 | 0.016294 | 0.079300 | 0.168200 |
| H.e | 0.087312 | 0.011520 | 0.061500 | 0.141700 |
| H.five | 0.084317 | 0.011962 | 0.059100 | 0.137300 |
| H.i | 0.074324 | 0.010366 | 0.054400 | 0.118300 |
| H.l | 0.094150 | 0.011782 | 0.071500 | 0.138600 |
| H.n | 0.081241 | 0.014164 | 0.041500 | 0.131400 |
| H.o | 0.090016 | 0.014939 | 0.024600 | 0.136500 |
| H.period | 0.105977 | 0.014132 | 0.058100 | 0.154100 |
| H.t | 0.084762 | 0.009978 | 0.065200 | 0.155200 |
| DD.Shift.r.o | 0.243458 | 0.127890 | 0.108600 | 0.811800 |
| DD.a.n | 0.133470 | 0.056744 | 0.011700 | 0.491400 |
| DD.e.five | 0.617483 | 0.232731 | 0.224100 | 1.441000 |
| DD.five.Shift.r | 0.509844 | 0.224372 | 0.303300 | 1.736000 |
| DD.i.e | 0.139634 | 0.073893 | 0.042300 | 1.016200 |
| DD.l.Return | 0.266533 | 0.113126 | 0.177800 | 1.266300 |
| DD.n.l | 0.215335 | 0.073408 | 0.146300 | 0.709900 |
| DD.o.a | 0.164851 | 0.052936 | 0.088300 | 0.451600 |
| DD.period.t | 0.183529 | 0.083852 | 0.066400 | 0.644100 |
| DD.t.i | 0.156147 | 0.034738 | 0.082900 | 0.296600 |
| UD.Shift.r.o | 0.133925 | 0.122412 | 0.010100 | 0.708600 |
| UD.a.n | 0.016758 | 0.054369 | -0.118000 | 0.408800 |
| UD.e.five | 0.530170 | 0.235554 | 0.134300 | 1.353200 |
| UD.five.Shift.r | 0.425528 | 0.223736 | 0.225200 | 1.671300 |
| UD.i.e | 0.065309 | 0.072719 | -0.027200 | 0.947600 |
| UD.l.Return | 0.172383 | 0.111172 | 0.098300 | 1.168100 |
| UD.n.l | 0.134095 | 0.067967 | 0.073900 | 0.623000 |
| UD.o.a | 0.074835 | 0.054960 | 0.005100 | 0.385900 |
| UD.period.t | 0.077552 | 0.082810 | -0.026000 | 0.552000 |
| UD.t.i | 0.071385 | 0.031680 | 0.009800 | 0.211100 |

### s003 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.092698 | 0.017740 | 0.018200 | 0.157800 |
| H.Shift.r | 0.121899 | 0.014463 | 0.078500 | 0.179200 |
| H.a | 0.134731 | 0.019181 | 0.071900 | 0.224200 |
| H.e | 0.115022 | 0.027518 | 0.053600 | 0.211700 |
| H.five | 0.099011 | 0.019143 | 0.031400 | 0.151000 |
| H.i | 0.115686 | 0.017038 | 0.048600 | 0.161800 |
| H.l | 0.098290 | 0.012451 | 0.039400 | 0.130400 |
| H.n | 0.104638 | 0.017998 | 0.046500 | 0.158400 |
| H.o | 0.122639 | 0.015284 | 0.080900 | 0.164700 |
| H.period | 0.138564 | 0.029058 | 0.037200 | 0.246500 |
| H.t | 0.125244 | 0.019751 | 0.033300 | 0.171700 |
| DD.Shift.r.o | 0.203398 | 0.055797 | 0.136600 | 0.664900 |
| DD.a.n | 0.137185 | 0.054352 | 0.030400 | 0.457400 |
| DD.e.five | 0.337261 | 0.204703 | 0.006300 | 1.268500 |
| DD.five.Shift.r | 0.414638 | 0.164532 | 0.263700 | 1.288100 |
| DD.i.e | 0.176793 | 0.124437 | 0.049400 | 0.929000 |
| DD.l.Return | 0.270794 | 0.141495 | 0.175500 | 1.498800 |
| DD.n.l | 0.152313 | 0.033882 | 0.079900 | 0.381100 |
| DD.o.a | 0.139250 | 0.053257 | 0.026500 | 0.418100 |
| DD.period.t | 0.194672 | 0.130958 | 0.027600 | 1.009100 |
| DD.t.i | 0.167867 | 0.081691 | 0.001100 | 0.702300 |
| UD.Shift.r.o | 0.081499 | 0.059123 | -0.001400 | 0.552700 |
| UD.a.n | 0.002454 | 0.055394 | -0.127200 | 0.367700 |
| UD.e.five | 0.222239 | 0.206426 | -0.065800 | 1.163400 |
| UD.five.Shift.r | 0.315627 | 0.164430 | 0.182200 | 1.184100 |
| UD.i.e | 0.061108 | 0.124091 | -0.078900 | 0.799100 |
| UD.l.Return | 0.172504 | 0.142346 | 0.097500 | 1.418800 |
| UD.n.l | 0.047675 | 0.033569 | -0.020100 | 0.272100 |
| UD.o.a | 0.016611 | 0.051632 | -0.093400 | 0.318600 |
| UD.period.t | 0.056109 | 0.139665 | -0.077200 | 0.869200 |
| UD.t.i | 0.042623 | 0.089493 | -0.119200 | 0.603800 |

### s005 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.093652 | 0.015349 | 0.050400 | 0.142000 |
| H.Shift.r | 0.131004 | 0.018064 | 0.066000 | 0.181100 |
| H.a | 0.133027 | 0.017934 | 0.048900 | 0.184000 |
| H.e | 0.115201 | 0.017890 | 0.051200 | 0.168700 |
| H.five | 0.108537 | 0.018515 | 0.038300 | 0.187400 |
| H.i | 0.088432 | 0.012176 | 0.050400 | 0.130100 |
| H.l | 0.085662 | 0.013261 | 0.048900 | 0.157600 |
| H.n | 0.081254 | 0.015743 | 0.011600 | 0.124900 |
| H.o | 0.094499 | 0.011481 | 0.064400 | 0.130100 |
| H.period | 0.081229 | 0.014596 | 0.041500 | 0.141200 |
| H.t | 0.114087 | 0.015050 | 0.059700 | 0.153300 |
| DD.Shift.r.o | 0.376928 | 0.095061 | 0.257200 | 0.923700 |
| DD.a.n | 0.211656 | 0.084573 | 0.124400 | 1.023200 |
| DD.e.five | 0.380408 | 0.173330 | 0.215100 | 1.165600 |
| DD.five.Shift.r | 0.584973 | 0.229038 | 0.381900 | 2.731400 |
| DD.i.e | 0.206574 | 0.084535 | 0.132800 | 0.981200 |
| DD.l.Return | 0.431221 | 0.118236 | 0.299200 | 1.316000 |
| DD.n.l | 0.265536 | 0.107357 | 0.192600 | 1.070800 |
| DD.o.a | 0.197710 | 0.097195 | 0.043900 | 1.372300 |
| DD.period.t | 0.306867 | 0.140928 | 0.193900 | 1.267100 |
| DD.t.i | 0.240661 | 0.045273 | 0.161800 | 0.431000 |
| UD.Shift.r.o | 0.245924 | 0.095633 | 0.098200 | 0.817800 |
| UD.a.n | 0.078629 | 0.083789 | -0.006600 | 0.857400 |
| UD.e.five | 0.265207 | 0.174354 | 0.100000 | 1.024100 |
| UD.five.Shift.r | 0.476435 | 0.225437 | 0.278700 | 2.610800 |
| UD.i.e | 0.118142 | 0.085825 | 0.039900 | 0.909400 |
| UD.l.Return | 0.345558 | 0.118153 | 0.233700 | 1.233900 |
| UD.n.l | 0.184282 | 0.111894 | 0.099200 | 1.022500 |
| UD.o.a | 0.103212 | 0.096820 | -0.039600 | 1.272300 |
| UD.period.t | 0.225638 | 0.142684 | 0.101300 | 1.204000 |
| UD.t.i | 0.126574 | 0.044124 | 0.050400 | 0.317700 |

### s007 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.079779 | 0.016288 | 0.024600 | 0.119600 |
| H.Shift.r | 0.081299 | 0.014159 | 0.039900 | 0.119800 |
| H.a | 0.093300 | 0.012873 | 0.056500 | 0.129100 |
| H.e | 0.086893 | 0.012902 | 0.017700 | 0.129300 |
| H.five | 0.066869 | 0.009165 | 0.029700 | 0.094800 |
| H.i | 0.073798 | 0.014738 | 0.040400 | 0.126400 |
| H.l | 0.091675 | 0.012217 | 0.056500 | 0.153900 |
| H.n | 0.118067 | 0.017262 | 0.073100 | 0.159200 |
| H.o | 0.084223 | 0.015338 | 0.011900 | 0.164200 |
| H.period | 0.092274 | 0.015709 | 0.047800 | 0.137000 |
| H.t | 0.067118 | 0.009096 | 0.037000 | 0.097400 |
| DD.Shift.r.o | 0.180616 | 0.060492 | 0.086600 | 0.623600 |
| DD.a.n | 0.123698 | 0.060937 | 0.061300 | 0.644800 |
| DD.e.five | 0.302062 | 0.159197 | 0.158900 | 1.046400 |
| DD.five.Shift.r | 0.339273 | 0.097335 | 0.229600 | 0.811100 |
| DD.i.e | 0.144520 | 0.122288 | 0.032500 | 1.002300 |
| DD.l.Return | 0.249874 | 0.125868 | 0.170700 | 0.946000 |
| DD.n.l | 0.122616 | 0.040725 | 0.060200 | 0.394200 |
| DD.o.a | 0.138924 | 0.032741 | 0.079600 | 0.378000 |
| DD.period.t | 0.208244 | 0.090999 | 0.096000 | 0.802700 |
| DD.t.i | 0.113829 | 0.028224 | 0.070000 | 0.295900 |
| UD.Shift.r.o | 0.099317 | 0.065710 | 0.010100 | 0.583700 |
| UD.a.n | 0.030398 | 0.062284 | -0.033500 | 0.554500 |
| UD.e.five | 0.215168 | 0.157645 | 0.062500 | 0.954500 |
| UD.five.Shift.r | 0.272403 | 0.097970 | 0.166800 | 0.753500 |
| UD.i.e | 0.070722 | 0.121655 | -0.030100 | 0.942900 |
| UD.l.Return | 0.158199 | 0.126691 | 0.074300 | 0.868600 |
| UD.n.l | 0.004549 | 0.044944 | -0.038600 | 0.297100 |
| UD.o.a | 0.054702 | 0.031207 | -0.015000 | 0.251100 |
| UD.period.t | 0.115971 | 0.088783 | 0.005700 | 0.730900 |
| UD.t.i | 0.046711 | 0.028378 | -0.003200 | 0.228800 |

### s008 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.104180 | 0.020307 | 0.054700 | 0.157000 |
| H.Shift.r | 0.096341 | 0.012111 | 0.034900 | 0.132600 |
| H.a | 0.117248 | 0.012863 | 0.077900 | 0.156400 |
| H.e | 0.092603 | 0.014641 | 0.044900 | 0.131700 |
| H.five | 0.092740 | 0.016007 | 0.028800 | 0.153600 |
| H.i | 0.078911 | 0.012680 | 0.023500 | 0.109400 |
| H.l | 0.088469 | 0.018917 | 0.009800 | 0.137500 |
| H.n | 0.094282 | 0.018432 | 0.051000 | 0.159700 |
| H.o | 0.091757 | 0.012963 | 0.034100 | 0.140200 |
| H.period | 0.096176 | 0.020847 | 0.046700 | 0.160200 |
| H.t | 0.084572 | 0.015737 | 0.041200 | 0.129900 |
| DD.Shift.r.o | 0.135321 | 0.074019 | 0.049400 | 0.632700 |
| DD.a.n | 0.119082 | 0.032245 | 0.023300 | 0.330500 |
| DD.e.five | 0.365370 | 0.176507 | 0.178900 | 1.728000 |
| DD.five.Shift.r | 0.353949 | 0.075552 | 0.267600 | 0.925100 |
| DD.i.e | 0.128437 | 0.062398 | 0.045400 | 0.697500 |
| DD.l.Return | 0.257006 | 0.099282 | 0.159300 | 0.783900 |
| DD.n.l | 0.161752 | 0.114986 | 0.039100 | 0.735500 |
| DD.o.a | 0.103376 | 0.031512 | 0.011400 | 0.388300 |
| DD.period.t | 0.246264 | 0.099927 | 0.122900 | 0.761800 |
| DD.t.i | 0.115094 | 0.035763 | 0.061100 | 0.361300 |
| UD.Shift.r.o | 0.038979 | 0.071430 | -0.039600 | 0.514400 |
| UD.a.n | 0.001834 | 0.033415 | -0.109200 | 0.244200 |
| UD.e.five | 0.272766 | 0.175403 | 0.103400 | 1.647000 |
| UD.five.Shift.r | 0.261209 | 0.074047 | 0.176300 | 0.831700 |
| UD.i.e | 0.049526 | 0.060790 | -0.026700 | 0.625200 |
| UD.l.Return | 0.168537 | 0.102968 | 0.061900 | 0.697300 |
| UD.n.l | 0.067471 | 0.124734 | -0.059700 | 0.665800 |
| UD.o.a | 0.011618 | 0.029574 | -0.061700 | 0.270600 |
| UD.period.t | 0.150088 | 0.097718 | 0.032900 | 0.662300 |
| UD.t.i | 0.030522 | 0.037182 | -0.019300 | 0.288700 |

### s010 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.069602 | 0.012538 | 0.028800 | 0.097400 |
| H.Shift.r | 0.061650 | 0.009201 | 0.039900 | 0.092200 |
| H.a | 0.062012 | 0.007888 | 0.041700 | 0.101400 |
| H.e | 0.065885 | 0.006830 | 0.048600 | 0.088400 |
| H.five | 0.060022 | 0.012592 | 0.037500 | 0.115900 |
| H.i | 0.079622 | 0.010721 | 0.037500 | 0.124300 |
| H.l | 0.077446 | 0.009780 | 0.049100 | 0.099800 |
| H.n | 0.078731 | 0.009911 | 0.018800 | 0.105600 |
| H.o | 0.085733 | 0.007132 | 0.063900 | 0.104300 |
| H.period | 0.091186 | 0.011657 | 0.058400 | 0.121200 |
| H.t | 0.065371 | 0.008562 | 0.044600 | 0.106400 |
| DD.Shift.r.o | 0.151397 | 0.070854 | 0.088200 | 0.844800 |
| DD.a.n | 0.127745 | 0.070593 | 0.073400 | 0.733400 |
| DD.e.five | 0.256221 | 0.100447 | 0.176300 | 0.790100 |
| DD.five.Shift.r | 0.260175 | 0.106081 | 0.175900 | 1.117600 |
| DD.i.e | 0.129148 | 0.028176 | 0.073200 | 0.282900 |
| DD.l.Return | 0.251186 | 0.101364 | 0.125100 | 0.812600 |
| DD.n.l | 0.232855 | 0.068271 | 0.184200 | 0.966600 |
| DD.o.a | 0.123570 | 0.058855 | 0.063200 | 0.632900 |
| DD.period.t | 0.160564 | 0.069801 | 0.074900 | 0.620000 |
| DD.t.i | 0.170468 | 0.082386 | 0.093700 | 0.645800 |
| UD.Shift.r.o | 0.089748 | 0.070231 | 0.019800 | 0.794300 |
| UD.a.n | 0.065734 | 0.069534 | 0.014800 | 0.669000 |
| UD.e.five | 0.190336 | 0.101816 | 0.114500 | 0.735200 |
| UD.five.Shift.r | 0.200152 | 0.100191 | 0.112500 | 1.018100 |
| UD.i.e | 0.049525 | 0.027905 | -0.002900 | 0.211100 |
| UD.l.Return | 0.173740 | 0.100298 | 0.044000 | 0.731600 |
| UD.n.l | 0.154125 | 0.068198 | 0.111100 | 0.885300 |
| UD.o.a | 0.037838 | 0.059045 | -0.015000 | 0.550000 |
| UD.period.t | 0.069377 | 0.068053 | -0.007000 | 0.526600 |
| UD.t.i | 0.105097 | 0.081478 | 0.029600 | 0.581600 |

### s012 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.140504 | 0.024691 | 0.077600 | 0.242800 |
| H.Shift.r | 0.136255 | 0.021533 | 0.074700 | 0.202500 |
| H.a | 0.156007 | 0.033990 | 0.082400 | 0.244200 |
| H.e | 0.127230 | 0.022385 | 0.073100 | 0.210300 |
| H.five | 0.107050 | 0.022480 | 0.036700 | 0.182100 |
| H.i | 0.128292 | 0.019537 | 0.032500 | 0.207700 |
| H.l | 0.149554 | 0.020923 | 0.091600 | 0.214300 |
| H.n | 0.139967 | 0.029175 | 0.045200 | 0.229100 |
| H.o | 0.118489 | 0.017718 | 0.022200 | 0.173400 |
| H.period | 0.140487 | 0.028395 | 0.068900 | 0.297400 |
| H.t | 0.123444 | 0.017848 | 0.016700 | 0.174700 |
| DD.Shift.r.o | 0.200822 | 0.071400 | 0.104300 | 0.597800 |
| DD.a.n | 0.142762 | 0.063314 | 0.013000 | 0.574700 |
| DD.e.five | 0.419188 | 0.210538 | 0.200300 | 1.372300 |
| DD.five.Shift.r | 0.384291 | 0.123314 | 0.254700 | 0.996000 |
| DD.i.e | 0.151753 | 0.130774 | 0.002400 | 1.152100 |
| DD.l.Return | 0.321789 | 0.108632 | 0.213400 | 1.193200 |
| DD.n.l | 0.147739 | 0.083132 | 0.058800 | 0.503000 |
| DD.o.a | 0.155835 | 0.066673 | 0.040500 | 0.549000 |
| DD.period.t | 0.203438 | 0.163173 | 0.068500 | 1.604900 |
| DD.t.i | 0.155243 | 0.076238 | 0.021000 | 0.846000 |
| UD.Shift.r.o | 0.064566 | 0.067845 | -0.039900 | 0.458100 |
| UD.a.n | -0.013245 | 0.072733 | -0.194500 | 0.423900 |
| UD.e.five | 0.291957 | 0.205242 | 0.082600 | 1.253200 |
| UD.five.Shift.r | 0.277242 | 0.115164 | 0.181400 | 0.855100 |
| UD.i.e | 0.023461 | 0.131190 | -0.157800 | 1.035200 |
| UD.l.Return | 0.172235 | 0.108227 | 0.084100 | 1.087900 |
| UD.n.l | 0.007771 | 0.094694 | -0.094800 | 0.426200 |
| UD.o.a | 0.037347 | 0.061577 | -0.048500 | 0.395900 |
| UD.period.t | 0.062951 | 0.152857 | -0.056900 | 1.393200 |
| UD.t.i | 0.031799 | 0.076860 | -0.035100 | 0.708000 |

### s013 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.086418 | 0.016628 | 0.048600 | 0.124100 |
| H.Shift.r | 0.082173 | 0.009470 | 0.055700 | 0.104100 |
| H.a | 0.093430 | 0.013022 | 0.029100 | 0.127300 |
| H.e | 0.058435 | 0.011091 | 0.014000 | 0.099300 |
| H.five | 0.068895 | 0.011835 | 0.027700 | 0.100600 |
| H.i | 0.082855 | 0.008660 | 0.041200 | 0.111500 |
| H.l | 0.071558 | 0.009784 | 0.047300 | 0.097400 |
| H.n | 0.056966 | 0.010874 | 0.027200 | 0.095600 |
| H.o | 0.082978 | 0.008755 | 0.061500 | 0.104800 |
| H.period | 0.077242 | 0.011946 | 0.050200 | 0.115400 |
| H.t | 0.067155 | 0.008065 | 0.024600 | 0.090000 |
| DD.Shift.r.o | 0.143232 | 0.052945 | 0.093300 | 0.519400 |
| DD.a.n | 0.099365 | 0.026693 | 0.031000 | 0.283200 |
| DD.e.five | 0.282743 | 0.206976 | 0.134700 | 1.419200 |
| DD.five.Shift.r | 0.327952 | 0.099673 | 0.221000 | 0.854000 |
| DD.i.e | 0.090224 | 0.072120 | 0.029600 | 0.532600 |
| DD.l.Return | 0.250518 | 0.093269 | 0.143400 | 0.754600 |
| DD.n.l | 0.144561 | 0.046765 | 0.103900 | 0.534700 |
| DD.o.a | 0.106365 | 0.068792 | 0.052600 | 0.606000 |
| DD.period.t | 0.143205 | 0.082031 | 0.064100 | 0.531400 |
| DD.t.i | 0.110946 | 0.031224 | 0.061300 | 0.376400 |
| UD.Shift.r.o | 0.061059 | 0.052429 | 0.002100 | 0.434900 |
| UD.a.n | 0.005934 | 0.031868 | -0.060200 | 0.201700 |
| UD.e.five | 0.224308 | 0.206028 | 0.072000 | 1.359200 |
| UD.five.Shift.r | 0.259057 | 0.098448 | 0.148100 | 0.781400 |
| UD.i.e | 0.007369 | 0.073512 | -0.049600 | 0.456800 |
| UD.l.Return | 0.178959 | 0.093711 | 0.074300 | 0.683600 |
| UD.n.l | 0.087595 | 0.045228 | 0.052300 | 0.480000 |
| UD.o.a | 0.023387 | 0.068933 | -0.034800 | 0.517000 |
| UD.period.t | 0.065962 | 0.079203 | -0.018300 | 0.442700 |
| UD.t.i | 0.043792 | 0.032392 | -0.002400 | 0.307200 |

### s016 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.090994 | 0.018730 | 0.035700 | 0.160200 |
| H.Shift.r | 0.102067 | 0.017325 | 0.040400 | 0.146800 |
| H.a | 0.112075 | 0.016588 | 0.036700 | 0.156300 |
| H.e | 0.086645 | 0.014390 | 0.048300 | 0.148900 |
| H.five | 0.067966 | 0.010716 | 0.033000 | 0.109800 |
| H.i | 0.082473 | 0.014661 | 0.035100 | 0.141500 |
| H.l | 0.094511 | 0.016989 | 0.061300 | 0.141500 |
| H.n | 0.099214 | 0.013529 | 0.060500 | 0.133600 |
| H.o | 0.095662 | 0.018684 | 0.047500 | 0.166000 |
| H.period | 0.103235 | 0.021927 | 0.060200 | 0.171600 |
| H.t | 0.064930 | 0.010151 | 0.027500 | 0.095000 |
| DD.Shift.r.o | 0.337460 | 0.111703 | 0.163800 | 0.831600 |
| DD.a.n | 0.359212 | 0.168132 | 0.199600 | 2.039400 |
| DD.e.five | 0.728447 | 0.179927 | 0.328600 | 1.708400 |
| DD.five.Shift.r | 0.951965 | 0.246009 | 0.602900 | 2.714900 |
| DD.i.e | 0.406247 | 0.328225 | 0.126000 | 2.654700 |
| DD.l.Return | 0.431515 | 0.237866 | 0.260300 | 2.304200 |
| DD.n.l | 0.336681 | 0.096353 | 0.194700 | 1.385400 |
| DD.o.a | 0.391174 | 0.154234 | 0.169800 | 1.062500 |
| DD.period.t | 0.477581 | 0.207031 | 0.230800 | 1.675900 |
| DD.t.i | 0.265943 | 0.090933 | 0.122300 | 0.953500 |
| UD.Shift.r.o | 0.235393 | 0.112604 | 0.065000 | 0.728900 |
| UD.a.n | 0.247138 | 0.172724 | 0.088500 | 1.966800 |
| UD.e.five | 0.641802 | 0.182638 | 0.239900 | 1.637100 |
| UD.five.Shift.r | 0.883999 | 0.244016 | 0.535800 | 2.627000 |
| UD.i.e | 0.323774 | 0.329463 | 0.042500 | 2.585500 |
| UD.l.Return | 0.337004 | 0.236766 | 0.165600 | 2.188300 |
| UD.n.l | 0.237467 | 0.095318 | 0.084400 | 1.281900 |
| UD.o.a | 0.295511 | 0.158051 | 0.063700 | 0.972500 |
| UD.period.t | 0.374347 | 0.211614 | 0.118400 | 1.541500 |
| UD.t.i | 0.201012 | 0.089769 | 0.082600 | 0.890100 |

### s017 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.060396 | 0.007166 | 0.042000 | 0.082900 |
| H.Shift.r | 0.055899 | 0.007593 | 0.038000 | 0.101100 |
| H.a | 0.068337 | 0.011096 | 0.013800 | 0.090500 |
| H.e | 0.061024 | 0.005159 | 0.043600 | 0.074700 |
| H.five | 0.054951 | 0.007264 | 0.008500 | 0.076000 |
| H.i | 0.045930 | 0.007780 | 0.009000 | 0.058900 |
| H.l | 0.066736 | 0.006033 | 0.044400 | 0.087400 |
| H.n | 0.053574 | 0.009581 | 0.003700 | 0.077100 |
| H.o | 0.064033 | 0.006593 | 0.044100 | 0.085300 |
| H.period | 0.067911 | 0.008981 | 0.044400 | 0.097100 |
| H.t | 0.050944 | 0.007119 | 0.011900 | 0.076600 |
| DD.Shift.r.o | 0.194625 | 0.043544 | 0.141800 | 0.427100 |
| DD.a.n | 0.083298 | 0.029297 | 0.004300 | 0.327500 |
| DD.e.five | 0.328359 | 0.201666 | 0.175000 | 1.357200 |
| DD.five.Shift.r | 0.390056 | 0.073927 | 0.303800 | 0.959300 |
| DD.i.e | 0.103689 | 0.047893 | 0.054700 | 0.480000 |
| DD.l.Return | 0.212198 | 0.080965 | 0.156600 | 0.606100 |
| DD.n.l | 0.171988 | 0.036618 | 0.133300 | 0.388200 |
| DD.o.a | 0.123335 | 0.023615 | 0.071900 | 0.216000 |
| DD.period.t | 0.214254 | 0.103449 | 0.080200 | 0.841200 |
| DD.t.i | 0.110863 | 0.031238 | 0.037200 | 0.408100 |
| UD.Shift.r.o | 0.138726 | 0.041076 | 0.092100 | 0.381400 |
| UD.a.n | 0.014960 | 0.028641 | -0.077100 | 0.264700 |
| UD.e.five | 0.267336 | 0.202263 | 0.111400 | 1.301200 |
| UD.five.Shift.r | 0.335106 | 0.073586 | 0.247600 | 0.897800 |
| UD.i.e | 0.057759 | 0.051236 | -0.003500 | 0.461000 |
| UD.l.Return | 0.145462 | 0.081660 | 0.093800 | 0.541400 |
| UD.n.l | 0.118414 | 0.036641 | 0.078400 | 0.326200 |
| UD.o.a | 0.059302 | 0.022572 | 0.005400 | 0.147900 |
| UD.period.t | 0.146342 | 0.103976 | 0.006300 | 0.775700 |
| UD.t.i | 0.059918 | 0.030778 | -0.020600 | 0.355000 |

### s018 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.074103 | 0.017429 | 0.011100 | 0.172300 |
| H.Shift.r | 0.115247 | 0.020982 | 0.049900 | 0.175300 |
| H.a | 0.110997 | 0.023233 | 0.038000 | 0.181800 |
| H.e | 0.082194 | 0.019456 | 0.018200 | 0.145700 |
| H.five | 0.105225 | 0.022663 | 0.025600 | 0.176600 |
| H.i | 0.071031 | 0.016696 | 0.022700 | 0.152300 |
| H.l | 0.117410 | 0.020409 | 0.051000 | 0.191400 |
| H.n | 0.117471 | 0.028124 | 0.041200 | 0.234900 |
| H.o | 0.083947 | 0.017028 | 0.034900 | 0.138600 |
| H.period | 0.089891 | 0.016225 | 0.043600 | 0.165500 |
| H.t | 0.092069 | 0.013995 | 0.039600 | 0.138300 |
| DD.Shift.r.o | 0.258698 | 0.113377 | 0.132000 | 0.802600 |
| DD.a.n | 0.143355 | 0.079484 | 0.044900 | 0.731300 |
| DD.e.five | 0.381804 | 0.217028 | 0.202200 | 1.495500 |
| DD.five.Shift.r | 0.442747 | 0.213803 | 0.274300 | 2.911300 |
| DD.i.e | 0.129526 | 0.149862 | 0.025100 | 1.292000 |
| DD.l.Return | 0.315611 | 0.092662 | 0.188100 | 0.742600 |
| DD.n.l | 0.099395 | 0.125918 | 0.007400 | 0.776400 |
| DD.o.a | 0.185066 | 0.103321 | 0.066600 | 0.710200 |
| DD.period.t | 0.171794 | 0.066269 | 0.104700 | 0.625400 |
| DD.t.i | 0.149049 | 0.085752 | 0.078700 | 1.069600 |
| UD.Shift.r.o | 0.143452 | 0.116826 | 0.021400 | 0.716500 |
| UD.a.n | 0.032357 | 0.079239 | -0.053300 | 0.626800 |
| UD.e.five | 0.299610 | 0.221356 | 0.123000 | 1.430500 |
| UD.five.Shift.r | 0.337523 | 0.217585 | 0.159000 | 2.836300 |
| UD.i.e | 0.058494 | 0.143908 | -0.039100 | 1.139700 |
| UD.l.Return | 0.198202 | 0.091499 | 0.074100 | 0.618200 |
| UD.n.l | -0.018076 | 0.133752 | -0.126800 | 0.653100 |
| UD.o.a | 0.101119 | 0.105832 | -0.014700 | 0.647900 |
| UD.period.t | 0.081904 | 0.061885 | 0.013900 | 0.459900 |
| UD.t.i | 0.056981 | 0.083581 | -0.034900 | 0.978000 |

### s020 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.108292 | 0.017191 | 0.017300 | 0.138400 |
| H.Shift.r | 0.101489 | 0.016058 | 0.046800 | 0.158900 |
| H.a | 0.136651 | 0.020521 | 0.052000 | 0.254900 |
| H.e | 0.097522 | 0.017529 | 0.052500 | 0.147000 |
| H.five | 0.074464 | 0.017086 | 0.005000 | 0.121700 |
| H.i | 0.117719 | 0.014999 | 0.060500 | 0.161000 |
| H.l | 0.132992 | 0.021582 | 0.078700 | 0.192500 |
| H.n | 0.117840 | 0.024286 | 0.061300 | 0.199000 |
| H.o | 0.102802 | 0.020284 | 0.057300 | 0.168400 |
| H.period | 0.114915 | 0.017571 | 0.039400 | 0.149700 |
| H.t | 0.093960 | 0.014067 | 0.040900 | 0.132000 |
| DD.Shift.r.o | 0.187879 | 0.147601 | 0.052100 | 0.982000 |
| DD.a.n | 0.204833 | 0.233320 | 0.020100 | 1.750000 |
| DD.e.five | 0.370808 | 0.239104 | 0.169200 | 1.662000 |
| DD.five.Shift.r | 0.451299 | 0.207271 | 0.246900 | 1.989200 |
| DD.i.e | 0.171211 | 0.160120 | 0.052300 | 1.017500 |
| DD.l.Return | 0.265323 | 0.186375 | 0.126600 | 1.400800 |
| DD.n.l | 0.163386 | 0.165502 | 0.020300 | 1.236600 |
| DD.o.a | 0.184052 | 0.139314 | 0.014900 | 0.953800 |
| DD.period.t | 0.239716 | 0.182123 | 0.069900 | 1.995400 |
| DD.t.i | 0.161587 | 0.102359 | 0.070300 | 0.608300 |
| UD.Shift.r.o | 0.086391 | 0.144520 | -0.039400 | 0.855800 |
| UD.a.n | 0.068182 | 0.225118 | -0.100300 | 1.596800 |
| UD.e.five | 0.273286 | 0.235176 | 0.084200 | 1.571700 |
| UD.five.Shift.r | 0.376835 | 0.211067 | 0.172500 | 1.917700 |
| UD.i.e | 0.053492 | 0.155788 | -0.065700 | 0.886100 |
| UD.l.Return | 0.132331 | 0.192761 | -0.022900 | 1.297100 |
| UD.n.l | 0.045546 | 0.156929 | -0.098200 | 1.138100 |
| UD.o.a | 0.081250 | 0.128759 | -0.085700 | 0.785400 |
| UD.period.t | 0.124802 | 0.184462 | -0.040000 | 1.894300 |
| UD.t.i | 0.067626 | 0.098872 | -0.022700 | 0.513800 |

### s021 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.087154 | 0.017451 | 0.033500 | 0.164400 |
| H.Shift.r | 0.079631 | 0.015758 | 0.040100 | 0.116500 |
| H.a | 0.062123 | 0.014863 | 0.025400 | 0.138800 |
| H.e | 0.073599 | 0.013189 | 0.030900 | 0.121400 |
| H.five | 0.072881 | 0.017059 | 0.023800 | 0.127200 |
| H.i | 0.095384 | 0.014905 | 0.050200 | 0.131200 |
| H.l | 0.097998 | 0.013271 | 0.053600 | 0.149900 |
| H.n | 0.077276 | 0.015310 | 0.021100 | 0.112200 |
| H.o | 0.080824 | 0.013794 | 0.043000 | 0.127200 |
| H.period | 0.100213 | 0.016472 | 0.065500 | 0.153300 |
| H.t | 0.080420 | 0.012582 | 0.043000 | 0.108800 |
| DD.Shift.r.o | 0.219643 | 0.068581 | 0.145500 | 0.689400 |
| DD.a.n | 0.189380 | 0.104081 | 0.094800 | 1.128000 |
| DD.e.five | 0.277778 | 0.140290 | 0.164100 | 1.041600 |
| DD.five.Shift.r | 0.328438 | 0.111125 | 0.218900 | 0.973900 |
| DD.i.e | 0.177095 | 0.146447 | 0.090000 | 1.594200 |
| DD.l.Return | 0.354290 | 0.100862 | 0.244500 | 1.163100 |
| DD.n.l | 0.238670 | 0.110329 | 0.149400 | 1.157700 |
| DD.o.a | 0.131461 | 0.065758 | 0.060400 | 0.611800 |
| DD.period.t | 0.262892 | 0.090495 | 0.138700 | 0.694000 |
| DD.t.i | 0.223875 | 0.178445 | 0.123300 | 1.426100 |
| UD.Shift.r.o | 0.140012 | 0.068304 | 0.061300 | 0.604100 |
| UD.a.n | 0.127257 | 0.100174 | 0.057600 | 1.047700 |
| UD.e.five | 0.204178 | 0.136518 | 0.089700 | 0.920200 |
| UD.five.Shift.r | 0.255557 | 0.108452 | 0.162100 | 0.914200 |
| UD.i.e | 0.081711 | 0.145667 | 0.001400 | 1.472500 |
| UD.l.Return | 0.256292 | 0.101143 | 0.152200 | 1.078400 |
| UD.n.l | 0.161393 | 0.111509 | 0.059100 | 1.126800 |
| UD.o.a | 0.050637 | 0.064916 | -0.006300 | 0.512300 |
| UD.period.t | 0.162679 | 0.095407 | 0.020800 | 0.609800 |
| UD.t.i | 0.143456 | 0.179704 | 0.048600 | 1.340000 |

### s022 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.059593 | 0.014296 | 0.019800 | 0.108800 |
| H.Shift.r | 0.052605 | 0.006903 | 0.039100 | 0.092900 |
| H.a | 0.087848 | 0.014983 | 0.037500 | 0.125100 |
| H.e | 0.087958 | 0.018998 | 0.030400 | 0.144900 |
| H.five | 0.053134 | 0.006411 | 0.038800 | 0.078100 |
| H.i | 0.043095 | 0.007701 | 0.011600 | 0.064700 |
| H.l | 0.048576 | 0.008560 | 0.026200 | 0.093700 |
| H.n | 0.051378 | 0.005586 | 0.038300 | 0.067300 |
| H.o | 0.047279 | 0.007656 | 0.011600 | 0.081800 |
| H.period | 0.051970 | 0.009086 | 0.032800 | 0.098500 |
| H.t | 0.050181 | 0.005484 | 0.030400 | 0.064700 |
| DD.Shift.r.o | 0.509840 | 0.230918 | 0.328100 | 1.823800 |
| DD.a.n | 0.231088 | 0.089837 | 0.117300 | 0.875700 |
| DD.e.five | 0.951507 | 0.489520 | 0.263600 | 3.828700 |
| DD.five.Shift.r | 0.816620 | 0.354218 | 0.481000 | 2.602800 |
| DD.i.e | 0.286115 | 0.110142 | 0.128100 | 1.047600 |
| DD.l.Return | 0.739890 | 0.351237 | 0.302600 | 2.450300 |
| DD.n.l | 0.454717 | 0.400235 | 0.256800 | 4.025200 |
| DD.o.a | 0.348235 | 0.139704 | 0.172200 | 0.959300 |
| DD.period.t | 0.546670 | 0.158140 | 0.384500 | 1.458700 |
| DD.t.i | 0.372634 | 0.199501 | 0.251000 | 1.946400 |
| UD.Shift.r.o | 0.457234 | 0.228499 | 0.280600 | 1.773600 |
| UD.a.n | 0.143239 | 0.091973 | 0.025700 | 0.810200 |
| UD.e.five | 0.863550 | 0.488201 | 0.175000 | 3.745800 |
| UD.five.Shift.r | 0.763487 | 0.354317 | 0.429200 | 2.551800 |
| UD.i.e | 0.243021 | 0.108989 | 0.082100 | 1.007500 |
| UD.l.Return | 0.691315 | 0.349256 | 0.258000 | 2.400400 |
| UD.n.l | 0.403340 | 0.400089 | 0.209300 | 3.978200 |
| UD.o.a | 0.300956 | 0.138828 | 0.127600 | 0.877500 |
| UD.period.t | 0.494700 | 0.156329 | 0.333000 | 1.408300 |
| UD.t.i | 0.322453 | 0.198505 | 0.189500 | 1.889400 |

### s025 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.085701 | 0.014040 | 0.002900 | 0.146800 |
| H.Shift.r | 0.082741 | 0.012500 | 0.057800 | 0.139600 |
| H.a | 0.064830 | 0.008441 | 0.042000 | 0.095600 |
| H.e | 0.074909 | 0.010106 | 0.026700 | 0.126200 |
| H.five | 0.073218 | 0.006269 | 0.062300 | 0.103700 |
| H.i | 0.064648 | 0.009005 | 0.027700 | 0.097900 |
| H.l | 0.089209 | 0.008357 | 0.067300 | 0.127500 |
| H.n | 0.073595 | 0.007698 | 0.038300 | 0.096900 |
| H.o | 0.077619 | 0.010508 | 0.042800 | 0.105900 |
| H.period | 0.107828 | 0.018862 | 0.074700 | 0.174500 |
| H.t | 0.071160 | 0.006653 | 0.053600 | 0.093700 |
| DD.Shift.r.o | 0.197576 | 0.110608 | 0.119700 | 1.181200 |
| DD.a.n | 0.114321 | 0.056141 | 0.034900 | 0.663800 |
| DD.e.five | 0.686774 | 0.211276 | 0.438400 | 1.790900 |
| DD.five.Shift.r | 0.474772 | 0.226094 | 0.286500 | 1.631200 |
| DD.i.e | 0.127761 | 0.049152 | 0.077200 | 0.631500 |
| DD.l.Return | 0.569287 | 0.122931 | 0.349300 | 1.284300 |
| DD.n.l | 0.196071 | 0.070252 | 0.141200 | 0.777400 |
| DD.o.a | 0.126274 | 0.068849 | 0.075800 | 0.862200 |
| DD.period.t | 0.222105 | 0.122770 | 0.116400 | 1.011200 |
| DD.t.i | 0.142839 | 0.076446 | 0.062600 | 1.096000 |
| UD.Shift.r.o | 0.114834 | 0.106950 | 0.035700 | 1.096700 |
| UD.a.n | 0.049490 | 0.053597 | -0.021100 | 0.593000 |
| UD.e.five | 0.611866 | 0.211826 | 0.350200 | 1.717200 |
| UD.five.Shift.r | 0.401554 | 0.225213 | 0.213600 | 1.561800 |
| UD.i.e | 0.063113 | 0.047981 | 0.019600 | 0.557600 |
| UD.l.Return | 0.480077 | 0.122927 | 0.248700 | 1.196100 |
| UD.n.l | 0.122476 | 0.072360 | 0.070500 | 0.722200 |
| UD.o.a | 0.048655 | 0.071832 | -0.001300 | 0.815500 |
| UD.period.t | 0.114277 | 0.124153 | -0.003000 | 0.880800 |
| UD.t.i | 0.071678 | 0.075680 | -0.003700 | 1.016800 |

### s027 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.089128 | 0.018065 | 0.019600 | 0.132200 |
| H.Shift.r | 0.108743 | 0.015038 | 0.067600 | 0.163600 |
| H.a | 0.113783 | 0.011004 | 0.051800 | 0.134900 |
| H.e | 0.087812 | 0.011928 | 0.048100 | 0.127000 |
| H.five | 0.064911 | 0.010026 | 0.022700 | 0.092700 |
| H.i | 0.109910 | 0.014203 | 0.059400 | 0.159200 |
| H.l | 0.114697 | 0.013866 | 0.072300 | 0.165000 |
| H.n | 0.102610 | 0.025062 | 0.033300 | 0.162800 |
| H.o | 0.125568 | 0.015109 | 0.087900 | 0.181100 |
| H.period | 0.100999 | 0.016885 | 0.058400 | 0.154100 |
| H.t | 0.089022 | 0.014061 | 0.025400 | 0.137300 |
| DD.Shift.r.o | 0.227437 | 0.062772 | 0.143100 | 0.662400 |
| DD.a.n | 0.199194 | 0.118560 | 0.091700 | 1.354900 |
| DD.e.five | 0.672681 | 0.196705 | 0.300600 | 1.722800 |
| DD.five.Shift.r | 0.447149 | 0.142922 | 0.314600 | 1.122400 |
| DD.i.e | 0.110894 | 0.032963 | 0.042300 | 0.427300 |
| DD.l.Return | 0.331890 | 0.098743 | 0.263300 | 1.378000 |
| DD.n.l | 0.183493 | 0.058327 | 0.128300 | 0.548300 |
| DD.o.a | 0.099898 | 0.034774 | 0.038400 | 0.399100 |
| DD.period.t | 0.312663 | 0.143494 | 0.137700 | 1.202400 |
| DD.t.i | 0.253488 | 0.112616 | 0.132800 | 0.781900 |
| UD.Shift.r.o | 0.118694 | 0.062790 | 0.019800 | 0.527500 |
| UD.a.n | 0.085411 | 0.123358 | -0.024300 | 1.303100 |
| UD.e.five | 0.584870 | 0.195975 | 0.215900 | 1.633100 |
| UD.five.Shift.r | 0.382238 | 0.140512 | 0.259700 | 1.050900 |
| UD.i.e | 0.000985 | 0.034538 | -0.086600 | 0.329100 |
| UD.l.Return | 0.217193 | 0.097319 | 0.154300 | 1.248100 |
| UD.n.l | 0.080883 | 0.066479 | 0.001300 | 0.503500 |
| UD.o.a | -0.025670 | 0.035541 | -0.072500 | 0.300900 |
| UD.period.t | 0.211664 | 0.144626 | -0.002200 | 1.114000 |
| UD.t.i | 0.164466 | 0.118304 | 0.042800 | 0.706400 |

### s030 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.082002 | 0.009600 | 0.052000 | 0.116900 |
| H.Shift.r | 0.098083 | 0.014281 | 0.062900 | 0.170500 |
| H.a | 0.104945 | 0.011892 | 0.082100 | 0.143800 |
| H.e | 0.136017 | 0.028852 | 0.058400 | 0.208500 |
| H.five | 0.086535 | 0.013250 | 0.063100 | 0.156300 |
| H.i | 0.111523 | 0.029162 | 0.011400 | 0.244100 |
| H.l | 0.086206 | 0.014300 | 0.044900 | 0.148900 |
| H.n | 0.092940 | 0.019331 | 0.014800 | 0.171300 |
| H.o | 0.128526 | 0.024279 | 0.063600 | 0.246800 |
| H.period | 0.120820 | 0.027583 | 0.068100 | 0.194800 |
| H.t | 0.115556 | 0.020737 | 0.063600 | 0.171800 |
| DD.Shift.r.o | 0.282827 | 0.226493 | 0.151600 | 2.811400 |
| DD.a.n | 0.244836 | 0.109377 | 0.121700 | 0.935800 |
| DD.e.five | 0.567609 | 0.217011 | 0.295300 | 1.589500 |
| DD.five.Shift.r | 0.472075 | 0.117146 | 0.323100 | 1.389400 |
| DD.i.e | 0.278471 | 0.104605 | 0.090600 | 0.958500 |
| DD.l.Return | 0.409012 | 0.270813 | 0.240100 | 1.414100 |
| DD.n.l | 0.271660 | 0.140142 | 0.142300 | 1.435800 |
| DD.o.a | 0.190430 | 0.097807 | 0.094600 | 0.904800 |
| DD.period.t | 0.299090 | 0.268660 | 0.105000 | 3.816000 |
| DD.t.i | 0.257477 | 0.294970 | 0.119600 | 4.010500 |
| UD.Shift.r.o | 0.184744 | 0.227949 | 0.059400 | 2.747800 |
| UD.a.n | 0.139891 | 0.106363 | 0.028300 | 0.833100 |
| UD.e.five | 0.431592 | 0.223343 | 0.169900 | 1.431900 |
| UD.five.Shift.r | 0.385539 | 0.114601 | 0.251800 | 1.251400 |
| UD.i.e | 0.166948 | 0.097563 | 0.048100 | 0.850800 |
| UD.l.Return | 0.322806 | 0.274773 | 0.166900 | 1.341200 |
| UD.n.l | 0.178721 | 0.138255 | 0.076300 | 1.346000 |
| UD.o.a | 0.061905 | 0.090875 | -0.024200 | 0.658000 |
| UD.period.t | 0.178270 | 0.274158 | -0.016800 | 3.705100 |
| UD.t.i | 0.141922 | 0.294313 | 0.014800 | 3.916500 |

### s031 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.070972 | 0.013710 | 0.005600 | 0.129300 |
| H.Shift.r | 0.098697 | 0.021813 | 0.053300 | 0.156300 |
| H.a | 0.123119 | 0.037021 | 0.018500 | 0.223900 |
| H.e | 0.074649 | 0.015446 | 0.045200 | 0.120100 |
| H.five | 0.068795 | 0.014390 | 0.037200 | 0.127200 |
| H.i | 0.073878 | 0.014236 | 0.039900 | 0.119600 |
| H.l | 0.085657 | 0.016952 | 0.049400 | 0.132000 |
| H.n | 0.078723 | 0.014011 | 0.054400 | 0.122200 |
| H.o | 0.071698 | 0.017087 | 0.034600 | 0.132000 |
| H.period | 0.095148 | 0.018521 | 0.056800 | 0.140700 |
| H.t | 0.073877 | 0.012956 | 0.028300 | 0.110300 |
| DD.Shift.r.o | 0.316283 | 0.150967 | 0.187400 | 1.904500 |
| DD.a.n | 0.157031 | 0.057112 | 0.011200 | 0.754400 |
| DD.e.five | 0.503219 | 0.289988 | 0.202700 | 1.657300 |
| DD.five.Shift.r | 0.474986 | 0.176240 | 0.299900 | 1.576600 |
| DD.i.e | 0.211981 | 0.162019 | 0.079500 | 1.288300 |
| DD.l.Return | 0.361553 | 0.185128 | 0.228700 | 2.113800 |
| DD.n.l | 0.216831 | 0.094671 | 0.140900 | 0.997500 |
| DD.o.a | 0.143936 | 0.066671 | 0.051800 | 0.771900 |
| DD.period.t | 0.314866 | 0.145991 | 0.109500 | 1.287400 |
| DD.t.i | 0.199129 | 0.221885 | 0.084000 | 2.612700 |
| UD.Shift.r.o | 0.217585 | 0.156765 | 0.050700 | 1.836100 |
| UD.a.n | 0.033911 | 0.050782 | -0.121400 | 0.530500 |
| UD.e.five | 0.428570 | 0.291486 | 0.148800 | 1.596800 |
| UD.five.Shift.r | 0.406190 | 0.175649 | 0.242600 | 1.523500 |
| UD.i.e | 0.138103 | 0.158372 | 0.008500 | 1.227300 |
| UD.l.Return | 0.275896 | 0.183826 | 0.146900 | 1.999800 |
| UD.n.l | 0.138108 | 0.092876 | 0.055900 | 0.920400 |
| UD.o.a | 0.072237 | 0.063066 | -0.033500 | 0.655000 |
| UD.period.t | 0.219718 | 0.147651 | 0.011000 | 1.209300 |
| UD.t.i | 0.125252 | 0.221564 | 0.016100 | 2.550700 |

### s032 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.080107 | 0.016005 | 0.046100 | 0.147200 |
| H.Shift.r | 0.095765 | 0.014975 | 0.060500 | 0.136400 |
| H.a | 0.088631 | 0.021440 | 0.005000 | 0.139500 |
| H.e | 0.085563 | 0.017019 | 0.038800 | 0.149000 |
| H.five | 0.080106 | 0.015384 | 0.010600 | 0.121900 |
| H.i | 0.079929 | 0.013078 | 0.003200 | 0.109500 |
| H.l | 0.083284 | 0.014827 | 0.013200 | 0.143500 |
| H.n | 0.068147 | 0.010681 | 0.035100 | 0.116900 |
| H.o | 0.074296 | 0.011891 | 0.042000 | 0.105300 |
| H.period | 0.100122 | 0.013122 | 0.058600 | 0.135600 |
| H.t | 0.092463 | 0.015403 | 0.054600 | 0.128500 |
| DD.Shift.r.o | 0.207396 | 0.118537 | 0.138500 | 1.211200 |
| DD.a.n | 0.143370 | 0.134232 | 0.047100 | 1.587800 |
| DD.e.five | 0.364359 | 0.229542 | 0.114000 | 1.960200 |
| DD.five.Shift.r | 0.361549 | 0.187697 | 0.206400 | 1.410800 |
| DD.i.e | 0.112246 | 0.098442 | 0.034600 | 0.849000 |
| DD.l.Return | 0.249677 | 0.180672 | 0.150700 | 2.042800 |
| DD.n.l | 0.219737 | 0.177270 | 0.105000 | 1.158700 |
| DD.o.a | 0.117142 | 0.055582 | 0.028100 | 0.398600 |
| DD.period.t | 0.197566 | 0.130243 | 0.063500 | 1.063500 |
| DD.t.i | 0.189348 | 0.054881 | 0.109000 | 0.563400 |
| UD.Shift.r.o | 0.111631 | 0.119144 | 0.034100 | 1.125200 |
| UD.a.n | 0.054739 | 0.131827 | -0.026600 | 1.496000 |
| UD.e.five | 0.278795 | 0.227970 | 0.044300 | 1.883400 |
| UD.five.Shift.r | 0.281442 | 0.188928 | 0.128200 | 1.310300 |
| UD.i.e | 0.032317 | 0.099570 | -0.042000 | 0.792500 |
| UD.l.Return | 0.166393 | 0.183145 | 0.067900 | 1.970500 |
| UD.n.l | 0.151590 | 0.176034 | -0.011900 | 1.081200 |
| UD.o.a | 0.042847 | 0.056009 | -0.041400 | 0.346900 |
| UD.period.t | 0.097445 | 0.132001 | -0.036300 | 1.004900 |
| UD.t.i | 0.096885 | 0.056347 | 0.032500 | 0.495300 |

### s033 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.105962 | 0.022451 | 0.041200 | 0.199100 |
| H.Shift.r | 0.149896 | 0.024836 | 0.055700 | 0.213700 |
| H.a | 0.219281 | 0.063296 | 0.136400 | 0.722100 |
| H.e | 0.111360 | 0.022595 | 0.042200 | 0.204900 |
| H.five | 0.083769 | 0.015841 | 0.041200 | 0.155900 |
| H.i | 0.119039 | 0.023016 | 0.032700 | 0.206500 |
| H.l | 0.111729 | 0.025742 | 0.003700 | 0.190400 |
| H.n | 0.109851 | 0.024168 | 0.046200 | 0.186200 |
| H.o | 0.123781 | 0.023473 | 0.047200 | 0.193900 |
| H.period | 0.096059 | 0.024502 | 0.014000 | 0.150100 |
| H.t | 0.158912 | 0.021771 | 0.101600 | 0.241100 |
| DD.Shift.r.o | 0.411269 | 0.250120 | 0.182000 | 1.606900 |
| DD.a.n | 0.276765 | 0.164674 | 0.130900 | 1.211300 |
| DD.e.five | 0.693109 | 0.244679 | 0.298600 | 2.110700 |
| DD.five.Shift.r | 0.656084 | 0.180410 | 0.458100 | 1.529400 |
| DD.i.e | 0.331500 | 0.208630 | 0.174400 | 1.487400 |
| DD.l.Return | 0.439343 | 0.191730 | 0.264700 | 1.296300 |
| DD.n.l | 0.443164 | 0.261054 | 0.218700 | 2.044300 |
| DD.o.a | 0.307780 | 0.144001 | 0.168900 | 1.268300 |
| DD.period.t | 0.448245 | 0.270229 | 0.204900 | 2.458200 |
| DD.t.i | 0.166155 | 0.094490 | 0.035400 | 1.181700 |
| UD.Shift.r.o | 0.261373 | 0.252832 | 0.028800 | 1.468900 |
| UD.a.n | 0.057483 | 0.145651 | -0.235500 | 0.694900 |
| UD.e.five | 0.581749 | 0.241557 | 0.174300 | 1.997000 |
| UD.five.Shift.r | 0.572315 | 0.174855 | 0.382400 | 1.406200 |
| UD.i.e | 0.212461 | 0.210994 | 0.034800 | 1.395300 |
| UD.l.Return | 0.327614 | 0.191718 | 0.171800 | 1.218700 |
| UD.n.l | 0.333312 | 0.262881 | 0.083300 | 1.936400 |
| UD.o.a | 0.184000 | 0.150551 | 0.020100 | 1.171200 |
| UD.period.t | 0.352186 | 0.273512 | 0.098600 | 2.354500 |
| UD.t.i | 0.007243 | 0.095998 | -0.132400 | 1.070900 |

### s035 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.069282 | 0.013388 | 0.042500 | 0.131400 |
| H.Shift.r | 0.108767 | 0.012943 | 0.071800 | 0.139100 |
| H.a | 0.072867 | 0.015854 | 0.039300 | 0.117900 |
| H.e | 0.071594 | 0.017110 | 0.042200 | 0.161200 |
| H.five | 0.067945 | 0.010982 | 0.042800 | 0.114500 |
| H.i | 0.056932 | 0.011111 | 0.028500 | 0.108200 |
| H.l | 0.071625 | 0.015798 | 0.030100 | 0.117900 |
| H.n | 0.069406 | 0.020642 | 0.040900 | 0.153200 |
| H.o | 0.066845 | 0.016511 | 0.012700 | 0.124200 |
| H.period | 0.050925 | 0.007419 | 0.032200 | 0.089200 |
| H.t | 0.073725 | 0.014758 | 0.043500 | 0.114200 |
| DD.Shift.r.o | 0.230716 | 0.144009 | 0.131700 | 1.312800 |
| DD.a.n | 0.119796 | 0.087067 | 0.001700 | 0.840300 |
| DD.e.five | 0.341851 | 0.196837 | 0.140300 | 1.271100 |
| DD.five.Shift.r | 0.473853 | 0.193144 | 0.306000 | 1.475800 |
| DD.i.e | 0.167876 | 0.126747 | 0.073900 | 0.949700 |
| DD.l.Return | 0.315863 | 0.080419 | 0.215900 | 0.694500 |
| DD.n.l | 0.192141 | 0.095712 | 0.122400 | 0.777400 |
| DD.o.a | 0.171683 | 0.125786 | 0.008000 | 0.622500 |
| DD.period.t | 0.320932 | 0.157101 | 0.156100 | 1.188200 |
| DD.t.i | 0.164344 | 0.137184 | 0.076800 | 1.773200 |
| UD.Shift.r.o | 0.121949 | 0.144744 | 0.022700 | 1.221000 |
| UD.a.n | 0.046929 | 0.086069 | -0.076700 | 0.751900 |
| UD.e.five | 0.270257 | 0.199432 | 0.015800 | 1.210400 |
| UD.five.Shift.r | 0.405908 | 0.192211 | 0.239800 | 1.406100 |
| UD.i.e | 0.110944 | 0.127474 | -0.011600 | 0.889500 |
| UD.l.Return | 0.244237 | 0.082464 | 0.140700 | 0.626700 |
| UD.n.l | 0.122735 | 0.098108 | 0.022900 | 0.717500 |
| UD.o.a | 0.104839 | 0.128001 | -0.066200 | 0.575000 |
| UD.period.t | 0.270006 | 0.158052 | 0.085400 | 1.139400 |
| UD.t.i | 0.090618 | 0.136742 | 0.005800 | 1.687200 |

### s036 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.044272 | 0.007204 | 0.018500 | 0.070200 |
| H.Shift.r | 0.038173 | 0.005705 | 0.020300 | 0.059600 |
| H.a | 0.055973 | 0.006935 | 0.009800 | 0.075500 |
| H.e | 0.049071 | 0.006474 | 0.023800 | 0.072300 |
| H.five | 0.045687 | 0.007573 | 0.022700 | 0.067300 |
| H.i | 0.043290 | 0.005691 | 0.018000 | 0.059900 |
| H.l | 0.040426 | 0.005652 | 0.024600 | 0.070400 |
| H.n | 0.037849 | 0.005478 | 0.024300 | 0.058800 |
| H.o | 0.042682 | 0.004726 | 0.031900 | 0.059600 |
| H.period | 0.044157 | 0.005787 | 0.028800 | 0.061700 |
| H.t | 0.049291 | 0.009380 | 0.022500 | 0.073300 |
| DD.Shift.r.o | 0.677548 | 0.202490 | 0.509800 | 2.040700 |
| DD.a.n | 0.381400 | 0.107500 | 0.131500 | 1.178800 |
| DD.e.five | 0.535658 | 0.535350 | 0.142700 | 3.604900 |
| DD.five.Shift.r | 0.700574 | 0.473403 | 0.306800 | 5.444500 |
| DD.i.e | 0.604824 | 0.262462 | 0.178300 | 2.209900 |
| DD.l.Return | 1.050996 | 0.727745 | 0.435300 | 5.883600 |
| DD.n.l | 0.653866 | 0.255680 | 0.428800 | 2.313800 |
| DD.o.a | 0.262067 | 0.135639 | 0.069300 | 0.989500 |
| DD.period.t | 0.849943 | 0.547174 | 0.305300 | 5.535300 |
| DD.t.i | 0.322557 | 0.265118 | 0.098400 | 2.742800 |
| UD.Shift.r.o | 0.639375 | 0.201084 | 0.478400 | 2.001600 |
| UD.a.n | 0.325427 | 0.107614 | 0.072900 | 1.122600 |
| UD.e.five | 0.486587 | 0.535786 | 0.096800 | 3.555000 |
| UD.five.Shift.r | 0.654887 | 0.474698 | 0.248700 | 5.411500 |
| UD.i.e | 0.561534 | 0.262700 | 0.136900 | 2.167900 |
| UD.l.Return | 1.010570 | 0.726111 | 0.402300 | 5.836400 |
| UD.n.l | 0.616017 | 0.254346 | 0.394500 | 2.272100 |
| UD.o.a | 0.219384 | 0.135561 | 0.022300 | 0.946000 |
| UD.period.t | 0.805787 | 0.547589 | 0.263600 | 5.488600 |
| UD.t.i | 0.273267 | 0.269924 | 0.047500 | 2.705300 |

### s037 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.102607 | 0.019858 | 0.052500 | 0.156100 |
| H.Shift.r | 0.091847 | 0.012939 | 0.066200 | 0.137200 |
| H.a | 0.102032 | 0.011223 | 0.074900 | 0.144300 |
| H.e | 0.096323 | 0.013768 | 0.052800 | 0.161400 |
| H.five | 0.073792 | 0.013207 | 0.048000 | 0.146700 |
| H.i | 0.072082 | 0.012496 | 0.045900 | 0.109800 |
| H.l | 0.127830 | 0.012947 | 0.089700 | 0.170100 |
| H.n | 0.087776 | 0.013516 | 0.054600 | 0.126900 |
| H.o | 0.098906 | 0.015522 | 0.051500 | 0.176200 |
| H.period | 0.114916 | 0.016241 | 0.038300 | 0.184900 |
| H.t | 0.082869 | 0.012919 | 0.036400 | 0.153800 |
| DD.Shift.r.o | 0.226813 | 0.144941 | 0.094800 | 1.699400 |
| DD.a.n | 0.140191 | 0.053359 | 0.046700 | 0.536700 |
| DD.e.five | 0.450723 | 0.145467 | 0.234200 | 1.459600 |
| DD.five.Shift.r | 0.370498 | 0.134873 | 0.266700 | 1.248600 |
| DD.i.e | 0.104846 | 0.042517 | 0.029000 | 0.320700 |
| DD.l.Return | 0.292859 | 0.150564 | 0.205100 | 1.892800 |
| DD.n.l | 0.180183 | 0.078448 | 0.099200 | 0.878500 |
| DD.o.a | 0.141825 | 0.066164 | 0.072400 | 0.650700 |
| DD.period.t | 0.172173 | 0.118489 | 0.018700 | 0.826900 |
| DD.t.i | 0.159658 | 0.115430 | 0.056200 | 1.502900 |
| UD.Shift.r.o | 0.134966 | 0.142651 | 0.020900 | 1.613100 |
| UD.a.n | 0.038160 | 0.050142 | -0.049300 | 0.405900 |
| UD.e.five | 0.354401 | 0.146657 | 0.146900 | 1.376000 |
| UD.five.Shift.r | 0.296706 | 0.133792 | 0.203700 | 1.178200 |
| UD.i.e | 0.032764 | 0.043795 | -0.061700 | 0.257100 |
| UD.l.Return | 0.165029 | 0.151420 | 0.066900 | 1.784100 |
| UD.n.l | 0.092408 | 0.079939 | -0.003000 | 0.791400 |
| UD.o.a | 0.042919 | 0.060945 | -0.029800 | 0.543300 |
| UD.period.t | 0.057257 | 0.114926 | -0.064200 | 0.706400 |
| UD.t.i | 0.076788 | 0.113654 | -0.012200 | 1.416100 |

### s038 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.075007 | 0.015115 | 0.004000 | 0.106800 |
| H.Shift.r | 0.079474 | 0.020869 | 0.047800 | 0.135300 |
| H.a | 0.110818 | 0.020856 | 0.007700 | 0.158300 |
| H.e | 0.070750 | 0.014346 | 0.007400 | 0.140100 |
| H.five | 0.060761 | 0.012123 | 0.036700 | 0.104700 |
| H.i | 0.070027 | 0.012793 | 0.042800 | 0.132400 |
| H.l | 0.091983 | 0.012007 | 0.061000 | 0.121100 |
| H.n | 0.062827 | 0.008521 | 0.043300 | 0.102600 |
| H.o | 0.090671 | 0.016736 | 0.010300 | 0.152000 |
| H.period | 0.092204 | 0.022957 | 0.040900 | 0.146400 |
| H.t | 0.080585 | 0.015971 | 0.042000 | 0.117100 |
| DD.Shift.r.o | 0.459730 | 0.258869 | 0.255000 | 2.210800 |
| DD.a.n | 0.144652 | 0.104579 | 0.050800 | 1.280400 |
| DD.e.five | 0.359497 | 0.261821 | 0.177200 | 1.761300 |
| DD.five.Shift.r | 0.550594 | 0.278942 | 0.357900 | 2.669300 |
| DD.i.e | 0.160550 | 0.156961 | 0.034100 | 1.375800 |
| DD.l.Return | 0.401797 | 0.232960 | 0.253400 | 1.899000 |
| DD.n.l | 0.231626 | 0.192788 | 0.113500 | 1.475200 |
| DD.o.a | 0.138850 | 0.053424 | 0.076300 | 0.681800 |
| DD.period.t | 0.385614 | 0.250374 | 0.170100 | 1.682900 |
| DD.t.i | 0.097187 | 0.033188 | 0.039100 | 0.339700 |
| UD.Shift.r.o | 0.380257 | 0.261773 | 0.199400 | 2.161700 |
| UD.a.n | 0.033834 | 0.107233 | -0.045400 | 1.183300 |
| UD.e.five | 0.288747 | 0.261413 | 0.104400 | 1.681600 |
| UD.five.Shift.r | 0.489834 | 0.279725 | 0.281700 | 2.616300 |
| UD.i.e | 0.090523 | 0.158658 | -0.030400 | 1.313800 |
| UD.l.Return | 0.309813 | 0.233859 | 0.160000 | 1.797700 |
| UD.n.l | 0.168800 | 0.192688 | 0.053500 | 1.395500 |
| UD.o.a | 0.048180 | 0.056201 | -0.013700 | 0.605800 |
| UD.period.t | 0.293409 | 0.254668 | 0.046100 | 1.617700 |
| UD.t.i | 0.016603 | 0.036409 | -0.055100 | 0.225000 |

### s039 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.091526 | 0.010084 | 0.058300 | 0.127900 |
| H.Shift.r | 0.150138 | 0.015298 | 0.104500 | 0.185900 |
| H.a | 0.101217 | 0.012525 | 0.023000 | 0.128200 |
| H.e | 0.080938 | 0.008023 | 0.058300 | 0.104200 |
| H.five | 0.090226 | 0.011023 | 0.066800 | 0.126100 |
| H.i | 0.074449 | 0.009740 | 0.030600 | 0.098700 |
| H.l | 0.087003 | 0.010810 | 0.023000 | 0.119800 |
| H.n | 0.085191 | 0.007781 | 0.055700 | 0.107400 |
| H.o | 0.085979 | 0.009378 | 0.063300 | 0.116900 |
| H.period | 0.079211 | 0.011825 | 0.059400 | 0.137700 |
| H.t | 0.091710 | 0.012596 | 0.035600 | 0.141400 |
| DD.Shift.r.o | 0.243182 | 0.169322 | 0.124800 | 1.490000 |
| DD.a.n | 0.146753 | 0.030239 | 0.098400 | 0.350100 |
| DD.e.five | 0.329347 | 0.199483 | 0.193100 | 1.232600 |
| DD.five.Shift.r | 0.520614 | 0.172072 | 0.370900 | 2.133700 |
| DD.i.e | 0.164318 | 0.024788 | 0.106100 | 0.334500 |
| DD.l.Return | 0.249857 | 0.181134 | 0.167900 | 2.376400 |
| DD.n.l | 0.225765 | 0.142851 | 0.170100 | 1.580400 |
| DD.o.a | 0.170036 | 0.152278 | 0.067300 | 1.750900 |
| DD.period.t | 0.228672 | 0.117073 | 0.112900 | 0.870500 |
| DD.t.i | 0.169898 | 0.120649 | 0.103900 | 1.506900 |
| UD.Shift.r.o | 0.093044 | 0.168057 | -0.010100 | 1.341200 |
| UD.a.n | 0.045536 | 0.032516 | -0.005000 | 0.259600 |
| UD.e.five | 0.248409 | 0.198780 | 0.117400 | 1.156400 |
| UD.five.Shift.r | 0.430388 | 0.171600 | 0.282000 | 2.032100 |
| UD.i.e | 0.089869 | 0.023847 | 0.041200 | 0.235800 |
| UD.l.Return | 0.162855 | 0.182106 | 0.076900 | 2.297000 |
| UD.n.l | 0.140574 | 0.143715 | 0.076200 | 1.493600 |
| UD.o.a | 0.084057 | 0.152957 | -0.009500 | 1.680700 |
| UD.period.t | 0.149461 | 0.117166 | 0.029500 | 0.798700 |
| UD.t.i | 0.078187 | 0.118818 | 0.004800 | 1.403200 |

### s040 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.108941 | 0.019019 | 0.065400 | 0.185400 |
| H.Shift.r | 0.087928 | 0.018717 | 0.010300 | 0.132400 |
| H.a | 0.176739 | 0.037110 | 0.069400 | 0.259400 |
| H.e | 0.130250 | 0.029193 | 0.013000 | 0.187300 |
| H.five | 0.087598 | 0.017959 | 0.034600 | 0.159600 |
| H.i | 0.114206 | 0.022450 | 0.068300 | 0.176000 |
| H.l | 0.125395 | 0.035806 | 0.026400 | 0.217600 |
| H.n | 0.130284 | 0.033346 | 0.054900 | 0.238800 |
| H.o | 0.110441 | 0.018314 | 0.074400 | 0.188100 |
| H.period | 0.107277 | 0.015851 | 0.065400 | 0.173800 |
| H.t | 0.128098 | 0.018632 | 0.075500 | 0.185700 |
| DD.Shift.r.o | 0.364718 | 0.120028 | 0.235600 | 1.563300 |
| DD.a.n | 0.187600 | 0.130152 | 0.003000 | 0.978700 |
| DD.e.five | 0.701203 | 0.326113 | 0.270400 | 2.251800 |
| DD.five.Shift.r | 0.594400 | 0.297499 | 0.323700 | 1.750100 |
| DD.i.e | 0.200967 | 0.138769 | 0.104000 | 1.069100 |
| DD.l.Return | 0.469307 | 0.280443 | 0.204800 | 2.546200 |
| DD.n.l | 0.235083 | 0.136320 | 0.069600 | 0.957300 |
| DD.o.a | 0.231786 | 0.138768 | 0.092700 | 0.766000 |
| DD.period.t | 0.424240 | 0.222697 | 0.245000 | 1.947200 |
| DD.t.i | 0.148163 | 0.098464 | 0.067000 | 1.015400 |
| UD.Shift.r.o | 0.276790 | 0.120529 | 0.126300 | 1.473900 |
| UD.a.n | 0.010861 | 0.143607 | -0.185200 | 0.844700 |
| UD.e.five | 0.570954 | 0.329071 | 0.124500 | 2.091400 |
| UD.five.Shift.r | 0.506803 | 0.298951 | 0.232700 | 1.674600 |
| UD.i.e | 0.086761 | 0.142228 | -0.046700 | 0.969100 |
| UD.l.Return | 0.343912 | 0.280992 | 0.115900 | 2.442300 |
| UD.n.l | 0.104798 | 0.150494 | -0.109000 | 0.805600 |
| UD.o.a | 0.121344 | 0.138702 | -0.014700 | 0.647800 |
| UD.period.t | 0.316963 | 0.221459 | 0.122000 | 1.839000 |
| UD.t.i | 0.020065 | 0.099996 | -0.062000 | 0.898500 |

### s042 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.122301 | 0.028968 | 0.042200 | 0.185400 |
| H.Shift.r | 0.068691 | 0.009365 | 0.047000 | 0.097300 |
| H.a | 0.071782 | 0.011113 | 0.009000 | 0.093100 |
| H.e | 0.077041 | 0.010776 | 0.053800 | 0.104500 |
| H.five | 0.061157 | 0.009645 | 0.027500 | 0.090200 |
| H.i | 0.067943 | 0.009931 | 0.045400 | 0.091300 |
| H.l | 0.129105 | 0.021372 | 0.020600 | 0.182800 |
| H.n | 0.141284 | 0.030307 | 0.070000 | 0.225500 |
| H.o | 0.087942 | 0.017814 | 0.043300 | 0.141400 |
| H.period | 0.098789 | 0.018324 | 0.061000 | 0.154600 |
| H.t | 0.062433 | 0.008361 | 0.041700 | 0.086500 |
| DD.Shift.r.o | 0.217025 | 0.126927 | 0.134700 | 0.991800 |
| DD.a.n | 0.184487 | 0.077982 | 0.121900 | 1.155900 |
| DD.e.five | 0.363632 | 0.173396 | 0.196200 | 1.295300 |
| DD.five.Shift.r | 0.381665 | 0.084825 | 0.258800 | 0.886700 |
| DD.i.e | 0.192824 | 0.067497 | 0.122400 | 0.778800 |
| DD.l.Return | 0.336496 | 0.097054 | 0.227500 | 1.242200 |
| DD.n.l | 0.141422 | 0.096225 | 0.041900 | 1.218600 |
| DD.o.a | 0.139859 | 0.028271 | 0.084200 | 0.370100 |
| DD.period.t | 0.315823 | 0.079831 | 0.219400 | 1.006000 |
| DD.t.i | 0.156582 | 0.068196 | 0.102100 | 1.071800 |
| UD.Shift.r.o | 0.148335 | 0.125832 | 0.053600 | 0.908700 |
| UD.a.n | 0.112705 | 0.078056 | 0.040900 | 1.090700 |
| UD.e.five | 0.286590 | 0.174425 | 0.131600 | 1.218800 |
| UD.five.Shift.r | 0.320509 | 0.082776 | 0.209500 | 0.816000 |
| UD.i.e | 0.124881 | 0.065790 | 0.059900 | 0.687500 |
| UD.l.Return | 0.207390 | 0.102481 | 0.108300 | 1.106100 |
| UD.n.l | 0.000138 | 0.106088 | -0.086600 | 1.142400 |
| UD.o.a | 0.051917 | 0.034844 | -0.010800 | 0.269900 |
| UD.period.t | 0.217034 | 0.077121 | 0.120700 | 0.901000 |
| UD.t.i | 0.094149 | 0.066560 | 0.043800 | 0.991600 |

### s043 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.049298 | 0.009659 | 0.027700 | 0.117600 |
| H.Shift.r | 0.050739 | 0.007670 | 0.030900 | 0.072300 |
| H.a | 0.110547 | 0.019729 | 0.010600 | 0.157500 |
| H.e | 0.082586 | 0.010875 | 0.062500 | 0.113400 |
| H.five | 0.058500 | 0.008170 | 0.033300 | 0.078100 |
| H.i | 0.062716 | 0.009536 | 0.015600 | 0.098100 |
| H.l | 0.055909 | 0.008046 | 0.018500 | 0.093600 |
| H.n | 0.056034 | 0.006482 | 0.037000 | 0.078400 |
| H.o | 0.061092 | 0.006718 | 0.043300 | 0.081800 |
| H.period | 0.060471 | 0.007480 | 0.036200 | 0.084200 |
| H.t | 0.065369 | 0.012740 | 0.038300 | 0.118700 |
| DD.Shift.r.o | 0.484148 | 0.149656 | 0.348500 | 1.570200 |
| DD.a.n | 0.253960 | 0.068273 | 0.157800 | 0.927600 |
| DD.e.five | 0.538473 | 0.305930 | 0.193300 | 2.502300 |
| DD.five.Shift.r | 0.622101 | 0.223361 | 0.396300 | 2.298500 |
| DD.i.e | 0.212429 | 0.107552 | 0.082300 | 0.692300 |
| DD.l.Return | 0.373494 | 0.223416 | 0.239200 | 2.358700 |
| DD.n.l | 0.321115 | 0.074151 | 0.244700 | 0.793000 |
| DD.o.a | 0.101644 | 0.030563 | 0.035400 | 0.231900 |
| DD.period.t | 0.584784 | 0.891490 | 0.108600 | 12.506100 |
| DD.t.i | 0.359853 | 0.221783 | 0.153600 | 1.616100 |
| UD.Shift.r.o | 0.433409 | 0.149327 | 0.297300 | 1.515100 |
| UD.a.n | 0.143413 | 0.069973 | 0.064400 | 0.827600 |
| UD.e.five | 0.455887 | 0.304830 | 0.125500 | 2.427100 |
| UD.five.Shift.r | 0.563602 | 0.223025 | 0.338200 | 2.241800 |
| UD.i.e | 0.149713 | 0.105438 | 0.022700 | 0.634800 |
| UD.l.Return | 0.317584 | 0.224566 | 0.181900 | 2.305400 |
| UD.n.l | 0.265080 | 0.073338 | 0.190900 | 0.721200 |
| UD.o.a | 0.040552 | 0.030418 | -0.025300 | 0.165700 |
| UD.period.t | 0.524313 | 0.892414 | 0.044000 | 12.451700 |
| UD.t.i | 0.294484 | 0.226815 | 0.075500 | 1.565400 |

### s047 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.099145 | 0.018709 | 0.019100 | 0.141100 |
| H.Shift.r | 0.089372 | 0.014059 | 0.039900 | 0.160600 |
| H.a | 0.080321 | 0.017330 | 0.025900 | 0.126600 |
| H.e | 0.070102 | 0.017175 | 0.020300 | 0.114500 |
| H.five | 0.080277 | 0.015543 | 0.031900 | 0.127900 |
| H.i | 0.064287 | 0.019670 | 0.013000 | 0.117900 |
| H.l | 0.107909 | 0.015841 | 0.068600 | 0.153200 |
| H.n | 0.090220 | 0.020368 | 0.048000 | 0.155400 |
| H.o | 0.070571 | 0.020860 | 0.013200 | 0.134500 |
| H.period | 0.079171 | 0.018694 | 0.018800 | 0.144800 |
| H.t | 0.083903 | 0.013581 | 0.039300 | 0.122100 |
| DD.Shift.r.o | 0.406887 | 0.208990 | 0.161800 | 1.473700 |
| DD.a.n | 0.162102 | 0.100836 | 0.034900 | 0.811800 |
| DD.e.five | 0.761925 | 0.312329 | 0.455700 | 3.022300 |
| DD.five.Shift.r | 0.522303 | 0.253033 | 0.346100 | 2.202900 |
| DD.i.e | 0.170025 | 0.100098 | 0.080200 | 1.101000 |
| DD.l.Return | 0.343389 | 0.405734 | 0.199800 | 5.302700 |
| DD.n.l | 0.310233 | 0.223026 | 0.116300 | 2.493100 |
| DD.o.a | 0.167870 | 0.055324 | 0.106200 | 0.588000 |
| DD.period.t | 0.349388 | 0.259720 | 0.128700 | 2.015700 |
| DD.t.i | 0.438514 | 0.222057 | 0.152500 | 1.791400 |
| UD.Shift.r.o | 0.317514 | 0.207909 | 0.077000 | 1.397400 |
| UD.a.n | 0.081781 | 0.097792 | -0.033500 | 0.717900 |
| UD.e.five | 0.691823 | 0.314057 | 0.375800 | 2.958400 |
| UD.five.Shift.r | 0.442025 | 0.255658 | 0.260400 | 2.143500 |
| UD.i.e | 0.105738 | 0.098518 | 0.027500 | 1.058800 |
| UD.l.Return | 0.235480 | 0.405046 | 0.092500 | 5.202500 |
| UD.n.l | 0.220013 | 0.227073 | -0.023500 | 2.426600 |
| UD.o.a | 0.097299 | 0.054006 | 0.025400 | 0.479800 |
| UD.period.t | 0.270217 | 0.261145 | 0.055100 | 1.926300 |
| UD.t.i | 0.354610 | 0.221782 | 0.089700 | 1.690900 |

### s050 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.086182 | 0.015382 | 0.009000 | 0.152200 |
| H.Shift.r | 0.107428 | 0.020113 | 0.027000 | 0.204500 |
| H.a | 0.114834 | 0.014869 | 0.072600 | 0.153200 |
| H.e | 0.082474 | 0.009296 | 0.055900 | 0.112600 |
| H.five | 0.066758 | 0.011867 | 0.040400 | 0.110500 |
| H.i | 0.074440 | 0.010649 | 0.016600 | 0.106000 |
| H.l | 0.086933 | 0.012611 | 0.009500 | 0.112600 |
| H.n | 0.086387 | 0.016364 | 0.010300 | 0.113400 |
| H.o | 0.093839 | 0.012169 | 0.065400 | 0.143000 |
| H.period | 0.091499 | 0.010985 | 0.063600 | 0.127700 |
| H.t | 0.066690 | 0.008666 | 0.017200 | 0.096800 |
| DD.Shift.r.o | 0.188801 | 0.097781 | 0.092700 | 0.992900 |
| DD.a.n | 0.132085 | 0.083354 | 0.052000 | 0.528500 |
| DD.e.five | 0.328145 | 0.132588 | 0.211000 | 1.187800 |
| DD.five.Shift.r | 0.370111 | 0.138838 | 0.248300 | 1.137500 |
| DD.i.e | 0.201933 | 0.177603 | 0.049900 | 1.394500 |
| DD.l.Return | 0.278470 | 0.101300 | 0.196400 | 0.749900 |
| DD.n.l | 0.219472 | 0.054981 | 0.121800 | 0.615000 |
| DD.o.a | 0.164007 | 0.043058 | 0.097500 | 0.324300 |
| DD.period.t | 0.294785 | 0.193887 | 0.112900 | 1.436800 |
| DD.t.i | 0.182892 | 0.103605 | 0.089500 | 1.024000 |
| UD.Shift.r.o | 0.081373 | 0.094901 | -0.004500 | 0.891600 |
| UD.a.n | 0.017252 | 0.086082 | -0.061400 | 0.425400 |
| UD.e.five | 0.245671 | 0.133292 | 0.127100 | 1.111000 |
| UD.five.Shift.r | 0.303353 | 0.136267 | 0.187900 | 1.058600 |
| UD.i.e | 0.127493 | 0.177407 | -0.027700 | 1.313000 |
| UD.l.Return | 0.191536 | 0.101726 | 0.112200 | 0.683900 |
| UD.n.l | 0.133085 | 0.059813 | 0.080200 | 0.581500 |
| UD.o.a | 0.070168 | 0.041849 | 0.012000 | 0.229500 |
| UD.period.t | 0.203286 | 0.193325 | 0.026600 | 1.331500 |
| UD.t.i | 0.116202 | 0.104514 | 0.035100 | 0.962800 |

### s051 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.061447 | 0.012146 | 0.023500 | 0.094400 |
| H.Shift.r | 0.080281 | 0.016432 | 0.010600 | 0.125000 |
| H.a | 0.083975 | 0.023083 | 0.007700 | 0.163600 |
| H.e | 0.075104 | 0.013750 | 0.041700 | 0.145300 |
| H.five | 0.071902 | 0.013940 | 0.024800 | 0.108900 |
| H.i | 0.072709 | 0.012062 | 0.011600 | 0.101600 |
| H.l | 0.081658 | 0.014436 | 0.017700 | 0.125000 |
| H.n | 0.089442 | 0.016868 | 0.005300 | 0.138200 |
| H.o | 0.073204 | 0.013425 | 0.024000 | 0.120800 |
| H.period | 0.099705 | 0.019361 | 0.008200 | 0.180100 |
| H.t | 0.070716 | 0.011273 | 0.040400 | 0.151900 |
| DD.Shift.r.o | 0.196056 | 0.060391 | 0.086400 | 0.481600 |
| DD.a.n | 0.161210 | 0.066492 | 0.009300 | 0.462900 |
| DD.e.five | 0.232088 | 0.107032 | 0.001300 | 0.677800 |
| DD.five.Shift.r | 0.354710 | 0.122332 | 0.221700 | 1.475100 |
| DD.i.e | 0.111311 | 0.060610 | 0.041700 | 0.544100 |
| DD.l.Return | 0.238986 | 0.085693 | 0.107300 | 0.926100 |
| DD.n.l | 0.127847 | 0.061773 | 0.037400 | 0.487100 |
| DD.o.a | 0.147527 | 0.064863 | 0.020400 | 0.438700 |
| DD.period.t | 0.197225 | 0.053648 | 0.111800 | 0.523600 |
| DD.t.i | 0.109496 | 0.045548 | 0.002700 | 0.466900 |
| UD.Shift.r.o | 0.115775 | 0.066037 | 0.007200 | 0.433100 |
| UD.a.n | 0.077234 | 0.062307 | -0.037700 | 0.358700 |
| UD.e.five | 0.156985 | 0.107019 | -0.136200 | 0.604700 |
| UD.five.Shift.r | 0.282808 | 0.125454 | 0.150700 | 1.450300 |
| UD.i.e | 0.038603 | 0.061961 | -0.020300 | 0.477600 |
| UD.l.Return | 0.157329 | 0.084644 | 0.075800 | 0.855900 |
| UD.n.l | 0.038405 | 0.065919 | -0.058600 | 0.433800 |
| UD.o.a | 0.074322 | 0.063672 | -0.004200 | 0.371700 |
| UD.period.t | 0.097520 | 0.054816 | 0.011300 | 0.449500 |
| UD.t.i | 0.038780 | 0.045513 | -0.062800 | 0.405900 |

### s052 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.075775 | 0.020465 | 0.014000 | 0.164600 |
| H.Shift.r | 0.078109 | 0.018546 | 0.011100 | 0.145400 |
| H.a | 0.101884 | 0.030771 | 0.042200 | 0.204100 |
| H.e | 0.081717 | 0.018762 | 0.050700 | 0.145600 |
| H.five | 0.068813 | 0.012822 | 0.045200 | 0.101800 |
| H.i | 0.110183 | 0.017686 | 0.028800 | 0.154800 |
| H.l | 0.112873 | 0.016664 | 0.019000 | 0.157700 |
| H.n | 0.100797 | 0.018253 | 0.049700 | 0.148000 |
| H.o | 0.069108 | 0.009873 | 0.035400 | 0.120300 |
| H.period | 0.058036 | 0.009299 | 0.012700 | 0.096600 |
| H.t | 0.081822 | 0.011517 | 0.046700 | 0.123200 |
| DD.Shift.r.o | 0.286736 | 0.106348 | 0.172900 | 1.012800 |
| DD.a.n | 0.204970 | 0.129118 | 0.030700 | 0.887500 |
| DD.e.five | 0.401899 | 0.222687 | 0.208100 | 1.362300 |
| DD.five.Shift.r | 0.357397 | 0.115281 | 0.238000 | 1.173300 |
| DD.i.e | 0.127395 | 0.038121 | 0.012700 | 0.469800 |
| DD.l.Return | 0.519848 | 0.218644 | 0.287100 | 1.512100 |
| DD.n.l | 0.070807 | 0.052894 | 0.001600 | 0.488700 |
| DD.o.a | 0.258824 | 0.139628 | 0.079200 | 0.954500 |
| DD.period.t | 0.619142 | 0.145138 | 0.284800 | 1.656000 |
| DD.t.i | 0.064640 | 0.019073 | 0.032700 | 0.165200 |
| UD.Shift.r.o | 0.208627 | 0.107050 | 0.106600 | 0.996900 |
| UD.a.n | 0.103085 | 0.124591 | -0.086300 | 0.775100 |
| UD.e.five | 0.320182 | 0.217998 | 0.130500 | 1.276600 |
| UD.five.Shift.r | 0.288584 | 0.112564 | 0.171800 | 1.088400 |
| UD.i.e | 0.017212 | 0.036019 | -0.023800 | 0.356600 |
| UD.l.Return | 0.406975 | 0.216841 | 0.203700 | 1.392000 |
| UD.n.l | -0.029989 | 0.052513 | -0.083400 | 0.401900 |
| UD.o.a | 0.189716 | 0.142389 | -0.008900 | 0.892200 |
| UD.period.t | 0.561106 | 0.144803 | 0.215400 | 1.596600 |
| UD.t.i | -0.017182 | 0.018351 | -0.041200 | 0.081800 |

### s053 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.097148 | 0.017939 | 0.010100 | 0.140900 |
| H.Shift.r | 0.055018 | 0.011149 | 0.001400 | 0.092900 |
| H.a | 0.111507 | 0.019954 | 0.033600 | 0.163800 |
| H.e | 0.058636 | 0.016886 | 0.015100 | 0.119000 |
| H.five | 0.070904 | 0.018167 | 0.020300 | 0.135800 |
| H.i | 0.096376 | 0.012581 | 0.059100 | 0.123200 |
| H.l | 0.105408 | 0.018152 | 0.052300 | 0.178300 |
| H.n | 0.111909 | 0.015098 | 0.047600 | 0.175100 |
| H.o | 0.082097 | 0.018589 | 0.032500 | 0.145900 |
| H.period | 0.112054 | 0.023187 | 0.018000 | 0.182000 |
| H.t | 0.058093 | 0.011082 | 0.026400 | 0.097100 |
| DD.Shift.r.o | 0.229252 | 0.065656 | 0.140700 | 0.622700 |
| DD.a.n | 0.081702 | 0.042366 | 0.002200 | 0.518600 |
| DD.e.five | 0.280265 | 0.178989 | 0.126600 | 1.113400 |
| DD.five.Shift.r | 0.360496 | 0.098401 | 0.243300 | 0.836600 |
| DD.i.e | 0.096076 | 0.073449 | 0.026900 | 0.940200 |
| DD.l.Return | 0.247800 | 0.092959 | 0.124700 | 0.803700 |
| DD.n.l | 0.085837 | 0.053079 | 0.008700 | 0.696800 |
| DD.o.a | 0.096783 | 0.038586 | 0.006200 | 0.355100 |
| DD.period.t | 0.206323 | 0.110596 | 0.099900 | 1.314300 |
| DD.t.i | 0.091546 | 0.024063 | 0.035900 | 0.193900 |
| UD.Shift.r.o | 0.174233 | 0.064083 | 0.085500 | 0.566200 |
| UD.a.n | -0.029804 | 0.045082 | -0.121300 | 0.388200 |
| UD.e.five | 0.221628 | 0.175824 | 0.088100 | 1.036100 |
| UD.five.Shift.r | 0.289592 | 0.095914 | 0.185700 | 0.751700 |
| UD.i.e | -0.000300 | 0.073346 | -0.068700 | 0.843100 |
| UD.l.Return | 0.142391 | 0.087669 | 0.057900 | 0.693200 |
| UD.n.l | -0.026071 | 0.050579 | -0.063100 | 0.579900 |
| UD.o.a | 0.014687 | 0.032054 | -0.075100 | 0.232400 |
| UD.period.t | 0.094269 | 0.105336 | -0.009100 | 1.170000 |
| UD.t.i | 0.033453 | 0.022939 | -0.010800 | 0.124000 |

### s054 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.099315 | 0.020594 | 0.025600 | 0.163800 |
| H.Shift.r | 0.081213 | 0.011831 | 0.014500 | 0.111100 |
| H.a | 0.104408 | 0.016522 | 0.012400 | 0.135900 |
| H.e | 0.099630 | 0.015271 | 0.041200 | 0.139800 |
| H.five | 0.074270 | 0.011642 | 0.005800 | 0.108900 |
| H.i | 0.095932 | 0.015295 | 0.039900 | 0.130800 |
| H.l | 0.102220 | 0.023820 | 0.063600 | 0.184400 |
| H.n | 0.096473 | 0.036509 | 0.013500 | 0.203400 |
| H.o | 0.082464 | 0.014020 | 0.013500 | 0.124500 |
| H.period | 0.095910 | 0.015321 | 0.027700 | 0.142700 |
| H.t | 0.081299 | 0.007917 | 0.049600 | 0.104700 |
| DD.Shift.r.o | 0.156052 | 0.095549 | 0.093500 | 0.808700 |
| DD.a.n | 0.126992 | 0.102004 | 0.030700 | 0.798300 |
| DD.e.five | 0.309466 | 0.139861 | 0.197800 | 1.081500 |
| DD.five.Shift.r | 0.369102 | 0.102043 | 0.254900 | 0.879600 |
| DD.i.e | 0.117324 | 0.043605 | 0.056800 | 0.652200 |
| DD.l.Return | 0.281777 | 0.125771 | 0.134700 | 1.070600 |
| DD.n.l | 0.197174 | 0.155896 | 0.010000 | 1.565100 |
| DD.o.a | 0.150097 | 0.082056 | 0.082400 | 1.099200 |
| DD.period.t | 0.332838 | 0.137627 | 0.212300 | 1.436900 |
| DD.t.i | 0.116872 | 0.051953 | 0.066000 | 0.706000 |
| UD.Shift.r.o | 0.074838 | 0.097921 | 0.007500 | 0.719000 |
| UD.a.n | 0.022583 | 0.106545 | -0.052200 | 0.700700 |
| UD.e.five | 0.209837 | 0.145982 | 0.092000 | 1.040300 |
| UD.five.Shift.r | 0.294832 | 0.101734 | 0.209800 | 0.807600 |
| UD.i.e | 0.021392 | 0.045211 | -0.027500 | 0.563600 |
| UD.l.Return | 0.179556 | 0.121724 | 0.046600 | 0.902300 |
| UD.n.l | 0.100701 | 0.167577 | -0.103700 | 1.460400 |
| UD.o.a | 0.067634 | 0.084770 | 0.007000 | 1.028000 |
| UD.period.t | 0.236927 | 0.137102 | 0.121000 | 1.334300 |
| UD.t.i | 0.035573 | 0.052179 | -0.009300 | 0.628700 |

### s055 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.105615 | 0.020973 | 0.038100 | 0.226300 |
| H.Shift.r | 0.116344 | 0.009970 | 0.075500 | 0.143600 |
| H.a | 0.086359 | 0.013112 | 0.049700 | 0.131600 |
| H.e | 0.103511 | 0.026857 | 0.040600 | 0.188000 |
| H.five | 0.102676 | 0.022464 | 0.002100 | 0.146100 |
| H.i | 0.083243 | 0.008756 | 0.051500 | 0.107600 |
| H.l | 0.094595 | 0.012440 | 0.064100 | 0.131900 |
| H.n | 0.064462 | 0.008025 | 0.024800 | 0.110000 |
| H.o | 0.083806 | 0.014287 | 0.023800 | 0.139600 |
| H.period | 0.095491 | 0.011141 | 0.058600 | 0.124000 |
| H.t | 0.102476 | 0.017184 | 0.015100 | 0.157200 |
| DD.Shift.r.o | 0.110406 | 0.024818 | 0.072300 | 0.235000 |
| DD.a.n | 0.070143 | 0.045645 | 0.006900 | 0.397200 |
| DD.e.five | 0.235293 | 0.149831 | 0.092300 | 0.965900 |
| DD.five.Shift.r | 0.255428 | 0.086101 | 0.170000 | 0.719500 |
| DD.i.e | 0.085931 | 0.046967 | 0.044900 | 0.373700 |
| DD.l.Return | 0.177718 | 0.040433 | 0.115400 | 0.468500 |
| DD.n.l | 0.180595 | 0.019195 | 0.157700 | 0.335500 |
| DD.o.a | 0.116713 | 0.033962 | 0.048900 | 0.459300 |
| DD.period.t | 0.150391 | 0.066295 | 0.080100 | 0.570100 |
| DD.t.i | 0.116979 | 0.033432 | 0.040700 | 0.522400 |
| UD.Shift.r.o | -0.005938 | 0.024592 | -0.038300 | 0.117600 |
| UD.a.n | -0.016217 | 0.045932 | -0.077900 | 0.297500 |
| UD.e.five | 0.131782 | 0.157778 | -0.045400 | 0.849000 |
| UD.five.Shift.r | 0.152752 | 0.083010 | 0.086800 | 0.581500 |
| UD.i.e | 0.002688 | 0.048993 | -0.052800 | 0.318300 |
| UD.l.Return | 0.083122 | 0.046493 | -0.004400 | 0.394100 |
| UD.n.l | 0.116133 | 0.018618 | 0.097000 | 0.296100 |
| UD.o.a | 0.032907 | 0.034425 | -0.041100 | 0.373800 |
| UD.period.t | 0.054900 | 0.069284 | -0.016500 | 0.479300 |
| UD.t.i | 0.014502 | 0.031393 | -0.023800 | 0.406100 |

### s056 (n=200 enrollment rows, BACKGROUND)

| Column | Mean (s) | Std (s) | Min (s) | Max (s) |
|--------|----------|---------|---------|----------|
| H.Return | 0.111749 | 0.018431 | 0.039100 | 0.170400 |
| H.Shift.r | 0.104339 | 0.019529 | 0.013500 | 0.156900 |
| H.a | 0.101565 | 0.014852 | 0.047000 | 0.164600 |
| H.e | 0.085402 | 0.018464 | 0.037000 | 0.173600 |
| H.five | 0.103127 | 0.021103 | 0.015900 | 0.187500 |
| H.i | 0.083856 | 0.020332 | 0.043300 | 0.153000 |
| H.l | 0.101161 | 0.021054 | 0.063600 | 0.179900 |
| H.n | 0.065356 | 0.011657 | 0.033300 | 0.119000 |
| H.o | 0.098073 | 0.013909 | 0.062800 | 0.138000 |
| H.period | 0.093794 | 0.017642 | 0.058800 | 0.153000 |
| H.t | 0.075799 | 0.013301 | 0.038800 | 0.112900 |
| DD.Shift.r.o | 0.228391 | 0.078817 | 0.112400 | 0.566000 |
| DD.a.n | 0.131274 | 0.046744 | 0.076000 | 0.555200 |
| DD.e.five | 0.328234 | 0.171036 | 0.134300 | 0.922400 |
| DD.five.Shift.r | 0.365114 | 0.151106 | 0.225200 | 1.290500 |
| DD.i.e | 0.092609 | 0.057325 | 0.033800 | 0.488000 |
| DD.l.Return | 0.183699 | 0.101596 | 0.095600 | 0.800300 |
| DD.n.l | 0.177590 | 0.068480 | 0.128400 | 0.847900 |
| DD.o.a | 0.125509 | 0.091280 | 0.063900 | 0.865900 |
| DD.period.t | 0.184621 | 0.102017 | 0.101200 | 1.341400 |
| DD.t.i | 0.145993 | 0.047201 | 0.087300 | 0.549600 |
| UD.Shift.r.o | 0.124053 | 0.080065 | 0.013200 | 0.499500 |
| UD.a.n | 0.029709 | 0.045099 | -0.033000 | 0.457600 |
| UD.e.five | 0.242833 | 0.170552 | 0.071500 | 0.825100 |
| UD.five.Shift.r | 0.261987 | 0.149604 | 0.149900 | 1.159700 |
| UD.i.e | 0.008753 | 0.059455 | -0.056700 | 0.410400 |
| UD.l.Return | 0.082537 | 0.101583 | -0.016000 | 0.671600 |
| UD.n.l | 0.112233 | 0.066293 | 0.058300 | 0.758700 |
| UD.o.a | 0.027436 | 0.086540 | -0.036600 | 0.760100 |
| UD.period.t | 0.090827 | 0.100900 | 0.002800 | 1.258300 |
| UD.t.i | 0.070194 | 0.045958 | 0.002700 | 0.452300 |

