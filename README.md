# ZuCo EEG preprocessing

This folder contains a single script, `preprocess_zuco.py`, that builds `zuco_merged_whiten_norm.df`: the merged ZuCo 1 + ZuCo 2 sentence-level EEG corpus after spectral whitening and robust z-score.

No other code files are required. MATLAB recordings and label CSVs are **data** you supply; they are not included here.

## Output: `zuco_merged_whiten_norm.df`

Pandas pickle. One row per sentence × subject trial after merging labels.

| Column | Type / shape | Description |
|--------|----------------|-------------|
| `eeg` | `np.ndarray (1280, 128)` float32 | Time × channels at 128 Hz (10 s pad). 104 EEG channels + 24 zero-pad channels. Whitened then robust z-scored. |
| `mask` | `np.ndarray (1280,)` int8 | `1` = valid time step, `0` = padding. |
| `input text` | `str` | Typo-corrected sentence (merge key). |
| `raw label` | `str` or NaN | Sentiment (task1), relation (task2/3), or NaN for non-control NR. |
| `length` | `int` | Word count of `input text`. |
| `text uid` | `int` | Sentence ID from `pd.factorize(input text)`. |
| `dataset` | `str` | `ZuCo1` or `ZuCo2`. |
| `task` | `str` | `task1` (SR), `task2` (NR), `task3` (TSR). |
| `subject` | `str` | e.g. `ZAB` (ZuCo1), `YAC` (ZuCo2). |
| `phase` | `str` | `train` / `val` / `test` (sentence-independent, seed 42). |

Expected size after a complete run: **23,446** rows — train **16,085**, val **2,628**, test **4,733**.

EEG is stored as `(time=1280, channels=128)`. Both signal transforms run on the last axis.

## Data you must provide

1. **ZuCo 1 MATLAB** under `--zuco1` (default: `../zuco_1`):

   - `task1- SR/Matlab files/`
   - `task2 - NR/Matlab files/`
   - `task3 - TSR/Matlab files/`

2. **ZuCo 2 MATLAB** under `--zuco2` (default: `../zuco_2`):

   - `task1 - NR/Matlab files/`
   - `task2 - TSR/Matlab files/`

3. **ZuCo 2 sentence CSVs** at `{zuco2}/task_materials/`:

   - `nr_{1..7}.csv`, `nr_{1..7}_control_questions.csv`
   - `tsr_{1..7}.csv`

4. **ZuCo 1 revised label CSVs** under `--revised-csv` (default: `../baselines/SemKey-main/preprocess/resource/revised_csv`):

   - `sentiment_labels_task1.csv`
   - `relations_labels_task2.csv`
   - `relations_labels_task3.csv`

ZuCo 1.0: Hollenstein et al. 2018. ZuCo 2.0: Hollenstein et al. 2020.

## Sequential steps

### 0. Install dependencies

```bash
pip install numpy scipy h5py pandas scikit-learn rich
```

Use a machine with roughly **32 GB RAM**. Peak usage is high because merged EEG pickles are ~14 GB.

### 1. Place the datasets

Download ZuCo 1 and ZuCo 2 and lay out folders as above. Point `--revised-csv` at the three ZuCo1 label CSVs.

### 2. Run the script

From this `github/` directory:

```bash
python preprocess_zuco.py --zuco1 ../zuco_1 --zuco2 ../zuco_2 --revised-csv ../baselines/SemKey-main/preprocess/resource/revised_csv --output ../preprocessed_data
```

Defaults match those paths when the repository root is one level above this folder. The script writes to `--output` and **skips a step if its output already exists**. Delete a pickle to re-run that step.

### 3. What the script does internally

**Step 1 — EEG load.** Read `sentenceData.rawData` from each MATLAB file (ZuCo1 via `scipy.io.loadmat`, ZuCo2 via `h5py`). Drop trials with NaN/Inf, non-2-D arrays, a non-zero last channel, or duration outside `[0.5 s, 10 s]` at 500 Hz. Drop the last all-zero channel (104 of 105 kept). Resample 500 Hz → 128 Hz (`scipy.signal.resample_poly` along time). Zero-pad to `(128, 1280)`, transpose to `(1280, 128)`, build a binary `mask`. Writes `zuco_eeg_128ch_1280len.df` (~22,335 rows). Intermediate per-task pickles live in `{output}/_eeg_tmp/` and may be deleted after a successful run.

**Step 2 — Labels.** Load ZuCo1 sentiment/relation CSVs and ZuCo2 NR/TSR materials. Apply the 24-entry typo table. Add `length` (whitespace word count) and `text uid`. Writes `zuco_label_input_text.df` and `.csv`.

**Step 3 — Merge and split.** Typo-correct EEG `text`. Inner join on `(text == input text, dataset, task)`. Split unique `text uid` values 70 / 10 / 20 with `random_state=42` (val is 12.5 % of the non-test pool so that val is 10 % of all sentences). Writes `zuco_merged.df` (23,446 rows).

**Step 4 — Whitening then robust z-score.** For each `eeg` row, in this order:

Spectral whitening (α = 0.95), last axis:

```
y[..., 0] = x[..., 0]
y[..., t] = x[..., t] - 0.95 * x[..., t-1]    for t >= 1
```

Robust z-score, last axis, zeros treated as padding:

```
valid = (x != 0)
y = (x - mean(x[valid])) / (std(x[valid]) + 1e-6)
y = clip(y, -10, +10)
y[~valid] = 0
```

Writes `zuco_merged_whiten_norm.df`. This file applies **both** transforms.

### 4. Confirm the run

The script prints smoke-test lines `[PASS] Step 1` … `[PASS] Step 4` and:

```
=== Preprocessing complete ===
```

Load the corpus with `pandas.read_pickle`.

## Intermediate files

| File | Typical size | Role |
|------|----------------|------|
| `zuco_eeg_128ch_1280len.df` | ~14 GB | Step 1 EEG only |
| `zuco_label_input_text.df` / `.csv` | < 1 MB | Step 2 labels |
| `zuco_merged.df` | ~14 GB | Step 3, no signal transforms |
| `zuco_merged_whiten_norm.df` | ~14 GB | Step 4 (training corpus) |
| `_eeg_tmp/*.pkl` | ~14 GB total | Per-task cache; deletable after Step 1 |

## License / data

ZuCo recordings and task materials are released by the original authors; obtain them from the official ZuCo distributions. This script only transforms those recordings into a padded, whitened, z-scored table.
