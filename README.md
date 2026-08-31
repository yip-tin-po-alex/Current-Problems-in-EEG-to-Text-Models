# EEG-to-Text Probe and Faithfulness Notebooks

Self-contained Jupyter notebooks for two frozen-LLM EEG→text studies:

| Notebook | Question |
| --- | --- |
| [`probe.ipynb`](probe.ipynb) | How much of generation is a **language prior**? Train an EEG encoder, then generate under `gt_text`, clean `eeg`, and mixed noise (`noise50`, `noise100`) with word prefills `{0, 1, 3, 5}`. |
| [`eeg_faith.ipynb`](eeg_faith.ipynb) | How **faithful** is the EEG memory to an oracle text encoder? Train EEG only, then compare `oracle_text` vs `eeg` with a tuned lens, Spearman, overlap / Fréchet, and linear probes. |

Neither notebook imports project-local code. Run them independently.

## Requirements

- Python 3.10+
- A CUDA GPU is strongly recommended (default `batch_size=72`, 100 epochs)
- Hugging Face access (the notebooks default `HF_ENDPOINT` to `https://hf-mirror.com` if unset)
- [ModelScope](https://www.modelscope.cn/) for **BART-large** weights

```bash
pip install -r requirements.txt
```

`eeg_faith.ipynb` also needs `tuned-lens` (installed automatically on first import if missing).

## Data

Both notebooks load a whitened ZuCo pickle named `zuco_merged_whiten_norm.df`. 
Set `CONFIG['data_path']` in the first code cell to your path.

Required columns: `eeg` `(1280, 128)`, `mask` `(1280,)`, `input text`, `text uid`, `phase` (`train` / `val` / `test`), `dataset`, `task`, `subject`.

# ZuCo EEG preprocessing

A single script, `preprocess_zuco.py`, builds `zuco_merged_whiten_norm.df`: the merged ZuCo 1 + ZuCo 2 sentence-level EEG corpus after spectral whitening and robust z-score.

MATLAB recordings and label CSVs are **data** you supply; they are not included here.

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


## How to run

Open the notebook in Jupyter or VS Code / Cursor and run **top to bottom** from a fresh kernel.

Cell order (both notebooks):

1. **Config** — paths, loss weights, which LLMs to run
2. **Data pipeline** — pickle load, split checks, GLIM sampler
3. **EEG encoder** — SemKey Q-Merger on raw 1280 samples
4. **ProbeSystem** — frozen EncDec LLM + trainable EEG path
5. **Train / evaluate** — E2E training; probe also generates and plots
6. **Analyses** (`eeg_faith` only) — tuned lens, Spearman, overlap / Fréchet, linear probes
7. **Encoder smoke** — offline checks (no LLM download)
8. **Main** — per-LLM smokes, then the full experiment

### Smoke only (recommended first)

In the first code cell:

```python
CONFIG['run_smoke_only'] = True
```

This runs encoder + per-LLM smokes and skips corpus load / training.

### Choose models

```python
CONFIG['llm_keys_to_run'] = ['bart_large']          # or ['flan_t5_large']
# default is both: ['bart_large', 'flan_t5_large']  # probe
# default is both: ['flan_t5_large', 'bart_large']  # eeg_faith
```

BART is loaded from ModelScope (`AI-ModelScope/bart-large`). Flan-T5-large is loaded from the Hugging Face Hub / mirror.

### Other useful knobs

| Key | Notebook | Meaning |
| --- | --- | --- |
| `seeds` | probe | EEG / noise seeds; `gt_text` always runs once (`n=1`) |
| `prefill_ns` | probe | Word prefills; `0` is labeled `bos` |
| `epochs` / `patience` | both | E2E train length and early stopping |
| `batch_size` | both | Default `72`; lower if you OOM |
| `output_dir` | both | See below |
| `model_cache_dir` | both | Hugging Face / ModelScope cache |

## Outputs

**probe** (`outputs/probe/seed_{seed}/{llm_key}/{data_key}/`):

- Prediction CSVs (raw decode + token-boundary continuation)
- EEG checkpoints and training curves
- Shared-fit before/after t-SNE figures

Aggregate BLEU / SBERT tables are printed and plotted, not written as a summary CSV.

**eeg_faith** (`outputs/eeg_faith/{llm_key}/`):

- EEG checkpoint
- Tuned-lens weights
- Layer / overlap / probe **figures**

Metric tables are displayed in the notebook, not written as CSV/JSON.

## Training objective

EEG training and checkpoint selection use the same teacher-forced composite:

`0.5 CLIP + 0.5 teacher-forced AR + 0.7 masked commitment`

Full-target greedy-rollout CE is display-only.

Decoder histories:

- **BART:** `decoder_start → BOS → optional instruction → lexical target → EOS`
- **T5:** `decoder_start → optional instruction → lexical target → EOS`

In `probe`, native `gt_text` generation uses start-only prefixes (Hugging Face forced-BOS). In `eeg_faith`, both `oracle_text` and `eeg` use the same instruction-only prefix.

## Hardware note

A full two-LLM run at the default settings is a long GPU job (100 epochs, patience 5–10, batch 72). Start with `run_smoke_only=True`, then a single LLM and fewer epochs if you are checking the pipeline.
