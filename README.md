# EEG-to-Text Probe and Faithfulness Notebooks

Self-contained Jupyter notebooks for two frozen-LLM EEG→text studies. Neither notebook imports project-local code; run them independently after building the whitened ZuCo pickle.

| File | Role |
| --- | --- |
| [`probe.ipynb`](probe.ipynb) | How much of generation is a **language prior**? Train an EEG encoder, then generate under `gt_text`, clean `eeg`, and mixed noise (`noise50`, `noise100`) with word prefills `{0, 1, 3, 5}`. |
| [`eeg_faith.ipynb`](eeg_faith.ipynb) | How **faithful** is the EEG memory to an oracle text encoder? Train EEG only, then compare `oracle_text` vs `eeg` with a tuned lens, Spearman, overlap / Fréchet, and linear probes. |
| [`preprocess_zuco.py`](preprocess_zuco.py) | Merge ZuCo 1 + ZuCo 2 sentence EEG into `zuco_merged_whiten_norm.df`. |
| [`requirements.txt`](requirements.txt) | Python dependencies for the notebooks. |

## Requirements

- Python 3.10+
- A CUDA GPU is strongly recommended (default `batch_size=72`, 100 epochs)
- Hugging Face access (the notebooks default `HF_ENDPOINT` to `https://hf-mirror.com` if unset)
- [ModelScope](https://www.modelscope.cn/) for **BART-large** weights

```bash
pip install -r requirements.txt
```

Preprocessing also needs `h5py` (ZuCo 2 MATLAB v7.3) and `rich` (progress bars):

```bash
pip install h5py rich
```

`eeg_faith.ipynb` also needs `tuned-lens` (installed automatically on first import if missing).

## 1. Download ZuCo

MATLAB recordings and task materials are **not** included here. Obtain them from the original authors:

- **ZuCo 1** (12 subjects, three reading tasks): [https://osf.io/q3zws/overview](https://osf.io/q3zws/overview)
- **ZuCo 2** (18 subjects, NR + TSR): [https://osf.io/2urht/overview](https://osf.io/2urht/overview)

Folder names must match exactly, including spaces:

```
zuco_1/
  task1- SR/Matlab files/*.mat
  task2 - NR/Matlab files/*.mat
  task3 - TSR/Matlab files/*.mat
  task_materials/          # sentiment + relation label CSVs
zuco_2/
  task1 - NR/Matlab files/*.mat
  task2 - TSR/Matlab files/*.mat
  task_materials/          # nr_*.csv, tsr_*.csv
```

ZuCo 1 labels live in OSF `task_materials`: `sentiment_labels_task1.csv`, `relations_labels_task2.csv`, `relations_labels_task3.csv`. Pass that directory as `--revised-csv`.

## 2. Preprocess

Pass **explicit paths**. Script defaults resolve two levels above this folder (`_REPO_ROOT = parent.parent`), which is easy to miss if you treat this directory as a standalone repo.

```bash
python preprocess_zuco.py \
  --zuco1 /path/to/zuco_1 \
  --zuco2 /path/to/zuco_2 \
  --revised-csv /path/to/zuco_1/task_materials \
  --output ./preprocessed_data
```

Four sequential steps. An existing output pickle is skipped (resume). Delete a file to re-run that step.

1. Load sentence EEG, resample 500 Hz → 128 Hz, pad to `(1280, 128)`.
2. Load sentence labels, apply typo corrections, assign text UIDs.
3. Inner-merge EEG with labels; sentence-independent 70 / 10 / 20 split.
4. Spectral whitening, then robust z-score on each EEG row.

| Output | Contents |
| --- | --- |
| `zuco_eeg_128ch_1280len.df` | Resampled EEG, one row per sentence × subject |
| `zuco_label_input_text.df` | Sentence labels (CSV sidecar as well) |
| `zuco_merged.df` | Merged EEG + labels + `phase` |
| `zuco_merged_whiten_norm.df` | Merged corpus after whitening and z-score (**notebooks load this**) |

Expected merged size: **23,446** rows (train 16,085 / val 2,628 / test 4,733).

Notebook columns: `eeg` `(1280, 128)`, `mask` `(1280,)`, `input text`, `text uid`, `phase` (`train` / `val` / `test`), `dataset`, `task`, `subject`.

## 3. Run the notebooks

Open [`probe.ipynb`](probe.ipynb) or [`eeg_faith.ipynb`](eeg_faith.ipynb) in Jupyter or VS Code / Cursor and run **top to bottom** from a fresh kernel.

If the pickle is not on a fallback path (`./preprocessed_data/zuco_merged_whiten_norm.df` or the AutoDL locations in cell 1), set it explicitly:

```python
CONFIG['data_path'] = './preprocessed_data/zuco_merged_whiten_norm.df'
```

### Smoke first (recommended)

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

### Useful knobs

| Key | Notebook | Meaning |
| --- | --- | --- |
| `seeds` | probe | EEG / noise seeds; `gt_text` always runs once (`n=1`) |
| `prefill_ns` | probe | Word prefills; `0` is labeled `bos` |
| `epochs` / `patience` | both | E2E train length and early stopping |
| `batch_size` | both | Default `72`; lower if you OOM |
| `output_dir` | both | See [Outputs](#outputs) |
| `model_cache_dir` | both | Hugging Face / ModelScope cache |

## Training objective

EEG training and checkpoint selection use the same teacher-forced composite:

`0.5 CLIP + 0.5 teacher-forced AR + 0.7 masked commitment`

Full-target greedy-rollout CE is display-only.

Decoder histories:

- **BART:** `decoder_start → BOS → optional instruction → lexical target → EOS`
- **T5:** `decoder_start → optional instruction → lexical target → EOS`

In `probe`, native `gt_text` generation uses start-only prefixes (Hugging Face forced-BOS). In `eeg_faith`, both `oracle_text` and `eeg` use the same instruction-only prefix.

## Outputs

**probe** (`outputs/probe/seed_{seed}/{llm_key}/{data_key}/`):

- Prediction CSVs (raw decode + token-boundary continuation)
- EEG checkpoints and training curves
- Shared-fit before/after t-SNE figures

Aggregate BLEU / SBERT tables are printed and plotted, not written as a summary CSV.

**eeg_faith** (`outputs/eeg_faith/{llm_key}/`):

- EEG checkpoint
- Tuned-lens weights
- Layer / overlap / probe figures

Metric tables are displayed in the notebook, not written as CSV/JSON.

## Hardware

A full two-LLM run at the default settings is a long GPU job (100 epochs, patience 5–10, batch 72). Start with `run_smoke_only=True`, then a single LLM and fewer epochs if you are checking the pipeline.

## License

Code in this folder is MIT. ZuCo recordings and task materials remain the original authors’ release; this repo only transforms those recordings into a padded, whitened, z-scored table.
