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

Both notebooks load a whitened ZuCo pickle named `zuco_merged_whiten_norm.df`. Place it at one of:

- `./autodl-tmp/preprocessed_data/zuco_merged_whiten_norm.df`
- `/root/autodl-tmp/preprocessed_data/zuco_merged_whiten_norm.df`
- `./preprocessed_data/zuco_merged_whiten_norm.df`

Or set `CONFIG['data_path']` in the first code cell to your path.

Required columns: `eeg` `(1280, 128)`, `mask` `(1280,)`, `input text`, `text uid`, `phase` (`train` / `val` / `test`), `dataset`, `task`, `subject`.

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
