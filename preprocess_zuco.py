#!/usr/bin/env python
"""
ZuCo 1 and 2 EEG preprocessing.

Four sequential steps:

  1. Load sentence-level EEG, resample 500 Hz to 128 Hz, pad to (1280, 128).
  2. Load sentence labels, apply typo corrections, assign text UIDs.
  3. Inner-merge EEG with labels; sentence-independent 70 / 10 / 20 split.
  4. Spectral whitening then robust z-score on each EEG row.

Outputs (under --output):

  zuco_eeg_128ch_1280len.df     resampled EEG, one row per sentence x subject
  zuco_label_input_text.df      sentence labels
  zuco_label_input_text.csv     CSV copy of the label table
  zuco_merged.df                merged EEG + labels + phase
  zuco_merged_whiten_norm.df    merged corpus after whitening and z-score

Existing outputs are skipped (resume). Delete an output to re-run that step.
"""

from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import rich.progress as rp
import scipy.io
import scipy.signal
from rich.console import Console
from sklearn.model_selection import train_test_split

_CONSOLE = Console(
    highlight=False, force_terminal=False, force_jupyter=False, legacy_windows=False
)

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent

SRC_SAMPLE_RATE: int = 500
TGT_SAMPLE_RATE: int = 128
TGT_MAX_LEN: int = 1280
TGT_WIDTH: int = 128

TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.10
SPLIT_SEED: int = 42

ZUCO1_SUBJECTS: List[str] = [
    "ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN",
    "ZJS", "ZKB", "ZKH", "ZKW", "ZMG", "ZPH",
]
ZUCO1_TASKS: Dict[str, Tuple[str, int]] = {
    "task1- SR": ("task1", 400),
    "task2 - NR": ("task2", 300),
    "task3 - TSR": ("task3", 407),
}

ZUCO2_SUBJECTS: List[str] = [
    "YAC", "YAG", "YAK", "YDG", "YDR", "YFR",
    "YFS", "YHS", "YIS", "YLS", "YMD", "YMS",
    "YRH", "YRK", "YRP", "YSD", "YSL", "YTL",
]
# On-disk "task1 - NR" uses internal key task2 (Normal Reading).
# On-disk "task2 - TSR" uses internal key task3 (Task-Specific Reading).
ZUCO2_TASKS: Dict[str, Tuple[str, int]] = {
    "task1 - NR": ("task2", 349),
    "task2 - TSR": ("task3", 390),
}

MERGED_LABEL_COLS: List[str] = [
    "input text",
    "raw label",
    "length",
    "text uid",
    "dataset",
    "task",
]

TYPOBOOK: Dict[str, str] = {
    "emp11111ty": "empty",
    "film.1": "film.",
    "–": "-",
    "’s": "'s",
    "�s": "'s",
    "`s": "'s",
    "Maria": "Marić",
    "1Universidad": "Universidad",
    "1902—19": "1902 - 19",
    "Wuerttemberg": "Württemberg",
    "long -time": "long-time",
    "Jose": "José",
    "Bucher": "Bôcher",
    "1839 ? May": "1839 - May",
    "G�n�ration": "Generation",
    "Bragança": "Bragana",
    "1837?October": "1837 - October",
    "nVera-Ellen": "Vera-Ellen",
    "write Ethics": "wrote Ethics",
    "Adams-Onis": "Adams-Onís",
    "(40 km?)": "(40 km²)",
    "(40 km˝)": "(40 km²)",
    " (IPA: /?g?nz?b?g/) ": " ",
    '""Canes""': '"Canes"',
}


def spectral_whitening(eeg_data: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """
    First-order pre-emphasis along the last axis to flatten 1/f noise.

    Formula: y[..., 0] = x[..., 0]; y[..., t] = x[..., t] - alpha * x[..., t-1].

    Args:
        eeg_data: EEG array, shape ``(n_time, n_channels)`` or
            ``(batch, n_time, n_channels)``.
        alpha: Pre-emphasis weight in ``[0, 1]``. Default ``0.95``.

    Returns:
        Whitened array, same shape as ``eeg_data``, dtype float32.
    """
    eeg_data = eeg_data.astype(np.float32)
    whitened = np.zeros_like(eeg_data)
    whitened[..., 0] = eeg_data[..., 0]
    whitened[..., 1:] = eeg_data[..., 1:] - alpha * eeg_data[..., :-1]
    return whitened


def robust_normalize_padded(
    eeg_data: np.ndarray,
    axis: int = -1,
    clip_value: Optional[float] = 10.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Z-score along ``axis``, ignoring zeros, then restore padding zeros.

    Args:
        eeg_data: EEG array, typically shape ``(n_time, n_channels)``.
        axis: Axis for mean and std. Default ``-1`` (channel axis).
        clip_value: Symmetric clip bound after z-score. ``None`` disables clipping.
        epsilon: Added to std to avoid division by zero.

    Returns:
        Normalized float32 array of the same shape. Positions that were zero
        in the input are zero in the output.
    """
    arr = eeg_data.astype(np.float32, copy=False)
    valid = arr != 0.0

    n_valid = valid.sum(axis=axis, keepdims=True).astype(np.float32)
    n_valid = np.maximum(n_valid, 1.0)

    mean = np.where(valid, arr, 0.0).sum(axis=axis, keepdims=True) / n_valid
    diff = np.where(valid, arr - mean, 0.0)
    std = np.sqrt((diff * diff).sum(axis=axis, keepdims=True) / n_valid)

    normalized = np.where(valid, (arr - mean) / (std + epsilon), 0.0)
    if clip_value is not None:
        normalized = np.clip(normalized, -clip_value, clip_value)
    normalized[~valid] = 0.0
    return normalized


def revise_typo(text: str, typobook: Dict[str, str] = TYPOBOOK) -> str:
    """
    Replace listed source fragments in a sentence with their corrections.

    Args:
        text: Raw sentence string.
        typobook: Mapping from substring to replacement. Default ``TYPOBOOK``.

    Returns:
        Corrected string. Unchanged if no listed fragment occurs.
    """
    for src, tgt in typobook.items():
        if src in text:
            text = text.replace(src, tgt)
    return text


def process_eeg_row(
    eeg_raw: np.ndarray,
    text_raw: str,
    dataset: str,
    task: str,
    subject: str,
) -> Tuple[dict, bool]:
    """
    Validate one sentence EEG recording and resample it to the storage layout.

    Drops NaN/Inf, non-2-D arrays, a non-zero last channel, and durations
    outside ``[0.5 s, 10 s]`` at 500 Hz. Drops the last (all-zero) channel,
    resamples 500 Hz to 128 Hz along time, zero-pads to ``(128, 1280)``,
    and transposes to ``(1280, 128)``.

    Args:
        eeg_raw: Raw EEG, shape ``(n_channels, n_time)``, typically
            ``(105, T)`` at 500 Hz.
        text_raw: Sentence string stored with the recording.
        dataset: ``'ZuCo1'`` or ``'ZuCo2'``.
        task: ``'task1'``, ``'task2'``, or ``'task3'``.
        subject: Subject identifier, e.g. ``'ZAB'``.

    Returns:
        Tuple ``(record, ok)``. If ``ok`` is True, ``record`` has:

        - ``eeg``: ``np.ndarray``, shape ``(1280, 128)``, dtype float32
        - ``mask``: ``np.ndarray``, shape ``(1280,)``, dtype int8 (1=valid)
        - ``text``, ``dataset``, ``task``, ``subject``: ``str``

        If ``ok`` is False, ``record`` is ``{}``.
    """
    if not np.all(np.isfinite(eeg_raw)):
        return {}, False

    if eeg_raw.ndim != 2:
        warnings.warn(
            f"{dataset} {subject} {task}: unexpected EEG shape {eeg_raw.shape} — dropping"
        )
        return {}, False

    if np.any(eeg_raw[-1]):
        warnings.warn(
            f"{dataset} {subject} {task}: last channel is not all-zero — dropping"
        )
        return {}, False
    eeg104 = eeg_raw[:-1, :]

    _, len_raw = eeg104.shape
    if len_raw < 0.5 * SRC_SAMPLE_RATE or len_raw > 10 * SRC_SAMPLE_RATE:
        return {}, False

    eeg = scipy.signal.resample_poly(
        eeg104, TGT_SAMPLE_RATE, SRC_SAMPLE_RATE, axis=1
    )
    len_new = eeg.shape[1]

    n_ch = eeg104.shape[0]
    eeg = np.pad(
        eeg,
        ((0, TGT_WIDTH - n_ch), (0, TGT_MAX_LEN - len_new)),
        mode="constant",
        constant_values=0,
    )

    mask = np.zeros(TGT_MAX_LEN, dtype=np.int8)
    mask[:len_new] = 1

    record = {
        "eeg": eeg.T.astype(np.float32),
        "mask": mask,
        "text": text_raw,
        "dataset": dataset,
        "task": task,
        "subject": subject,
    }
    return record, True


def load_one_task(
    dataset_name: str,
    folder_name: str,
    task_key: str,
    expected_n: int,
    mat_dir: Path,
    subject_keys: List[str],
    prog: "rp.Progress",
) -> pd.DataFrame:
    """
    Load every MATLAB recording in one task folder.

    ZuCo1 files are MATLAB < v7.3; ZuCo2 files are HDF5 v7.3. Each sentence
    is passed through :func:`process_eeg_row`.

    Args:
        dataset_name: ``'ZuCo1'`` or ``'ZuCo2'``.
        folder_name: On-disk task folder name (may contain spaces).
        task_key: ``'task1'``, ``'task2'``, or ``'task3'``.
        expected_n: Expected sentence count per subject file.
        mat_dir: Directory containing one MATLAB file per subject.
        subject_keys: Allowed subject identifiers.
        prog: Active progress bar.

    Returns:
        DataFrame with columns ``eeg``, ``mask``, ``text``, ``dataset``,
        ``task``, ``subject``. ``eeg`` is ``(1280, 128)`` float32; ``mask``
        is ``(1280,)`` int8.
    """
    mat_paths = sorted(mat_dir.glob("*.mat"))
    assert len(mat_paths) == len(subject_keys), (
        f"{folder_name}: expected {len(subject_keys)} MATLAB files, "
        f"found {len(mat_paths)}"
    )
    records: List[dict] = []
    n_rec = n_drop = 0
    color = "cyan" if dataset_name == "ZuCo1" else "magenta"
    task_bar = prog.add_task(
        f"[{color}]{dataset_name} {task_key}",
        total=expected_n * len(subject_keys),
        n_rec=n_rec,
        n_drop=n_drop,
    )

    for mat_path in mat_paths:
        subject_key = mat_path.stem.replace("results", "").split("_")[0]
        assert subject_key in subject_keys, (
            f"Unrecognised subject: {subject_key!r} in {mat_path.name}"
        )

        if dataset_name == "ZuCo1":
            sentence_data = scipy.io.loadmat(
                str(mat_path), squeeze_me=True, struct_as_record=False
            )["sentenceData"]
            n = len(sentence_data)
            if n != expected_n:
                warnings.warn(
                    f"{mat_path.name}: got {n} sentences (expected {expected_n})"
                )
            for j in range(n):
                try:
                    eeg_raw: np.ndarray = sentence_data[j].rawData
                    text_raw: str = str(sentence_data[j].content)
                except Exception as exc:
                    warnings.warn(
                        f"{dataset_name} {subject_key} {task_key} j={j}: "
                        f"read error — {exc}"
                    )
                    n_drop += 1
                    prog.update(task_bar, advance=1, n_rec=n_rec, n_drop=n_drop)
                    continue
                row, ok = process_eeg_row(
                    eeg_raw, text_raw, dataset_name, task_key, subject_key
                )
                if ok:
                    records.append(row)
                    n_rec += 1
                else:
                    n_drop += 1
                prog.update(task_bar, advance=1, n_rec=n_rec, n_drop=n_drop)
            del sentence_data

        else:
            with h5py.File(str(mat_path), "r") as mat:
                n = len(mat["sentenceData"]["rawData"])
                if n != expected_n:
                    warnings.warn(
                        f"{mat_path.name}: got {n} sentences (expected {expected_n})"
                    )
                for j in range(n):
                    try:
                        eeg_ref = mat["sentenceData"]["rawData"][j][0]
                        # HDF5 stores arrays in C-order; transpose restores
                        # (channels, time).
                        eeg_raw = mat[eeg_ref][:].T.astype(np.float32)
                        text_ref = mat["sentenceData"]["content"][j][0]
                        text_raw = "".join(
                            chr(int(k)) for k in mat[text_ref][:].squeeze()
                        )
                    except Exception as exc:
                        warnings.warn(
                            f"{dataset_name} {subject_key} {task_key} j={j}: "
                            f"read error — {exc}"
                        )
                        n_drop += 1
                        prog.update(task_bar, advance=1, n_rec=n_rec, n_drop=n_drop)
                        continue
                    row, ok = process_eeg_row(
                        eeg_raw, text_raw, dataset_name, task_key, subject_key
                    )
                    if ok:
                        records.append(row)
                        n_rec += 1
                    else:
                        n_drop += 1
                    prog.update(task_bar, advance=1, n_rec=n_rec, n_drop=n_drop)

    return pd.DataFrame(records)


def step1_load_eeg_mat(
    zuco1_dir: Path,
    zuco2_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Load and resample all ZuCo 1 and ZuCo 2 sentence EEG recordings.

    Processes one task at a time, caches each task DataFrame under
    ``output_dir / _eeg_tmp``, then concatenates. Peak in-memory EEG is
    one task (~3 GB) rather than the full corpus.

    Args:
        zuco1_dir: ZuCo 1 root. Task folders contain a ``Matlab files`` subfolder.
        zuco2_dir: ZuCo 2 root. Same layout as ZuCo 1.
        output_dir: Directory for per-task cache pickles.

    Returns:
        DataFrame, one row per valid sentence x subject:

        - ``eeg``: ``np.ndarray (1280, 128)`` float32
        - ``mask``: ``np.ndarray (1280,)`` int8
        - ``text``: ``str``
        - ``dataset``: ``'ZuCo1'`` or ``'ZuCo2'``
        - ``task``: ``'task1'`` / ``'task2'`` / ``'task3'``
        - ``subject``: ``str``
    """
    tmp_dir = output_dir / "_eeg_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    task_pickle_paths: List[Path] = []

    with rp.Progress(
        rp.SpinnerColumn(),
        rp.TextColumn("[progress.description]{task.description}"),
        rp.BarColumn(),
        rp.TaskProgressColumn(),
        "|",
        rp.TextColumn("Rec:{task.fields[n_rec]}  Drop:{task.fields[n_drop]}"),
        "|",
        rp.TimeElapsedColumn(),
        console=_CONSOLE,
    ) as prog:

        for folder_name, (task_key, expected_n) in ZUCO1_TASKS.items():
            mat_dir = zuco1_dir / folder_name / "Matlab files"
            tmp_path = tmp_dir / f"ZuCo1_{task_key}.pkl"
            if tmp_path.exists():
                print(f"  [cache] {tmp_path.name} already exists — skipping", flush=True)
                task_pickle_paths.append(tmp_path)
                continue
            df_task = load_one_task(
                "ZuCo1", folder_name, task_key, expected_n, mat_dir,
                ZUCO1_SUBJECTS, prog,
            )
            pd.to_pickle(df_task, tmp_path)
            task_pickle_paths.append(tmp_path)
            del df_task
            gc.collect()

        for folder_name, (task_key, expected_n) in ZUCO2_TASKS.items():
            mat_dir = zuco2_dir / folder_name / "Matlab files"
            tmp_path = tmp_dir / f"ZuCo2_{task_key}.pkl"
            if tmp_path.exists():
                print(f"  [cache] {tmp_path.name} already exists — skipping", flush=True)
                task_pickle_paths.append(tmp_path)
                continue
            df_task = load_one_task(
                "ZuCo2", folder_name, task_key, expected_n, mat_dir,
                ZUCO2_SUBJECTS, prog,
            )
            pd.to_pickle(df_task, tmp_path)
            task_pickle_paths.append(tmp_path)
            del df_task
            gc.collect()

    print(
        f"\n[Step 1] Concatenating {len(task_pickle_paths)} task DataFrames...",
        flush=True,
    )
    all_dfs: List[pd.DataFrame] = []
    for p in task_pickle_paths:
        chunk = pd.read_pickle(p)
        all_dfs.append(chunk)
        if len(all_dfs) >= 3:
            merged_so_far = pd.concat(all_dfs, ignore_index=True)
            all_dfs = [merged_so_far]
            gc.collect()
    df = pd.concat(all_dfs, ignore_index=True) if len(all_dfs) > 1 else all_dfs[0]
    print(
        f"[Step 1] {len(df):,} EEG records loaded "
        f"(ZuCo1: {(df['dataset']=='ZuCo1').sum():,}  "
        f"ZuCo2: {(df['dataset']=='ZuCo2').sum():,})"
    )
    return df


def merge_qa(question: object, answer: object) -> object:
    """
    Join a ZuCo2 NR control question with its correct answer.

    Args:
        question: Control question string, or NaN for non-control sentences.
        answer: Correct answer string.

    Returns:
        Combined label string, or ``np.nan`` when ``question`` is NaN.

    Raises:
        ValueError: If a non-NaN question does not end in ``'...'`` or ``'?'``.
    """
    if pd.isna(question):
        return np.nan
    q, a = str(question), str(answer)
    if q.endswith("..."):
        return q.replace("...", " " + a)
    if q.endswith("?"):
        return q + " " + a
    raise ValueError(f"Unexpected control question format: {q!r}")


def load_zuco2_tsr(file_index: int, materials_dir: Path) -> pd.DataFrame:
    """
    Load one ZuCo2 Task-Specific Reading label table.

    Each file has one non-CONTROL relation type; every row (including CONTROL)
    is assigned that type.

    Args:
        file_index: Integer ``1``–``7`` selecting the TSR materials table.
        materials_dir: ZuCo2 ``task_materials`` directory.

    Returns:
        DataFrame with columns ``raw text`` (str), ``raw label`` (str),
        ``dataset`` (``'ZuCo2'``), ``task`` (``'task3'``).
    """
    valid_labels = {
        "AWARD", "EDUCATION", "EMPLOYER", "FOUNDER", "JOB_TITLE",
        "NATIONALITY", "POLITICAL_AFFILIATION", "VISITED", "WIFE", "CONTROL",
    }
    df = pd.read_csv(
        materials_dir / f"tsr_{file_index}.csv",
        sep=";", encoding="utf-8", header=None,
        names=["paragraph_id", "sentence_id", "sentence", "label"],
        dtype=str,
    ).rename(columns={"sentence": "raw text", "label": "raw label"})

    for lbl in df["raw label"].dropna():
        assert lbl in valid_labels, f"tsr_{file_index}: unexpected label {lbl!r}"

    unique_types = [lab for lab in df["raw label"].unique() if lab != "CONTROL"]
    assert len(unique_types) == 1, (
        f"tsr_{file_index}: expected 1 non-CONTROL relation type, got {unique_types}"
    )
    relation_type = unique_types[0]
    df["raw label"] = [relation_type] * len(df)
    df = df.reindex(columns=["raw text", "raw label"])
    df["dataset"] = "ZuCo2"
    df["task"] = "task3"
    return df


def step2_load_labels(revised_csv_dir: Path, zuco2_materials_dir: Path) -> pd.DataFrame:
    """
    Load and unify sentence labels for ZuCo 1 and ZuCo 2.

    ZuCo1: sentiment (task1) and relation (task2, task3) CSVs.
    ZuCo2: NR sentences plus control questions (task2); TSR tables (task3).

    Adds typo-corrected ``input text``, word-count ``length``, and integer
    ``text uid`` from ``pd.factorize`` on ``input text``.

    Args:
        revised_csv_dir: Directory of ZuCo1 label CSVs
            (``sentiment_labels_task1.csv``, ``relations_labels_task2.csv``,
            ``relations_labels_task3.csv``).
        zuco2_materials_dir: ZuCo2 ``task_materials`` directory.

    Returns:
        DataFrame with columns ``raw text`` (str), ``input text`` (str),
        ``raw label`` (str or NaN), ``length`` (int), ``text uid`` (int),
        ``dataset`` (str), ``task`` (str).
    """
    frames: List[pd.DataFrame] = []

    df11 = pd.read_csv(
        revised_csv_dir / "sentiment_labels_task1.csv",
        sep=";", header=0, skiprows=[1], encoding="utf-8",
        dtype={"sentence": str, "control": str, "sentiment_label": str},
    ).rename(columns={"sentence": "raw text", "sentiment_label": "raw label"})
    df11 = df11.reindex(columns=["raw text", "raw label"])
    df11["dataset"] = "ZuCo1"
    df11["task"] = "task1"
    frames.append(df11)

    df12 = pd.read_csv(
        revised_csv_dir / "relations_labels_task2.csv",
        sep=",", header=0, encoding="utf-8",
        dtype={"sentence": str, "control": str, "relation_types": str},
    ).rename(columns={"sentence": "raw text", "relation_types": "raw label"})
    df12 = df12.reindex(columns=["raw text", "raw label"])
    df12["dataset"] = "ZuCo1"
    df12["task"] = "task2"
    frames.append(df12)

    df13 = pd.read_csv(
        revised_csv_dir / "relations_labels_task3.csv",
        sep=";", header=0, encoding="utf-8",
        dtype={"sentence": str, "relation-type": str},
    ).rename(columns={"sentence": "raw text", "relation-type": "raw label"})
    df13 = df13.reindex(columns=["raw text", "raw label"])
    df13["dataset"] = "ZuCo1"
    df13["task"] = "task3"
    frames.append(df13)

    nr_frames: List[pd.DataFrame] = []
    for i in range(1, 8):
        df_sent = pd.read_csv(
            zuco2_materials_dir / f"nr_{i}.csv",
            sep=";", encoding="utf-8", header=None,
            names=["paragraph_id", "sentence_id", "sentence", "control"],
            dtype=str,
        )
        df_ctrl = pd.read_csv(
            zuco2_materials_dir / f"nr_{i}_control_questions.csv",
            sep=";", encoding="utf-8", header=0,
            dtype=str,
        )
        n_control = (df_sent["control"] == "CONTROL").sum()
        assert n_control == len(df_ctrl), (
            f"nr_{i}: {n_control} CONTROL sentences but {len(df_ctrl)} control questions"
        )
        df_merged_nr = pd.merge(
            df_sent, df_ctrl, how="left", on=["paragraph_id", "sentence_id"]
        )
        nr_frames.append(df_merged_nr)

    df22 = pd.concat(nr_frames, ignore_index=True)
    df22["raw label"] = [
        merge_qa(df22["control_question"].iloc[i], df22["correct_answer"].iloc[i])
        for i in range(len(df22))
    ]
    df22 = df22.rename(columns={"sentence": "raw text"})
    df22 = df22.reindex(columns=["raw text", "raw label"])
    df22["dataset"] = "ZuCo2"
    df22["task"] = "task2"
    frames.append(df22)

    tsr_frames = [load_zuco2_tsr(i, zuco2_materials_dir) for i in range(1, 8)]
    df23 = pd.concat(tsr_frames, ignore_index=True)
    frames.append(df23)

    df = pd.concat(frames, ignore_index=True)
    df["input text"] = df["raw text"].apply(lambda t: revise_typo(str(t)))
    df["length"] = df["input text"].apply(lambda t: len(str(t).split()))
    uids, _ = pd.factorize(df["input text"])
    df["text uid"] = uids.tolist()

    print(
        f"\n[Step 2] {len(df):,} label rows loaded, "
        f"{df['text uid'].nunique():,} unique sentences"
    )
    return df


def step3_merge_and_split(
    df_eeg: pd.DataFrame,
    df_labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-merge EEG with labels and assign sentence-independent phases.

    Typo-corrects EEG ``text``, joins on ``(text == input text, dataset, task)``,
    then splits unique ``text uid`` values 70 % train / 10 % val / 20 % test
    with ``random_state=42``. All trials of one sentence share one phase.

    Args:
        df_eeg: Step 1 DataFrame. Requires ``eeg`` ``(1280, 128)``, ``mask``
            ``(1280,)``, ``text``, ``dataset``, ``task``, ``subject``.
        df_labels: Step 2 DataFrame. Requires ``MERGED_LABEL_COLS``.

    Returns:
        Merged DataFrame: EEG columns (``text`` dropped), label columns, and
        ``phase`` in ``{'train', 'val', 'test'}``.
    """
    df_eeg = df_eeg.copy()
    df_eeg["text"] = df_eeg["text"].apply(lambda t: revise_typo(str(t)))

    df_merged = pd.merge(
        df_eeg,
        df_labels[MERGED_LABEL_COLS],
        left_on=["text", "dataset", "task"],
        right_on=["input text", "dataset", "task"],
        how="inner",
    )
    df_merged = df_merged.drop(columns=["text"])

    print(
        f"\n[Step 3] {len(df_eeg):,} EEG rows × {len(df_labels):,} label rows "
        f"→ {len(df_merged):,} merged rows  "
        f"({len(df_eeg) - len(df_merged):,} EEG rows with no matching label)"
    )

    unique_uids = df_merged["text uid"].unique()
    train_uids, test_uids = train_test_split(
        unique_uids, test_size=TEST_SIZE, random_state=SPLIT_SEED
    )
    val_split_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    train_uids, val_uids = train_test_split(
        train_uids, test_size=val_split_ratio, random_state=SPLIT_SEED
    )
    train_set, val_set = set(train_uids), set(val_uids)

    def _assign_phase(uid: int) -> str:
        """Map one text UID to ``'train'``, ``'val'``, or ``'test'``."""
        if uid in train_set:
            return "train"
        if uid in val_set:
            return "val"
        return "test"

    df_merged["phase"] = df_merged["text uid"].map(_assign_phase)
    counts = df_merged["phase"].value_counts()
    print(
        f"  train={counts.get('train', 0):,}  "
        f"val={counts.get('val', 0):,}  "
        f"test={counts.get('test', 0):,}"
    )
    return df_merged


def step4_apply_signal_transforms(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Replace each EEG row with whitening followed by robust z-score.

    Order: ``robust_normalize_padded(spectral_whitening(eeg))``. Updates the
    ``eeg`` column in place to avoid duplicating the ~14 GB frame.

    Args:
        df_merged: Step 3 DataFrame. ``eeg`` entries are ``(1280, 128)``
            float32. Modified in place.

    Returns:
        The same DataFrame; ``eeg`` entries are whitened and z-scored
        ``(1280, 128)`` float32 arrays.
    """
    eeg_col = df_merged["eeg"].to_numpy().copy()

    with rp.Progress(
        rp.SpinnerColumn(),
        rp.TextColumn("[progress.description]{task.description}"),
        rp.BarColumn(),
        rp.TaskProgressColumn(),
        rp.TimeElapsedColumn(),
        console=_CONSOLE,
    ) as prog:
        task_id = prog.add_task(
            "[green]Spectral whitening + robust z-score…", total=len(eeg_col)
        )
        for i in range(len(eeg_col)):
            eeg_col[i] = robust_normalize_padded(spectral_whitening(eeg_col[i]))
            prog.advance(task_id)

    df_merged["eeg"] = eeg_col
    print(f"\n[Step 4] Signal transforms applied to {len(df_merged):,} rows")
    return df_merged


def run_smoke_tests(
    df_eeg: pd.DataFrame,
    df_labels: pd.DataFrame,
    df_merged: pd.DataFrame,
    df_transformed: pd.DataFrame,
) -> None:
    """
    Check shapes, required columns, split integrity, and finite EEG.

    Args:
        df_eeg: Frame with ``eeg`` ``(1280, 128)`` and ``mask`` ``(1280,)``.
        df_labels: Frame with ``input text``, ``text uid``, ``length``,
            ``dataset``, ``task``.
        df_merged: Frame with ``phase`` in ``{train, val, test}`` and one
            phase per ``text uid``.
        df_transformed: Frame whose ``eeg`` sample is finite ``(1280, 128)``.

    Returns:
        None.

    Raises:
        AssertionError: If a check fails.
    """
    print("\n--- Smoke tests ---")

    eeg_required = {"eeg", "mask", "dataset", "task", "subject"}
    assert eeg_required <= set(df_eeg.columns), (
        f"Missing EEG columns: {eeg_required - set(df_eeg.columns)}"
    )
    sample_eeg = df_eeg["eeg"].iloc[0]
    assert sample_eeg.shape == (TGT_MAX_LEN, TGT_WIDTH), (
        f"EEG shape: expected ({TGT_MAX_LEN}, {TGT_WIDTH}), got {sample_eeg.shape}"
    )
    assert sample_eeg.dtype == np.float32
    sample_mask = df_eeg["mask"].iloc[0]
    assert sample_mask.shape == (TGT_MAX_LEN,)
    assert set(np.unique(sample_mask)) <= {0, 1}
    assert set(df_eeg["dataset"].unique()) <= {"ZuCo1", "ZuCo2"}
    print(f"  [PASS] Step 1 — {len(df_eeg):,} rows, EEG shape {sample_eeg.shape}")

    for col in ("input text", "text uid", "length", "dataset", "task"):
        assert col in df_labels.columns, f"Missing label column: {col!r}"
    assert int(df_labels["length"].min()) >= 1, "Minimum sentence length is 0"
    print(
        f"  [PASS] Step 2 — {len(df_labels):,} rows, "
        f"{df_labels['text uid'].nunique():,} unique sentence UIDs"
    )

    assert {"train", "val", "test"} == set(df_merged["phase"].unique()), (
        f"Unexpected phases: {df_merged['phase'].unique()}"
    )
    uid_phases = df_merged.groupby("text uid")["phase"].nunique()
    multi_phase = uid_phases[uid_phases > 1]
    assert len(multi_phase) == 0, (
        f"{len(multi_phase)} UIDs span multiple phases — split is not sentence-independent"
    )
    print(
        f"  [PASS] Step 3 — {len(df_merged):,} merged rows, "
        f"sentence-independent split confirmed"
    )

    sample_xfm = df_transformed["eeg"].iloc[0]
    assert sample_xfm.shape == (TGT_MAX_LEN, TGT_WIDTH), (
        f"Transformed EEG shape: expected ({TGT_MAX_LEN}, {TGT_WIDTH}), got {sample_xfm.shape}"
    )
    assert np.all(np.isfinite(sample_xfm)), "NaN or Inf in transformed EEG sample"
    print(f"  [PASS] Step 4 — transformed EEG shape {sample_xfm.shape}, no NaN/Inf")

    print("--- All smoke tests passed ---\n")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line paths for ZuCo roots, ZuCo1 label CSVs, and output.

    Returns:
        Namespace with ``zuco1``, ``zuco2``, ``revised_csv``, and ``output``,
        each a ``pathlib.Path``.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess ZuCo 1 and 2 EEG into a merged whitened/normalized corpus."
    )
    parser.add_argument(
        "--zuco1",
        type=Path,
        default=_REPO_ROOT / "zuco_1",
        help="ZuCo 1 dataset root (task folders with Matlab files).",
    )
    parser.add_argument(
        "--zuco2",
        type=Path,
        default=_REPO_ROOT / "zuco_2",
        help="ZuCo 2 dataset root (task folders with Matlab files, plus task_materials).",
    )
    parser.add_argument(
        "--revised-csv",
        type=Path,
        default=(
            _REPO_ROOT
            / "baselines"
            / "SemKey-main"
            / "preprocess"
            / "resource"
            / "revised_csv"
        ),
        help="Directory of ZuCo1 sentiment and relation label CSVs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "preprocessed_data",
        help="Directory for pickle/CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run steps 1–4 and write outputs under ``--output``.

    Resumes from any output that already exists. After a full run, prints
    the four pickle paths and a CSV sidecar. Expected merged size is 23,446
    rows with phase counts train 16,085 / val 2,628 / test 4,733.

    Returns:
        None.
    """
    args = parse_args()
    zuco1_dir = args.zuco1
    zuco2_dir = args.zuco2
    revised_csv_dir = args.revised_csv
    output_dir = args.output
    zuco2_materials_dir = zuco2_dir / "task_materials"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n", flush=True)

    out_eeg = output_dir / "zuco_eeg_128ch_1280len.df"
    out_lbl = output_dir / "zuco_label_input_text.df"
    out_merged = output_dir / "zuco_merged.df"
    out_xfm = output_dir / "zuco_merged_whiten_norm.df"

    if out_eeg.exists():
        print(f"=== Step 1: Skipped (found {out_eeg.name}) ===", flush=True)
        df_eeg = None
    else:
        print("=== Step 1: Loading EEG MATLAB files ===", flush=True)
        df_eeg = step1_load_eeg_mat(zuco1_dir, zuco2_dir, output_dir)
        pd.to_pickle(df_eeg, out_eeg)
        print(f"Saved → {out_eeg}", flush=True)

    if out_lbl.exists():
        print(f"\n=== Step 2: Skipped (found {out_lbl.name}) ===", flush=True)
        df_labels = pd.read_pickle(out_lbl)
    else:
        print("\n=== Step 2: Loading sentence labels ===", flush=True)
        df_labels = step2_load_labels(revised_csv_dir, zuco2_materials_dir)
        pd.to_pickle(df_labels, out_lbl)
        df_labels.to_csv(output_dir / "zuco_label_input_text.csv", index=False)
        print(f"Saved → {out_lbl}  +  .csv", flush=True)

    if out_merged.exists():
        print(f"\n=== Step 3: Skipped (found {out_merged.name}) ===", flush=True)
        df_merged = None
    else:
        print("\n=== Step 3: Merging EEG + labels and splitting ===", flush=True)
        if df_eeg is None:
            df_eeg = pd.read_pickle(out_eeg)
        df_merged = step3_merge_and_split(df_eeg, df_labels)
        pd.to_pickle(df_merged, out_merged)
        print(f"Saved → {out_merged}", flush=True)

    del df_eeg
    gc.collect()

    if out_xfm.exists():
        print(f"\n=== Step 4: Skipped (found {out_xfm.name}) ===", flush=True)
        df_transformed = pd.read_pickle(out_xfm)
    else:
        print("\n=== Step 4: Offline spectral whitening + robust z-score ===", flush=True)
        if df_merged is None:
            df_merged = pd.read_pickle(out_merged)
        df_transformed = step4_apply_signal_transforms(df_merged)
        pd.to_pickle(df_transformed, out_xfm)
        print(f"Saved → {out_xfm}", flush=True)
        del df_merged
        gc.collect()

    run_smoke_tests(df_transformed, df_labels, df_transformed, df_transformed)

    print("=== Preprocessing complete ===")
    print(f"  {out_eeg}")
    print(f"  {out_lbl}  +  .csv")
    print(f"  {out_merged}")
    print(f"  {out_xfm}")


if __name__ == "__main__":
    main()
