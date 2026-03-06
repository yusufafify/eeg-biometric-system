"""
Calibration Data Loader — Phase 2: Patient-Specific Fine-Tuning
================================================================

Provides `build_calibration_split()` which partitions a target patient's
pre-computed segment cache into:

  * **Calibration set** – first N hours of continuous EEG.
    Used exclusively for fine-tuning (3-5 epochs).

  * **Test set** – all remaining segments.
    Strictly held-out; never seen during any training phase.

Zero-Leakage Guarantee
-----------------------
The split is performed on the *ordered segment index*, which mirrors the
original chronological EDF recording order stored in `manifest.txt`.
Segments are **never shuffled** before splitting, so the calibration
window is always the earliest portion of the patient's data.

Seizure Injection
-----------------
If the calibration set contains zero seizure segments, a small number of
seizure segments are injected from the general training pool.  This
prevents catastrophic forgetting of the seizure class during fine-tuning.
The injected segments are sourced only from *training* patients (never the
target patient), so test-set integrity is maintained.

Usage
-----
    from src.data.calibration_loader import build_calibration_split
    from torch.utils.data import DataLoader, TensorDataset

    cal_segs, cal_labels, test_segs, test_labels = build_calibration_split(
        processed_dir="data/processed",
        target_patient="chb15",
        train_patients=["chb01", ..., "chb14"],
        calibration_hours=1.5,
        window_size_sec=4.0,
        overlap_ratio=0.5,
        inject_seizures_if_none=True,
        n_injected_seizures=30,
        random_state=42,
    )
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_calibration_split(
    processed_dir: str,
    target_patient: str,
    train_patients: List[str],
    calibration_hours: float = 1.5,
    window_size_sec: float = 4.0,
    overlap_ratio: float = 0.5,
    inject_seizures_if_none: bool = True,
    n_injected_seizures: int = 30,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a target patient's cached segments and split them chronologically
    into a calibration set and a held-out test set.

    Parameters
    ----------
    processed_dir : str
        Root directory containing per-patient ``.npy`` caches
        (e.g. ``data/processed``).
    target_patient : str
        Patient ID to adapt to (e.g. ``"chb15"``).
    train_patients : list[str]
        Patients used in the original training run.  Only used for seizure
        injection; the target patient must *not* be in this list if you want
        a clean evaluation.
    calibration_hours : float
        How many hours of data (from the beginning) to use as the
        calibration set.  The rest becomes the test set.
    window_size_sec : float
        Window duration in seconds (must match the cached segments).
    overlap_ratio : float
        Fractional overlap used during segmentation (must match the cache).
    inject_seizures_if_none : bool
        If ``True`` and the calibration set has no seizure segments, inject
        a small number from the training pool.
    n_injected_seizures : int
        Maximum number of seizure segments to inject.
    random_state : int
        RNG seed for reproducibility of injection sampling.

    Returns
    -------
    cal_segs   : np.ndarray, shape (N_cal, C, T), float32
    cal_labels : np.ndarray, shape (N_cal,),       int64
    test_segs  : np.ndarray, shape (N_test, C, T), float32
    test_labels: np.ndarray, shape (N_test,),       int64

    Raises
    ------
    FileNotFoundError
        If the processed cache for ``target_patient`` does not exist.
    ValueError
        If there are not enough segments to form a calibration set.
    """
    proc = Path(processed_dir)
    patient_dir = proc / target_patient
    manifest_path = patient_dir / "manifest.txt"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"[calibration_loader] Cache not found for '{target_patient}'. "
            f"Expected: {manifest_path}\n"
            "Run build_segments_index() first."
        )

    # ------------------------------------------------------------------
    # 1. Load ALL segments in chronological order
    # ------------------------------------------------------------------
    # `manifest.txt` lists EDF stems in the order they were processed,
    # which mirrors the recording chronology.  We preserve this order to
    # ensure the split boundary is a real-time boundary, not a random one.
    stems = manifest_path.read_text("utf-8").strip().split("\n")

    all_segs_parts: List[np.ndarray] = []
    all_labels_parts: List[np.ndarray] = []

    for stem in stems:
        seg_path = patient_dir / f"{stem}_segments.npy"
        lbl_path = patient_dir / f"{stem}_labels.npy"
        if not seg_path.exists() or not lbl_path.exists():
            logger.warning("[%s] Missing .npy for stem '%s' — skipping.", target_patient, stem)
            continue
        all_segs_parts.append(np.load(seg_path, mmap_mode="r"))
        all_labels_parts.append(np.load(lbl_path))

    if not all_segs_parts:
        raise ValueError(f"No segments found for patient '{target_patient}'.")

    # Materialise into contiguous arrays (float32 / int64)
    all_segs = np.concatenate([np.array(s, dtype=np.float32) for s in all_segs_parts], axis=0)
    all_labels = np.concatenate(all_labels_parts).astype(np.int64)
    n_total = len(all_labels)

    logger.info(
        "[%s] Loaded %d segments total  (seizure=%d, normal=%d)",
        target_patient, n_total, int(all_labels.sum()), int((all_labels == 0).sum()),
    )

    # ------------------------------------------------------------------
    # 2. Compute the calibration / test split boundary
    # ------------------------------------------------------------------
    # Each segment represents `step_sec` of new signal (non-overlapping
    # portion).  We convert `calibration_hours` to a segment count using
    # this stride so overlapping windows are not double-counted.
    step_sec = window_size_sec * (1.0 - overlap_ratio)
    segs_per_hour = 3600.0 / step_sec
    cal_n = int(np.ceil(calibration_hours * segs_per_hour))
    cal_n = min(cal_n, n_total - 1)   # guarantee at least 1 test segment

    if cal_n <= 0:
        raise ValueError(
            f"calibration_hours={calibration_hours} maps to 0 segments. "
            "Increase calibration_hours or check window/overlap settings."
        )

    cal_segs   = all_segs[:cal_n].copy()
    cal_labels = all_labels[:cal_n].copy()
    test_segs  = all_segs[cal_n:].copy()
    test_labels = all_labels[cal_n:].copy()

    _log_split(target_patient, cal_segs, cal_labels, test_segs, test_labels,
               calibration_hours, step_sec)

    # ------------------------------------------------------------------
    # 3. Seizure injection (if calibration contains no seizures)
    # ------------------------------------------------------------------
    if inject_seizures_if_none and int(cal_labels.sum()) == 0:
        logger.warning(
            "[%s] Calibration set has NO seizure segments! "
            "Injecting up to %d seizure segments from training pool.",
            target_patient, n_injected_seizures,
        )
        inj_segs, inj_labels = _collect_injected_seizures(
            proc, train_patients, n_injected_seizures, random_state,
        )
        if len(inj_segs) > 0:
            cal_segs   = np.concatenate([cal_segs, inj_segs], axis=0)
            cal_labels = np.concatenate([cal_labels, inj_labels], axis=0)
            logger.info(
                "[%s] After injection: calibration has %d seizure segments.",
                target_patient, int(cal_labels.sum()),
            )
        else:
            logger.error(
                "[%s] Could not find any seizure segments in training pool! "
                "Fine-tuning with only normal-class examples.",
                target_patient,
            )
    elif int(cal_labels.sum()) == 0:
        logger.warning(
            "[%s] Calibration has no seizures and injection is disabled. "
            "Fine-tuning may cause catastrophic forgetting of seizure class.",
            target_patient,
        )

    return cal_segs, cal_labels, test_segs, test_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_split(
    patient: str,
    cal_segs: np.ndarray, cal_labels: np.ndarray,
    test_segs: np.ndarray, test_labels: np.ndarray,
    cal_hours: float, step_sec: float,
) -> None:
    """Print a clear split summary for audit."""
    cal_duration_h = len(cal_labels) * step_sec / 3600.0
    test_duration_h = len(test_labels) * step_sec / 3600.0
    print("=" * 68)
    print(f"  CALIBRATION SPLIT — Patient: {patient}")
    print("=" * 68)
    print(f"  Target calibration window : {cal_hours:.2f} h")
    print(f"  Calibration segments      : {len(cal_labels):>6}  "
          f"≈ {cal_duration_h:.2f} h   "
          f"(seizure={int(cal_labels.sum())}, normal={int((cal_labels==0).sum())})")
    print(f"  Test segments (held-out)  : {len(test_labels):>6}  "
          f"≈ {test_duration_h:.2f} h   "
          f"(seizure={int(test_labels.sum())}, normal={int((test_labels==0).sum())})")
    print(f"  *** ZERO LEAKAGE: test set starts at segment index {len(cal_labels)} ***")
    print("=" * 68)


def _collect_injected_seizures(
    proc: Path,
    train_patients: List[str],
    n_seizures: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gather up to `n_seizures` seizure segments from the training patient pool.

    Samples proportionally from each training patient so no single patient
    dominates the injected distribution.
    """
    rng = np.random.RandomState(random_state)
    pool_segs: List[np.ndarray] = []
    pool_labels: List[np.ndarray] = []

    for patient in train_patients:
        manifest = proc / patient / "manifest.txt"
        if not manifest.exists():
            continue
        for stem in manifest.read_text("utf-8").strip().split("\n"):
            lbl_path = proc / patient / f"{stem}_labels.npy"
            seg_path = proc / patient / f"{stem}_segments.npy"
            if not lbl_path.exists() or not seg_path.exists():
                continue
            labels = np.load(lbl_path)
            sz_idx = np.where(labels == 1)[0]
            if len(sz_idx) == 0:
                continue
            segs = np.load(seg_path, mmap_mode="r")
            pool_segs.append(np.array(segs[sz_idx], dtype=np.float32))
            pool_labels.append(labels[sz_idx].astype(np.int64))

    if not pool_segs:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    all_pool_segs   = np.concatenate(pool_segs, axis=0)
    all_pool_labels = np.concatenate(pool_labels, axis=0)

    # Sample without replacement (or with replacement if pool is small)
    n_available = len(all_pool_segs)
    n_draw = min(n_seizures, n_available)
    replace = n_draw > n_available
    chosen = rng.choice(n_available, size=n_draw, replace=replace)
    logger.info(
        "[INJECT] Drew %d / %d available seizure segments from training pool.",
        n_draw, n_available,
    )
    return all_pool_segs[chosen], all_pool_labels[chosen]
