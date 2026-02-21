"""
EEG Dataset Module for CHB-MIT Seizure Detection

Handles:
- Parsing seizure annotations from summary files
- Building preprocessed segment caches (.npy, per-EDF-file)
- Memory-mapped PyTorch Dataset with optional z-score normalization
- DataLoader creation with WeightedRandomSampler for class imbalance
- Subject-wise stratified train/val splitting
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.data.preprocess import load_eeg_data, segment_eeg, TARGET_CHANNELS

logger = logging.getLogger(__name__)


# =============================================================================
# Summary Parsing
# =============================================================================

def parse_summary(summary_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parse a CHB-MIT summary file to extract per-file seizure intervals.

    Maps summary filenames to disk filenames (handles + to %2B encoding).

    Args:
        summary_path: Path to the chbXX-summary.txt file

    Returns:
        Dictionary mapping disk_filename to list of (start_sec, end_sec) tuples.
        Files with no seizures map to an empty list.
    """
    summary_path_p = Path(summary_path)
    subject_dir = summary_path_p.parent

    file_records: Dict[str, List[Tuple[int, int]]] = {}
    current_file: Optional[str] = None
    starts: List[int] = []
    ends: List[int] = []

    def _save_current():
        nonlocal current_file, starts, ends
        if current_file is not None:
            file_records[current_file] = list(zip(starts, ends))
        starts.clear()
        ends.clear()

    with open(summary_path_p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("File Name:"):
                _save_current()
                summary_name = line.split(":", 1)[1].strip()
                disk_name = summary_name.replace("+", "%2B")

                if (subject_dir / disk_name).exists():
                    current_file = disk_name
                elif (subject_dir / summary_name).exists():
                    current_file = summary_name
                else:
                    current_file = disk_name
                    logger.warning("EDF not found on disk: %s", summary_name)

            elif "Seizure" in line and "Start Time" in line:
                match = re.search(r"(\d+)\s*seconds", line)
                if match:
                    starts.append(int(match.group(1)))

            elif "Seizure" in line and "End Time" in line:
                match = re.search(r"(\d+)\s*seconds", line)
                if match:
                    ends.append(int(match.group(1)))

    _save_current()
    return file_records


# =============================================================================
# Labeling
# =============================================================================

def _compute_label(
    window_start_sec: float,
    window_end_sec: float,
    seizure_intervals: List[Tuple[int, int]],
    threshold: float = 0.25,
) -> int:
    """
    Assign binary seizure label based on overlap ratio.

    Label 1 if >= *threshold* of the window overlaps a seizure interval.
    """
    window_length = window_end_sec - window_start_sec
    if window_length <= 0:
        return 0

    total_overlap = 0.0
    for sz_start, sz_end in seizure_intervals:
        overlap = max(0.0, min(window_end_sec, sz_end) - max(window_start_sec, sz_start))
        total_overlap += overlap

    return 1 if (total_overlap / window_length) >= threshold else 0


# =============================================================================
# Segment Index Builder (per-EDF-file caching)
# =============================================================================

def build_segments_index(
    data_dir: str,
    subjects: List[str],
    output_dir: str = "data/processed",
    window_size_sec: float = 4.0,
    overlap_ratio: float = 0.5,
    label_threshold: float = 0.25,
    target_channels: Optional[List[str]] = None,
    force_rebuild: bool = False,
    verbose: bool = False,
) -> None:
    """
    Preprocess EDF files and cache segments + labels as ``.npy`` files.

    Each EDF recording is saved to its own pair of files to keep peak
    memory usage low:

    * ``<stem>_segments.npy``  (float32, ``(N, C, T)``)
    * ``<stem>_labels.npy``    (int64,   ``(N,)``)

    A ``manifest.txt`` listing all stems is written per subject.
    """
    if target_channels is None:
        target_channels = list(TARGET_CHANNELS)

    data_dir_p = Path(data_dir)
    out_p = Path(output_dir)

    for subject in subjects:
        subj_dir = data_dir_p / subject
        cache_dir = out_p / subject
        manifest = cache_dir / "manifest.txt"

        if manifest.exists() and not force_rebuild:
            logger.info("[%s] Cache exists, skipping.", subject)
            continue

        summary = subj_dir / f"{subject}-summary.txt"
        if not summary.exists():
            logger.warning("[%s] No summary file, skipping.", subject)
            continue

        logger.info("[%s] Parsing summary...", subject)
        records = parse_summary(str(summary))

        cache_dir.mkdir(parents=True, exist_ok=True)
        stems: List[str] = []
        total_segs = total_sz = 0

        for fname, sz_intervals in records.items():
            edf_path = subj_dir / fname
            if not edf_path.exists() or fname.endswith(".tmp"):
                continue

            logger.info("  Loading %s ...", fname)
            try:
                data, _, sfreq = load_eeg_data(
                    str(edf_path), target_channels=target_channels, verbose=verbose,
                )
                overlap_sec = window_size_sec * overlap_ratio
                segs = segment_eeg(data, sfreq, window_size_sec, overlap_sec)

                stride_sec = window_size_sec - overlap_sec
                labels = np.array(
                    [_compute_label(i * stride_sec, i * stride_sec + window_size_sec,
                                    sz_intervals, label_threshold)
                     for i in range(segs.shape[0])],
                    dtype=np.int64,
                )

                stem = Path(fname).stem
                np.save(cache_dir / f"{stem}_segments.npy", segs.astype(np.float32))
                np.save(cache_dir / f"{stem}_labels.npy", labels)
                stems.append(stem)
                total_segs += len(labels)
                total_sz += int(labels.sum())
                del data, segs, labels

            except Exception as e:
                logger.error("  Error processing %s: %s", fname, e)

        if not stems:
            logger.warning("[%s] No segments extracted.", subject)
            continue

        manifest.write_text("\n".join(stems), encoding="utf-8")
        logger.info("[%s] %d segments (sz=%d, normal=%d) in %d files",
                    subject, total_segs, total_sz, total_segs - total_sz, len(stems))


# =============================================================================
# PyTorch Dataset (memory-mapped)
# =============================================================================

class EEGDataset(Dataset):
    """
    Dataset backed by memory-mapped ``.npy`` files.

    Segments are read on-demand via ``np.load(mmap_mode='r')`` so that
    the full dataset never needs to reside in RAM simultaneously.
    """

    def __init__(
        self,
        processed_dir: str,
        subjects: List[str],
        normalize: bool = True,
    ):
        self.processed_dir = Path(processed_dir)
        self.normalize = normalize

        self._seg_mmaps: List[np.ndarray] = []
        self._cum: List[int] = [0]
        lab_parts: List[np.ndarray] = []

        for subject in subjects:
            sd = self.processed_dir / subject
            mf = sd / "manifest.txt"
            if not mf.exists():
                raise FileNotFoundError(
                    f"Cache not found for {subject}. "
                    "Run build_segments_index first."
                )
            for stem in mf.read_text("utf-8").strip().split("\n"):
                seg_mm = np.load(sd / f"{stem}_segments.npy", mmap_mode="r")
                lab = np.load(sd / f"{stem}_labels.npy")
                self._seg_mmaps.append(seg_mm)
                lab_parts.append(lab)
                self._cum.append(self._cum[-1] + len(lab))

        self.labels = np.concatenate(lab_parts) if lab_parts else np.array([], dtype=np.int64)
        self._total = self._cum[-1]

        # Normalisation stats from a subsample (max 10 000 segments)
        if self.normalize and self._total > 0:
            rng = np.random.RandomState(0)
            n = min(10_000, self._total)
            idxs = rng.choice(self._total, size=n, replace=False)
            sample = np.stack([self._raw(i) for i in idxs])
            self.ch_mean = sample.mean(axis=(0, 2), keepdims=True).astype(np.float32)
            self.ch_std = sample.std(axis=(0, 2), keepdims=True).astype(np.float32)
            self.ch_std[self.ch_std == 0] = 1.0
            del sample

    def _locate(self, idx: int) -> Tuple[int, int]:
        lo, hi = 0, len(self._seg_mmaps) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cum[mid + 1] <= idx:
                lo = mid + 1
            else:
                hi = mid
        return lo, idx - self._cum[lo]

    def _raw(self, idx: int) -> np.ndarray:
        fi, li = self._locate(idx)
        return np.array(self._seg_mmaps[fi][li], dtype=np.float32)

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seg = self._raw(idx)
        if self.normalize:
            seg = (seg - self.ch_mean[0]) / self.ch_std[0]
        return torch.from_numpy(seg), int(self.labels[idx])


# =============================================================================
# DataLoader Factory
# =============================================================================

def create_dataloaders(
    data_dir: str,
    subjects: List[str],
    processed_dir: str = "data/processed",
    batch_size: int = 32,
    val_ratio: float = 0.2,
    random_state: int = 42,
    normalize: bool = True,
    num_workers: int = 0,
    window_size_sec: float = 4.0,
    overlap_ratio: float = 0.5,
    label_threshold: float = 0.25,
    force_rebuild: bool = False,
) -> Tuple[DataLoader, DataLoader, List[str], List[str]]:
    """
    Build segment cache and return train / val DataLoaders.

    * **Subject-wise split** prevents data leakage.
    * **Stratified** so validation always contains seizure subjects.
    * **WeightedRandomSampler** on train set to counter class imbalance.

    Returns:
        (train_loader, val_loader, train_subjects, val_subjects)
    """
    logger.info("Building segment cache (existing caches are reused)...")
    build_segments_index(
        data_dir=data_dir, subjects=subjects, output_dir=processed_dir,
        window_size_sec=window_size_sec, overlap_ratio=overlap_ratio,
        label_threshold=label_threshold, force_rebuild=force_rebuild,
    )

    proc = Path(processed_dir)
    available = [s for s in subjects if (proc / s / "manifest.txt").exists()]
    if len(available) < 2:
        raise ValueError(f"Need >=2 subjects for split, got {len(available)}.")

    # ------------------------------------------------------------------
    # Subject-level seizure-segment-proportional stratified split
    # ------------------------------------------------------------------
    # 1. Count seizure segments per subject
    subj_sz_count: Dict[str, int] = {}
    subj_total_count: Dict[str, int] = {}
    for subj in available:
        mf = proc / subj / "manifest.txt"
        stems = mf.read_text("utf-8").strip().split("\n")
        sz_total = 0
        seg_total = 0
        for stem in stems:
            labels = np.load(proc / subj / f"{stem}_labels.npy")
            sz_total += int(labels.sum())
            seg_total += len(labels)
        subj_sz_count[subj] = sz_total
        subj_total_count[subj] = seg_total

    total_seizure = sum(subj_sz_count.values())
    target_val_seizure = val_ratio * total_seizure

    logger.info("[SPLIT] Total seizure segments across all subjects: %d", total_seizure)
    logger.info("[SPLIT] Target val seizure segments (%.0f%%): %d",
                val_ratio * 100, int(target_val_seizure))

    # 2. Separate seizure-bearing vs normal-only subjects
    sz_subjects = [s for s in available if subj_sz_count[s] > 0]
    no_subjects = [s for s in available if subj_sz_count[s] == 0]

    # 3. Sort seizure subjects by count descending for greedy packing
    rng = np.random.RandomState(random_state)
    rng.shuffle(sz_subjects)  # randomise before stable sort for tie-breaking
    sz_subjects.sort(key=lambda s: subj_sz_count[s], reverse=True)

    # 4. Greedily assign seizure subjects to val until target is met
    val_subjects: List[str] = []
    train_subjects: List[str] = []
    val_sz_so_far = 0

    for subj in sz_subjects:
        if val_sz_so_far < target_val_seizure and len(val_subjects) < len(sz_subjects) - 1:
            # Keep at least 1 seizure subject in training
            val_subjects.append(subj)
            val_sz_so_far += subj_sz_count[subj]
        else:
            train_subjects.append(subj)

    # 5. Split normal-only subjects proportionally
    rng.shuffle(no_subjects)
    n_val_no = max(0, int(len(no_subjects) * val_ratio))
    val_subjects.extend(no_subjects[:n_val_no])
    train_subjects.extend(no_subjects[n_val_no:])

    # Log the split details
    val_sz_total = sum(subj_sz_count[s] for s in val_subjects)
    train_sz_total = sum(subj_sz_count[s] for s in train_subjects)
    logger.info("[SPLIT] Val seizure subjects & counts: %s",
                {s: subj_sz_count[s] for s in val_subjects if subj_sz_count[s] > 0})
    logger.info("[SPLIT] Val seizure segments: %d / %d (%.1f%%)",
                val_sz_total, total_seizure,
                100.0 * val_sz_total / max(total_seizure, 1))
    logger.info("[SPLIT] Train seizure segments: %d / %d (%.1f%%)",
                train_sz_total, total_seizure,
                100.0 * train_sz_total / max(total_seizure, 1))

    logger.info("Train subjects (%d): %s", len(train_subjects), train_subjects)
    logger.info("Val subjects   (%d): %s", len(val_subjects), val_subjects)

    # ------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------
    train_ds = EEGDataset(processed_dir, train_subjects, normalize=normalize)
    val_ds = EEGDataset(processed_dir, val_subjects, normalize=normalize)

    # WeightedRandomSampler for training ONLY
    counts = np.bincount(train_ds.labels, minlength=2)
    w = 1.0 / np.maximum(counts, 1).astype(np.float64)
    sampler = WeightedRandomSampler(
        torch.from_numpy(w[train_ds.labels]).double(), len(train_ds), replacement=True
    )

    logger.info("Train: %d samples (sz=%d, normal=%d)",
                len(train_ds), counts[1] if len(counts) > 1 else 0, counts[0])
    logger.info("Val:   %d samples (natural distribution, no sampler)", len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, train_subjects, val_subjects
