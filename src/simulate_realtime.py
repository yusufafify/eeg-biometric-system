"""
Patient-Independent Real-Time Clinical Simulation & Threshold Calibration
=========================================================================

Simulates continuous real-time inference on the held-out test portion of a
target patient's EEG recording using the *original patient-independent*
``best_model.pth`` from Phase 1.

Normalization
-------------
Data is loaded through ``EEGDataset`` which applies the same Z-score
normalization (``ch_mean`` / ``ch_std``) that the model was trained with
in Phase 1.  This guarantees the feature-extraction backbone receives
activations in the exact distribution it learned on.

Zero-Leakage Guarantee
-----------------------
``EEGDataset`` loads the **entire** patient recording in chronological
order.  The first ``cal_n_segments`` segments (≈ calibration_hours) are
skipped during inference so they are never evaluated.  Only segments
beyond that boundary contribute to the clinical report.

Usage
-----
    python -m src.simulate_realtime \\
        --target-patient chb15 \\
        --calibration-hours 1.5 \\
        --threshold 0.50 0.75 0.85 0.90
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import EEGDataset
from src.models.cnn_lstm import CNN_LSTM_Classifier
from src.utils.config import load_config
from src.utils.metrics import (
    calculate_macro_f1,
    calculate_seizure_sensitivity,
    calculate_specificity,
    calculate_false_alarm_rate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _safe(fn, y_true, y_pred, **kwargs) -> float:
    try:
        return fn(y_true, y_pred, **kwargs)
    except (ValueError, ZeroDivisionError):
        return 0.0


# =============================================================================
# Threshold Calibration Engine
# =============================================================================

def apply_threshold(
    probs_seizure: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Apply a custom probability threshold to the seizure-class probability.

    Parameters
    ----------
    probs_seizure : (N,) array of P(class=1 | segment), in [0, 1]
    threshold     : Decision boundary.  Segments with P >= threshold are
                    classified as seizure (label=1).

    Returns
    -------
    preds : (N,) int array of binary predictions (0 or 1).
    """
    return (probs_seizure >= threshold).astype(np.int64)


# =============================================================================
# Streaming Inference Engine
# =============================================================================

def stream_inference(
    model: torch.nn.Module,
    test_loader: DataLoader,
    cal_n_segments: int,
    thresholds: List[float],
    step_sec: float,
    device: torch.device,
    print_every: int = 500,
) -> Dict[float, Dict]:
    """
    Stream test segments through the model in chronological order,
    skipping the first ``cal_n_segments`` to enforce the zero-leakage
    calibration / test boundary.

    Data arrives pre-normalized from ``EEGDataset`` (Z-score, matching
    Phase 1 training), so no additional scaling is applied here.

    Returns a dict mapping each threshold to its clinical metrics,
    plus an array of per-batch latencies.
    """
    model.eval()

    all_probs: List[float] = []        # P(seizure) per TEST segment
    all_labels: List[int] = []         # true labels for TEST segments
    latencies_ms: List[float] = []

    prev_pred = {t: 0 for t in thresholds}             # for onset detection
    onsets    = {t: [] for t in thresholds}             # (threshold -> [sec])

    abs_idx = 0                        # absolute segment counter across batches
    n_test  = 0                        # number of test segments processed

    for batch_inputs, batch_targets in test_loader:
        batch_size_actual = len(batch_targets)

        # ---------------------------------------------------------------
        # Determine which segments in this batch are past the cal boundary
        # ---------------------------------------------------------------
        batch_end_idx = abs_idx + batch_size_actual
        if batch_end_idx <= cal_n_segments:
            # Entire batch is within calibration → skip
            abs_idx = batch_end_idx
            continue

        if abs_idx < cal_n_segments:
            # Partial overlap — trim the calibration prefix off this batch
            skip = cal_n_segments - abs_idx
            batch_inputs  = batch_inputs[skip:]
            batch_targets = batch_targets[skip:]
            abs_idx = cal_n_segments
            batch_size_actual = len(batch_targets)

        # ---------------------------------------------------------------
        # Inference on test-only segments
        # ---------------------------------------------------------------
        x = batch_inputs.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            logits = model(x)                              # (B, 2)
            probs  = F.softmax(logits, dim=1)[:, 1]       # P(seizure), (B,)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0 / batch_size_actual)

        probs_np = probs.cpu().numpy()
        labels_np = batch_targets.numpy()
        all_probs.extend(probs_np.tolist())
        all_labels.extend(labels_np.tolist())

        # Live onset detection for each threshold
        for seg_offset, p_sei in enumerate(probs_np):
            test_seg_idx = n_test + seg_offset
            window_sec   = test_seg_idx * step_sec       # time into the TEST recording
            for thr in thresholds:
                pred = int(p_sei >= thr)
                if pred == 1 and prev_pred[thr] == 0:
                    onsets[thr].append(window_sec)
                    if thr == thresholds[0]:     # only print for primary thr
                        print(f"  *** SEIZURE DETECTED @ {format_time(window_sec)} "
                              f"[thr={thr:.2f}]  test segment {test_seg_idx + 1}")
                prev_pred[thr] = pred

        n_test += batch_size_actual
        abs_idx += batch_size_actual

        # Progress every N segments
        if (n_test % print_every < batch_size_actual) or abs_idx >= len(test_loader.dataset):
            elapsed_t = (n_test - 1) * step_sec
            avg_lat   = np.mean(latencies_ms[-max(1, len(latencies_ms) // 10):])
            print(f"  Segment [{n_test:>6}] | "
                  f"time: {format_time(elapsed_t)} | "
                  f"avg latency: {avg_lat:.2f} ms/seg")

    # ------------------------------------------------------------------
    # Compute clinical metrics at each threshold
    # ------------------------------------------------------------------
    probs_arr  = np.array(all_probs, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    # --- DEBOUNCE FILTER ---
    # Smooth probabilities over a 3-segment (12-second) rolling window
    # to eliminate isolated 4-second false alarms.
    from scipy.ndimage import median_filter
    probs_arr = median_filter(probs_arr, size=3)

    total_hours = n_test * step_sec / 3600.0

    results: Dict[float, Dict] = {}
    for thr in thresholds:
        preds   = apply_threshold(probs_arr, thr)
        y_true  = labels_arr

        tp = int(((y_true == 1) & (preds == 1)).sum())
        tn = int(((y_true == 0) & (preds == 0)).sum())
        fp = int(((y_true == 0) & (preds == 1)).sum())
        fn = int(((y_true == 1) & (preds == 0)).sum())

        sensitivity  = _safe(calculate_seizure_sensitivity, y_true, preds)
        specificity  = _safe(calculate_specificity,         y_true, preds)
        macro_f1     = _safe(calculate_macro_f1,            y_true, preds)
        fpr_per_hour = fp / max(total_hours, 1e-9)

        results[thr] = {
            "threshold"   : thr,
            "sensitivity" : sensitivity,
            "specificity" : specificity,
            "macro_f1"    : macro_f1,
            "fpr_per_hour": fpr_per_hour,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "n_onsets"    : len(onsets[thr]),
            "onsets_sec"  : onsets[thr],
            "total_hours" : total_hours,
        }

    return results, np.array(latencies_ms)


# =============================================================================
# Report Printer
# =============================================================================

def print_report(
    results: Dict[float, Dict],
    latencies_ms: np.ndarray,
    target_patient: str,
    n_test: int,
    n_true_seiz: int,
    n_true_norm: int,
    step_sec: float,
) -> None:
    total_hours = n_test * step_sec / 3600.0

    print()
    print("=" * 68)
    print("  REAL-TIME SIMULATION REPORT  —  Patient:", target_patient)
    print("=" * 68)
    print(f"  Test duration     : {format_time(int(total_hours * 3600))}  "
          f"({total_hours:.2f} h)")
    print(f"  Test segments     : {n_test:,}")
    print(f"  True seizure segs : {n_true_seiz:,}")
    print(f"  True normal segs  : {n_true_norm:,}")
    print()

    # Latency
    print("-" * 68)
    print("  INFERENCE LATENCY (per segment, batched)")
    print("-" * 68)
    if len(latencies_ms) > 0:
        print(f"    Mean    : {latencies_ms.mean():>8.3f} ms")
        print(f"    Median  : {np.percentile(latencies_ms, 50):>8.3f} ms")
        print(f"    P95     : {np.percentile(latencies_ms, 95):>8.3f} ms")
        print(f"    P99     : {np.percentile(latencies_ms, 99):>8.3f} ms")
    print()

    # Per-threshold table
    print("-" * 68)
    print("  CLINICAL METRICS vs DECISION THRESHOLD")
    print("-" * 68)
    header = (f"  {'Threshold':>10}  {'Macro-F1':>9}  {'Sensitivity':>11}  "
              f"{'Specificity':>11}  {'FPR/h':>7}  {'TP':>5}  {'FP':>5}  "
              f"{'FN':>5}  {'Onsets':>6}")
    print(header)
    print("  " + "-" * 65)
    for thr, m in sorted(results.items()):
        print(
            f"  {thr:>10.2f}  "
            f"{m['macro_f1']:>9.4f}  "
            f"{m['sensitivity']:>11.4f}  "
            f"{m['specificity']:>11.4f}  "
            f"{m['fpr_per_hour']:>7.2f}  "
            f"{m['tp']:>5}  {m['fp']:>5}  "
            f"{m['fn']:>5}  {m['n_onsets']:>6}"
        )
    print()

    # Seizure onset timestamps for primary threshold
    primary_thr = min(results.keys())
    onsets = results[primary_thr]["onsets_sec"]
    print(f"-" * 68)
    print(f"  SEIZURE ONSET EVENTS  (threshold={primary_thr:.2f})")
    print("-" * 68)
    if onsets:
        for i, t in enumerate(onsets):
            print(f"    [{i + 1:>3}]  {format_time(t)}  ({t:.1f} s into test recording)")
    else:
        print("    (no seizure onsets detected at this threshold)")
    print("=" * 68)


# =============================================================================
# Main
# =============================================================================

def main(args: argparse.Namespace) -> None:
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_patient = args.target_patient
    thresholds     = sorted(set(args.threshold))   # deduplicate & sort
    batch_size     = args.batch_size
    cal_hours      = args.calibration_hours

    step_sec = cfg.data.window_size_sec * (1.0 - cfg.data.overlap_ratio)

    print("=" * 68)
    print("  PHASE 2 — REAL-TIME CLINICAL SIMULATION")
    print("=" * 68)
    print(f"  Target patient   : {target_patient}")
    print(f"  Calibration hours: {cal_hours:.2f} h  (skipped for zero-leakage)")
    print(f"  Decision thresholds: {thresholds}")
    print(f"  Device           : {device}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load the full patient dataset via EEGDataset
    # ------------------------------------------------------------------
    # EEGDataset applies the same Z-score normalization used during
    # Phase 1 training, so the model receives correctly scaled inputs.
    print("[STEP 1] Loading patient data via EEGDataset (with Z-score norm) ...")
    test_ds = EEGDataset(
        processed_dir = cfg.data.processed_dir,
        subjects      = [target_patient],
        normalize     = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,           # MUST preserve chronological order
        num_workers = 0,
        pin_memory  = device.type == "cuda",
    )

    # Compute the calibration boundary in segments
    segs_per_hour   = 3600.0 / step_sec
    cal_n_segments  = int(np.ceil(cal_hours * segs_per_hour))
    n_total         = len(test_ds)
    n_test          = n_total - cal_n_segments

    # Count true labels in the test portion only
    test_labels     = test_ds.labels[cal_n_segments:]
    n_true_seiz     = int((test_labels == 1).sum())
    n_true_norm     = int((test_labels == 0).sum())

    print(f"  Total segments     : {n_total:,}")
    print(f"  Calibration (skip) : {cal_n_segments:,}  ≈ {cal_hours:.2f} h")
    print(f"  Test segments      : {n_test:,}  "
          f"(seizure={n_true_seiz}, normal={n_true_norm})")
    print()

    # ------------------------------------------------------------------
    # Step 2: Load patient-independent model (best_model.pth)
    # ------------------------------------------------------------------
    print("[STEP 2] Loading patient-specific model ...")
    ckpt_path = Path(cfg.training.checkpoint_dir) / "patient_specific_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CNN_LSTM_Classifier(
        num_channels     = cfg.model.num_channels,
        sequence_length  = cfg.model.sequence_length,
        num_classes      = cfg.model.num_classes,
        cnn_channels     = cfg.model.cnn_channels,
        kernel_size      = cfg.model.kernel_size,
        lstm_hidden_size = cfg.model.lstm_hidden_size,
        lstm_num_layers  = cfg.model.lstm_num_layers,
        dropout_rate     = cfg.model.dropout_rate,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Handle both best_model.pth and patient_specific_model.pth formats
    epochs_info = ckpt.get("epoch", ckpt.get("epochs_finetuned", "?"))
    metrics     = ckpt.get("metrics", ckpt.get("finetune_metrics", {}))
    f1_val      = metrics.get("f1", metrics.get("macro_f1", "N/A"))
    f1_str      = f"{f1_val:.4f}" if isinstance(f1_val, (int, float)) else str(f1_val)
    print(f"  Loaded checkpoint (epochs={epochs_info}, F1={f1_str})")
    print(f"  Normalization: Z-score via EEGDataset (matches Phase 1 training)")
    print()

    # ------------------------------------------------------------------
    # Step 3: Stream inference (skipping calibration segments)
    # ------------------------------------------------------------------
    print("[STEP 3] Running real-time simulation ...")
    print(f"[STREAM] Streaming {n_test:,} test segments "
          f"(≈ {n_test * step_sec / 3600:.2f} h) — "
          f"skipping first {cal_n_segments:,} calibration segments ...")
    print("-" * 68)

    results, latencies_ms = stream_inference(
        model          = model,
        test_loader    = test_loader,
        cal_n_segments = cal_n_segments,
        thresholds     = thresholds,
        step_sec       = step_sec,
        device         = device,
    )

    # ------------------------------------------------------------------
    # Step 4: Print final clinical report
    # ------------------------------------------------------------------
    print_report(
        results        = results,
        latencies_ms   = latencies_ms,
        target_patient = target_patient,
        n_test         = n_test,
        n_true_seiz    = n_true_seiz,
        n_true_norm    = n_true_norm,
        step_sec       = step_sec,
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time simulation of EEG seizure detector (patient-independent model)"
    )
    parser.add_argument(
        "--target-patient", type=str, required=True,
        help="Patient ID to simulate on (e.g. chb15)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--threshold", type=float, nargs="+", default=[0.50, 0.75, 0.85, 0.90],
        help="Decision threshold(s) (P(seizure) >= threshold → alarm). "
             "Pass multiple values to sweep, e.g. --threshold 0.75 0.85 0.90",
    )
    parser.add_argument(
        "--calibration-hours", type=float, default=1.5,
        help="Hours of initial data to skip as calibration (default: 1.5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Inference batch size (default: 256)",
    )
    args = parser.parse_args()
    main(args)
