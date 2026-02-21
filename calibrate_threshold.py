"""
Threshold Calibration for EEG Seizure Detection

Loads the best checkpoint, runs inference on the VALIDATION set, and sweeps
classification thresholds to find the optimal operating point that maximises
clinical F1.

Usage:
    python calibrate_threshold.py --config config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.cnn_lstm import CNN_LSTM_Classifier
from src.data.dataset import build_segments_index, EEGDataset, create_dataloaders
from src.utils.config import load_config
from src.utils.metrics import (
    calculate_seizure_sensitivity,
    calculate_specificity,
    calculate_f1_score,
    calculate_false_alarm_rate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Inference (collects softmax probabilities)
# =============================================================================

def collect_probabilities(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return (y_true, seizure_probs)."""
    model.eval()
    all_targets: List[int] = []
    all_seizure_probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            seizure_prob = probs[:, 1]  # P(seizure)

            all_targets.extend(targets.tolist())
            all_seizure_probs.append(seizure_prob)

            if (batch_idx + 1) % 200 == 0:
                print(f"  Inference batch [{batch_idx + 1}/{len(dataloader)}]")

    y_true = np.array(all_targets)
    seizure_probs = np.concatenate(all_seizure_probs)
    return y_true, seizure_probs


# =============================================================================
# Threshold sweep
# =============================================================================

def sweep_thresholds(
    y_true: np.ndarray,
    seizure_probs: np.ndarray,
    window_sec: float,
    overlap_ratio: float,
    thresholds: np.ndarray = None,
) -> List[Dict]:
    """Evaluate clinical metrics at each threshold."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    step_sec = window_sec * (1.0 - overlap_ratio)
    total_hours = (len(y_true) * step_sec) / 3600.0

    results = []
    for t in thresholds:
        y_pred = (seizure_probs >= t).astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        recall = sens
        f1 = 2 * prec * recall / max(prec + recall, 1e-9)
        fpr_h = fp / max(total_hours, 1e-9)

        results.append({
            "threshold": float(t),
            "f1": f1,
            "sensitivity": sens,
            "specificity": spec,
            "precision": prec,
            "fpr_per_hour": fpr_h,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_threshold_curves(
    results: List[Dict],
    optimal_idx: int,
    save_path: Path,
) -> None:
    """Plot F1, Sensitivity, Specificity vs threshold."""
    thresholds = [r["threshold"] for r in results]
    f1s = [r["f1"] for r in results]
    senss = [r["sensitivity"] for r in results]
    specs = [r["specificity"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: F1, Sensitivity, Specificity vs Threshold
    ax1.plot(thresholds, f1s, "b-", linewidth=2, label="F1 (seizure)")
    ax1.plot(thresholds, senss, "g--", linewidth=1.5, label="Sensitivity")
    ax1.plot(thresholds, specs, "r--", linewidth=1.5, label="Specificity")
    opt_t = results[optimal_idx]["threshold"]
    opt_f1 = results[optimal_idx]["f1"]
    ax1.axvline(opt_t, color="orange", linestyle=":", linewidth=2,
                label=f"Optimal = {opt_t:.2f}")
    ax1.scatter([opt_t], [opt_f1], color="orange", s=100, zorder=5)
    ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Clinical Metrics vs Classification Threshold", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)

    # Right: FPR/h vs Threshold
    fprs = [r["fpr_per_hour"] for r in results]
    ax2.plot(thresholds, fprs, "r-", linewidth=2)
    ax2.axvline(opt_t, color="orange", linestyle=":", linewidth=2,
                label=f"Optimal = {opt_t:.2f}")
    ax2.axhline(5.0, color="green", linestyle="--", alpha=0.7,
                label="Clinical target (5 FP/h)")
    ax2.set_xlabel("Threshold", fontsize=12)
    ax2.set_ylabel("False Positives / Hour", fontsize=12)
    ax2.set_title("False Alarm Rate vs Threshold", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_yscale("log")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] Threshold calibration plot: {save_path}")


# =============================================================================
# Main
# =============================================================================

def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(cfg.training.checkpoint_dir) / "best_model.pth"

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print("=" * 72)
    print("THRESHOLD CALIBRATION — EEG SEIZURE DETECTION")
    print("=" * 72)
    print(f"  Device:     {device}")
    print(f"  Checkpoint: {ckpt_path}")
    print()

    # ------------------------------------------------------------------
    # Reconstruct the validation set (same split as training)
    # ------------------------------------------------------------------
    print("[DATA] Reconstructing train/val split to get val subjects...")
    _, val_loader, train_subjects, val_subjects = create_dataloaders(
        data_dir=cfg.data.raw_dir,
        subjects=cfg.data.subjects,
        processed_dir=cfg.data.processed_dir,
        batch_size=cfg.training.batch_size,
        val_ratio=cfg.data.val_ratio,
        random_state=cfg.training.seed,
        normalize=True,
        num_workers=cfg.training.num_workers,
        window_size_sec=cfg.data.window_size_sec,
        overlap_ratio=cfg.data.overlap_ratio,
        label_threshold=cfg.data.label_threshold,
    )

    val_labels = val_loader.dataset.labels
    n_sz = int((val_labels == 1).sum())
    n_no = int((val_labels == 0).sum())
    print(f"[DATA] Val subjects:  {val_subjects}")
    print(f"[DATA] Val segments:  {len(val_labels):,} (seizure={n_sz:,}, normal={n_no:,})")
    print()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = CNN_LSTM_Classifier(
        num_channels=cfg.model.num_channels,
        sequence_length=cfg.model.sequence_length,
        num_classes=cfg.model.num_classes,
        cnn_channels=cfg.model.cnn_channels,
        kernel_size=cfg.model.kernel_size,
        lstm_hidden_size=cfg.model.lstm_hidden_size,
        lstm_num_layers=cfg.model.lstm_num_layers,
        dropout_rate=cfg.model.dropout_rate,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[MODEL] Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"[MODEL] Val F1 at save: {checkpoint['metrics'].get('f1', 'N/A')}")
    print()

    # ------------------------------------------------------------------
    # Collect probabilities on val set
    # ------------------------------------------------------------------
    print("[INFER] Running inference on validation set...")
    y_true, seizure_probs = collect_probabilities(model, val_loader, device)
    print(f"[INFER] Collected {len(y_true):,} predictions")
    print(f"[INFER] P(seizure) range: [{seizure_probs.min():.6f}, {seizure_probs.max():.6f}]")
    print(f"[INFER] P(seizure) mean:  {seizure_probs.mean():.6f}")
    print()

    # ------------------------------------------------------------------
    # Sweep thresholds
    # ------------------------------------------------------------------
    print("[SWEEP] Evaluating thresholds 0.01 → 0.99...")
    results = sweep_thresholds(
        y_true, seizure_probs,
        window_sec=cfg.data.window_size_sec,
        overlap_ratio=cfg.data.overlap_ratio,
    )

    # Find optimal threshold (max clinical F1)
    optimal_idx = max(range(len(results)), key=lambda i: results[i]["f1"])
    opt = results[optimal_idx]

    # Also find default threshold=0.5 result
    default_idx = min(range(len(results)),
                      key=lambda i: abs(results[i]["threshold"] - 0.50))
    default = results[default_idx]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    w = 72
    print()
    print("=" * w)
    print("  THRESHOLD CALIBRATION RESULTS")
    print("=" * w)

    print()
    opt_header = f"Optimal ({opt['threshold']:.2f})"
    print(f"  {'Metric':<22} {'Default (0.50)':>16} {opt_header:>16}")
    print("  " + "-" * 54)
    print(f"  {'Clinical F1':<22} {default['f1']:>16.4f} {opt['f1']:>16.4f}")
    print(f"  {'Sensitivity':<22} {default['sensitivity']:>16.4f} {opt['sensitivity']:>16.4f}")
    print(f"  {'Specificity':<22} {default['specificity']:>16.4f} {opt['specificity']:>16.4f}")
    print(f"  {'Precision':<22} {default['precision']:>16.4f} {opt['precision']:>16.4f}")
    print(f"  {'FPR/h':<22} {default['fpr_per_hour']:>16.2f} {opt['fpr_per_hour']:>16.2f}")
    print(f"  {'TP':<22} {default['tp']:>16,} {opt['tp']:>16,}")
    print(f"  {'FP':<22} {default['fp']:>16,} {opt['fp']:>16,}")
    print(f"  {'FN':<22} {default['fn']:>16,} {opt['fn']:>16,}")
    print(f"  {'TN':<22} {default['tn']:>16,} {opt['tn']:>16,}")

    print()
    print("=" * w)
    print("  TOP 10 THRESHOLDS BY CLINICAL F1")
    print("=" * w)
    sorted_results = sorted(results, key=lambda r: r["f1"], reverse=True)[:10]
    print(f"  {'Thresh':>7} {'F1':>8} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'FPR/h':>10}")
    print("  " + "-" * 55)
    for r in sorted_results:
        print(f"  {r['threshold']:>7.2f} {r['f1']:>8.4f} "
              f"{r['sensitivity']:>8.4f} {r['specificity']:>8.4f} "
              f"{r['precision']:>8.4f} {r['fpr_per_hour']:>10.2f}")

    # Also show where FPR/h drops below clinical target
    print()
    print("=" * w)
    print("  THRESHOLDS WHERE FPR/h < 5.0 (CLINICAL TARGET)")
    print("=" * w)
    clinical_ok = [r for r in results if r["fpr_per_hour"] <= 5.0]
    if clinical_ok:
        best_clinical = max(clinical_ok, key=lambda r: r["f1"])
        print(f"  {'Thresh':>7} {'F1':>8} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'FPR/h':>10}")
        print("  " + "-" * 55)
        for r in sorted(clinical_ok, key=lambda r: r["f1"], reverse=True)[:10]:
            marker = " <-- BEST" if r["threshold"] == best_clinical["threshold"] else ""
            print(f"  {r['threshold']:>7.2f} {r['f1']:>8.4f} "
                  f"{r['sensitivity']:>8.4f} {r['specificity']:>8.4f} "
                  f"{r['precision']:>8.4f} {r['fpr_per_hour']:>10.2f}{marker}")
    else:
        print("  None found — model needs retraining (Phase 2: Focal Loss)")

    print()
    print("=" * w)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plot_path = Path(cfg.training.log_dir) / "threshold_calibration.png"
    plot_threshold_curves(results, optimal_idx, plot_path)

    print()
    print("[DONE] Use the optimal threshold in production inference:")
    print(f"       seizure = P(seizure) >= {opt['threshold']:.2f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find optimal classification threshold for seizure detection"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(args)
