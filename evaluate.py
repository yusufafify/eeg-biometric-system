"""
Clinical Evaluation Script for EEG Seizure Detection Model

Loads the best checkpoint and runs inference on a held-out test set.
Produces a thesis-ready text report with:
  - Sensitivity, Specificity, F1, FPR/h
  - Per-class counts (TP, TN, FP, FN)
  - Confusion matrix saved as runs/confusion_matrix.png

Usage:
    python evaluate.py --config config.yaml
    python evaluate.py --config config.yaml --test_subjects chb19,chb20
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- render to file only
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.cnn_lstm import CNN_LSTM_Classifier
from src.data.dataset import build_segments_index, EEGDataset
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
# Inference
# =============================================================================

def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple:
    """Run model inference on the entire dataloader.

    Returns:
        (y_true, y_pred, y_probs) as numpy arrays.
    """
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().tolist()

            all_targets.extend(targets.tolist())
            all_preds.extend(preds)
            all_probs.append(probs)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.vstack(all_probs)
    return y_true, y_pred, y_probs


# =============================================================================
# Confusion Matrix
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    class_names: tuple = ("Normal", "Seizure"),
) -> None:
    """Generate and save a publication-quality confusion matrix."""
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix - EEG Seizure Detection",
    )
    ax.xaxis.label.set_fontsize(13)
    ax.yaxis.label.set_fontsize(13)
    ax.title.set_fontsize(15)

    # Annotate cells with counts and percentages
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        row_total = cm[i].sum()
        for j in range(n_classes):
            pct = 100.0 * cm[i, j] / max(row_total, 1)
            ax.text(
                j, i,
                f"{cm[i, j]:,}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] Confusion matrix saved: {save_path}")


# =============================================================================
# Report
# =============================================================================

def _safe(fn, y_true, y_pred):
    try:
        return fn(y_true, y_pred)
    except ValueError:
        return 0.0


def print_thesis_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_sec: float,
    overlap_ratio: float,
    test_subjects: list,
) -> None:
    """Print a thesis-ready clinical evaluation report."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    sensitivity = _safe(calculate_seizure_sensitivity, y_true, y_pred)
    specificity = _safe(calculate_specificity, y_true, y_pred)
    f1 = _safe(calculate_f1_score, y_true, y_pred)
    far = _safe(calculate_false_alarm_rate, y_true, y_pred)

    step_sec = window_sec * (1.0 - overlap_ratio)
    total_hours = (len(y_true) * step_sec) / 3600.0
    fpr_per_hour = fp / max(total_hours, 1e-9)

    w = 72
    print()
    print("=" * w)
    print("     CLINICAL EVALUATION REPORT -- EEG SEIZURE DETECTION")
    print("=" * w)
    print()
    print(f"  Test Subjects:       {', '.join(test_subjects)}")
    print(f"  Total Segments:      {len(y_true):,}")
    print(f"  Seizure Segments:    {int((y_true == 1).sum()):,}")
    print(f"  Normal Segments:     {int((y_true == 0).sum()):,}")
    print(f"  Monitored Duration:  {total_hours:.2f} hours")
    print()
    print("-" * w)
    print("  CONFUSION MATRIX")
    print("-" * w)
    print(f"    True Positives  (TP):  {tp:>8,}")
    print(f"    True Negatives  (TN):  {tn:>8,}")
    print(f"    False Positives (FP):  {fp:>8,}")
    print(f"    False Negatives (FN):  {fn:>8,}")
    print()
    print("-" * w)
    print("  CLINICAL METRICS")
    print("-" * w)
    print(f"    Accuracy:              {100.0 * (tp + tn) / max(len(y_true), 1):>8.2f}%")
    print(f"    Sensitivity (Recall):  {sensitivity:>8.4f}   (seizure detection rate)")
    print(f"    Specificity:           {specificity:>8.4f}   (normal rejection rate)")
    print(f"    F1-Score:              {f1:>8.4f}   (harmonic mean)")
    print(f"    False Alarm Rate:      {far:>8.4f}")
    print(f"    FPR / hour:            {fpr_per_hour:>8.2f}   (false positives per hour)")
    print()
    print("-" * w)
    print("  CLINICAL INTERPRETATION")
    print("-" * w)
    if sensitivity >= 0.95:
        print("    [PASS] Sensitivity >= 95% -- meets clinical threshold")
    else:
        print(f"    [WARN] Sensitivity = {sensitivity:.2%} -- below 95% clinical threshold")
    if fpr_per_hour <= 5.0:
        print(f"    [PASS] FPR/h = {fpr_per_hour:.2f} -- acceptable alarm burden")
    else:
        print(f"    [WARN] FPR/h = {fpr_per_hour:.2f} -- may cause alarm fatigue")
    print("=" * w)
    print()


# =============================================================================
# Main
# =============================================================================

def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    # Determine test subjects
    if args.test_subjects:
        test_subjects = [s.strip() for s in args.test_subjects.split(",")]
    else:
        test_subjects = cfg.evaluation.test_subjects

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(cfg.training.checkpoint_dir) / "best_model.pth"

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print("=" * 72)
    print("EEG BIOMETRIC SYSTEM - CLINICAL EVALUATION")
    print("=" * 72)
    print(f"  Device:       {device}")
    print(f"  Checkpoint:   {ckpt_path}")
    print(f"  Test Subjects: {test_subjects}")
    print()

    # ------------------------------------------------------------------
    # Build test segments (reuses cache)
    # ------------------------------------------------------------------
    build_segments_index(
        data_dir=cfg.data.raw_dir,
        subjects=test_subjects,
        output_dir=cfg.data.processed_dir,
        window_size_sec=cfg.data.window_size_sec,
        overlap_ratio=cfg.data.overlap_ratio,
        label_threshold=cfg.data.label_threshold,
    )

    test_ds = EEGDataset(cfg.data.processed_dir, test_subjects, normalize=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    if len(test_ds) == 0:
        print("[ERROR] No test segments found. Are the test subjects processed?")
        sys.exit(1)

    print(f"[DATA] Test segments: {len(test_ds):,}")

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
    print(f"[MODEL] Training Val F1 at save: {checkpoint['metrics'].get('f1', 'N/A')}")
    print()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print("[EVAL] Running inference on test set...")
    y_true, y_pred, _ = run_inference(model, test_loader, device)

    # ------------------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------------------
    cm_path = Path(cfg.training.log_dir) / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, cm_path)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print_thesis_report(
        y_true, y_pred,
        window_sec=cfg.data.window_size_sec,
        overlap_ratio=cfg.data.overlap_ratio,
        test_subjects=test_subjects,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate EEG seizure detection model on held-out test set"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--test_subjects", type=str, default=None,
        help="Override test subjects (comma-separated, e.g. chb19,chb20)",
    )
    args = parser.parse_args()
    main(args)
