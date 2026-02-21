"""
Training Script for EEG-Based Biometric Analysis System

Production pipeline with:
- Centralized config via config.yaml
- Clinical metrics (Sensitivity, Specificity, F1, FPR/h)
- F1-based best-model checkpointing (strict >)
- Early stopping (patience configurable)
- Gradient accumulation for large effective batch sizes
- ReduceLROnPlateau scheduler on validation F1
- TensorBoard logging
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.cnn_lstm import CNN_LSTM_Classifier
from src.models.focal_loss import FocalLoss
from src.data.dataset import create_dataloaders
from src.utils.config import load_config
from src.utils.metrics import (
    calculate_seizure_sensitivity,
    calculate_specificity,
    calculate_f1_score,
    calculate_macro_f1,
    calculate_false_alarm_rate,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Metric helpers (safe wrappers around src.utils.metrics)
# =============================================================================

def _safe_metric(fn, y_true, y_pred, **kwargs) -> float:
    """Call a metrics function, returning 0.0 on edge-case ValueError."""
    try:
        return fn(y_true, y_pred, **kwargs)
    except ValueError:
        return 0.0


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_segments: int,
    window_sec: float,
    overlap_ratio: float,
) -> Dict[str, float]:
    """
    Compute the full suite of clinical metrics.

    FPR/h uses step_sec = window_sec * (1 - overlap_ratio) to avoid
    double-counting time from overlapping windows.
    """
    correct = int((y_true == y_pred).sum())
    accuracy = 100.0 * correct / max(len(y_true), 1)

    sensitivity = _safe_metric(calculate_seizure_sensitivity, y_true, y_pred)
    specificity = _safe_metric(calculate_specificity, y_true, y_pred)
    f1 = _safe_metric(calculate_f1_score, y_true, y_pred)
    macro_f1 = _safe_metric(calculate_macro_f1, y_true, y_pred)
    far = _safe_metric(calculate_false_alarm_rate, y_true, y_pred)

    step_sec = window_sec * (1.0 - overlap_ratio)
    total_hours = (n_segments * step_sec) / 3600.0
    fp_count = int(((y_true == 0) & (y_pred == 1)).sum())
    fpr_per_hour = fp_count / max(total_hours, 1e-9)

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "macro_f1": macro_f1,
        "far": far,
        "fpr_per_hour": fpr_per_hour,
    }


# =============================================================================
# Training Loop (with gradient accumulation)
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    accum_steps: int = 1,
) -> Dict[str, float]:
    """
    Train for one epoch with optional gradient accumulation.

    Args:
        accum_steps: Number of micro-batches to accumulate before
                     calling optimizer.step().  Effective batch size
                     = dataloader.batch_size * accum_steps.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets: List[int] = []
    all_preds: List[int] = []

    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets) / accum_steps  # scale loss
        loss.backward()

        # Accumulate raw (unscaled) loss for logging
        running_loss += loss.item() * accum_steps
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(predicted.cpu().tolist())

        # Step optimizer every accum_steps batches, or on the last batch
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        if (batch_idx + 1) % 50 == 0:
            running_macro_f1 = _safe_metric(
                calculate_macro_f1,
                np.array(all_targets),
                np.array(all_preds),
            )
            print(
                f"  Batch [{batch_idx + 1}/{len(dataloader)}] | "
                f"Loss: {loss.item() * accum_steps:.4f} | "
                f"Acc: {100.0 * correct / total:.2f}% | "
                f"Macro-F1: {running_macro_f1:.4f}"
            )

    # Compute train macro F1
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    train_macro_f1 = _safe_metric(calculate_macro_f1, y_true, y_pred)

    return {
        "loss": running_loss / max(len(dataloader), 1),
        "accuracy": 100.0 * correct / max(total, 1),
        "macro_f1": train_macro_f1,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    window_sec: float,
    overlap_ratio: float,
) -> Dict[str, float]:
    """Validate and compute all clinical metrics."""
    model.eval()
    running_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(predicted.cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    n_segments = len(y_true)

    metrics = compute_clinical_metrics(
        y_true, y_pred, n_segments, window_sec, overlap_ratio,
    )
    metrics["loss"] = running_loss / max(len(dataloader), 1)
    return metrics


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Dict[str, float],
    filepath: Path,
) -> None:
    """Save model checkpoint with full training state (includes scheduler)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, filepath)
    print(f"[SAVE] Checkpoint saved: {filepath}")


# =============================================================================
# Scheduler factory
# =============================================================================

def build_scheduler(optimizer, cfg_sched):
    """Build a LR scheduler from config."""
    name = cfg_sched.name
    params = cfg_sched.params.to_dict()

    if name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, **params)
    elif name == "StepLR":
        return StepLR(optimizer, **params)
    elif name == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, **params)
    else:
        raise ValueError(f"Unknown scheduler '{name}'")


# =============================================================================
# Main
# =============================================================================

def main(args: argparse.Namespace) -> None:
    """Main training pipeline driven by config.yaml."""
    cfg = load_config(args.config)

    # Allow CLI override of subjects
    if args.subjects:
        cfg.data.subjects = [s.strip() for s in args.subjects.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)

    accum_steps = cfg.training.gradient_accumulation_steps
    effective_batch = cfg.training.batch_size * accum_steps

    print("=" * 70)
    print("EEG BIOMETRIC SYSTEM - TRAINING PIPELINE")
    print("=" * 70)
    print(f"  Device:          {device}")
    print(f"  Config:          {args.config}")
    print(f"  Subjects:        {cfg.data.subjects}")
    print(f"  Batch Size:      {cfg.training.batch_size} x {accum_steps} accum = {effective_batch} effective")
    print(f"  Optimizer:       {cfg.training.optimizer.name} (lr={cfg.training.optimizer.params.get('lr', 'N/A')})")
    print(f"  Scheduler:       {cfg.training.scheduler.name}")
    print(f"  Epochs:          {cfg.training.epochs}")
    print(f"  Early Stop:      patience={cfg.training.early_stopping_patience}")
    print(f"  Checkpoint Metric: Val F1-Score (strict >)")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, train_subjects, val_subjects = create_dataloaders(
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

    print(f"[DATA] Train subjects: {train_subjects}")
    print(f"[DATA] Val subjects:   {val_subjects}")
    print(f"[DATA] Train batches:  {len(train_loader)}")
    print(f"[DATA] Val batches:    {len(val_loader)}")
    print()

    # ------------------------------------------------------------------
    # Model
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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] CNN_LSTM_Classifier ({total_params:,} trainable params)")
    print()

    # ------------------------------------------------------------------
    # Loss — Focal Loss handles class imbalance
    # ------------------------------------------------------------------
    # Using RetinaNet optimal baseline for extreme imbalance: 
    # alpha=[0.10, 0.90] (Normal, Seizure) and gamma=2.0.
    # This avoids destroying majority class gradients.
    # The corrected PyTorch Focal Loss initialization
    alpha = torch.tensor([0.10, 0.90], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    
    print(f"[LOSS] FocalLoss(gamma=2.0) with static alpha: {alpha.cpu().numpy()}")

    # Build optimizer from config
    opt_name = cfg.training.optimizer.name
    opt_params = cfg.training.optimizer.params.to_dict()
    opt_cls_map = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop,
    }
    if opt_name not in opt_cls_map:
        raise ValueError(
            f"Unknown optimizer '{opt_name}'. "
            f"Supported: {list(opt_cls_map.keys())}"
        )
    optimizer = opt_cls_map[opt_name](model.parameters(), **opt_params)
    print(f"[OPTIM] {opt_name} with {opt_params}")

    # Build scheduler
    scheduler = build_scheduler(optimizer, cfg.training.scheduler)
    print(f"[SCHED] {cfg.training.scheduler.name} with {cfg.training.scheduler.params.to_dict()}")

    tb_dir = Path(cfg.training.log_dir)
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    ckpt_dir = Path(cfg.training.checkpoint_dir)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_f1 = -1.0
    epochs_no_improve = 0
    patience = cfg.training.early_stopping_patience

    for epoch in range(1, cfg.training.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch}/{cfg.training.epochs}]  (LR: {current_lr:.2e})")
        print("-" * 70)

        t0 = time.time()

        # ---- Train ----
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            accum_steps=accum_steps,
        )

        # ---- Validate ----
        val_metrics = validate(
            model, val_loader, criterion, device,
            window_sec=cfg.data.window_size_sec,
            overlap_ratio=cfg.data.overlap_ratio,
        )

        elapsed = time.time() - t0

        # ---- LR Scheduler step ----
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_metrics["f1"])
        else:
            scheduler.step()

        # ---- Console output ----
        print(f"\n[EPOCH {epoch}] Time: {elapsed:.1f}s")
        print(
            f"  Train  | Loss: {train_metrics['loss']:.4f} "
            f"| Acc: {train_metrics['accuracy']:.2f}% "
            f"| Macro-F1: {train_metrics['macro_f1']:.4f}"
        )
        print(
            f"  Val    | Loss: {val_metrics['loss']:.4f} "
            f"| Acc: {val_metrics['accuracy']:.2f}% "
            f"| Macro-F1: {val_metrics['macro_f1']:.4f}"
        )
        print(
            f"  Clinical | F1: {val_metrics['f1']:.4f} "
            f"| Sens: {val_metrics['sensitivity']:.4f} "
            f"| Spec: {val_metrics['specificity']:.4f} "
            f"| FPR/h: {val_metrics['fpr_per_hour']:.2f}"
        )

        # ---- TensorBoard ----
        writer.add_scalars("Loss", {
            "train": train_metrics["loss"],
            "val": val_metrics["loss"],
        }, epoch)
        writer.add_scalars("Accuracy", {
            "train": train_metrics["accuracy"],
            "val": val_metrics["accuracy"],
        }, epoch)
        writer.add_scalar("Val/F1", val_metrics["f1"], epoch)
        writer.add_scalar("Val/Sensitivity", val_metrics["sensitivity"], epoch)
        writer.add_scalar("Val/Specificity", val_metrics["specificity"], epoch)
        writer.add_scalar("Val/FPR_per_hour", val_metrics["fpr_per_hour"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # ---- F1-based checkpointing (strict >) ----
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                ckpt_dir / "best_model.pth",
            )
            print(f"  >> New best model! Val F1: {best_val_f1:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(
                f"  -- No F1 improvement for {epochs_no_improve}/{patience} epochs"
            )

        # ---- Early stopping ----
        if epochs_no_improve >= patience:
            print(f"\n[EARLY STOP] No F1 improvement for {patience} epochs. Stopping.")
            break

    # Always save a final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, epoch, val_metrics,
        ckpt_dir / "last_model.pth",
    )

    writer.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Best Val F1:   {best_val_f1:.4f}")
    print(f"  Best model:    {ckpt_dir / 'best_model.pth'}")
    print(f"  Last model:    {ckpt_dir / 'last_model.pth'}")
    print(f"  TensorBoard:   tensorboard --logdir {cfg.training.log_dir}")
    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EEG seizure detection model"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--subjects", type=str, default=None,
        help="Override config subjects (comma-separated, e.g. chb01,chb02)",
    )

    args = parser.parse_args()
    main(args)
