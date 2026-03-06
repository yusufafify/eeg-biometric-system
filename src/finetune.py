"""
Phase 2 — Patient-Specific Fine-Tuning Script
==============================================

Adapts the pre-trained patient-independent CNN-LSTM model to a specific
target patient's baseline EEG charactertics using a "frozen backbone /
trainable head" strategy.

Strategy
--------
1. Load ``best_model.pth`` (trained on all source patients).
2. **Freeze** the feature-extraction backbone:
   - ``model.cnn``  — 1D CNN layers
   - ``model.lstm`` — temporal LSTM encoder
3. **Leave unfrozen** only:
   - ``model.fc``   — the final MLP classification head
4. Run a micro-training loop (3–5 epochs) with:
   - AdamW at lr=1e-5  (very conservative to avoid overfitting the tiny cal set)
   - FocalLoss(gamma=2.0, alpha=[0.25, 0.75])  (identical to the base training)
5. Save the adapted model to ``models/checkpoints/patient_specific_model.pth``.

Zero-Leakage Guarantee
-----------------------
Only calibration segments (first N hours) are used here.
The test split is NEVER touched by this script.

Usage
-----
    # Fine-tune on patient chb15, using first 1.5 hours as calibration:
    python -m src.finetune \\
        --target-patient chb15 \\
        --config config.yaml \\
        --calibration-hours 1.5 \\
        --epochs 5 \\
        --lr 1e-5

    # The calibration split is printed at startup so you can verify no leakage.
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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.data.calibration_loader import build_calibration_split
from src.data.dataset import EEGDataset
from src.models.cnn_lstm import CNN_LSTM_Classifier
from src.models.focal_loss import FocalLoss
from src.utils.config import load_config
from src.utils.metrics import (
    calculate_macro_f1,
    calculate_seizure_sensitivity,
    calculate_specificity,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Freeze helpers
# =============================================================================

def freeze_backbone(model: CNN_LSTM_Classifier) -> None:
    """
    Freeze all parameters in the CNN and LSTM backbone.

    Only ``model.fc`` parameters remain trainable.  This forces the
    adapted model to keep the universal feature representations learned
    during Phase 1 and adjusts only the final decision boundary to the
    target patient's background noise.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the classification head
    for param in model.fc.parameters():
        param.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# Safe metric wrapper
# =============================================================================

def _safe(fn, y_true, y_pred, **kwargs) -> float:
    try:
        return fn(y_true, y_pred, **kwargs)
    except (ValueError, ZeroDivisionError):
        return 0.0


# =============================================================================
# Fine-Tuning loop
# =============================================================================

def finetune(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    n_epochs: int,
) -> Dict[str, float]:
    """
    Micro-training loop for patient-specific fine-tuning.

    Returns the metrics from the final epoch.
    """
    model.train()
    last_metrics: Dict[str, float] = {}

    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        all_targets: List[int] = []
        all_preds: List[int] = []

        t0 = time.time()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs  = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

        y_true = np.array(all_targets, dtype=np.int64)
        y_pred = np.array(all_preds, dtype=np.int64)

        avg_loss    = running_loss / max(len(dataloader), 1)
        macro_f1    = _safe(calculate_macro_f1, y_true, y_pred)
        sensitivity = _safe(calculate_seizure_sensitivity, y_true, y_pred)
        specificity = _safe(calculate_specificity, y_true, y_pred)
        elapsed     = time.time() - t0

        sz_in_batch = int((y_true == 1).sum())
        print(
            f"  [Epoch {epoch}/{n_epochs}]  Loss: {avg_loss:.4f}  "
            f"Macro-F1: {macro_f1:.4f}  "
            f"Sens: {sensitivity:.4f}  Spec: {specificity:.4f}  "
            f"SeizureSegs: {sz_in_batch}  Time: {elapsed:.1f}s"
        )

        last_metrics = {
            "loss": avg_loss,
            "macro_f1": macro_f1,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    return last_metrics


# =============================================================================
# Main
# =============================================================================

def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_patient = args.target_patient
    cal_hours      = args.calibration_hours
    n_epochs       = args.epochs
    lr             = args.lr
    batch_size     = args.batch_size

    # Training patients are all subjects from config.yaml MINUS the target
    all_subjects   = cfg.data.subjects   # type: ignore
    train_patients = [s for s in all_subjects if s != target_patient]

    print("=" * 68)
    print("  PHASE 2 — PATIENT-SPECIFIC FINE-TUNING")
    print("=" * 68)
    print(f"  Target patient   : {target_patient}")
    print(f"  Calibration hours: {cal_hours:.2f} h  (FIRST portion of data)")
    print(f"  Backbone frozen  : model.cnn + model.lstm")
    print(f"  Head unfrozen    : model.fc  (only these weights update)")
    print(f"  Epochs           : {n_epochs}")
    print(f"  Learning rate    : {lr}")
    print(f"  Device           : {device}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Build calibration split
    # ------------------------------------------------------------------
    print("[STEP 1] Building calibration / test split ...")
    cal_segs, cal_labels, test_segs, test_labels = build_calibration_split(
        processed_dir       = cfg.data.processed_dir,
        target_patient      = target_patient,
        train_patients      = train_patients,
        calibration_hours   = cal_hours,
        window_size_sec     = cfg.data.window_size_sec,
        overlap_ratio       = cfg.data.overlap_ratio,
        inject_seizures_if_none = True,
        n_injected_seizures = 150,
        random_state        = cfg.training.seed,
    )
    print()

    # ------------------------------------------------------------------
    # Step 2: Normalise calibration data (Z-Score via EEGDataset)
    # ------------------------------------------------------------------
    # Instantiate dataset just to extract the exact baseline statistics
    # that will be used during real-time inference.
    dummy_ds = EEGDataset(
        processed_dir=cfg.data.processed_dir,
        subjects=[target_patient],
        normalize=True,
    )
    ch_mean = dummy_ds.ch_mean
    ch_std  = dummy_ds.ch_std

    cal_segs_norm = (cal_segs - ch_mean) / ch_std

    # Convert to tensors
    X_cal = torch.from_numpy(cal_segs_norm.astype(np.float32))  # (N, C, T)
    y_cal = torch.from_numpy(cal_labels)                        # (N,)

    cal_dataset = TensorDataset(X_cal, y_cal)

    # ------------------------------------------------------------------
    # WeightedRandomSampler — enforce a 50/50 class split per batch
    # ------------------------------------------------------------------
    counts         = np.bincount(cal_labels)                # [n_normal, n_seizure]
    class_weights  = 1.0 / counts.astype(np.float64)       # inverse frequency
    sample_weights = class_weights[cal_labels]              # one weight per sample
    sampler = WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(cal_dataset),
        replacement = True,
    )

    cal_loader = DataLoader(
        cal_dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        num_workers = 0,
        pin_memory  = device.type == "cuda",
    )

    # We no longer need to persist norm_stats because simulate_realtime.py
    # uses EEGDataset directly.
    norm_stats = {}

    print(f"[DATA] Calibration loader: {len(cal_dataset)} segments, "
          f"{len(cal_loader)} batches")
    print(f"       Label distribution — "
          f"normal: {int((cal_labels == 0).sum())}, "
          f"seizure: {int((cal_labels == 1).sum())}")
    print(f"[DATA] Test set (strictly held-out): {len(test_labels)} segments")
    print()

    # ------------------------------------------------------------------
    # Step 3: Load pre-trained model
    # ------------------------------------------------------------------
    print("[STEP 2] Loading pre-trained model ...")
    ckpt_path = Path(cfg.training.checkpoint_dir) / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val F1={ckpt['metrics'].get('f1', 'N/A'):.4f})")

    # ------------------------------------------------------------------
    # Step 4: Freeze backbone, leave only FC head trainable
    # ------------------------------------------------------------------
    print("\n[STEP 3] Freezing backbone layers ...")
    freeze_backbone(model)
    trainable = count_trainable(model)
    total     = count_total(model)
    print(f"  Trainable parameters : {trainable:,}  /  {total:,}  total")
    print(f"  Frozen parameters    : {total - trainable:,}")
    print(f"  Unfrozen module      : model.fc")
    print()

    # ------------------------------------------------------------------
    # Step 5: Set up loss + optimizer (only over unfrozen params)
    # ------------------------------------------------------------------
    # Alpha is now symmetric [0.5, 0.5] because WeightedRandomSampler
    # already guarantees ~50/50 class representation per batch.  An
    # asymmetric alpha on top of a balanced sampler would over-penalise
    # normal-class predictions and destabilise the FC head at lr=1e-5.
    alpha     = torch.tensor([0.5, 0.5], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    # filter_func ensures the frozen params are truly excluded from the
    # optimiser's update graph
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = lr,
        weight_decay = 1e-4,
    )

    print(f"[STEP 4] Fine-tuning for {n_epochs} epoch(s)  "
          f"(AdamW lr={lr}, FocalLoss γ=2.0, α=[0.5, 0.5], WeightedSampler=ON)")
    print("-" * 68)

    # ------------------------------------------------------------------
    # Step 6: Micro-training loop
    # ------------------------------------------------------------------
    final_metrics = finetune(
        model      = model,
        dataloader = cal_loader,
        criterion  = criterion,
        optimizer  = optimizer,
        device     = device,
        n_epochs   = n_epochs,
    )

    # ------------------------------------------------------------------
    # Step 7: Save patient-specific model
    # ------------------------------------------------------------------
    out_dir  = Path(cfg.training.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "patient_specific_model.pth"

    torch.save({
        "target_patient"  : target_patient,
        "calibration_hours": cal_hours,
        "epochs_finetuned": n_epochs,
        "model_state_dict": model.state_dict(),
        "norm_stats"      : norm_stats,          # mean/std for test-time normalisation
        "finetune_metrics": final_metrics,
        "model_config"    : {
            "num_channels"    : cfg.model.num_channels,
            "sequence_length" : cfg.model.sequence_length,
            "num_classes"     : cfg.model.num_classes,
            "cnn_channels"    : cfg.model.cnn_channels,
            "kernel_size"     : cfg.model.kernel_size,
            "lstm_hidden_size": cfg.model.lstm_hidden_size,
            "lstm_num_layers" : cfg.model.lstm_num_layers,
            "dropout_rate"    : cfg.model.dropout_rate,
        },
    }, out_path)

    print()
    print("=" * 68)
    print("  FINE-TUNING COMPLETE")
    print("=" * 68)
    print(f"  Patient-specific model saved : {out_path}")
    print(f"  Final Macro-F1 (cal set)     : {final_metrics.get('macro_f1', 0):.4f}")
    print(f"  Final Sensitivity (cal set)  : {final_metrics.get('sensitivity', 0):.4f}")
    print()
    print("  Next step — run the real-time simulation:")
    print(f"    python -m src.simulate_realtime "
          f"--target-patient {target_patient} --threshold 0.80")
    print("=" * 68)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Patient-specific fine-tuning of EEG seizure detector"
    )
    parser.add_argument(
        "--target-patient", type=str, required=True,
        help="Patient ID to adapt to (e.g. chb15)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--calibration-hours", type=float, default=1.5,
        help="Hours of initial data to use as calibration (default: 1.5)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of fine-tuning epochs (default: 5, recommended: 3-5)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="Learning rate for AdamW (default: 1e-5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for calibration loader (default: 64)",
    )
    args = parser.parse_args()
    main(args)
