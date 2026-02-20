"""
Real-Time Inference Simulator for EEG Seizure Detection

Streams a raw .edf file chunk-by-chunk through the trained model,
simulating a live 4-second rolling window at 256 Hz.

Outputs:
  - Per-window predictions with timestamps
  - Average / P95 / P99 inference latency (ms)
  - Total seizure detections with onset timestamps

Usage:
    python simulate_realtime.py --edf data/raw/chbmit/chb01/chb01_03.edf
    python simulate_realtime.py --edf data/raw/chbmit/chb01/chb01_03.edf --config config.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.models.cnn_lstm import CNN_LSTM_Classifier
from src.data.preprocess import load_eeg_data
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    edf_path = Path(args.edf)
    if not edf_path.exists():
        print(f"[ERROR] EDF file not found: {edf_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(cfg.training.checkpoint_dir) / "best_model.pth"

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    window_sec = cfg.data.window_size_sec
    overlap_ratio = cfg.data.overlap_ratio
    step_sec = window_sec * (1.0 - overlap_ratio)

    print("=" * 72)
    print("EEG BIOMETRIC SYSTEM - REAL-TIME INFERENCE SIMULATOR")
    print("=" * 72)
    print(f"  Device:       {device}")
    print(f"  EDF File:     {edf_path}")
    print(f"  Checkpoint:   {ckpt_path}")
    print(f"  Window:       {window_sec}s ({int(window_sec * 256)} samples)")
    print(f"  Step:         {step_sec}s ({int(step_sec * 256)} samples)")
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
    model.eval()
    print(f"[MODEL] Loaded checkpoint (epoch {checkpoint['epoch']})")

    # ------------------------------------------------------------------
    # Load & preprocess EDF
    # ------------------------------------------------------------------
    print(f"[DATA] Loading EDF: {edf_path.name} ...")
    data, times, sfreq = load_eeg_data(str(edf_path))
    n_channels, n_samples = data.shape
    duration_sec = n_samples / sfreq
    print(f"[DATA] Channels: {n_channels}, Samples: {n_samples:,}, "
          f"Duration: {format_time(duration_sec)}, Sfreq: {sfreq} Hz")
    print()

    # ------------------------------------------------------------------
    # Warmup (discard first few runs for GPU/JIT initialization)
    # ------------------------------------------------------------------
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)

    dummy_input = torch.randn(1, n_channels, window_samples, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("[WARMUP] 10 warmup passes complete")

    # ------------------------------------------------------------------
    # Stream windows and measure latency
    # ------------------------------------------------------------------
    latencies_ms = []
    predictions = []
    seizure_onsets = []
    prev_pred = 0

    n_windows = max(0, (n_samples - window_samples) // step_samples + 1)
    print(f"[STREAM] Simulating {n_windows:,} windows ...")
    print("-" * 72)

    # Compute normalization stats from a subsample (consistent with training)
    subsample_idx = np.linspace(0, n_samples - window_samples, min(100, n_windows), dtype=int)
    sample_data = np.stack([data[:, i:i + window_samples] for i in subsample_idx])
    ch_mean = sample_data.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
    ch_std = sample_data.std(axis=(0, 2), keepdims=True) + 1e-8

    for w_idx in range(n_windows):
        start = w_idx * step_samples
        end = start + window_samples
        chunk = data[:, start:end].astype(np.float32)  # (C, T)

        # Z-score normalize (per-channel)
        chunk = (chunk - ch_mean[0]) / ch_std[0]

        # Convert to tensor
        x = torch.from_numpy(chunk).unsqueeze(0).to(device)  # (1, C, T)

        # Inference with timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            logits = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        latencies_ms.append(latency_ms)

        pred = logits.argmax(dim=1).item()
        predictions.append(pred)

        # Detect seizure onset (transition from 0 -> 1)
        window_time = start / sfreq
        if pred == 1 and prev_pred == 0:
            seizure_onsets.append(window_time)
            print(f"  *** SEIZURE DETECTED at {format_time(window_time)} "
                  f"(window {w_idx + 1}) | latency: {latency_ms:.2f} ms")
        prev_pred = pred

        # Progress indicator every 500 windows
        if (w_idx + 1) % 500 == 0 or (w_idx + 1) == n_windows:
            avg_lat = np.mean(latencies_ms[-500:])
            print(f"  Window [{w_idx + 1:>6}/{n_windows}] | "
                  f"time: {format_time(window_time)} | "
                  f"avg latency: {avg_lat:.2f} ms")

    # ------------------------------------------------------------------
    # Summary Report
    # ------------------------------------------------------------------
    latencies = np.array(latencies_ms)
    preds = np.array(predictions)
    n_seizure_windows = int((preds == 1).sum())

    print()
    print("=" * 72)
    print("REAL-TIME SIMULATION REPORT")
    print("=" * 72)
    print(f"  EDF File:            {edf_path.name}")
    print(f"  Duration:            {format_time(duration_sec)}")
    print(f"  Windows Processed:   {n_windows:,}")
    print()
    print("-" * 72)
    print("  LATENCY BENCHMARKS")
    print("-" * 72)
    print(f"    Mean:              {latencies.mean():>8.2f} ms")
    print(f"    Std:               {latencies.std():>8.2f} ms")
    print(f"    Median (P50):      {np.percentile(latencies, 50):>8.2f} ms")
    print(f"    P95:               {np.percentile(latencies, 95):>8.2f} ms")
    print(f"    P99:               {np.percentile(latencies, 99):>8.2f} ms")
    print(f"    Min:               {latencies.min():>8.2f} ms")
    print(f"    Max:               {latencies.max():>8.2f} ms")
    print()

    budget_ms = window_sec * 1000  # 4000 ms for a 4s window
    if latencies.mean() < budget_ms:
        ratio = budget_ms / latencies.mean()
        print(f"    [PASS] Mean latency ({latencies.mean():.2f} ms) << "
              f"window budget ({budget_ms:.0f} ms)")
        print(f"           Real-time capable: {ratio:.0f}x faster than required")
    else:
        print(f"    [FAIL] Mean latency ({latencies.mean():.2f} ms) > "
              f"window budget ({budget_ms:.0f} ms)")
        print("           Model is too slow for real-time deployment")

    print()
    print("-" * 72)
    print("  SEIZURE DETECTIONS")
    print("-" * 72)
    print(f"    Seizure Windows:   {n_seizure_windows:,} / {n_windows:,} "
          f"({100.0 * n_seizure_windows / max(n_windows, 1):.2f}%)")
    print(f"    Onset Events:      {len(seizure_onsets)}")
    for i, onset in enumerate(seizure_onsets):
        print(f"      [{i + 1}] {format_time(onset)}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate real-time EEG seizure detection"
    )
    parser.add_argument(
        "--edf", type=str, required=True,
        help="Path to a single .edf file to stream",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(args)
