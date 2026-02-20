"""
Overfit Test -- Verify CNN-LSTM can memorize a single batch.

Loads a small batch (with guaranteed seizure samples) from the cached
chb01 data and trains on it for 100 epochs.

Success criteria:
  - Training loss approaches 0.0
  - Training accuracy reaches 100%
  - No gradient explosions or dimension mismatches

Usage:
    python overfit_test.py
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

from src.data.dataset import EEGDataset
from src.models.cnn_lstm import CNN_LSTM_Classifier


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LR = 0.001
    SUBJECT = "chb01"
    PROCESSED_DIR = "data/processed"

    print("=" * 60)
    print("OVERFIT TEST -- Single Batch Memorization")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print()

    # 1. Load dataset
    ds = EEGDataset(PROCESSED_DIR, subjects=[SUBJECT], normalize=True)
    print(f"Dataset size: {len(ds)} segments")

    # 2. Build a batch with guaranteed seizure samples
    #    Pick all seizure indices + pad with normal indices to fill the batch
    sz_indices = np.where(ds.labels == 1)[0]
    no_indices = np.where(ds.labels == 0)[0]

    n_sz = min(len(sz_indices), BATCH_SIZE // 2)  # half seizure
    n_no = BATCH_SIZE - n_sz                       # half normal

    rng = np.random.RandomState(42)
    chosen_sz = rng.choice(sz_indices, size=n_sz, replace=False)
    chosen_no = rng.choice(no_indices, size=n_no, replace=False)
    chosen = np.concatenate([chosen_sz, chosen_no])
    rng.shuffle(chosen)

    # Materialize the batch
    xs, ys = [], []
    for idx in chosen:
        x, y = ds[int(idx)]
        xs.append(x)
        ys.append(y)

    batch_x = torch.stack(xs).to(DEVICE)       # (B, 18, 1024)
    batch_y = torch.tensor(ys).to(DEVICE)       # (B,)

    print(f"Batch shape:  {tuple(batch_x.shape)}")
    print(f"Batch labels: {batch_y.tolist()}")
    print(f"  Seizure: {(batch_y == 1).sum().item()}, Normal: {(batch_y == 0).sum().item()}")
    print()

    # 3. Build model
    model = CNN_LSTM_Classifier(
        num_channels=18,
        sequence_length=1024,
        num_classes=2,
        cnn_channels=[64, 128, 256],
        kernel_size=5,
        lstm_hidden_size=128,
        lstm_num_layers=2,
        dropout_rate=0.0,  # no dropout for overfit test
    ).to(DEVICE)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Train on the same batch for N epochs
    print(f"\nTraining on 1 batch for {NUM_EPOCHS} epochs...")
    print("-" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Check for gradient explosion
        max_grad = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)

        optimizer.step()

        _, preds = outputs.max(1)
        correct = preds.eq(batch_y).sum().item()
        acc = 100.0 * correct / len(batch_y)

        if epoch <= 10 or epoch % 10 == 0 or acc == 100.0:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.6f} | "
                  f"Acc: {acc:.1f}% | Max Grad: {max_grad:.4f}")

        if acc == 100.0 and loss.item() < 0.01:
            print(f"\n>> Converged at epoch {epoch}!")
            break

    print()
    print("=" * 60)
    if acc == 100.0:
        print("OVERFIT TEST PASSED -- Model can memorize a single batch")
    else:
        print(f"OVERFIT TEST INCOMPLETE -- Final acc: {acc:.1f}%")
        print("Consider: more epochs, higher LR, or check architecture")
    print("=" * 60)


if __name__ == "__main__":
    main()
