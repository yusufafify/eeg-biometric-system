"""
Smoke Test — validates the end-to-end data pipeline on chb01.

Run:
    python smoke_test.py
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

from src.data.dataset import parse_summary, build_segments_index, EEGDataset

SUBJECT = "chb01"
DATA_DIR = "data/raw/chbmit"
PROCESSED_DIR = "data/processed"
EXPECTED_CHANNELS = 18
EXPECTED_SEQ_LEN = 1024


def main() -> None:
    print("=" * 60)
    print("SMOKE TEST — CHB-MIT Data Pipeline")
    print("=" * 60)

    # 1. Parse summary
    print("\n[1] Parsing summary file...")
    summary_path = f"{DATA_DIR}/{SUBJECT}/{SUBJECT}-summary.txt"
    file_records = parse_summary(summary_path)
    print(f"    Parsed {len(file_records)} files from summary.")
    assert len(file_records) > 0, "No files parsed from summary!"

    # 2. Verify seizure labels exist
    seizure_files = [f for f, intervals in file_records.items() if intervals]
    print(f"    Files with seizures: {seizure_files}")
    assert len(seizure_files) > 0, "No seizure annotations found!"
    print("    [OK] Seizure annotations found")

    # 3. Build segments (cached)
    print(f"\n[2] Building segments for {SUBJECT} (this may take a few minutes)...")
    build_segments_index(
        data_dir=DATA_DIR,
        subjects=[SUBJECT],
        output_dir=PROCESSED_DIR,
        window_size_sec=4.0,
        overlap_ratio=0.5,
        label_threshold=0.25,
    )
    print("    [OK] Segments built / cached")

    # 4. Load dataset and verify shapes
    print("\n[3] Loading EEGDataset...")
    ds = EEGDataset(PROCESSED_DIR, subjects=[SUBJECT])
    x, y = ds[0]
    print(f"    Dataset size:   {len(ds)}")
    print(f"    Segment shape:  {tuple(x.shape)}")
    print(f"    Label (first):  {y}")

    assert x.shape[0] == EXPECTED_CHANNELS, (
        f"Expected {EXPECTED_CHANNELS} channels, got {x.shape[0]}"
    )
    assert x.shape[1] == EXPECTED_SEQ_LEN, (
        f"Expected sequence length {EXPECTED_SEQ_LEN}, got {x.shape[1]}"
    )
    print(f"    [OK] Shape is ({EXPECTED_CHANNELS}, {EXPECTED_SEQ_LEN})")

    # 5. Class distribution
    import numpy as np
    labels = ds.labels
    n_seizure = int(labels.sum())
    n_normal = len(labels) - n_seizure
    print(f"\n[4] Class distribution:")
    print(f"    Normal:  {n_normal} ({100*n_normal/len(labels):.1f}%)")
    print(f"    Seizure: {n_seizure} ({100*n_seizure/len(labels):.1f}%)")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
