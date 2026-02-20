"""
EEG Signal Preprocessing Module

This module handles loading, filtering, and preprocessing of EEG signals stored
in European Data Format (.edf). It uses MNE-Python for robust signal processing.

Key preprocessing steps:
1. Notch filter at 50Hz to remove power line interference
2. Butterworth bandpass filter (1-45Hz) to retain relevant EEG frequency bands
3. Extraction of raw NumPy arrays for downstream deep learning pipelines
"""

import mne
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


# The 18 standard bipolar EEG channels shared across all CHB-MIT subjects.
# Channels 19-23+ vary between subjects and are excluded for consistency.
TARGET_CHANNELS: List[str] = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]


def select_channels(
    raw,
    target_channels: Optional[List[str]] = None,
):
    """
    Pick and reorder channels from an MNE Raw object.

    Performs case-insensitive matching and strips common MNE prefixes
    (e.g. ``EEG ``) so that channel names from the summary files match
    the names stored inside the EDF.

    Args:
        raw: MNE Raw object (modified in-place).
        target_channels: Channel names to keep.  Defaults to TARGET_CHANNELS.

    Returns:
        The same Raw object with only the requested channels.

    Raises:
        ValueError: If a requested channel cannot be found.
    """
    if target_channels is None:
        target_channels = TARGET_CHANNELS

    # Build normalised-name → actual-name mapping
    def _norm(name: str) -> str:
        n = name.strip().upper().replace(".", "-")
        for prefix in ("EEG ", "EMG ", "EOG ", "ECG "):
            if n.startswith(prefix):
                n = n[len(prefix):]
        return n

    raw_map = {_norm(ch): ch for ch in raw.ch_names}

    picks = []
    for target in target_channels:
        key = _norm(target)
        if key in raw_map:
            picks.append(raw_map[key])
        # MNE appends "-0", "-1", … when channel names are duplicated
        # in the EDF file (e.g. T8-P8 → T8-P8-0, T8-P8-1).
        # We pick the first occurrence ("-0") as the canonical channel.
        elif key + "-0" in raw_map:
            picks.append(raw_map[key + "-0"])
        else:
            raise ValueError(
                f"Channel '{target}' not found.  "
                f"Available (normalised): {sorted(raw_map.keys())}"
            )

    raw.pick(picks)
    return raw


def load_eeg_data(
    edf_file_path: str,
    notch_freq: float = 50.0,
    bandpass_low: float = 1.0,
    bandpass_high: float = 45.0,
    target_channels: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load and preprocess EEG data from an EDF file.
    
    Args:
        edf_file_path: Path to the .edf file
        notch_freq: Frequency for notch filter to remove line noise (Hz)
        bandpass_low: Lower cutoff frequency for bandpass filter (Hz)
        bandpass_high: Upper cutoff frequency for bandpass filter (Hz)
        verbose: Whether to display MNE logging output
        
    Returns:
        data: Preprocessed EEG data as NumPy array of shape (n_channels, n_samples)
        times: Time vector corresponding to samples (seconds)
        sfreq: Sampling frequency (Hz)
        
    Raises:
        FileNotFoundError: If the EDF file does not exist
        ValueError: If filter parameters are invalid
    """
    # Verify file exists
    edf_path = Path(edf_file_path)
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_file_path}")
    
    # Set MNE logging verbosity
    mne.set_log_level(verbose='INFO' if verbose else 'WARNING')
    
    # Load EDF file using MNE
    # preload=True loads data into memory for faster filtering
    raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=verbose)
    
    # Select target channels (before filtering for efficiency)
    if target_channels is not None:
        raw = select_channels(raw, target_channels)
    
    # Extract sampling frequency
    sfreq = raw.info['sfreq']
    
    # Validate filter parameters
    if bandpass_low >= bandpass_high:
        raise ValueError("bandpass_low must be less than bandpass_high")
    if bandpass_high > sfreq / 2:
        raise ValueError(f"bandpass_high ({bandpass_high} Hz) exceeds Nyquist frequency ({sfreq/2} Hz)")
    
    # Apply notch filter to remove power line noise (50Hz or 60Hz depending on region)
    # This removes interference from electrical infrastructure
    raw.notch_filter(
        freqs=notch_freq,
        verbose=verbose
    )
    
    # Apply Butterworth bandpass filter to retain EEG-relevant frequencies
    # 1-45 Hz captures all standard EEG bands:
    #   - Delta (1-4 Hz): Deep sleep, unconscious states
    #   - Theta (4-8 Hz): Drowsiness, meditation
    #   - Alpha (8-13 Hz): Relaxed wakefulness
    #   - Beta (13-30 Hz): Active thinking, focus
    #   - Gamma (30-45 Hz): High-level cognitive processing
    raw.filter(
        l_freq=bandpass_low,
        h_freq=bandpass_high,
        method='iir',  # Butterworth is an IIR filter
        iir_params={'order': 4, 'ftype': 'butter'},  # 4th-order Butterworth
        verbose=verbose
    )
    
    # Extract the underlying NumPy array
    # Shape: (n_channels, n_samples)
    data = raw.get_data()
    
    # Extract time vector (in seconds)
    times = raw.times
    
    return data, times, sfreq


def segment_eeg(
    data: np.ndarray,
    sfreq: float,
    window_size_sec: float = 4.0,
    overlap_sec: float = 2.0
) -> np.ndarray:
    """
    Segment continuous EEG data into fixed-length windows with optional overlap.
    
    Args:
        data: EEG data array of shape (n_channels, n_samples)
        sfreq: Sampling frequency (Hz)
        window_size_sec: Length of each window in seconds
        overlap_sec: Overlap between consecutive windows in seconds
        
    Returns:
        segments: Array of shape (n_windows, n_channels, window_samples)
    """
    n_channels, n_samples = data.shape
    
    # Convert time parameters to sample counts
    window_samples = int(window_size_sec * sfreq)
    stride_samples = int((window_size_sec - overlap_sec) * sfreq)
    
    # Calculate number of windows
    n_windows = (n_samples - window_samples) // stride_samples + 1
    
    # Preallocate output array
    segments = np.zeros((n_windows, n_channels, window_samples), dtype=data.dtype)
    
    # Extract windows
    for i in range(n_windows):
        start_idx = i * stride_samples
        end_idx = start_idx + window_samples
        segments[i] = data[:, start_idx:end_idx]
    
    return segments


if __name__ == "__main__":
    """
    Example usage demonstrating the preprocessing pipeline.
    """
    # Example: Process a sample EDF file
    sample_edf = "data/raw/sample.edf"  # Update with actual file path
    
    try:
        # Load and preprocess
        data, times, sfreq = load_eeg_data(
            edf_file_path=sample_edf,
            notch_freq=50.0,
            bandpass_low=1.0,
            bandpass_high=45.0,
            verbose=True
        )
        
        print(f"Loaded EEG data:")
        print(f"  Shape: {data.shape}")
        print(f"  Sampling frequency: {sfreq} Hz")
        print(f"  Duration: {times[-1]:.2f} seconds")
        print(f"  Data type: {data.dtype}")
        print(f"  Value range: [{data.min():.2e}, {data.max():.2e}]")
        
        # Segment into windows
        segments = segment_eeg(data, sfreq, window_size_sec=4.0, overlap_sec=2.0)
        print(f"\nSegmented into {segments.shape[0]} windows of {segments.shape[2]} samples each")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update the sample_edf path with a valid EDF file.")
