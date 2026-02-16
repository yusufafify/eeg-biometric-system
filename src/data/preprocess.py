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
from typing import Tuple, Optional


def load_eeg_data(
    edf_file_path: str,
    notch_freq: float = 50.0,
    bandpass_low: float = 1.0,
    bandpass_high: float = 45.0,
    verbose: bool = False
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
