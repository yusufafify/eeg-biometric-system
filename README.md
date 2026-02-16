# EEG-Based Biometric System for Medical Anomaly Detection

A PyTorch-based deep learning system for analyzing continuous 1D EEG signals to perform binary classification of medical anomalies, specifically detecting **pre-ictal epilepsy states** and **major depressive disorder markers**.

## 🧠 Project Overview

This project implements a hybrid **1D-CNN + LSTM** architecture designed to process multi-channel EEG time-series data. The system:

- Processes raw `.edf` (European Data Format) EEG recordings
- Applies clinical-grade signal preprocessing (notch filtering, bandpass filtering)
- Extracts spatial and temporal features using deep learning
- Performs binary classification for medical anomaly detection

### Datasets

- **CHB-MIT Scalp EEG Database**: Epileptic seizure detection
- **MODMA Dataset**: Depression marker identification

## 🏗️ Architecture

### Hybrid CNN-LSTM Pipeline

```
Input EEG Signal (batch, channels, time)
    ↓
[1D Convolutional Layers]
    • Spatial/frequency feature extraction
    • Multi-channel pattern recognition
    • Dimensionality reduction via pooling
    ↓
[LSTM Layers]
    • Temporal dependency modeling
    • Long-range pattern capture
    ↓
[Fully Connected Classifier]
    • Binary classification (normal vs. anomaly)
    ↓
Output Logits (batch, 2)
```

### Key Components

1. **Signal Preprocessing** (`src/data/preprocess.py`)
   - Notch filter at 50Hz (removes power line interference)
   - Butterworth bandpass filter (1-45Hz): Retains all physiological EEG bands
     - Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-45 Hz)

2. **Deep Learning Model** (`src/models/cnn_lstm.py`)
   - **1D-CNN**: Extracts inter-channel spatial patterns and frequency features
   - **LSTM**: Captures temporal dependencies across time
   - **Fully Connected Head**: Maps features to binary classification logits

## 📁 Project Structure

```
eeg-biometric-system/
├── data/
│   ├── raw/              # Raw .edf files
│   ├── processed/        # Preprocessed NumPy arrays
│   └── splits/           # Train/val/test splits
├── models/
│   ├── checkpoints/      # Training checkpoints
│   └── final/            # Best trained models
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── data/
│   │   └── preprocess.py # EEG signal preprocessing
│   ├── models/
│   │   └── cnn_lstm.py   # Hybrid CNN-LSTM model
│   └── utils/            # Helper functions
├── logs/                 # Training logs and metrics
├── outputs/              # Predictions and visualizations
├── requirements.txt      # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yusufafify/eeg-biometric-system.git
cd eeg-biometric-system
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

**Test the model architecture:**
```bash
python src/models/cnn_lstm.py
```

Expected output:
```
Input shape: torch.Size([8, 23, 1024])
...
Output shape: torch.Size([8, 2])
✓ Forward pass successful!
```

**Test the preprocessing pipeline:**
```bash
# Note: Requires a valid .edf file in data/raw/
python src/data/preprocess.py
```

## 📊 Usage Example

### Preprocessing EEG Data

```python
from src.data.preprocess import load_eeg_data, segment_eeg

# Load and filter EEG data
data, times, sfreq = load_eeg_data(
    edf_file_path="data/raw/patient01.edf",
    notch_freq=50.0,
    bandpass_low=1.0,
    bandpass_high=45.0
)

# Segment into fixed-length windows
segments = segment_eeg(data, sfreq, window_size_sec=4.0, overlap_sec=2.0)
```

### Running the Model

```python
import torch
from src.models.cnn_lstm import CNN_LSTM_Classifier

# Initialize model
model = CNN_LSTM_Classifier(
    num_channels=23,
    sequence_length=1024,
    num_classes=2,
    cnn_channels=[64, 128, 256],
    lstm_hidden_size=128,
    lstm_num_layers=2,
    dropout_rate=0.5
)

# Create dummy input (batch_size=8, channels=23, time=1024)
x = torch.randn(8, 23, 1024)

# Forward pass
logits = model(x)
probabilities = torch.softmax(logits, dim=1)
```

## 🔬 Technical Details

### Signal Processing

- **Sampling Rate**: Typically 256 Hz (CHB-MIT) or 128 Hz (MODMA)
- **EEG Frequency Bands**:
  - Delta (1-4 Hz): Deep sleep, unconscious states
  - Theta (4-8 Hz): Drowsiness, meditation
  - Alpha (8-13 Hz): Relaxed wakefulness
  - Beta (13-30 Hz): Active thinking, focus
  - Gamma (30-45 Hz): High-level cognition

### Model Specifications

- **Input**: Multi-channel EEG segments (e.g., 23 channels × 1024 time points)
- **CNN Layers**: 3-layer 1D convolution with batch normalization and max pooling
- **LSTM Layers**: 2-layer unidirectional LSTM with 128 hidden units
- **Output**: Binary classification logits (normal vs. anomaly)
- **Regularization**: Dropout (0.5) and batch normalization

## 📝 Development Roadmap

- [ ] Implement data loading pipeline for CHB-MIT and MODMA datasets
- [ ] Create training script with cross-validation
- [ ] Add TensorBoard logging for experiment tracking
- [ ] Implement class balancing for imbalanced datasets
- [ ] Add model evaluation metrics (accuracy, precision, recall, F1, AUC-ROC)
- [ ] Hyperparameter tuning with Optuna or Ray Tune
- [ ] Export model to ONNX for deployment

## 🛠️ Dependencies

Core libraries:
- **PyTorch**: Deep learning framework
- **MNE-Python**: EEG signal processing
- **NumPy/SciPy**: Numerical computing
- **scikit-learn**: Data splitting and metrics

See `requirements.txt` for complete dependency list.

## 📄 License

This project is intended for research and educational purposes in biomedical signal processing and clinical machine learning.

## 🤝 Contributing

This is a scaffold repository. Contributions for improved preprocessing techniques, model architectures, or training strategies are welcome.

---

**Note**: This system is designed for research purposes and is NOT approved for clinical diagnosis. Any medical applications require rigorous validation and regulatory approval.
