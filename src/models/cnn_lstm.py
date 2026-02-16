"""
Hybrid 1D-CNN + LSTM Architecture for EEG-Based Biometric Classification

This module implements a deep learning architecture combining:
1. 1D Convolutional Neural Networks (CNN) for spatial/frequency feature extraction
2. Long Short-Term Memory (LSTM) for temporal dependency modeling

The architecture is designed for binary classification of medical anomalies in EEG signals,
specifically targeting pre-ictal epilepsy states and major depressive disorder markers.

Input shape: (batch_size, num_channels, sequence_length)
Output shape: (batch_size, 2) - logits for binary classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM_Classifier(nn.Module):
    """
    Hybrid 1D-CNN + LSTM model for EEG signal classification.
    
    Architecture:
        1. Multi-layer 1D CNN: Extracts spatial and frequency features across channels
        2. LSTM: Captures temporal dependencies in the extracted features
        3. Fully connected layers: Maps LSTM outputs to binary classification logits
    """
    
    def __init__(
        self,
        num_channels: int = 23,
        sequence_length: int = 1024,
        num_classes: int = 2,
        cnn_channels: list = [64, 128, 256],
        kernel_size: int = 5,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout_rate: float = 0.5
    ):
        """
        Initialize the hybrid CNN-LSTM architecture.
        
        Args:
            num_channels: Number of EEG channels (e.g., 23 for CHB-MIT)
            sequence_length: Length of input sequence (number of time samples)
            num_classes: Number of output classes (2 for binary classification)
            cnn_channels: List of channel sizes for each CNN layer
            kernel_size: Kernel size for 1D convolutions
            lstm_hidden_size: Number of hidden units in LSTM
            lstm_num_layers: Number of stacked LSTM layers
            dropout_rate: Dropout probability for regularization
        """
        super(CNN_LSTM_Classifier, self).__init__()
        
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # ==================== 1D CNN Feature Extractor ====================
        # The CNN operates on the channel dimension, treating each EEG channel
        # as a separate input feature. It learns spatial filters that capture
        # frequency-domain patterns and inter-channel correlations.
        
        cnn_layers = []
        in_channels = num_channels
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2  # Same padding to preserve sequence length
                ),
                nn.BatchNorm1d(out_channels),  # Stabilizes training
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),  # Downsamples by factor of 2
                nn.Dropout(dropout_rate)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate the output sequence length after CNN pooling
        # Each MaxPool1d layer reduces length by factor of 2
        self.cnn_output_length = sequence_length // (2 ** len(cnn_channels))
        self.cnn_output_channels = cnn_channels[-1]
        
        # ==================== LSTM Temporal Encoder ====================
        # The LSTM processes the CNN feature maps as a temporal sequence,
        # capturing long-range dependencies and temporal patterns that are
        # critical for detecting pre-ictal states and depressive episodes.
        
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_channels,  # Each time step has cnn_channels[-1] features
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,  # Input shape: (batch, seq, feature)
            dropout=dropout_rate if lstm_num_layers > 1 else 0,  # Dropout between LSTM layers
            bidirectional=False  # Unidirectional for real-time inference
        )
        
        # ==================== Fully Connected Classifier ====================
        # Maps the final LSTM hidden state to class logits
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)  # Output logits (no activation)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid CNN-LSTM network.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, sequence_length)
               Represents multi-channel EEG signals
               
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
                    Raw logits for binary classification (apply softmax for probabilities)
        """
        batch_size = x.size(0)
        
        # ==================== CNN Feature Extraction ====================
        # Input: (batch_size, num_channels, sequence_length)
        # Output: (batch_size, cnn_output_channels, cnn_output_length)
        cnn_out = self.cnn(x)
        
        # ==================== Prepare for LSTM ====================
        # LSTM expects (batch, seq_len, features)
        # Permute from (batch, features, seq_len) to (batch, seq_len, features)
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # ==================== LSTM Temporal Encoding ====================
        # lstm_out: (batch_size, seq_len, lstm_hidden_size)
        # hidden: Tuple of (h_n, c_n) where h_n has shape (num_layers, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)
        
        # Extract the final hidden state from the last LSTM layer
        # h_n[-1]: (batch_size, lstm_hidden_size)
        final_hidden_state = h_n[-1]
        
        # ==================== Classification ====================
        # Map final LSTM state to class logits
        logits = self.fc(final_hidden_state)
        
        return logits


if __name__ == "__main__":
    """
    Test the model architecture with a dummy input tensor.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Model hyperparameters
    batch_size = 8
    num_channels = 23  # CHB-MIT has 23 EEG channels
    sequence_length = 1024  # 4 seconds at 256 Hz sampling rate
    num_classes = 2
    
    # Create dummy input: random EEG data
    dummy_input = torch.randn(batch_size, num_channels, sequence_length)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Input tensor range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
    
    # Instantiate model
    model = CNN_LSTM_Classifier(
        num_channels=num_channels,
        sequence_length=sequence_length,
        num_classes=num_classes,
        cnn_channels=[64, 128, 256],
        kernel_size=5,
        lstm_hidden_size=128,
        lstm_num_layers=2,
        dropout_rate=0.5
    )
    
    # Print model architecture
    print("\n" + "="*70)
    print("Model Architecture:")
    print("="*70)
    print(model)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "="*70)
    print(f"Total trainable parameters: {total_params:,}")
    print("="*70)
    
    # Forward pass
    model.eval()  # Set to evaluation mode (disables dropout)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output logits:\n{output}")
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1)
    print(f"\nClass probabilities (after softmax):\n{probabilities}")
    
    # Verify output shape
    assert output.shape == (batch_size, num_classes), \
        f"Expected output shape ({batch_size}, {num_classes}), got {output.shape}"
    
    print("\n✓ Forward pass successful!")
