"""
Domain-Specific Metrics for EEG Biometric Analysis

This module provides specialized evaluation metrics for medical anomaly detection
in EEG signals, focusing on clinical relevance (sensitivity, specificity, latency).

Status: Placeholder implementation (ready for real metric calculation)
"""

from typing import List, Tuple, Dict, Any
import time
import numpy as np
import torch


# =============================================================================
# Clinical Performance Metrics
# =============================================================================

def calculate_seizure_sensitivity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_class: int = 1
) -> float:
    """
    Calculate seizure detection sensitivity (True Positive Rate).
    
    Sensitivity = TP / (TP + FN)
    
    In medical context, this measures the percentage of actual seizure events
    that the model correctly identifies. High sensitivity (≥95%) is CRITICAL
    to avoid missing dangerous medical events.
    
    Args:
        y_true: Ground truth labels (0=normal, 1=seizure)
        y_pred: Predicted labels (0=normal, 1=seizure)
        positive_class: Label for positive class (default: 1)
    
    Returns:
        Sensitivity value in range [0.0, 1.0]
    
    Example:
        >>> y_true = np.array([1, 1, 0, 1, 0])
        >>> y_pred = np.array([1, 1, 0, 0, 0])
        >>> calculate_seizure_sensitivity(y_true, y_pred)
        0.6666666666666666  # 2 out of 3 seizures detected
    """
    # Filter only positive class samples
    mask = (y_true == positive_class)
    
    if mask.sum() == 0:
        raise ValueError("No positive samples in ground truth")
    
    # True Positives: correctly identified seizures
    true_positives = ((y_true == positive_class) & (y_pred == positive_class)).sum()
    
    # All actual seizures
    condition_positive = mask.sum()
    
    # Calculate sensitivity
    sensitivity = true_positives / condition_positive
    
    return float(sensitivity)


def calculate_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    negative_class: int = 0
) -> float:
    """
    Calculate specificity (True Negative Rate).
    
    Specificity = TN / (TN + FP)
    
    Measures the percentage of normal (non-seizure) periods correctly classified.
    High specificity reduces alarm fatigue in clinical settings.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        negative_class: Label for negative class (default: 0)
    
    Returns:
        Specificity value in range [0.0, 1.0]
    """
    # Filter only negative class samples
    mask = (y_true == negative_class)
    
    if mask.sum() == 0:
        raise ValueError("No negative samples in ground truth")
    
    # True Negatives: correctly identified normal periods
    true_negatives = ((y_true == negative_class) & (y_pred == negative_class)).sum()
    
    # All actual normal periods
    condition_negative = mask.sum()
    
    # Calculate specificity
    specificity = true_negatives / condition_negative
    
    return float(specificity)


def calculate_false_alarm_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_class: int = 1
) -> float:
    """
    Calculate false alarm rate (False Positive Rate).
    
    FAR = FP / (FP + TN) = 1 - Specificity
    
    Critical metric for clinical deployment. High false alarm rates cause:
    - Alarm fatigue in medical staff
    - Unnecessary interventions
    - Reduced trust in the system
    
    Target: ≤5% for production deployment
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        positive_class: Label for positive class (default: 1)
    
    Returns:
        False alarm rate in range [0.0, 1.0]
    
    Example:
        >>> y_true = np.array([0, 0, 0, 1, 1])
        >>> y_pred = np.array([1, 0, 0, 1, 1])
        >>> calculate_false_alarm_rate(y_true, y_pred)
        0.3333333333333333  # 1 false alarm out of 3 normal periods
    """
    # Filter only negative class samples
    mask = (y_true != positive_class)
    
    if mask.sum() == 0:
        raise ValueError("No negative samples in ground truth")
    
    # False Positives: normal periods incorrectly flagged as seizures
    false_positives = ((y_true != positive_class) & (y_pred == positive_class)).sum()
    
    # All actual normal periods
    condition_negative = mask.sum()
    
    # Calculate false alarm rate
    false_alarm_rate = false_positives / condition_negative
    
    return float(false_alarm_rate)


def calculate_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_class: int = 1
) -> float:
    """
    Calculate F1 score (harmonic mean of precision and recall).
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Balances precision and recall, useful when class distribution is imbalanced.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        positive_class: Label for positive class (default: 1)
    
    Returns:
        F1 score in range [0.0, 1.0]
    """
    # True Positives
    tp = ((y_true == positive_class) & (y_pred == positive_class)).sum()
    
    # False Positives
    fp = ((y_true != positive_class) & (y_pred == positive_class)).sum()
    
    # False Negatives
    fn = ((y_true == positive_class) & (y_pred != positive_class)).sum()
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 Score
    if (precision + recall) == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return float(f1)


def calculate_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate macro-averaged F1 score across all classes.

    Computes F1 for each class independently and returns the unweighted mean.
    This gives equal importance to each class regardless of support.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Macro F1 score in range [0.0, 1.0]
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1_scores = []
    for cls in classes:
        tp = ((y_true == cls) & (y_pred == cls)).sum()
        fp = ((y_true != cls) & (y_pred == cls)).sum()
        fn = ((y_true == cls) & (y_pred != cls)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if (precision + recall) == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return float(np.mean(f1_scores)) if f1_scores else 0.0


# =============================================================================
# Real-Time Performance Metrics
# =============================================================================

def measure_inference_latency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    num_iterations: int = 100
) -> Dict[str, float]:
    """
    Measure model inference latency for real-time deployment validation.
    
    Critical for clinical systems where sub-second response is required.
    Target: <500ms for real-time seizure detection.
    
    Args:
        model: PyTorch model to benchmark
        input_tensor: Sample input tensor
        device: Device to run inference on
        num_iterations: Number of inference runs for averaging (default: 100)
    
    Returns:
        Dictionary with latency statistics (mean, std, percentiles)
    
    Example:
        >>> model = MyModel().to('cuda')
        >>> sample_input = torch.randn(1, 19, 1280).to('cuda')
        >>> latency_stats = measure_inference_latency(model, sample_input, 'cuda')
        >>> print(f"Mean latency: {latency_stats['mean_ms']:.2f} ms")
    """
    model.eval()
    latencies = []
    
    # Warm-up runs (GPU initialization)
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Benchmark runs
    with torch.no_grad():
        for _ in range(num_iterations):
            # Synchronize GPU before timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(input_tensor)
            
            # Synchronize GPU after inference
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # Convert to milliseconds
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
    
    # Calculate statistics
    latencies = np.array(latencies)
    
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "num_iterations": num_iterations
    }


def calculate_throughput(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    duration_seconds: float = 10.0
) -> float:
    """
    Calculate model throughput (samples per second).
    
    Useful for batch processing scenarios and capacity planning.
    
    Args:
        model: PyTorch model
        input_tensor: Sample input tensor
        device: Device to run inference on
        duration_seconds: Duration to run benchmark (default: 10 seconds)
    
    Returns:
        Throughput in samples per second
    """
    model.eval()
    
    start_time = time.time()
    num_samples = 0
    
    with torch.no_grad():
        while (time.time() - start_time) < duration_seconds:
            _ = model(input_tensor)
            num_samples += input_tensor.size(0)
    
    elapsed_time = time.time() - start_time
    throughput = num_samples / elapsed_time
    
    return float(throughput)


# =============================================================================
# Comprehensive Evaluation Report
# =============================================================================

def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive evaluation report with all relevant metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names (default: ["Normal", "Anomaly"])
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    if class_names is None:
        class_names = ["Normal", "Anomaly"]
    
    # Calculate all metrics
    sensitivity = calculate_seizure_sensitivity(y_true, y_pred)
    specificity = calculate_specificity(y_true, y_pred)
    false_alarm_rate = calculate_false_alarm_rate(y_true, y_pred)
    f1 = calculate_f1_score(y_true, y_pred)
    
    # Overall accuracy
    accuracy = (y_true == y_pred).sum() / len(y_true)
    
    report = {
        "clinical_metrics": {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "false_alarm_rate": false_alarm_rate
        },
        "classification_metrics": {
            "accuracy": float(accuracy),
            "f1_score": f1
        },
        "class_names": class_names,
        "num_samples": len(y_true)
    }
    
    return report


# =============================================================================
# Utility Functions
# =============================================================================

def print_evaluation_report(report: Dict[str, Any]) -> None:
    """
    Pretty-print evaluation report to console.
    
    Args:
        report: Evaluation report from generate_evaluation_report()
    """
    print("=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    print(f"Total Samples: {report['num_samples']}")
    print()
    
    print("Clinical Metrics:")
    print(f"  Sensitivity (Recall):    {report['clinical_metrics']['sensitivity']:.4f}")
    print(f"  Specificity:             {report['clinical_metrics']['specificity']:.4f}")
    print(f"  False Alarm Rate:        {report['clinical_metrics']['false_alarm_rate']:.4f}")
    print()
    
    print("Classification Metrics:")
    print(f"  Accuracy:                {report['classification_metrics']['accuracy']:.4f}")
    print(f"  F1 Score:                {report['classification_metrics']['f1_score']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    # Example usage / unit test
    print("Testing metrics module...")
    print()
    
    # Create dummy predictions
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 1])
    
    # Generate report
    report = generate_evaluation_report(y_true, y_pred)
    print_evaluation_report(report)
