"""
integrated_single_process.py

Compute Integrated Gradients explanations for a single EEG window.

This script:

Loads a trained EEG classification model.

Loads one EEG window (.npy file) from the dataset.

Computes Integrated Gradients to identify important features.

Extracts abnormal segments from attributions based on a threshold.

Compares predicted segments to ground-truth abnormal segments
encoded in the filename (e.g., 'subject1_1.23_1.56.npy').

Reports Coverage, Precision, and IoU metrics.

Plots the EEG signal with highlighted important segments.

"""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Step 1: Integrated Gradients Computation
# ------------------------------------------------------------
@tf.function
def compute_gradients(inputs, target_class_idx):
    """Compute gradients of the target class output with respect to inputs."""
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs)
        target_class_predictions = predictions[:, target_class_idx]
    return tape.gradient(target_class_predictions, inputs)


def integrated_gradients(inputs, baseline, target_class_idx, m_steps=50):
    """Compute Integrated Gradients for a given input and baseline."""
    inputs = tf.convert_to_tensor(inputs)
    baseline = tf.convert_to_tensor(baseline)
    scaled_inputs = [baseline + (float(i) / m_steps) * (inputs - baseline) for i in range(m_steps + 1)]

    avg_gradients = tf.zeros_like(inputs)
    for scaled_input in scaled_inputs:
        gradients = compute_gradients(tf.convert_to_tensor(scaled_input), target_class_idx)
        avg_gradients += gradients

    avg_gradients /= tf.cast(m_steps, dtype=avg_gradients.dtype)
    return (inputs - baseline) * avg_gradients


# ------------------------------------------------------------
# Step 2: Utility Functions
# ------------------------------------------------------------
def extract_abnormal_segment_from_filename(file_path):
    """Extract ground-truth abnormal segment (start, end) from filename."""
    try:
        parts = re.findall(r'_(\d+\.\d+)', os.path.basename(file_path))
        return float(parts[-2]), float(parts[-1])
    except Exception:
        raise ValueError(f"❌ Could not extract abnormal segment from filename: {file_path}")


def calculate_metrics(real_segment, identified_segments):
    """Compute Coverage, Precision, IoU, and overlap statistics."""
    real_start, real_end = real_segment
    real_duration = real_end - real_start
    correct_overlap, identified_duration = 0, 0

    for (start, end) in identified_segments:
        identified_duration += (end - start)
        overlap_start = max(real_start, start)
        overlap_end = min(real_end, end)
        if overlap_start < overlap_end:
            correct_overlap += (overlap_end - overlap_start)

    coverage = correct_overlap / real_duration if real_duration > 0 else 0
    precision = correct_overlap / identified_duration if identified_duration > 0 else 0
    union = real_duration + identified_duration - correct_overlap
    iou = correct_overlap / union if union > 0 else 0

    return coverage, precision, iou


def get_time_segments(time_in_seconds, attributions, threshold, merge_gap=0.1):
    """Group consecutive time points exceeding threshold into segments."""
    segments = []
    start = None

    for i, attr in enumerate(attributions):
        if attr > threshold:
            if start is None:
                start = time_in_seconds[i]
        elif start is not None:
            end = time_in_seconds[i - 1]
            if segments and (start - segments[-1][1] <= merge_gap):
                segments[-1] = (segments[-1][0], end)
            else:
                segments.append((start, end))
            start = None

    if start is not None:
        end = time_in_seconds[-1]
        if segments and (start - segments[-1][1] <= merge_gap):
            segments[-1] = (segments[-1][0], end)
        else:
            segments.append((start, end))
    return segments


def load_eeg_signal(file_path):
    """Load EEG signal from a .npy file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    signal = np.load(file_path)
    print(f"📂 Loaded {file_path} | Shape: {signal.shape}")
    return signal


def set_baseline(signal):
    """Return a zero baseline of the same shape as the input signal."""
    return np.zeros(signal.shape)


# ------------------------------------------------------------
# Step 3: Visualization
# ------------------------------------------------------------
def plot_integrated_gradients(signal, attributions, title, file_path, sampling_frequency=200, threshold=30):
    """Visualize EEG signal and Integrated Gradients importance."""
    time_in_seconds = np.arange(len(signal)) / sampling_frequency
    attributions = np.array(attributions).flatten()

    # Normalize attributions
    positive_attrs = attributions[attributions > 0]
    if len(positive_attrs) > 0:
        min_attr, max_attr = np.min(positive_attrs), np.max(positive_attrs)
        normalized = np.zeros_like(attributions)
        normalized[attributions > 0] = 1 + 99 * (attributions[attributions > 0] - min_attr) / (max_attr - min_attr)
    else:
        normalized = np.zeros_like(attributions)

    segments = get_time_segments(time_in_seconds, normalized, threshold)
    real_segment = extract_abnormal_segment_from_filename(file_path)
    coverage, precision, iou = calculate_metrics(real_segment, segments)

    print(f"\n📊 Metrics for {os.path.basename(file_path)}")
    print(f"• Coverage:  {coverage:.2f}")
    print(f"• Precision: {precision:.2f}")
    print(f"• IoU:       {iou:.2f}")
    print(f"• Predicted Segments: {segments}\n")

    plt.figure(figsize=(10, 5))
    plt.plot(time_in_seconds, signal, color='blue', label='EEG Signal')
    plt.plot(time_in_seconds, normalized, color='red', alpha=0.7, label='Importance (Normalized)')
    plt.scatter(
        time_in_seconds[normalized > threshold],
        normalized[normalized > threshold],
        color='lime', label=f'Importance > {threshold}'
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude / Importance")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Step 4: Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    # Example EEG window path (replace with your local path)
    FILE_PATH = "data/abnormal_windows/example_abnormal_window_0_1.23_1.56.npy"

    # Load model
    MODEL_PATH = "models/eeg_classifier.h5"
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load signal and prepare for model
    signal = load_eeg_signal(FILE_PATH)
    signal = signal.reshape(1, len(signal), 1)
    baseline = set_baseline(signal)

    # Compute Integrated Gradients
    target_class_idx = 0  # 0 = abnormal class
    attributions = integrated_gradients(signal, baseline, target_class_idx)

    # Plot and evaluate
    plot_integrated_gradients(
        signal[0],
        attributions[0],
        title="Integrated Gradients – EEG Abnormality Explanation",
        file_path=FILE_PATH,
        sampling_frequency=200,
        threshold=30
    )
