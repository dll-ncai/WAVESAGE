"""
Shap_single_process.py
----------------
Perform SHAP-based explainability on EEG window classifications.

This script:
1. Loads a trained EEG classification model and EEG window.
2. Uses SHAP (SHapley Additive exPlanations) to identify time regions contributing
   to the abnormal prediction.
3. Computes evaluation metrics (Coverage, Precision, IoU) by comparing SHAP-based
   important regions with hand-labeled abnormal segments.
4. Visualizes the EEG signal with both SHAP-highlighted and hand-labeled segments.

Before running:
---------------
- Ensure the following files are available:
    models/eeg_classifier.h5
    models/scaler.pkl
    data/train_samples.npy
    data/abnormal_windows/sample_file_0_1.01_1.83.npy
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import shap
    # Example: load your model and training data
from tensorflow.keras.models import load_model
import joblib
# ============================================================
# Configuration
# ============================================================

SAMPLING_RATE = 200   # Hz (Adjust to your EEG setup)
DURATION = 2          # seconds
MAX_LENGTH = SAMPLING_RATE * DURATION
THRESHOLD = 3         # Threshold for normalized SHAP values (1–10)
MAX_GAP = 0.1         # Maximum gap (in seconds) for grouping segments

# ============================================================
# Utility Functions
# ============================================================

def extract_hand_labeled_segment(file_path):
    """Extract start and end times of the hand-labeled abnormal segment from filename."""
    file_name = os.path.basename(file_path).replace('.npy', '')
    start_time, end_time = map(float, file_name.split('_')[-2:])
    return start_time, end_time


def load_eeg_data(file_path):
    """Load EEG signal from .npy file."""
    return np.load(file_path)


def group_segments(time_steps, max_gap=0.1):
    """Group time steps into continuous abnormal segments."""
    if len(time_steps) == 0:
        return []

    segments = []
    current_segment = [time_steps[0]]

    for i in range(1, len(time_steps)):
        if time_steps[i] - time_steps[i - 1] <= max_gap:
            current_segment.append(time_steps[i])
        else:
            segments.append(current_segment)
            current_segment = [time_steps[i]]

    if current_segment:
        segments.append(current_segment)

    return segments


def calculate_metrics(hand_start, hand_end, predicted_segments):
    """Compute Coverage, Precision, and IoU between ground-truth and predicted segments."""
    real_duration = hand_end - hand_start
    correct_overlap = identified_duration = 0

    for segment in predicted_segments:
        identified_start = segment[0]
        identified_end = segment[-1]
        identified_duration += (identified_end - identified_start)

        overlap_start = max(hand_start, identified_start)
        overlap_end = min(hand_end, identified_end)
        if overlap_start < overlap_end:
            correct_overlap += (overlap_end - overlap_start)

    coverage = correct_overlap / real_duration if real_duration > 0 else 0
    precision = correct_overlap / identified_duration if identified_duration > 0 else 0
    union_duration = real_duration + identified_duration - correct_overlap
    iou = correct_overlap / union_duration if union_duration > 0 else 0

    return coverage, precision, iou


def plot_eeg_with_segments(time_steps, eeg_signal, hand_segment, shap_segments, threshold):
    """Visualize EEG signal with SHAP-important and hand-labeled segments."""
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, eeg_signal.flatten(), label='EEG Signal')

    # Highlight SHAP-important segments
    for segment in shap_segments:
        plt.axvspan(segment[0], segment[-1], color='red', alpha=0.5, label='Important Segment (SHAP)')

    # Highlight hand-labeled abnormal segment
    plt.axvspan(hand_segment[0], hand_segment[1], color='green', alpha=0.3, label='Hand-labeled Abnormal Segment')

    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys())

    plt.title(f'EEG Signal with Important Segments Highlighted (Threshold: {threshold})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('EEG Signal Amplitude')
    plt.tight_layout()
    plt.show()


# ============================================================
# SHAP Analysis
# ============================================================

def run_shap_analysis(model, X_train, eeg_file_path, threshold=THRESHOLD, sampling_rate=SAMPLING_RATE):
    """Run SHAP-based abnormal EEG segment detection on a single EEG window."""
    shap.initjs()

    # Prepare SHAP background set
    background = X_train[np.random.choice(X_train.shape[0], 1000, replace=False)]

    # Load EEG data and extract ground truth
    data = load_eeg_data(eeg_file_path)
    start_time_hand, end_time_hand = extract_hand_labeled_segment(eeg_file_path)

    print(f"Hand-labeled abnormal segment: [{start_time_hand:.2f}s, {end_time_hand:.2f}s]")
    print(f"EEG data shape: {data.shape}")

    # Reshape test sample
    if len(data.shape) == 1:
        test_sample = data.reshape(1, -1, 1)
    else:
        test_sample = data

    # Compute SHAP values
    explainer = shap.GradientExplainer((model.input, model.output), background)
    shap_values = explainer.shap_values(test_sample)
    shap_values_flat = shap_values[0].reshape(-1)

    # Filter positive SHAP contributions
    shap_values_pos = shap_values_flat[shap_values_flat > 0]

    # Normalize SHAP values to 1–10 scale
    shap_values_norm = 1 + 9 * (shap_values_pos - np.min(shap_values_pos)) / (
        np.max(shap_values_pos) - np.min(shap_values_pos)
    )

    # Filter significant SHAP values
    time_steps = np.arange(len(shap_values_flat)) / sampling_rate
    significant_times = time_steps[shap_values_flat > 0][shap_values_norm > threshold]
    significant_values = shap_values_norm[shap_values_norm > threshold]

    # Group into segments
    segments = group_segments(significant_times, max_gap=MAX_GAP)

    print(f"\nAbnormal Time Segments (Threshold={threshold}):")
    for seg in segments:
        print(f"Segment: [{seg[0]:.3f}s, {seg[-1]:.3f}s]")

    # Evaluate performance
    coverage, precision, iou = calculate_metrics(start_time_hand, end_time_hand, segments)
    print(f"\nCoverage: {coverage * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"IoU: {iou:.2f}")

    # Plot EEG with important regions
    plot_eeg_with_segments(time_steps, test_sample, (start_time_hand, end_time_hand), segments, threshold)


# ============================================================
# Example Run (User sets paths)
# ============================================================

if __name__ == "__main__":
    # Example placeholders — replace with your own paths
    MODEL_PATH = "models/eeg_classifier.h5"
    SCALER_PATH = "models/scaler.pkl"
    EEG_FILE_PATH = "data/abnormal_windows/sample_file_0_1.01_1.83.npy"

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_train = np.load("data/train_samples.npy")

    run_shap_analysis(model, X_train, EEG_FILE_PATH)
