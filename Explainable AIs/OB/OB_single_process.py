"""
OB_single_process.py
----------------------------
Perform occlusion-based explainability on a single EEG window.

This script:
1. Loads a trained EEG classifier and its associated scaler.
2. Applies occlusion sensitivity analysis to identify time regions most critical
   for abnormal EEG predictions.
3. Extracts predicted abnormal segments from the importance map.
4. Compares them with hand-labeled ground-truth abnormal intervals.
5. Computes and prints evaluation metrics (Coverage, Precision, IoU).
6. Visualizes both the EEG signal and its occlusion-based importance map.

Before running:
---------------
- Ensure the following files are available:
    models/
        eeg_classifier.h5
        scaler.pkl
    data/
        abnormal_windows/sample_file_0_1.01_1.83.npy
"""


# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def extract_hand_labeled_segment(file_path):
    """
    Extract hand-labeled abnormal start and end times from the file name.
    Example filename: 'sample_0_1.01_1.83.npy'
    """
    file_name = os.path.basename(file_path)
    start_time, end_time = map(float, file_name.replace(".npy", "").split("_")[-2:])
    return start_time, end_time


def extract_predicted_segments(importance_map, threshold=0.5, step=None, threshold_gap=0.1):
    """
    Identify predicted abnormal segments from occlusion importance values.

    Args:
        importance_map (np.ndarray): Array of importance values over time.
        threshold (float): Minimum importance to consider as abnormal.
        step (float): Time step per sample (in seconds).
        threshold_gap (float): Maximum gap to merge nearby segments.

    Returns:
        list of tuples: [(start_time, end_time), ...]
    """
    predicted_segments = []
    important_times = []

    if step is None:
        step = 2 / len(importance_map)  # Default assumes 2-second EEG window

    # Identify time indices above threshold
    for i, value in enumerate(importance_map):
        if value > threshold:
            important_times.append(i * step)

    # Merge close time points into segments
    if important_times:
        start = important_times[0]
        for i in range(1, len(important_times)):
            if important_times[i] - important_times[i - 1] > threshold_gap:
                end = important_times[i - 1]
                predicted_segments.append((start, end))
                start = important_times[i]
        predicted_segments.append((start, important_times[-1]))

    return predicted_segments


def calculate_metrics(actual_start, actual_end, predicted_segments):
    """
    Compute Coverage, Precision, and IoU metrics.

    Args:
        actual_start (float): Ground-truth abnormal start time.
        actual_end (float): Ground-truth abnormal end time.
        predicted_segments (list): List of predicted abnormal segments.

    Returns:
        tuple: (coverage, precision, iou)
    """
    real_duration = actual_end - actual_start
    correct_overlap = 0
    identified_duration = 0

    for segment in predicted_segments:
        pred_start, pred_end = segment
        identified_duration += (pred_end - pred_start)

        overlap_start = max(actual_start, pred_start)
        overlap_end = min(actual_end, pred_end)
        if overlap_start < overlap_end:
            correct_overlap += (overlap_end - overlap_start)

    coverage = correct_overlap / real_duration if real_duration > 0 else 0
    precision = correct_overlap / identified_duration if identified_duration > 0 else 0
    union_duration = real_duration + identified_duration - correct_overlap
    iou = correct_overlap / union_duration if union_duration > 0 else 0

    return coverage, precision, iou

# ------------------------------------------------------------
# Load Model and Scaler
# ------------------------------------------------------------
model = tf.keras.models.load_model("models/eeg_classifier.h5")
scaler = joblib.load("models/scaler.pkl")

# Example EEG file (replace with your own)
file_path = "data/abnormal_windows/sample_file_0_1.01_1.83.npy"

# ------------------------------------------------------------
# EEG Preprocessing
# ------------------------------------------------------------
eeg_window = np.load(file_path)
eeg_window = eeg_window.reshape(1, -1, 1)
eeg_scaled = scaler.transform(eeg_window.reshape(1, -1)).reshape(eeg_window.shape)

# ------------------------------------------------------------
# Occlusion Analysis
# ------------------------------------------------------------
timesteps = eeg_window.shape[1]
window_size = 20
stride = 5
occlusion_importance = np.zeros(timesteps)

# Baseline prediction
baseline_prob = model.predict(eeg_scaled)[0, 0]
print(f"Baseline prediction probability: {baseline_prob:.4f}")

# Sliding window occlusion
for start in range(0, timesteps - window_size + 1, stride):
    modified_input = np.copy(eeg_scaled)
    modified_input[0, start:start + window_size, 0] = 0
    pred = model.predict(modified_input)[0, 0]
    delta = baseline_prob - pred
    occlusion_importance[start:start + window_size] += delta

# Normalize importance values
occlusion_importance /= np.max(np.abs(occlusion_importance))

# ------------------------------------------------------------
# Thresholding
# ------------------------------------------------------------
threshold = 0.4
above_threshold = occlusion_importance >= threshold
occlusion_display = np.where(above_threshold, occlusion_importance, 0)

print(f"Threshold: {threshold}")
print(f"Timesteps above threshold: {np.sum(above_threshold)}")

# ------------------------------------------------------------
# Segment Extraction and Metric Calculation
# ------------------------------------------------------------
actual_start, actual_end = extract_hand_labeled_segment(file_path)
predicted_segments = extract_predicted_segments(occlusion_importance, threshold, step=2 / timesteps)

coverage, precision, iou = calculate_metrics(actual_start, actual_end, predicted_segments)

print(f"Coverage: {coverage:.2f}")
print(f"Precision: {precision:.2f}")
print(f"IoU: {iou:.2f}")
print(f"Predicted segments: {predicted_segments}")

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
time = np.linspace(0, 2, timesteps)
plt.figure(figsize=(15, 6))

# EEG signal
plt.subplot(2, 1, 1)
plt.plot(time, eeg_scaled[0, :, 0], label="EEG Signal")
plt.title("Scaled EEG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Occlusion importance
plt.subplot(2, 1, 2)
plt.plot(time, occlusion_importance, label="Occlusion Importance", color="m")
plt.plot(time, occlusion_display, label="Above Threshold", color="r")

# Highlight regions above threshold
in_region = False
region_start = None
for i in range(timesteps):
    if above_threshold[i] and not in_region:
        in_region = True
        region_start = time[i]
    elif not above_threshold[i] and in_region:
        in_region = False
        plt.axvspan(region_start, time[i], color="red", alpha=0.2)
if in_region:
    plt.axvspan(region_start, time[-1], color="red", alpha=0.2)

plt.title(f"Occlusion-Based Importance Map (Threshold = {threshold})")
plt.xlabel("Time (s)")
plt.ylabel("Relative Importance")
plt.legend()
plt.tight_layout()
plt.show()
