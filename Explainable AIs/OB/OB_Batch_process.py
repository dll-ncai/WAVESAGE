"""
occlusion_single_process.py
----------------------------
Perform occlusion-based explainability on a single EEG window to identify
time segments most relevant for abnormal classification.

This script:
1. Loads a trained EEG classification model and its scaler.
2. Loads a single EEG window (.npy file) specified by the user.
3. Applies occlusion sensitivity to compute importance of each time segment.
4. Extracts predicted abnormal segments based on occlusion importance.
5. Calculates evaluation metrics (Coverage, Precision, IoU) against the
   hand-labeled ground-truth segment from the filename.
6. Visualizes the EEG signal and occlusion importance map for interpretation.

Before running:
---------------
- Ensure the following files exist:
    models/
        eeg_classifier.h5
        scaler.pkl
    data/
        abnormal_windows/
            sample_1_1.00_1.83.npy
            sample_2_1.66_1.97.npy
            ...
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
    Extract the hand-labeled abnormal start and end times from the filename.
    Example filename: 'sample_1_1.01_1.83.npy'
    """
    file_name = os.path.basename(file_path)
    start_time, end_time = map(float, file_name.replace('.npy', '').split('_')[-2:])
    return start_time, end_time


def extract_predicted_segments(importance_map, threshold=0.5, step=None, threshold_gap=0.1):
    """
    Identify predicted abnormal segments based on occlusion importance values.

    Args:
        importance_map (np.ndarray): Array of importance scores.
        threshold (float): Threshold for identifying significant regions.
        step (float): Time step corresponding to each importance value.
        threshold_gap (float): Maximum allowed gap (seconds) between consecutive
                               significant points to group them as a single segment.

    Returns:
        list of tuples: [(start_time, end_time), ...]
    """
    if step is None:
        step = 2 / len(importance_map)  # Default assumes 2-second EEG window

    important_times = [i * step for i, v in enumerate(importance_map) if v > threshold]
    if not important_times:
        return []

    predicted_segments = []
    start = important_times[0]

    for i in range(1, len(important_times)):
        if important_times[i] - important_times[i - 1] > threshold_gap:
            predicted_segments.append((start, important_times[i - 1]))
            start = important_times[i]

    predicted_segments.append((start, important_times[-1]))
    return predicted_segments


def calculate_metrics(actual_start, actual_end, predicted_segments):
    """
    Calculate coverage, precision, and IoU for predicted abnormal segments.

    Args:
        actual_start (float): Ground truth start time.
        actual_end (float): Ground truth end time.
        predicted_segments (list): Predicted time segments.

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
# Model and Data Setup
# ------------------------------------------------------------
MODEL_PATH = "models/eeg_classifier.h5"
SCALER_PATH = "models/scaler.pkl"
DATA_FOLDER = "data/abnormal_windows"

WINDOW_SIZE = 20
STRIDE = 5
THRESHOLD = 0.4

# Load model and scaler
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ------------------------------------------------------------
# Process All EEG Files
# ------------------------------------------------------------
file_list = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".npy")]
all_coverages, all_precisions, all_ious = [], [], []

for idx, file_name in enumerate(file_list, 1):
    file_path = os.path.join(DATA_FOLDER, file_name)
    print(f"\nProcessing file {idx}/{len(file_list)}: {file_name}")

    # Load and scale EEG window
    eeg_window = np.load(file_path).reshape(1, -1, 1)
    eeg_scaled = scaler.transform(eeg_window.reshape(1, -1)).reshape(eeg_window.shape)

    # Baseline prediction
    baseline_prob = model.predict(eeg_scaled, verbose=0)[0, 0]
    print(f"Baseline probability: {baseline_prob:.4f}")

    # Occlusion sensitivity
    timesteps = eeg_window.shape[1]
    occlusion_importance = np.zeros(timesteps)

    for start in range(0, timesteps - WINDOW_SIZE + 1, STRIDE):
        modified_input = np.copy(eeg_scaled)
        modified_input[0, start:start + WINDOW_SIZE, 0] = 0
        pred = model.predict(modified_input, verbose=0)[0, 0]
        delta = baseline_prob - pred
        occlusion_importance[start:start + WINDOW_SIZE] += delta

    occlusion_importance /= np.max(np.abs(occlusion_importance))
    above_threshold = occlusion_importance >= THRESHOLD

    # Extract predicted segments and calculate metrics
    actual_start, actual_end = extract_hand_labeled_segment(file_path)
    predicted_segments = extract_predicted_segments(
        occlusion_importance, threshold=THRESHOLD, step=2 / timesteps
    )
    coverage, precision, iou = calculate_metrics(actual_start, actual_end, predicted_segments)

    print(f"Coverage: {coverage:.3f}, Precision: {precision:.3f}, IoU: {iou:.3f}")
    print(f"Predicted segments: {predicted_segments}")

    all_coverages.append(coverage)
    all_precisions.append(precision)
    all_ious.append(iou)

    # Save the last file's data for visualization
    if idx == len(file_list):
        last_eeg_scaled = eeg_scaled
        last_occlusion_importance = occlusion_importance
        last_above_threshold = above_threshold
        last_timesteps = timesteps
        last_file_name = file_name

# ------------------------------------------------------------
# Print Average Metrics
# ------------------------------------------------------------
avg_coverage = np.mean(all_coverages)
avg_precision = np.mean(all_precisions)
avg_iou = np.mean(all_ious)

print("\n=== Average Metrics Across All Files ===")
print(f"Average Coverage: {avg_coverage:.3f}")
print(f"Average Precision: {avg_precision:.3f}")
print(f"Average IoU: {avg_iou:.3f}")

# ------------------------------------------------------------
# Visualization for the Last File
# ------------------------------------------------------------
time = np.linspace(0, 2, last_timesteps)
occlusion_display = np.where(last_above_threshold, last_occlusion_importance, 0)

plt.figure(figsize=(15, 6))

# Plot EEG Signal
plt.subplot(2, 1, 1)
plt.plot(time, last_eeg_scaled[0, :, 0], label="EEG Signal")
plt.title(f"Scaled EEG Signal - {last_file_name}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot Occlusion Importance
plt.subplot(2, 1, 2)
plt.plot(time, last_occlusion_importance, label="Occlusion Importance", color="m")
plt.plot(time, occlusion_display, label="Above Threshold", color="r")

# Highlight Regions Above Threshold
in_region, region_start = False, None
for i in range(last_timesteps):
    if last_above_threshold[i] and not in_region:
        in_region, region_start = True, time[i]
    elif not last_above_threshold[i] and in_region:
        in_region = False
        plt.axvspan(region_start, time[i], color="red", alpha=0.2)
if in_region:
    plt.axvspan(region_start, time[-1], color="red", alpha=0.2)

plt.title(f"Occlusion-Based Importance Map (Threshold = {THRESHOLD})")
plt.xlabel("Time (s)")
plt.ylabel("Relative Importance")
plt.legend()
plt.tight_layout()
plt.show()
