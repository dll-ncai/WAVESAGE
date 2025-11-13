"""
shap_batch_analysis.py
----------------------
Batch evaluation of EEG window explanations using SHAP (SHapley Additive Explanations).

This script:
1. Loads a trained EEG classification model and background samples.
2. Iterates through all EEG `.npy` files in a folder (abnormal EEG windows).
3. Computes SHAP values for each EEG sample to identify time regions contributing
   to the abnormal classification.
4. Groups SHAP-important time points into segments.
5. Calculates evaluation metrics (Coverage, Precision, IoU) by comparing SHAP-predicted
   abnormal segments against hand-labeled ground-truth intervals.
6. Prints and aggregates average metrics across all test files.

Before running:
---------------
- Ensure you have the following files available:
    data/
        X_train.npy
        abnormal_windows/test_set/
            sample_0_1.01_1.83.npy
            sample_1_0.25_1.75.npy
            ...
    models/
        eeg_classifier.h5
        scaler.pkl

- Make sure the model is already loaded as `model` and `X_train` is accessible.
"""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# SHAP Initialization
# ------------------------------------------------------------
shap.initjs()

# Create background sample set (subset of training data)
background = X_train[np.random.choice(X_train.shape[0], 400, replace=False)]

# EEG sampling parameters
sampling_rate = 200
duration = 2
time_steps = np.arange(sampling_rate * duration) / sampling_rate

# ------------------------------------------------------------
# Metric Computation Function
# ------------------------------------------------------------
def calculate_metrics(hand_start, hand_end, predicted_segments):
    """Compute Coverage, Precision, and IoU for predicted abnormal segments."""
    real_duration = hand_end - hand_start
    correct_overlap = 0
    identified_duration = 0

    for segment in predicted_segments:
        identified_start = segment[0]
        identified_end = segment[-1]
        identified_duration += (identified_end - identified_start)

        overlap_start = max(hand_start, identified_start)
        overlap_end = min(hand_end, identified_end)
        if overlap_start < overlap_end:
            correct_overlap += (overlap_end - overlap_start)

    coverage = correct_overlap / real_duration
    precision = correct_overlap / identified_duration if identified_duration > 0 else 0
    union_duration = real_duration + identified_duration - correct_overlap
    iou = correct_overlap / union_duration if union_duration > 0 else 0

    return coverage, precision, iou

# ------------------------------------------------------------
# Segment Grouping Function
# ------------------------------------------------------------
def group_segments(time_steps, max_gap=0.1):
    """Group consecutive time steps into continuous abnormal segments."""
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

# ------------------------------------------------------------
# SHAP Evaluation Loop
# ------------------------------------------------------------
# Path to EEG windows (anonymous version)
eeg_folder = "data/abnormal_windows/test_set"

iou_list, precision_list, coverage_list = [], [], []
file_counter = 0

for file_name in sorted(os.listdir(eeg_folder)):
    if not file_name.endswith(".npy"):
        continue

    file_counter += 1
    file_path = os.path.join(eeg_folder, file_name)
    print(f"\n🧠 Processing file #{file_counter}: {file_name}")

    # Load EEG data
    data = np.load(file_path)

    # Extract labeled segment from file name (e.g., *_1.01_1.83.npy)
    clean_name = file_name.replace(".npy", "")
    start_time_hand, end_time_hand = map(float, clean_name.split("_")[-2:])
    print(f"   → Hand-labeled segment: [{start_time_hand:.2f}s, {end_time_hand:.2f}s]")

    # Reshape for model input
    test_sample = data.reshape(1, -1, 1) if len(data.shape) == 1 else data

    # SHAP Explanation
    explainer = shap.GradientExplainer((model.input, model.output), background)
    shap_values = explainer.shap_values(test_sample)

    # Flatten SHAP values
    shap_flat = shap_values[0].reshape(-1)
    shap_pos = shap_flat[shap_flat > 0]

    # Normalize SHAP values to range [1, 10]
    shap_norm = 1 + 9 * (shap_pos - np.min(shap_pos)) / (np.max(shap_pos) - np.min(shap_pos))

    # Filter by threshold
    threshold = 3
    sig_times = time_steps[shap_flat > 0][shap_norm > threshold]
    sig_values = shap_norm[shap_norm > threshold]

    # Group time steps into segments
    segments = group_segments(sig_times, max_gap=0.1)

    # Compute metrics
    coverage, precision, iou = calculate_metrics(start_time_hand, end_time_hand, segments)
    print(f"   Coverage: {coverage:.2f}, Precision: {precision:.2f}, IoU: {iou:.2f}")

    # Append to lists
    iou_list.append(iou)
    precision_list.append(precision)
    coverage_list.append(coverage)

# ------------------------------------------------------------
# Results Summary
# ------------------------------------------------------------
print("\n==================== SUMMARY ====================")
print(f"Files processed: {file_counter}")
print(f"Average Coverage:  {np.mean(coverage_list) * 100:.2f}%")
print(f"Average Precision: {np.mean(precision_list) * 100:.2f}%")
print(f"Average IoU:       {np.mean(iou_list):.2f}")
print("=================================================")
