"""
rise_explainer_single.py
------------------------
Apply RISE (Randomized Input Sampling for Explanation) to identify
time segments in an EEG signal that contribute most to abnormal
classification.

This script:
1. Loads a trained EEG classification model and associated scaler.
2. Loads a single EEG window from `.npy` format.
3. Applies the RISE explainability method to compute importance maps.
4. Identifies abnormal intervals using a threshold.
5. Compares predicted intervals with hand-labeled ground truth
   from the filename to compute Coverage, Precision, and IoU.
6. Visualizes the EEG signal with important segments highlighted.

Before running:
---------------
- Place your model and scaler in:
    models/
        eeg_classifier.h5
        scaler.pkl

- Place your test EEG windows in:
    data/abnormal_windows/
        sample_6_C3_0.00_0.85.npy
        sample_7_T4_1.66_1.97.npy
        ...
"""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib
import os


# ------------------------------------------------------------
# Step 1: Load model and scaler
# ------------------------------------------------------------
MODEL_PATH = "models/eeg_classifier.h5"
SCALER_PATH = "models/scaler.pkl"
FILE_PATH = "data/abnormal_windows/sample_6_C3_0.00_0.85.npy"  # <-- replace with your file

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"EEG file not found at {FILE_PATH}")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ------------------------------------------------------------
# Step 2: Load and preprocess EEG sample
# ------------------------------------------------------------
raw_data = np.load(FILE_PATH)
if raw_data.ndim == 1:
    raw_data = raw_data.reshape(-1, 1)  # shape: (400, 1)

flat = raw_data.reshape(1, -1)
scaled_flat = scaler.transform(flat)
sample = scaled_flat.reshape(-1, 1)


# ------------------------------------------------------------
# Step 3: RISE Implementation
# ------------------------------------------------------------
def generate_masks(num_masks, length, p=0.5):
    """Generate random binary masks for RISE."""
    return np.random.binomial(1, p, size=(num_masks, length)).astype(np.float32)


def apply_masks(input_signal, masks):
    """Apply masks to input signal."""
    return np.expand_dims(masks, axis=2) * input_signal


def rise_1d(model, input_signal, num_masks=10000, p=0.5):
    """Compute RISE importance map for 1D input."""
    T = input_signal.shape[0]
    masks = generate_masks(num_masks, T, p)
    masked_inputs = apply_masks(input_signal, masks)
    preds = model.predict(masked_inputs, verbose=0).flatten()
    importance = preds.dot(masks) / num_masks
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    return importance


# ------------------------------------------------------------
# Step 4: Helper functions
# ------------------------------------------------------------
def get_intervals(indices, fs=200, merge_gap_sec=0.05):
    """Convert index array to time intervals and merge nearby ones."""
    from itertools import groupby
    from operator import itemgetter

    intervals = []
    for _, group in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
        g = list(map(itemgetter(1), group))
        intervals.append((g[0], g[-1]))

    # Merge close intervals
    merged = []
    for start, end in intervals:
        if not merged:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= int(merge_gap_sec * fs):
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
    return [(s / fs, e / fs) for s, e in merged]


def extract_hand_labeled_segment(file_path):
    """Extract ground truth abnormal segment (start, end) from filename."""
    file_name = os.path.basename(file_path)
    start_time, end_time = map(float, file_name.replace('.npy', '').split('_')[-2:])
    return start_time, end_time


def calculate_metrics(actual_start, actual_end, predicted_segments):
    """Compute coverage, precision, and IoU."""
    real_duration = actual_end - actual_start
    correct_overlap = 0
    identified_duration = 0

    for segment in predicted_segments:
        identified_start, identified_end = segment
        identified_duration += (identified_end - identified_start)

        overlap_start = max(actual_start, identified_start)
        overlap_end = min(actual_end, identified_end)
        if overlap_start < overlap_end:
            correct_overlap += (overlap_end - overlap_start)

    coverage = correct_overlap / real_duration if real_duration > 0 else 0
    precision = correct_overlap / identified_duration if identified_duration > 0 else 0
    union_duration = real_duration + identified_duration - correct_overlap
    iou = correct_overlap / union_duration if union_duration > 0 else 0
    return coverage, precision, iou


# ------------------------------------------------------------
# Step 5: Run RISE and compute importance map
# ------------------------------------------------------------
importance_map = rise_1d(model, sample, num_masks=10000, p=0.7)

# ------------------------------------------------------------
# Step 6: Identify abnormal intervals
# ------------------------------------------------------------
threshold = 0.5
abnormal_idxs = np.where(importance_map >= threshold)[0]
abnormal_intervals = get_intervals(abnormal_idxs, fs=200, merge_gap_sec=0.05)

print("\nAbnormal time intervals (seconds):")
if abnormal_intervals:
    for s, e in abnormal_intervals:
        print(f" • From {s:.3f}s to {e:.3f}s")
else:
    print(" • No regions found above threshold.")


# ------------------------------------------------------------
# Step 7: Plot EEG signal with highlighted intervals
# ------------------------------------------------------------
timesteps = np.arange(len(sample)) / 200.0
plt.figure(figsize=(12, 4))
plt.plot(timesteps, sample.squeeze(), label='EEG Signal')
plt.fill_between(timesteps, 0, importance_map, color='red', alpha=0.3, label='RISE Importance')
for s, e in abnormal_intervals:
    plt.axvspan(s, e, color='orange', alpha=0.4)
plt.title('EEG + RISE Importance (Abnormal Segments Highlighted)')
plt.xlabel('Time (s)')
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Step 8: Compute and print performance metrics
# ------------------------------------------------------------
actual_start, actual_end = extract_hand_labeled_segment(FILE_PATH)
coverage, precision, iou = calculate_metrics(actual_start, actual_end, abnormal_intervals)

print("\nPerformance Metrics:")
print(f" • Ground Truth Abnormal Segment: {actual_start:.2f}s to {actual_end:.2f}s")
print(f" • Coverage: {coverage:.2f}")
print(f" • Precision: {precision:.2f}")
print(f" • IoU: {iou:.2f}")
