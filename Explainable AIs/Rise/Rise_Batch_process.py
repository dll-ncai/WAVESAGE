"""
rise_single_process.py
-----------------------
Apply RISE (Randomized Input Sampling for Explanation) on a single EEG window
to identify abnormal time segments and compute explainability metrics.

This script:
------------
1. Loads a trained EEG classification model and scaler.
2. Loads a single EEG window stored as a `.npy` file.
3. Extracts the ground-truth abnormal segment from the filename. 
   The filename should encode the abnormal interval at the end, e.g.,
   `patientX_window_0.56_1.23.npy` indicates the abnormal segment spans
   from 0.56s to 1.23s within the 2-second EEG window.
4. Applies RISE to compute an importance map highlighting which time points
   contribute most to the abnormal classification.
5. Thresholds the importance map to identify predicted abnormal segments.
6. Computes Coverage, Precision, and IoU between predicted and ground-truth
   abnormal segments.
7. Visualizes both the EEG signal and the RISE-derived importance map,
   highlighting predicted abnormal regions.

Before running:
---------------
- Ensure the following files exist:
    models/
        eeg_classifier.h5
        scaler.pkl
    data/abnormal_windows_278/
        sample_file_0.56_1.23.npy
"""


# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib
import os
import pandas as pd


# ------------------------------------------------------------
# Step 1: Paths and setup
# ------------------------------------------------------------
MODEL_PATH = "models/eeg_classifier.h5"
SCALER_PATH = "models/scaler.pkl"
DATA_DIR = "data/abnormal_windows_278"        # directory containing .npy EEG windows
OUTPUT_CSV = "results/rise_explainer_278.csv"
os.makedirs("results", exist_ok=True)

# Load model and scaler
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"EEG directory not found at {DATA_DIR}")

print("✅ Loading model and scaler...")
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ------------------------------------------------------------
# Step 2: RISE and Utility Functions
# ------------------------------------------------------------
def generate_masks(num_masks, length, p=0.5):
    """Generate random binary masks for RISE."""
    return np.random.binomial(1, p, size=(num_masks, length)).astype(np.float32)

def apply_masks(input_signal, masks):
    """Apply binary masks to a 1D EEG signal."""
    return np.expand_dims(masks, axis=2) * input_signal

def rise_1d(model, input_signal, num_masks=10000, p=0.7):
    """Compute RISE importance map for 1D EEG input."""
    T = input_signal.shape[0]
    masks = generate_masks(num_masks, T, p)
    masked_inputs = apply_masks(input_signal, masks)
    preds = model.predict(masked_inputs, verbose=0).flatten()
    importance = preds.dot(masks) / num_masks
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    return importance

def get_intervals(indices, fs=200, merge_gap_sec=0.05):
    """Convert indices to time intervals and merge nearby intervals."""
    from itertools import groupby
    from operator import itemgetter
    intervals = []
    for _, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
        group = list(map(itemgetter(1), g))
        intervals.append((group[0], group[-1]))
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
    """Extract ground truth (start, end) times from filename."""
    fname = os.path.basename(file_path)
    start, end = map(float, fname.replace(".npy", "").split("_")[-2:])
    return start, end

def calculate_metrics(actual_start, actual_end, predicted_segments):
    """Compute coverage, precision, and IoU for predicted vs actual segments."""
    real_duration = actual_end - actual_start
    correct_overlap = 0
    identified_duration = 0
    for seg in predicted_segments:
        s, e = seg
        identified_duration += (e - s)
        overlap_start = max(actual_start, s)
        overlap_end = min(actual_end, e)
        if overlap_start < overlap_end:
            correct_overlap += (overlap_end - overlap_start)
    coverage = correct_overlap / real_duration if real_duration > 0 else 0
    precision = correct_overlap / identified_duration if identified_duration > 0 else 0
    union_duration = real_duration + identified_duration - correct_overlap
    iou = correct_overlap / union_duration if union_duration > 0 else 0
    return coverage, precision, iou


# ------------------------------------------------------------
# Step 3: Process all EEG windows
# ------------------------------------------------------------
results = []
file_list = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")]
file_list.sort()

print(f"\n🧠 Processing {len(file_list)} EEG windows...\n")

for i, fname in enumerate(file_list, 1):
    file_path = os.path.join(DATA_DIR, fname)
    print(f"[{i}/{len(file_list)}] Processing: {fname}")

    # Load EEG window
    eeg = np.load(file_path)
    if eeg.ndim == 1:
        eeg = eeg.reshape(-1, 1)
    flat = eeg.reshape(1, -1)
    scaled = scaler.transform(flat).reshape(-1, 1)

    # Apply RISE
    importance = rise_1d(model, scaled, num_masks=10000, p=0.7)
    threshold = 0.6
    idxs = np.where(importance >= threshold)[0]
    predicted_segments = get_intervals(idxs, fs=200, merge_gap_sec=0.05)

    # Compute metrics
    actual_start, actual_end = extract_hand_labeled_segment(file_path)
    coverage, precision, iou = calculate_metrics(actual_start, actual_end, predicted_segments)

    # Append result
    results.append({
        "File": fname,
        "Actual_Start": actual_start,
        "Actual_End": actual_end,
        "Predicted_Intervals": [(float(s), float(e)) for s, e in predicted_segments],
        "Coverage": round(coverage, 4),
        "Precision": round(precision, 4),
        "IoU": round(iou, 4)
    })

    print(f"  • Coverage: {coverage:.3f}, Precision: {precision:.3f}, IoU: {iou:.3f}")

# ------------------------------------------------------------
# Step 4: Save results to CSV
# ------------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Results saved to: {OUTPUT_CSV}")

# ------------------------------------------------------------
# Step 5: Summary Statistics
# ------------------------------------------------------------
avg_cov = df["Coverage"].mean()
avg_prec = df["Precision"].mean()
avg_iou = df["IoU"].mean()

print("\n📊 Average Metrics Across All Files:")
print(f" • Average Coverage:  {avg_cov:.3f}")
print(f" • Average Precision: {avg_prec:.3f}")
print(f" • Average IoU:       {avg_iou:.3f}")
