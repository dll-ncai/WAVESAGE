"""
Shap_Batch_process.py

Perform SHAP-based explainability on multiple EEG windows.

- Loads a pre-trained Conv1D model and saved scaler
- Loads EEG windows from a folder
- Computes SHAP values per wavelet coefficient for predicted abnormal windows
- Reconstructs important time segments contributing to abnormality
- Compares detected segments with ground-truth extracted from filenames
    (abnormal segments are encoded in the filename, e.g.,
     "subject1_0.6_1.45.npy" → start=0.6s, end=1.45s)
- Calculates and reports Coverage, Precision, and IoU metrics
"""


import os
import re
import joblib
import shap
import pywt
import numpy as np
import warnings
from tensorflow.keras.models import load_model


# ===============================
# 1. SUPPRESS WARNINGS & LOGS
# ===============================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


# ===============================
# 2. CONFIGURATION
# ===============================
# 🔧 These should be updated by whoever clones the repo
MODEL_PATH = "path/to/conv1d_wavelet_model.keras"
SCALER_PATH = "path/to/scaler_wavelet.pkl"
INPUT_FOLDER = "path/to/abnormal_windows"
SHAP_SAVE_FOLDER = "path/to/shap_values"

# Wavelet decomposition settings
WAVELET = "db4"
LEVEL = 7
SELECTED_COEFFS = [5, 4, 3, 2, 1]

# SHAP settings
TOP_PERCENT_BY_COEFF = {
    "Coeff 5": 0.25,
    "Coeff 4": 0.25,
    "Coeff 3": 0.25,
    "Coeff 2": 0.25,
    "Coeff 1": 0.25,
}

THRESHOLDS_BY_COEFF = {
    "Coeff 5": 1e-10,
    "Coeff 4": 3e-9,
    "Coeff 3": 5e-9,
    "Coeff 2": 5e-8,
    "Coeff 1": 5e-8,
}


# ===============================
# 3. LOAD MODEL AND SCALER
# ===============================
print("Loading model and scaler...")
scaler = joblib.load(SCALER_PATH)
model = load_model(MODEL_PATH)
print("✅ Model and scaler loaded successfully.\n")


# ===============================
# 4. HELPER FUNCTIONS
# ===============================
def build_coeff_slices_from_coeffs(coeffs):
    """Create index slices for each wavelet coefficient level."""
    slices, start = {}, 0
    for i, c in enumerate(coeffs):
        end = start + len(c)
        slices[f"Coeff {i}"] = (start, end)
        start = end
    return slices


def merge_close_segments(segments, gap=0.05):
    """Merge time segments that are closer than 'gap' seconds apart."""
    if not segments:
        return []
    merged = [segments[0]]
    for curr in segments[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = curr
        if curr_start - prev_end <= gap:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(curr)
    return merged


def majority_vote_segments(segments_dict, k=3):
    """Combine segments from multiple coefficient levels using majority voting."""
    all_points = []
    for segs in segments_dict.values():
        for s, e in segs:
            all_points.append((s, 'start'))
            all_points.append((e, 'end'))
    all_points.sort()

    active = 0
    merged, seg_start = [], None
    for t, typ in all_points:
        if typ == 'start':
            active += 1
            if active == k:
                seg_start = t
        else:
            if active == k and seg_start is not None:
                merged.append((seg_start, t))
            active -= 1
    return merge_close_segments(merged, gap=0.05)


def calculate_metrics(actual_start, actual_end, predicted_segments):
    """Compute Coverage, Precision, and IoU for detected abnormal segments."""
    real_duration = actual_end - actual_start
    correct_overlap = identified_duration = 0

    for s, e in predicted_segments:
        identified_duration += (e - s)
        overlap_s = max(actual_start, s)
        overlap_e = min(actual_end, e)
        if overlap_s < overlap_e:
            correct_overlap += (overlap_e - overlap_s)

    coverage = correct_overlap / real_duration if real_duration > 0 else 0
    precision = correct_overlap / identified_duration if identified_duration > 0 else 0
    union_duration = real_duration + identified_duration - correct_overlap
    iou = correct_overlap / union_duration if union_duration > 0 else 0
    return coverage, precision, iou


# ===============================
# 5. MAIN PROCESSING LOOP
# ===============================
def process_windows():
    coverages, precisions, ious = [], [], []
    processed_files, skipped_files = 0, 0

    files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith('.npy')])
    print(f"Processing {len(files)} files...\n")

    for i, filename in enumerate(files, 1):
        file_path = os.path.join(INPUT_FOLDER, filename)
        raw = np.load(file_path)
        signal_length = len(raw)
        time_axis = np.linspace(0, 2, signal_length)  # assuming 2s windows

        # Wavelet decomposition
        full_coeffs = pywt.wavedec(raw, WAVELET, level=LEVEL)
        coeffs_used = full_coeffs[:6]
        features = np.concatenate(coeffs_used)
        coeff_slices = build_coeff_slices_from_coeffs(coeffs_used)

        # Ground truth from filename
        match = re.search(r'_(\d+\.\d+)_(\d+\.\d+)\.npy$', filename)
        if not match:
            continue
        abnormal_start = float(match.group(1))
        abnormal_end = float(match.group(2))

        # Model prediction
        features_scaled = scaler.transform([features]).reshape(1, -1, 1)
        pred_prob = model.predict(features_scaled, verbose=0)[0][0]

        if pred_prob <= 0.5:
            print(f"[{i}/{len(files)}] Skipped {filename} (pred_prob={pred_prob:.3f}) — predicted normal.\n")
            skipped_files += 1
            continue
        else:
            print(f"[{i}/{len(files)}] Processing {filename} (pred_prob={pred_prob:.3f}) — predicted abnormal.\n")
            processed_files += 1

        # SHAP computation
        # (User should define background data in their workspace)
        background = X_train_scaled[y_train == 0]  # placeholder for normal EEGs
        background = background[np.random.choice(background.shape[0], 400, replace=False)]

        explainer = shap.DeepExplainer(model, background)
        shap_vals = explainer.shap_values(features_scaled)
        shap_flat = shap_vals[0].reshape(-1)
        shap_pos = np.where(shap_flat > 0, shap_flat, 0)

        # Save SHAP values
        os.makedirs(SHAP_SAVE_FOLDER, exist_ok=True)
        shap_filename = os.path.splitext(filename)[0] + "_shap.npy"
        shap_path = os.path.join(SHAP_SAVE_FOLDER, shap_filename)
        np.save(shap_path, shap_flat)

        # Identify important time segments
        important_mask = np.zeros_like(shap_pos, dtype=bool)
        for coeff_name, (start, end) in coeff_slices.items():
            pct = TOP_PERCENT_BY_COEFF.get(coeff_name, 0.20)
            coeff_shap = shap_pos[start:end]
            k = max(1, int(len(coeff_shap) * pct))
            top_idx_local = np.argsort(coeff_shap)[-k:] + start
            important_mask[top_idx_local] = True

        segments_by_coeff = {}
        for coeff_id in SELECTED_COEFFS:
            coeff_name = f"Coeff {coeff_id}"
            start, end = coeff_slices[coeff_name]
            coeff_full = full_coeffs[coeff_id]
            coeff_mask = important_mask[start:end]

            coeffs_selected = [np.zeros_like(c) for c in full_coeffs]
            coeffs_selected[coeff_id][coeff_mask] = coeff_full[coeff_mask]
            reconstructed = pywt.waverec(coeffs_selected, WAVELET)

            threshold = THRESHOLDS_BY_COEFF.get(coeff_name, 1e-9)
            nz_mask = np.abs(reconstructed) > threshold

            segs, in_seg = [], False
            for j, v in enumerate(nz_mask):
                if v and not in_seg:
                    seg_start = j
                    in_seg = True
                elif not v and in_seg:
                    segs.append((seg_start, j))
                    in_seg = False
            if in_seg:
                segs.append((seg_start, len(nz_mask)))

            time_segs = [(time_axis[s], time_axis[e-1]) for s, e in segs] if segs else []
            merged = merge_close_segments(time_segs, gap=0.1)
            segments_by_coeff[coeff_id] = merged

        # Majority voting and metrics
        majority_segments = majority_vote_segments(segments_by_coeff, k=3)
        cov, pre, iou = calculate_metrics(abnormal_start, abnormal_end, majority_segments)
        coverages.append(cov)
        precisions.append(pre)
        ious.append(iou)

        print(f"  Segments: {len(majority_segments)}")
        print(f"  Coverage: {cov:.3f}, Precision: {pre:.3f}, IoU: {iou:.3f}\n")

    # Summary
    print("\n======== FINAL RESULTS ========")
    print(f"Total files processed: {processed_files}")
    print(f"Files skipped (predicted normal): {skipped_files}")

    if coverages:
        print(f"Average Coverage:  {np.mean(coverages):.3f}")
        print(f"Average Precision: {np.mean(precisions):.3f}")
        print(f"Average IoU:       {np.mean(ious):.3f}")
    else:
        print("No valid abnormal-predicted files processed.")


# ===============================
# 6. SCRIPT ENTRY POINT
# ===============================
if __name__ == "__main__":
    process_windows()
