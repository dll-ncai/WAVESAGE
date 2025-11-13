"""
shap_single_process.py

Explain EEG abnormal windows using SHAP on wavelet features.

- Loads trained Conv1D model and saved scaler
- Loads a single EEG window (.npy)
- Extracts SHAP values per wavelet coefficient
- Reconstructs important time segments contributing to abnormality
- Compares detected segments with ground-truth from filename
    (abnormal segments are encoded in the filename, e.g.,
     "subject1_0.6_1.45.npy" → start=0.6s, end=1.45s)
- Reports Coverage, Precision, and IoU
"""


import os
import re
import numpy as np
import pywt
import shap
import matplotlib.pyplot as plt
from joblib import load
from tensorflow.keras.models import load_model


# ================================================================
# 1. CONFIGURATION (Anonymous Paths)
# ================================================================
BASE_DIR = os.getcwd()

# Relative paths
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Example file (user should replace this with their own)
EEG_FILE = os.path.join(DATA_DIR, "abnormal_windows", "sample_abnormal.npy")

# Load model/scaler
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_wavelet.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "conv1d_wavelet_model.h5")

# Wavelet configuration
WAVELET = "db4"
LEVEL = 7
SELECTED_COEFFS = [5, 4, 3, 2, 1]

# Percent of SHAP importance to select from each coefficient level
TOP_PERCENT_BY_COEFF = {
    "Coeff 5": 0.25,
    "Coeff 4": 0.25,
    "Coeff 3": 0.25,
    "Coeff 2": 0.25,
    "Coeff 1": 0.25,
}

# Threshold for identifying active reconstruction segments
THRESHOLDS_BY_COEFF = {
    "Coeff 5": 1e-9,
    "Coeff 4": 5e-9,
    "Coeff 3": 8e-9,
    "Coeff 2": 5e-8,
    "Coeff 1": 5e-8,
}


# ================================================================
# 2. HELPER FUNCTIONS
# ================================================================
def build_coeff_slices_from_coeffs(coeffs):
    """Build index ranges for each wavelet coefficient array."""
    slices = {}
    start = 0
    for i, c in enumerate(coeffs):
        end = start + len(c)
        slices[f"Coeff {i}"] = (start, end)
        start = end
    return slices


def merge_close_segments(segments, gap=0.05):
    """Merge time segments that are very close together."""
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


def calculate_metrics(actual_start, actual_end, predicted_segments):
    """Compute coverage, precision, and IoU between ground truth and predicted."""
    real_duration = actual_end - actual_start
    correct_overlap = 0
    identified_duration = 0
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


def intersect_segments(seg_lists):
    """Find overlapping (common) segments across multiple coefficient levels."""
    if not seg_lists:
        return []
    common = seg_lists[0]
    for segs in seg_lists[1:]:
        new_common = []
        for a_s, a_e in common:
            for b_s, b_e in segs:
                s = max(a_s, b_s)
                e = min(a_e, b_e)
                if s < e:
                    new_common.append((s, e))
        common = new_common
        if not common:
            break
    return merge_close_segments(common, gap=0.01)


def majority_vote_segments(segments_dict, k=3):
    """Identify segments supported by at least 'k' coefficient levels."""
    all_points = []
    for segs in segments_dict.values():
        for s, e in segs:
            all_points.append((s, "start"))
            all_points.append((e, "end"))
    all_points.sort()
    active = 0
    merged = []
    seg_start = None
    for t, typ in all_points:
        if typ == "start":
            active += 1
            if active == k:
                seg_start = t
        else:
            if active == k and seg_start is not None:
                merged.append((seg_start, t))
            active -= 1
    return merge_close_segments(merged, gap=0.05)


# ================================================================
# 3. MAIN EXPLANATION LOGIC
# ================================================================
def main():
    # Load model and scaler
    if not (os.path.exists(SCALER_PATH) and os.path.exists(MODEL_PATH)):
        raise FileNotFoundError("Model or scaler not found. Train model.py first.")

    scaler = load(SCALER_PATH)
    model = load_model(MODEL_PATH)

    # Load EEG window
    if not os.path.exists(EEG_FILE):
        raise FileNotFoundError(f"EEG file not found: {EEG_FILE}")
    raw = np.load(EEG_FILE)

    # Generate time axis (assuming 2s window)
    signal_length = len(raw)
    time_axis = np.linspace(0, 2, signal_length)

    # Wavelet decomposition
    full_coeffs = pywt.wavedec(raw, WAVELET, level=LEVEL)
    coeffs_used = full_coeffs[:6]
    features = np.concatenate(coeffs_used)
    coeff_slices = build_coeff_slices_from_coeffs(coeffs_used)

    # Ground truth extraction from filename
    match = re.search(r"_(\d+\.\d+)_(\d+\.\d+)\.npy$", EEG_FILE)
    abnormal_start, abnormal_end = None, None
    if match:
        abnormal_start = float(match.group(1))
        abnormal_end = float(match.group(2))
        print(f"Ground truth: {abnormal_start:.2f}s – {abnormal_end:.2f}s")

    # Scale features
    features_scaled = scaler.transform([features]).reshape(1, -1, 1)

    # Background data (normal EEGs)
    background_path = os.path.join(DATA_DIR, "X_train_scaled.npy")
    y_train_path = os.path.join(DATA_DIR, "y_train.npy")
    if not os.path.exists(background_path) or not os.path.exists(y_train_path):
        raise FileNotFoundError("X_train_scaled.npy or y_train.npy not found. Required for SHAP baseline.")

    X_train_scaled = np.load(background_path)
    y_train = np.load(y_train_path)
    background = X_train_scaled[y_train == 0]
    background = background[np.random.choice(background.shape[0], 400, replace=False)]

    # Compute SHAP values
    explainer = shap.DeepExplainer(model, background)
    shap_vals = explainer.shap_values(features_scaled)
    shap_flat = shap_vals[0].reshape(-1)
    shap_pos = np.where(shap_flat > 0, shap_flat, 0)

    # ------------------------------------------------------------
    # Per-coefficient SHAP-based reconstruction
    # ------------------------------------------------------------
    important_mask = np.zeros_like(shap_pos, dtype=bool)
    for coeff_name, (start, end) in coeff_slices.items():
        pct = TOP_PERCENT_BY_COEFF.get(coeff_name, 0.20)
        coeff_shap = shap_pos[start:end]
        k = max(1, int(len(coeff_shap) * pct))
        top_idx_local = np.argsort(coeff_shap)[-k:] + start
        important_mask[top_idx_local] = True

    results = {}
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

        # Segment extraction
        segs, in_seg = [], False
        for i, v in enumerate(nz_mask):
            if v and not in_seg:
                seg_start = i
                in_seg = True
            elif not v and in_seg:
                segs.append((seg_start, i))
                in_seg = False
        if in_seg:
            segs.append((seg_start, len(nz_mask)))

        time_segs = [(time_axis[s], time_axis[e - 1]) for s, e in segs] if segs else []
        merged = merge_close_segments(time_segs, gap=0.1)
        segments_by_coeff[coeff_id] = merged

        if abnormal_start is not None:
            cov, pre, iou = calculate_metrics(abnormal_start, abnormal_end, merged)
            results[coeff_name] = {"segments": merged, "coverage": cov, "precision": pre, "iou": iou}

        # Plot
        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, raw, label="Original EEG", color="black", alpha=0.7)
        plt.plot(time_axis, reconstructed[:len(time_axis)], label=f"{coeff_name}", color="red", linewidth=1.5)
        for s, e in merged:
            plt.axvspan(s, e, color="red", alpha=0.3)
        if abnormal_start is not None:
            plt.axvspan(abnormal_start, abnormal_end, color="green", alpha=0.3, label="Ground Truth")
        plt.title(f"Important Reconstruction - {coeff_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Intersection and majority-vote segments
    common_segments = intersect_segments([segments_by_coeff[c] for c in SELECTED_COEFFS])
    majority_segments = majority_vote_segments(segments_by_coeff, k=3)

    # Plot intersection and majority
    for title, segs, color in [
        ("Intersection (Coeff 5–2)", common_segments, "orange"),
        ("Majority Vote (≥3 coeffs)", majority_segments, "blue"),
    ]:
        plt.figure(figsize=(12, 5))
        plt.plot(time_axis, raw, label="Original EEG", alpha=0.7)
        for s, e in segs:
            plt.axvspan(s, e, color=color, alpha=0.3)
        if abnormal_start is not None:
            plt.axvspan(abnormal_start, abnormal_end, color="green", alpha=0.3, label="Ground Truth")
        plt.legend()
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

    # Print results
    for coeff_name, res in results.items():
        print(f"\n{coeff_name}:")
        for seg in res["segments"]:
            print(f"  Segment: {seg[0]:.2f}s – {seg[1]:.2f}s")
        print(f"  Coverage: {res['coverage']:.3f}")
        print(f"  Precision: {res['precision']:.3f}")
        print(f"  IoU: {res['iou']:.3f}")

    if abnormal_start is not None:
        cov, pre, iou = calculate_metrics(abnormal_start, abnormal_end, common_segments)
        print(f"\nIntersection (Coeff 5–2): Coverage={cov:.3f}, Precision={pre:.3f}, IoU={iou:.3f}")
        cov, pre, iou = calculate_metrics(abnormal_start, abnormal_end, majority_segments)
        print(f"Majority-vote (≥3 coeffs): Coverage={cov:.3f}, Precision={pre:.3f}, IoU={iou:.3f}")


# ================================================================
# Entry Point
# ================================================================
if __name__ == "__main__":
    main()
