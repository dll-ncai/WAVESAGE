"""
Integrated Gradients Evaluation for EEG Abnormal Windows
-------------------------------------------------------
This script applies Integrated Gradients (IG) on 1D EEG window signals,
extracts important time intervals, compares them with ground-truth
abnormal segments from filenames, and calculates coverage, precision,
and IoU metrics for interpretability evaluation.


This is for batch process of windows
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ============================================================
# 1. Gradient Computation
# ============================================================

@tf.function
def compute_gradients(model, inputs, target_class_idx):
    """
    Compute gradients of target class output w.r.t. inputs.
    """
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs)
        target_predictions = predictions[:, target_class_idx]
    return tape.gradient(target_predictions, inputs)


def integrated_gradients(model, inputs, baseline, target_class_idx, m_steps=50):
    """
    Compute Integrated Gradients (IG) for given input.
    """
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)

    # Linear interpolation between baseline and input
    scaled_inputs = [
        baseline + (float(i) / m_steps) * (inputs - baseline)
        for i in range(m_steps + 1)
    ]

    avg_gradients = tf.zeros_like(inputs)
    for scaled_input in scaled_inputs:
        grads = compute_gradients(model, scaled_input, target_class_idx)
        avg_gradients += grads

    avg_gradients /= tf.cast(m_steps, tf.float32)
    return (inputs - baseline) * avg_gradients


# ============================================================
# 2. Utility Functions
# ============================================================

def extract_abnormal_segment_from_filename(file_path):
    """
    Extract real abnormal start and end times from filename.
    Example: 'patientX_T4_abnormal_window_10_0.56_1.02.npy'
    """
    match = re.search(r'_(\d+\.\d+)_(\d+\.\d+)\.npy$', file_path)
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        raise ValueError(f"Cannot extract segment from filename: {file_path}")


def calculate_metrics(real_segment, predicted_segments):
    """
    Calculate coverage, precision, and IoU for predicted abnormal regions.
    """
    real_start, real_end = real_segment
    real_duration = real_end - real_start
    correct_overlap = 0
    identified_duration = 0

    for seg in predicted_segments:
        seg_start, seg_end = seg
        identified_duration += seg_end - seg_start

        overlap_start = max(real_start, seg_start)
        overlap_end = min(real_end, seg_end)
        if overlap_start < overlap_end:
            correct_overlap += overlap_end - overlap_start

    coverage = correct_overlap / real_duration if real_duration > 0 else 0
    precision = correct_overlap / identified_duration if identified_duration > 0 else 0
    union = real_duration + identified_duration - correct_overlap
    iou = correct_overlap / union if union > 0 else 0

    return coverage, precision, iou


def get_time_segments(time_in_seconds, attributions, threshold, merge_gap=0.1):
    """
    Merge contiguous high-importance points into time intervals.
    """
    segments = []
    start = None

    for i, val in enumerate(attributions):
        if val > threshold:
            if start is None:
                start = time_in_seconds[i]
        else:
            if start is not None:
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


# ============================================================
# 3. Visualization and Metrics
# ============================================================

def plot_integrated_gradients(signal, attributions, file_path, sampling_frequency=200, threshold=50):
    """
    Plot EEG signal and IG importance values. Return metrics.
    """
    time_in_seconds = np.arange(len(signal)) / sampling_frequency
    attributions = np.array(attributions).flatten()

    # Normalize positive attributions (1–100 scale)
    pos_attr = attributions[attributions > 0]
    if len(pos_attr) > 0:
        min_val, max_val = np.min(pos_attr), np.max(pos_attr)
        norm_attr = np.zeros_like(attributions)
        norm_attr[attributions > 0] = 1 + 99 * (attributions[attributions > 0] - min_val) / (max_val - min_val)
    else:
        norm_attr = np.zeros_like(attributions)

    segments = get_time_segments(time_in_seconds, norm_attr, threshold)
    real_segment = extract_abnormal_segment_from_filename(file_path)
    coverage, precision, iou = calculate_metrics(real_segment, segments)

    print(f"\n🧠 File: {os.path.basename(file_path)}")
    print(f"  - Coverage: {coverage:.3f}")
    print(f"  - Precision: {precision:.3f}")
    print(f"  - IoU: {iou:.3f}")
    print(f"  - Predicted Segments: {segments}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_in_seconds, signal, label="EEG Signal", color="blue")
    plt.plot(time_in_seconds, norm_attr, label="IG Importance", color="red", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude / Importance")
    plt.title("Integrated Gradients - EEG Signal")
    plt.legend()
    plt.show()

    return coverage, precision, iou


# ============================================================
# 4. Core Loop
# ============================================================

def evaluate_folder(model, folder_path, sampling_frequency=200, threshold=50):
    """
    Run IG interpretation on all EEG .npy files in a folder.
    """
    coverage_list, precision_list, iou_list = [], [], []

    for fname in os.listdir(folder_path):
        if fname.endswith('.npy'):
            file_path = os.path.join(folder_path, fname)
            signal = np.load(file_path)
            signal = signal.reshape(1, len(signal), 1)
            baseline = np.zeros_like(signal)

            attributions = integrated_gradients(model, signal, baseline, target_class_idx=0)
            coverage, precision, iou = plot_integrated_gradients(signal[0], attributions[0],
                                                                 file_path, sampling_frequency, threshold)
            coverage_list.append(coverage)
            precision_list.append(precision)
            iou_list.append(iou)

    # === Average metrics ===
    avg_cov = np.mean(coverage_list) if coverage_list else 0
    avg_prec = np.mean(precision_list) if precision_list else 0
    avg_iou = np.mean(iou_list) if iou_list else 0

    print("\n📊 Average Results:")
    print(f"  - Coverage: {avg_cov:.3f}")
    print(f"  - Precision: {avg_prec:.3f}")
    print(f"  - IoU: {avg_iou:.3f}")

    return avg_cov, avg_prec, avg_iou


# ============================================================
# 5. Main Entry Point
# ============================================================

if __name__ == "__main__":
    # Load trained model
    from keras.models import load_model
    model = load_model("model_anonymized.h5")

    # === Anonymized input directory ===
    folder_path = "/path/to/anonymized_abnormal_windows/"

    # Run evaluation
    evaluate_folder(model, folder_path, sampling_frequency=200, threshold=40)
