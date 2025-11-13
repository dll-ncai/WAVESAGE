# ============================================================
# GRAD-CAM BASED EEG SEGMENT EXPLANATION AND EVALUATION SCRIPT
# ============================================================
"""
Gradcam_Batch_process.py

Explain and evaluate abnormal EEG segments using Grad-CAM.

- Loads a pre-trained Conv1D model and saved scaler.
- Processes EEG windows from a folder (files in .npy format).
- Computes Grad-CAM heatmaps to identify important regions for abnormality.
- Extracts predicted abnormal segments from heatmaps.
- Compares predictions with ground-truth abnormal segments 
  (encoded in filenames, e.g., 'subject1_1.66_1.97.npy' → start=1.66s, end=1.97s).
- Calculates Coverage, Precision, and IoU metrics for each EEG window.
- Saves detailed results in 'results/gradcam_results.csv'.
"""

import os
import numpy as np
import tensorflow as tf
import joblib
from scipy.ndimage import gaussian_filter1d
import scipy.signal
import csv

# ------------------------------------------------------------
# 1. FILE HELPERS
# ------------------------------------------------------------

def extract_hand_labeled_segment(file_path):
    """
    Extract start and end times of abnormal segments from filename.
    Example filename: 'subject1_1.66_1.97.npy' → (1.66, 1.97)
    """
    file_name = os.path.basename(file_path)
    segments = file_name.split('_')[-2:]
    start_time = float(segments[0])
    end_time = float(segments[1].replace('.npy', ''))
    return start_time, end_time


def load_eeg_data(file_path):
    """Load EEG data from .npy file."""
    return np.load(file_path)


# ------------------------------------------------------------
# 2. GRAD-CAM CORE IMPLEMENTATION
# ------------------------------------------------------------

def grad_cam(model, eeg_signal, layer_name='conv1d_3'):
    """
    Compute Grad-CAM heatmap for a given EEG signal.
    Smooths and resamples the map to match signal length (400 points).
    """
    eeg_signal = tf.convert_to_tensor(eeg_signal, dtype=tf.float32)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(eeg_signal, training=False)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=[0, 1])

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-6)

    # Smooth and resample for better temporal alignment
    heatmap = gaussian_filter1d(heatmap, sigma=10)
    heatmap = scipy.signal.resample(heatmap, 400)

    return heatmap


# ------------------------------------------------------------
# 3. SEGMENT EXTRACTION FROM HEATMAP
# ------------------------------------------------------------

def extract_predicted_segments(heatmap, threshold=0.05, sampling_rate=200, threshold_gap=0.1):
    """
    Extract predicted abnormal segments from Grad-CAM heatmap.
    Returns list of (start_time, end_time) tuples.
    """
    predicted_segments = []
    important_indices = np.where(heatmap > threshold)[0]

    if len(important_indices) > 0:
        gap_samples = int(threshold_gap * sampling_rate)
        start_idx = important_indices[0]

        for i in range(1, len(important_indices)):
            if important_indices[i] - important_indices[i - 1] > gap_samples:
                end_idx = important_indices[i - 1]
                predicted_segments.append((
                    round(start_idx / sampling_rate, 2),
                    round(end_idx / sampling_rate, 2)
                ))
                start_idx = important_indices[i]

        predicted_segments.append((
            round(start_idx / sampling_rate, 2),
            round(important_indices[-1] / sampling_rate, 2)
        ))

    return predicted_segments


# ------------------------------------------------------------
# 4. EVALUATION METRICS
# ------------------------------------------------------------

def calculate_metrics(actual_start, actual_end, predicted_segments):
    """
    Compute Coverage, Precision, and IoU between predicted and ground-truth segments.
    """
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

    coverage = correct_overlap / real_duration
    precision = correct_overlap / identified_duration if identified_duration > 0 else 0
    union_duration = real_duration + identified_duration - correct_overlap
    iou = correct_overlap / union_duration if union_duration > 0 else 0

    return coverage, precision, iou


# ------------------------------------------------------------
# 5. MODEL PREDICTION
# ------------------------------------------------------------

def predict_eeg(model, sample_input):
    """Run model prediction on a single EEG input sample."""
    return model.predict(sample_input)


# ------------------------------------------------------------
# 6. MAIN PIPELINE
# ------------------------------------------------------------

def run_gradcam_on_folder(data_folder, model, scaler, threshold=0.05, layer_name='conv1d_3'):
    """
    Run Grad-CAM on all EEG files in a folder and compute metrics.
    Saves results in 'results/gradcam_results.csv'.
    """
    eeg_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.npy')])
    if not eeg_files:
        print("⚠️ No EEG files found in the specified folder.")
        return

    os.makedirs("results", exist_ok=True)
    results = []
    total_coverage, total_precision, total_iou = 0, 0, 0

    for filename in eeg_files:
        file_path = os.path.join(data_folder, filename)

        # --- Extract true abnormal segment ---
        actual_start, actual_end = extract_hand_labeled_segment(file_path)
        eeg_signal = load_eeg_data(file_path)

        # --- Scale and reshape ---
        eeg_signal_reshaped = eeg_signal.reshape(1, -1)
        eeg_signal_scaled = scaler.transform(eeg_signal_reshaped).flatten()
        sample_input = np.expand_dims(eeg_signal_scaled, axis=(0, -1))

        # --- Predict and explain ---
        predictions = predict_eeg(model, sample_input)
        prediction_score = float(predictions[0][0])
        heatmap = grad_cam(model, sample_input, layer_name=layer_name)
        predicted_segments = extract_predicted_segments(heatmap, threshold=threshold)

        # --- Metrics ---
        coverage, precision, iou = calculate_metrics(actual_start, actual_end, predicted_segments)

        print(f"\n📄 {filename}")
        print(f"   Ground truth: {actual_start:.2f}s – {actual_end:.2f}s")
        print(f"   Prediction score: {prediction_score:.4f}")
        print(f"   Predicted segments: {predicted_segments}")
        print(f"   Coverage={coverage:.2f}, Precision={precision:.2f}, IoU={iou:.2f}")

        results.append({
            "filename": filename,
            "prediction_score": prediction_score,
            "coverage": coverage,
            "precision": precision,
            "iou": iou
        })

        total_coverage += coverage
        total_precision += precision
        total_iou += iou

    # --- Summary ---
    avg_coverage = total_coverage / len(results)
    avg_precision = total_precision / len(results)
    avg_iou = total_iou / len(results)

    print("\n==================== SUMMARY ====================")
    print(f"Files processed: {len(results)}")
    print(f"Average Coverage:  {avg_coverage:.2f}")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average IoU:       {avg_iou:.2f}")
    print("=================================================\n")

    # --- Save results ---
    output_csv = os.path.join("results", "gradcam_results.csv")
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "prediction_score", "coverage", "precision", "iou"])
        writer.writeheader()
        writer.writerows(results)

    print(f"💾 Results saved at: {output_csv}")


# ------------------------------------------------------------
# 7. SCRIPT ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":
    from tensorflow.keras.models import load_model

    # Default paths (can be changed by user)
    DATA_FOLDER = "data/abnormal_windows"
    MODEL_PATH = "models/model.h5"
    SCALER_PATH = "models/scaler.pkl"

    # Load model and scaler
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Run Grad-CAM evaluation
    run_gradcam_on_folder(
        data_folder=DATA_FOLDER,
        model=model,
        scaler=scaler,
        threshold=0.06,
        layer_name='conv1d_3'
    )
