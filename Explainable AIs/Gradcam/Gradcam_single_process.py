"""
GradCAM_single_process.py

Explain a single abnormal EEG window using Grad-CAM.

- Loads a pre-trained Conv1D model and saved scaler.
- Loads one EEG window (.npy file).
- Computes Grad-CAM heatmap to highlight regions important for abnormality.
- Extracts predicted abnormal segments from the heatmap.
- Compares predictions with ground-truth abnormal segments
  (encoded in filename, e.g., 'subject1_1.66_1.97.npy' → start=1.66s, end=1.97s).
- Calculates Coverage, Precision, and IoU metrics for that EEG window.
- Optionally displays plots and prints metrics.
"""

import os
import numpy as np
import tensorflow as tf
import joblib
from scipy.ndimage import gaussian_filter1d
import scipy.signal
import matplotlib.pyplot as plt

# -------------------------------
# 1. HELPERS
# -------------------------------

def extract_hand_labeled_segment(file_path):
    """Extract start and end times of abnormal segments from filename."""
    file_name = os.path.basename(file_path)
    segments = file_name.split('_')[-2:]
    start_time = float(segments[0])
    end_time = float(segments[1].replace('.npy', ''))
    return start_time, end_time


def load_eeg_data(file_path):
    """Load EEG data from .npy file."""
    return np.load(file_path)


def grad_cam(model, eeg_signal, layer_name='conv1d_3'):
    """Compute Grad-CAM heatmap for a given EEG signal."""
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

    # Smooth and resample
    heatmap = gaussian_filter1d(heatmap, sigma=10)
    heatmap = scipy.signal.resample(heatmap, 400)
    return heatmap


def extract_predicted_segments(heatmap, threshold=0.05, sampling_rate=200, threshold_gap=0.1):
    """Extract abnormal segments from Grad-CAM heatmap."""
    predicted_segments = []
    important_indices = np.where(heatmap > threshold)[0]

    if len(important_indices) > 0:
        gap_samples = int(threshold_gap * sampling_rate)
        start_idx = important_indices[0]

        for i in range(1, len(important_indices)):
            if important_indices[i] - important_indices[i - 1] > gap_samples:
                end_idx = important_indices[i - 1]
                predicted_segments.append((round(start_idx/sampling_rate, 2), round(end_idx/sampling_rate, 2)))
                start_idx = important_indices[i]

        predicted_segments.append((round(start_idx/sampling_rate, 2), round(important_indices[-1]/sampling_rate, 2)))

    return predicted_segments


def calculate_metrics(actual_start, actual_end, predicted_segments):
    """Compute Coverage, Precision, and IoU between predicted and true segments."""
    real_duration = actual_end - actual_start
    correct_overlap = 0
    identified_duration = 0

    for segment in predicted_segments:
        s, e = segment
        identified_duration += (e - s)
        overlap_start = max(actual_start, s)
        overlap_end = min(actual_end, e)
        if overlap_start < overlap_end:
            correct_overlap += (overlap_end - overlap_start)

    coverage = correct_overlap / real_duration
    precision = correct_overlap / identified_duration if identified_duration > 0 else 0
    union_duration = real_duration + identified_duration - correct_overlap
    iou = correct_overlap / union_duration if union_duration > 0 else 0
    return coverage, precision, iou


# -------------------------------
# 2. MAIN PROCESS
# -------------------------------

def run_gradcam_single(eeg_file, model, scaler, layer_name='conv1d_3', threshold=0.05):
    actual_start, actual_end = extract_hand_labeled_segment(eeg_file)
    eeg_signal = load_eeg_data(eeg_file)

    # Scale and reshape
    eeg_signal_reshaped = eeg_signal.reshape(1, -1)
    eeg_signal_scaled = scaler.transform(eeg_signal_reshaped).flatten()
    sample_input = np.expand_dims(eeg_signal_scaled, axis=(0, -1))

    # Predict and explain
    pred_prob = float(model.predict(sample_input, verbose=0)[0][0])
    heatmap = grad_cam(model, sample_input, layer_name=layer_name)
    predicted_segments = extract_predicted_segments(heatmap, threshold=threshold)

    # Metrics
    coverage, precision, iou = calculate_metrics(actual_start, actual_end, predicted_segments)

    # Print results
    print(f"\n📄 {os.path.basename(eeg_file)}")
    print(f"   Ground truth: {actual_start:.2f}s – {actual_end:.2f}s")
    print(f"   Prediction probability: {pred_prob:.3f}")
    print(f"   Predicted segments: {predicted_segments}")
    print(f"   Coverage={coverage:.3f}, Precision={precision:.3f}, IoU={iou:.3f}")

    # Optional plot
    plt.figure(figsize=(12, 4))
    plt.plot(np.linspace(0, 2, len(eeg_signal)), eeg_signal, label='EEG signal', color='black', alpha=0.7)
    for s, e in predicted_segments:
        plt.axvspan(s, e, color='red', alpha=0.3)
    plt.axvspan(actual_start, actual_end, color='green', alpha=0.3, label='Ground Truth')
    plt.title('Grad-CAM Predicted Abnormal Segments')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


# -------------------------------
# 3. SCRIPT ENTRY POINT
# -------------------------------

if __name__ == "__main__":
    # Paths to your model, scaler, and EEG file
    MODEL_PATH = "models/model.h5"
    SCALER_PATH = "models/scaler.pkl"
    EEG_FILE = "data/abnormal_windows/sample_abnormal_1.66_1.97.npy"

    # Load model and scaler
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Run Grad-CAM explanation
    run_gradcam_single(EEG_FILE, model, scaler, layer_name='conv1d_3', threshold=0.06)
