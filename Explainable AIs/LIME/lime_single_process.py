"""
LIME Explanation for EEG 2-Second Window
----------------------------------------
This script loads an EEG window (.npy), predicts abnormality using a trained model,
runs LIME to compute feature importances, aggregates them into 0.2s chunks, 
and visualizes the chunk-level explanation scores.
"""

import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
from keras.models import load_model
import lime.lime_tabular


# -------------------------------------------------
# Utility Functions
# -------------------------------------------------

def extract_time_from_filename(filepath):
    """Extract abnormal start/end timestamps from filename."""
    match = re.search(r'_(\d+\.\d+)_(\d+\.\d+)\.npy$', filepath)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def load_and_preprocess_window(filepath, scaler):
    """Load a 2-sec EEG window and scale it."""
    window = np.load(filepath)                      # shape (400,)
    window_scaled = scaler.transform(window.reshape(1, -1)) \
                       .reshape(1, 400, 1)          # shape (1, 400, 1)
    return window, window_scaled


def define_predict_fn(model, scaler):
    """Create a predict_fn for LIME that returns two-class probabilities."""
    def predict_fn(data):
        reshaped = data.reshape(data.shape[0], 400)
        normalized = scaler.transform(reshaped).reshape(len(data), 400, 1)
        preds = model.predict(normalized)
        return np.hstack([1 - preds, preds])
    return predict_fn


def aggregate_chunk_importance(feature_importance, chunk_size=40, total_length=400):
    """Convert timestep importance to chunk-level importance."""
    num_chunks = total_length // chunk_size
    chunk_scores = np.zeros(num_chunks)

    for i in range(num_chunks):
        start, end = i * chunk_size, (i + 1) * chunk_size
        chunk_scores[i] = sum(feature_importance.get(j, 0) for j in range(start, end))

    return chunk_scores / np.max(chunk_scores)  # normalize


def plot_chunk_importance(chunk_scores):
    """Plot 0.2s chunk-level LIME importance."""
    num_chunks = len(chunk_scores)
    chunk_labels = [f"{i*0.2:.1f}-{(i+1)*0.2:.1f}s" for i in range(num_chunks)]

    plt.figure(figsize=(10, 5))
    plt.bar(range(num_chunks), chunk_scores, alpha=0.7)
    plt.xticks(range(num_chunks), chunk_labels, rotation=45)
    plt.xlabel("Chunk (0.2s each)")
    plt.ylabel("Importance Score")
    plt.title("LIME Chunk Importance for EEG Window")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# Main LIME Explanation Pipeline
# -------------------------------------------------

def explain_window_with_lime(
    window_path,
    model_path,
    scaler_path,
    X_train_scaled
):
    sampling_rate = 200
    window_length = 400
    chunk_size = 40

    # Load model and scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Load and preprocess window
    original_window, scaled_window = load_and_preprocess_window(window_path, scaler)

    # Predict abnormality
    prob = model.predict(scaled_window)[0][0]
    pred_class = int(prob >= 0.5)
    print(f"Model Prediction: {prob:.4f} (Class: {pred_class})")

    # Prepare LIME
    X_train_flat = X_train_scaled.reshape(X_train_scaled.shape[0], 400)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_flat,
        mode="classification",
        feature_names=[f"Timestep_{i}" for i in range(400)],
        discretize_continuous=False
    )

    predict_fn = define_predict_fn(model, scaler)

    explanation = explainer.explain_instance(
        data_row=original_window.reshape(400),
        predict_fn=predict_fn,
        num_features=400,
        num_samples=2000
    )

    # Importance for class 1 (abnormal)
    feature_importance = dict(explanation.local_exp[1])

    # Aggregate into 0.2s chunks
    chunk_scores = aggregate_chunk_importance(
        feature_importance, chunk_size=chunk_size, total_length=window_length
    )

    # Visualize
    plot_chunk_importance(chunk_scores)

    return chunk_scores


# -------------------------------------------------
# Example Usage (Paths Anonymized)
# -------------------------------------------------

if __name__ == "__main__":
    window_path = "data/eeg_windows/sample_abnormal_window.npy"
    model_path = "models/eeg_classifier.h5"
    scaler_path = "models/scaler.pkl"

    # Load preprocessed training data (shape: N x 400 x 1)
    # Example placeholder:
    # X_train_scaled = np.load("data/train/X_train_scaled.npy")

    # chunk_scores = explain_window_with_lime(window_path, model_path, scaler_path, X_train_scaled)
    pass
