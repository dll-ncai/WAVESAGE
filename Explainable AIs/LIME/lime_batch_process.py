"""
Batch LIME Explanation for EEG Windows
--------------------------------------
This script processes all EEG windows (.npy files) in a directory,
runs predictions for each, computes LIME explanations,
aggregates importance into 0.2-second chunks,
and saves the results for each window.
"""

import os
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
from keras.models import load_model
import lime.lime_tabular


# -------------------------------------------------
# Utility Functions
# -------------------------------------------------

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


def plot_chunk_importance(chunk_scores, save_path=None):
    """Plot 0.2s chunk-level LIME importance."""
    num_chunks = len(chunk_scores)
    chunk_labels = [f"{i*0.2:.1f}-{(i+1)*0.2:.1f}s" for i in range(num_chunks)]

    plt.figure(figsize=(10, 5))
    plt.bar(range(num_chunks), chunk_scores, alpha=0.7)
    plt.xticks(range(num_chunks), chunk_labels, rotation=45)
    plt.xlabel("Chunk (0.2s each)")
    plt.ylabel("Importance Score")
    plt.title("LIME Chunk Importance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# -------------------------------------------------
# Batch LIME Processing Pipeline
# -------------------------------------------------

def process_folder_with_lime(
    folder_path,
    model_path,
    scaler_path,
    X_train_scaled,
    save_results_to="lime_results"
):
    """
    Run LIME on every .npy window file in a directory.
    Saves chunk importance arrays into .npy files and plots.
    """

    os.makedirs(save_results_to, exist_ok=True)

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    predict_fn = define_predict_fn(model, scaler)

    X_train_flat = X_train_scaled.reshape(X_train_scaled.shape[0], 400)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_flat,
        mode="classification",
        feature_names=[f"Timestep_{i}" for i in range(400)],
        discretize_continuous=False
    )

    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])

    print(f"Found {len(files)} EEG windows.")

    for idx, filename in enumerate(files):
        file_path = os.path.join(folder_path, filename)

        print(f"\n[{idx+1}/{len(files)}] Processing: {filename}")

        # Load & preprocess window
        original_window, scaled_window = load_and_preprocess_window(file_path, scaler)

        # Predict abnormality
        prob = model.predict(scaled_window)[0][0]
        pred_class = int(prob >= 0.5)
        print(f"Prediction: {prob:.4f} (Class {pred_class})")

        # Run LIME
        explanation = explainer.explain_instance(
            data_row=original_window.reshape(400),
            predict_fn=predict_fn,
            num_features=400,
            num_samples=2000
        )

        feature_importance = dict(explanation.local_exp[1])
        chunk_scores = aggregate_chunk_importance(feature_importance)

        # Save chunk importance
        save_array_path = os.path.join(save_results_to, filename.replace(".npy", "_lime.npy"))
        np.save(save_array_path, chunk_scores)

        # Save plot
        save_plot_path = os.path.join(save_results_to, filename.replace(".npy", "_lime.png"))
        plot_chunk_importance(chunk_scores, save_plot_path)

    print("\nBatch processing completed.")


# -------------------------------------------------
# Example Usage (Paths Anonymized)
# -------------------------------------------------

if __name__ == "__main__":
    folder_path = "data/eeg_windows/"            # Folder containing .npy EEG windows
    model_path = "models/eeg_classifier.h5"
    scaler_path = "models/scaler.pkl"

    # Example: load X_train_scaled (shape: N x 400 x 1)
    # X_train_scaled = np.load("data/train/X_train_scaled.npy")

    # process_folder_with_lime(folder_path, model_path, scaler_path, X_train_scaled)
    pass
