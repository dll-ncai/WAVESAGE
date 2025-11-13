"""
model.py

Train a Conv1D model on wavelet-decomposed EEG windows to classify
them as normal or abnormal.

- Loads EEG window datasets
- Extracts wavelet features
- Standardizes features and saves the scaler
- Defines and trains a Conv1D network
- Saves the trained model for later inference
"""


import os
import numpy as np
import pywt
from sklearn.preprocessing import StandardScaler
import joblib
from keras.models import Model
from keras.layers import (
    Input, Conv1D, MaxPooling1D, Dense, Dropout,
    GlobalAveragePooling1D, BatchNormalization
)
from keras.optimizers import Adam


# ================================================================
# Step 1: Configuration (Anonymous Paths)
# ================================================================
# Base directory: the current folder where model.py resides
BASE_DIR = os.getcwd()

# Data and model directories
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths
X_TRAIN_FILE = os.path.join(DATA_DIR, "x_train.npy")
X_TEST_FILE = os.path.join(DATA_DIR, "x_test.npy")
Y_TRAIN_FILE = os.path.join(DATA_DIR, "y_train.npy")
Y_TEST_FILE = os.path.join(DATA_DIR, "y_test.npy")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler_wavelet.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "conv1d_wavelet_model.h5")


# ================================================================
# Step 2: Load Data
# ================================================================
def load_data():
    """Load training and testing datasets."""
    X_train = np.load(X_TRAIN_FILE)
    X_test = np.load(X_TEST_FILE)
    y_train = np.load(Y_TRAIN_FILE)
    y_test = np.load(Y_TEST_FILE)

    print(f"✅ Data loaded successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape:  {y_test.shape}")

    return X_train, X_test, y_train, y_test


# ================================================================
# Step 3: Wavelet Feature Extraction
# ================================================================
def extract_wavelet_features(x, wavelet="db4", level=7):
    """
    Extract wavelet coefficients as features.

    Args:
        x (np.ndarray): EEG window (single sample).
        wavelet (str): Type of wavelet to use (default 'db4').
        level (int): Decomposition level (default 7).

    Returns:
        np.ndarray: Flattened feature vector from wavelet coefficients.
    """
    signal = x.squeeze()
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    modified_coeffs = coeffs[:6]  # Keep first 6 levels (most relevant)
    feature_vec = np.concatenate(modified_coeffs)
    return feature_vec


def build_feature_matrix(X):
    """Apply wavelet feature extraction to all samples."""
    print("Extracting wavelet features...")
    return np.array([extract_wavelet_features(x) for x in X])


# ================================================================
# Step 4: Feature Scaling
# ================================================================
def scale_features(X_train_features, X_test_features):
    """
    Standardize features and reshape them for Conv1D input.
    Saves the scaler for later use.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)

    # Save the scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    # Reshape for Conv1D (samples, timesteps, features=1)
    X_train_scaled = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
    X_test_scaled = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

    return X_train_scaled, X_test_scaled


# ================================================================
# Step 5: Conv1D Model Definition
# ================================================================
def build_conv1d_model(input_shape):
    """
    Define and compile a 1D CNN model for binary classification.
    """
    inputs = Input(shape=input_shape, name="input_layer")

    x = Conv1D(128, kernel_size=3, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(256, kernel_size=3, activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(256, kernel_size=3, activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    return model


# ================================================================
# Step 6: Main Training Pipeline
# ================================================================
def main():
    """Run the full training pipeline."""
    print("🚀 Starting training pipeline...")
    X_train, X_test, y_train, y_test = load_data()

    # Extract features
    X_train_features = build_feature_matrix(X_train)
    X_test_features = build_feature_matrix(X_test)
    print(f"Feature shape: {X_train_features.shape}")

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train_features, X_test_features)

    # Build model
    model = build_conv1d_model(input_shape=(X_train_scaled.shape[1], 1))

    # Train
    print("Training model...")
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=40,
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        verbose=1
    )

    # Save model
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    print("🎉 Training complete!")


# ================================================================
# Entry Point
# ================================================================
if __name__ == "__main__":
    main()
