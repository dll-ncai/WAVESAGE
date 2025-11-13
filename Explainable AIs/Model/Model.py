"""
Model Training for EEG Window Classification
---------------------------------------------
This script trains a 1D Convolutional Neural Network (Conv1D) to classify EEG windows
as normal or abnormal.

The workflow:
1. Loads preprocessed EEG window data (`x_train.npy`, `x_test.npy`, `y_train.npy`, `y_test.npy`).
2. Standardizes the EEG signals using a `StandardScaler`.
3. Builds a multi-layer Conv1D model with BatchNormalization, MaxPooling, and Dropout.
4. Trains the model and evaluates on a test set.
5. Saves the trained model and scaler for later use.

Usage:
------
- Place your preprocessed EEG window data in the `data/` folder.
- Run the script:
    python Model.py

Output:
-------
- Trained Conv1D model saved at `models/conv1d_eeg_model.h5`.
- StandardScaler saved at `models/scaler.pkl`.
- Training logs including accuracy and loss per epoch.
"""


# ================================================================
# 1. IMPORTS
# ================================================================
import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import (
    Input, Conv1D, MaxPooling1D,
    Dense, Dropout, GlobalAveragePooling1D,
    BatchNormalization
)

# ================================================================
# 2. PATH CONFIGURATION
# ================================================================
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths
X_TRAIN_FILE = os.path.join(DATA_DIR, "x_train.npy")
X_TEST_FILE  = os.path.join(DATA_DIR, "x_test.npy")
Y_TRAIN_FILE = os.path.join(DATA_DIR, "y_train.npy")
Y_TEST_FILE  = os.path.join(DATA_DIR, "y_test.npy")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH   = os.path.join(MODEL_DIR, "conv1d_eeg_model.h5")

# ================================================================
# 3. LOAD DATA
# ================================================================
print("\n📥 Loading EEG data...")

X_train = np.load(X_TRAIN_FILE)
X_test  = np.load(X_TEST_FILE)
y_train = np.load(Y_TRAIN_FILE)
y_test  = np.load(Y_TEST_FILE)

print(f"✅ X_train shape: {X_train.shape}")
print(f"✅ X_test shape:  {X_test.shape}")
print(f"✅ y_train shape: {y_train.shape}")
print(f"✅ y_test shape:  {y_test.shape}")

# ================================================================
# 4. STANDARDIZE SIGNALS
# ================================================================
print("\n⚙️ Standardizing EEG signals...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
X_test_scaled  = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

joblib.dump(scaler, SCALER_PATH)
print(f"✅ Scaler saved to {SCALER_PATH}")

# ================================================================
# 5. BUILD CONV1D MODEL
# ================================================================
print("\n🧠 Building Conv1D model...")

input_layer = Input(shape=(X_train_scaled.shape[1], 1), name='input_layer')

# Convolutional Block 1
conv1 = Conv1D(128, kernel_size=3, activation='relu', name="conv1d_1")(input_layer)
bn1 = BatchNormalization()(conv1)
pool1 = MaxPooling1D(pool_size=2, name="max_pooling1d_1")(bn1)
dropout1 = Dropout(0.2, name="dropout_1")(pool1)

# Convolutional Block 2
conv2 = Conv1D(256, kernel_size=3, activation='relu', name="conv1d_2")(dropout1)
bn2 = BatchNormalization()(conv2)
pool2 = MaxPooling1D(pool_size=2, name="max_pooling1d_2")(bn2)
dropout2 = Dropout(0.2, name="dropout_2")(pool2)

# Convolutional Block 3
conv3 = Conv1D(256, kernel_size=3, activation='relu', name="conv1d_3")(dropout2)
bn3 = BatchNormalization()(conv3)
pool3 = MaxPooling1D(pool_size=2, name="max_pooling1d_3")(bn3)
dropout3 = Dropout(0.2, name="dropout_3")(pool3)

# Global Average Pooling
global_avg_pool = GlobalAveragePooling1D()(dropout3)

# Output layer
output_layer = Dense(1, activation='sigmoid', name="output_layer")(global_avg_pool)

# Compile model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ================================================================
# 6. TRAIN MODEL
# ================================================================
print("\n🚀 Training model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test_scaled, y_test)
)

# ================================================================
# 7. SAVE MODEL
# ================================================================
model.save(MODEL_PATH)
print(f"\n💾 Model saved successfully to {MODEL_PATH}")
