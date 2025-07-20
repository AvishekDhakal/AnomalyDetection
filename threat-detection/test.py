#!/usr/bin/env python3
"""
test.py

This script loads the final trained models (Logistic Regression, Random Forest, XGBoost)
from the final_models folder, loads the saved feature mapping (features.pkl), and processes
the inference CSV (produced by enhanced_features.py) so that its features exactly match those 
used during training. It then applies the corresponding scaler (if available), runs predictions,
and merges the predictions back with the metadata (LogID and Timestamp) from the original CSV.
This final output CSV helps trace back which log (by LogID and Timestamp) was predicted as anomalous.
"""

import os
import pandas as pd
import numpy as np
import pickle
import joblib

# -----------------------------
# 1. File Paths and Configurations
# -----------------------------
# Inference CSV produced by enhanced_features.py (this CSV contains all columns, including metadata)
INFERENCE_CSV = "data/preprocessed_inference.csv"

# Feature mapping saved during training (placed in final_models)
FEATURE_MAPPING_PATH = "final_models/features.pkl"

# Model files (placed in final_models)
MODEL_PATHS = {
    "Logistic Regression": "final_models/logistic_reg.pkl",
    "Random Forest": "final_models/random_forest.pkl",
    "XGBoost": "final_models/xgb.pkl"
}

# Corresponding scaler files (saved during training)
SCALER_PATHS = {
    "Logistic Regression": "final_models/logistic_reg_scaler.pkl",
    "Random Forest": "final_models/random_forest_scaler.pkl",
    "XGBoost": "final_models/xgb_scaler.pkl"
}

# Numerical columns that were scaled during training
NUMERICAL_COLS = ['req_freq_5min', 'inventory_change_rate', 'role_risk']

# REQUIRED_FEATURES in the order used during training (as printed earlier):
REQUIRED_FEATURES = [
    'HTTP_Method_POST',
    'HTTP_Method_HEAD',
    'endpoint_l1_encoded',
    'HTTP_Method_OPTIONS',
    'inventory_change_rate',
    'is_unusual_time',
    'is_internal_ip',
    'HTTP_Method_PATCH',
    'HTTP_Method_GET',
    'HTTP_Method_DELETE',
    'hour',
    'is_authorized',
    'day_of_week',
    'role_risk',
    'req_freq_5min',
    'role_encoded',
    'HTTP_Method_PUT'
]

# Define metadata columns that we want to preserve in the final output.
META_COLUMNS = ['LogID', 'Timestamp']

# -----------------------------
# 2. Load the Feature Mapping and Inference Data
# -----------------------------
# Try to load the saved feature mapping from training.
try:
    with open(FEATURE_MAPPING_PATH, "rb") as f:
        feature_mapping = pickle.load(f)
    saved_features = feature_mapping.get("feature_columns", None)
    if saved_features is not None:
        print("Saved feature order from mapping:")
        print(saved_features)
    else:
        print("No feature_columns found in saved mapping; using hard-coded REQUIRED_FEATURES.")
except Exception as e:
    print(f"Warning: Could not load feature mapping from {FEATURE_MAPPING_PATH} ({e}).")
    saved_features = None

# For our purposes, enforce the expected feature order:
expected_features = REQUIRED_FEATURES

# Load the raw inference CSV.
if not os.path.exists(INFERENCE_CSV):
    raise FileNotFoundError(f"Inference CSV file {INFERENCE_CSV} not found.")

df_raw = pd.read_csv(INFERENCE_CSV)
# Clean column names if necessary.
df_raw.columns = df_raw.columns.str.strip().str.replace(' ', '_')

print("Original inference data columns:")
print(df_raw.columns.tolist())

# Extract metadata (e.g. LogID and Timestamp) for traceability.
df_meta = df_raw[META_COLUMNS].copy()

# Now, create a features DataFrame that will be passed to the models.
# Keep only the expected feature columns.
df_features = df_raw[[col for col in df_raw.columns if col in expected_features]].copy()

# Add any missing expected features with default value 0.
for feat in expected_features:
    if feat not in df_features.columns:
        df_features[feat] = 0

# Reorder columns to exactly match expected_features.
df_features = df_features[expected_features]

print("\nInference data features (in order):")
print(df_features.columns.tolist())

# -----------------------------
# 3. Define a Function to Scale the Numerical Features
# -----------------------------
def apply_scaler(df, scaler, numerical_cols):
    """Apply the scaler to all specified numerical columns at once."""
    df_copy = df.copy()
    df_copy[numerical_cols] = scaler.transform(df_copy[numerical_cols])
    return df_copy

# -----------------------------
# 4. Load Each Model, Apply the Corresponding Scaler, and Get Predictions
# -----------------------------
# We'll collect predictions for each model in a dictionary.
model_predictions = {}

print("\nStarting model predictions on inference data...\n")

# For each model, work on a fresh copy of df_features.
for model_name, model_path in MODEL_PATHS.items():
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Skipping {model_name}.")
        continue

    # Fresh copy of features.
    df_model_features = df_features.copy()

    print(f"Loading {model_name} from {model_path}...")
    model = joblib.load(model_path)

    # Load the corresponding scaler if available.
    scaler_path = SCALER_PATHS.get(model_name)
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        df_model_features = apply_scaler(df_model_features, scaler, NUMERICAL_COLS)
    else:
        print(f"No scaler found for {model_name}. Using unscaled features.")
    
    # Predict anomalies. (Assume model.predict returns 1 for anomalies.)
    try:
        y_pred = model.predict(df_model_features)
    except Exception as e:
        print(f"Error predicting with {model_name}: {e}")
        continue

    # Store the prediction array in our dictionary.
    model_predictions[model_name] = y_pred
    print(f"{model_name} detected {sum(y_pred==1)} anomalous log(s).")

# -----------------------------
# 5. Combine Predictions with Metadata
# -----------------------------
# Create a DataFrame for predictions (using the index of df_features).
df_preds = pd.DataFrame(index=df_features.index)
for model_name, preds in model_predictions.items():
    df_preds[f"{model_name}_predicted_anomaly"] = preds

# Merge the prediction columns with the metadata.
df_final = pd.concat([df_meta.reset_index(drop=True), df_preds.reset_index(drop=True)], axis=1)

# -----------------------------
# 6. Save Combined Results
# -----------------------------
OUTPUT_CSV = "data/inference_with_model_predictions.csv"
df_final.to_csv(OUTPUT_CSV, index=False)
print(f"\nCombined inference results with model predictions and metadata saved to {OUTPUT_CSV}")
