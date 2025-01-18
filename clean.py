#!/usr/bin/env python3

"""
model_testing.py

This script loads the saved best model and tests it on the enriched inference dataset.
It generates predictions and probabilities and saves the results to a new CSV file.

Author: [Your Name]
Date: YYYY-MM-DD
"""

import os
import pandas as pd
import joblib
import logging

# Configuration
BEST_MODEL_PATH = "models/best_model.pkl"
INFERENCE_DATA_PATH = "data/test_enriched.csv"
OUTPUT_RESULTS_PATH = "data/inference_results.csv"
FEATURE_NAMES_PATH = "models/feature_names.pkl"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the CSV dataset into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Successfully loaded data from {filepath} with shape {df.shape}.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise

def align_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Aligns DataFrame features with the expected feature names.
    Adds missing features with default values and ensures correct column order.
    """
    missing_features = set(feature_names) - set(df.columns)
    for feature in missing_features:
        df[feature] = 0  # Default value for missing features
        logging.warning(f"Missing feature '{feature}' added with default value 0.")

    df = df[feature_names]  # Reorder columns to match training
    logging.info("Aligned inference data with training features.")
    return df

def main():
    # Load the best model
    if not os.path.exists(BEST_MODEL_PATH):
        logging.error(f"Best model file not found at '{BEST_MODEL_PATH}'.")
        raise FileNotFoundError(f"Best model file not found at '{BEST_MODEL_PATH}'.")
    model = joblib.load(BEST_MODEL_PATH)
    logging.info(f"Loaded best model from '{BEST_MODEL_PATH}'.")

    # Load the inference data
    df = load_data(INFERENCE_DATA_PATH)

    # Load feature names
    if not os.path.exists(FEATURE_NAMES_PATH):
        logging.error(f"Feature names file not found at '{FEATURE_NAMES_PATH}'.")
        raise FileNotFoundError(f"Feature names file not found at '{FEATURE_NAMES_PATH}'.")
    feature_names = joblib.load(FEATURE_NAMES_PATH)

    # Align the inference data with training features
    non_feature_cols = ['LogID', 'Timestamp']
    X_inference = df.drop(columns=[col for col in non_feature_cols if col in df.columns], errors='ignore')
    X_inference = align_features(X_inference, feature_names)

    # Make predictions
    logging.info("Making predictions on the inference dataset...")
    y_pred = model.predict(X_inference)
    y_proba = model.predict_proba(X_inference)[:, 1] if hasattr(model, "predict_proba") else None

    # Append predictions and probabilities to the inference dataset
    df['Predicted_Label'] = y_pred
    if y_proba is not None:
        df['Predicted_Probability'] = y_proba

    # Save the results to a new CSV file
    df.to_csv(OUTPUT_RESULTS_PATH, index=False)
    logging.info(f"Inference results saved to '{OUTPUT_RESULTS_PATH}'.")

    # Print a preview of the results
    print("Inference Results:")
    print(df.head())

if __name__ == "__main__":
    main()
