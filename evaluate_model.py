#!/usr/bin/env python3

"""
evaluate_model.py

Loads a trained ExtraTreesClassifier, processes enriched test logs,
makes predictions, and evaluates performance.

Usage:
    python evaluate_model.py \
        --model_path models/ExtraTrees_best.pkl \
        --test_data_csv data/test_enriched_logs.csv \
        --feature_names models/feature_names.pkl \
        --output_report models/et_evaluation_report.txt \
        --output_confusion_matrix models/confusion_matrix.png

Author: [Your Name]
Date: YYYY-MM-DD
"""

import os
import argparse
import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import numpy as np
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Trained ExtraTreesClassifier on Test Logs")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained ExtraTrees model (e.g., models/ExtraTrees_best.pkl)")
    parser.add_argument("--test_data_csv", type=str, required=True,
                        help="Path to the enriched test CSV file (e.g., data/test_enriched_logs.csv)")
    parser.add_argument("--feature_names", type=str, required=True,
                        help="Path to the saved feature names file (e.g., models/feature_names.pkl)")
    parser.add_argument("--output_report", type=str, default="models/et_evaluation_report.txt",
                        help="Path to save the evaluation report.")
    parser.add_argument("--output_confusion_matrix", type=str, default="models/confusion_matrix.png",
                        help="Path to save the confusion matrix plot.")
    return parser.parse_args()

def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logging.info("Trained ExtraTreesClassifier loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        raise

def load_test_data(test_data_csv: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(test_data_csv)
        logging.info(f"Loaded test data from {test_data_csv} with shape {df.shape}.")
        return df
    except Exception as e:
        logging.error(f"Error loading test data from {test_data_csv}: {e}")
        raise

def load_feature_names(feature_names_path: str) -> list:
    try:
        feature_names = joblib.load(feature_names_path)
        logging.info(f"Loaded feature names from {feature_names_path}.")
        return feature_names
    except Exception as e:
        logging.error(f"Error loading feature names from {feature_names_path}: {e}")
        raise

def align_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Aligns test data features with training data features.
    Adds missing features with default value 0 and drops extra features.
    """
    # Identify missing and extra features
    test_features = set(df.columns)
    training_features = set(feature_names)
    
    # Features to add (present in training but missing in test)
    missing_features = training_features - test_features
    # Features to drop (present in test but not in training)
    extra_features = test_features - training_features - {'Anomalous', 'LogID', 'Timestamp'}
    
    logging.info(f"Missing features to add: {missing_features}")
    logging.info(f"Extra features to drop: {extra_features}")
    
    # Add missing features with default value 0
    for feature in missing_features:
        df[feature] = 0
        logging.info(f"Added missing feature '{feature}' with default value 0.")
    
    # Drop extra features
    if extra_features:
        df = df.drop(columns=list(extra_features))
        logging.info(f"Dropped extra features: {extra_features}")
    
    # Reorder columns to match training data
    df = df.reindex(columns=feature_names, fill_value=0)
    logging.info("Aligned test data features with training data features.")
    
    return df

def make_predictions(model, X_test: pd.DataFrame) -> np.ndarray:
    """
    Makes predictions using the trained model.
    """
    try:
        y_pred = model.predict(X_test)
        logging.info("Made predictions on test data.")
        return y_pred
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        raise

def generate_evaluation_report(y_test: pd.Series, y_pred: np.ndarray, report_path: str) -> None:
    """
    Generates and saves the classification report and confusion matrix.
    """
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    
    with open(report_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    
    logging.info(f"Saved classification report to {report_path}.")
    return cm

def plot_confusion_matrix(cm: np.ndarray, output_path: str) -> None:
    """
    Plots and saves the confusion matrix.
    """
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomalous'], yticklabels=['Normal', 'Anomalous'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Saved confusion matrix plot to {output_path}.")

def main():
    args = parse_arguments()
    
    # Load the trained model
    model = load_model(args.model_path)
    
    # Load feature names
    feature_names = load_feature_names(args.feature_names)
    
    # Load test data
    df_test = load_test_data(args.test_data_csv)
    
    # Ensure 'Anomalous' column exists for evaluation
    if 'Anomalous' not in df_test.columns:
        logging.error("'Anomalous' column is missing in test data for evaluation.")
        raise ValueError("'Anomalous' column is required in test data for evaluation.")
    
    # Separate features and labels
    y_test = df_test['Anomalous'].copy()
    X_test = df_test.drop(columns=['LogID', 'Timestamp', 'Anomalous'], errors='ignore').copy()
    
    # Align features
    X_test_aligned = align_features(X_test, feature_names)
    
    # Make predictions
    y_pred = make_predictions(model, X_test_aligned)
    
    # Evaluate
    cm = generate_evaluation_report(y_test, y_pred, args.output_report)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, args.output_confusion_matrix)
    
    logging.info("Model evaluation completed successfully.")

if __name__ == "__main__":
    main()
