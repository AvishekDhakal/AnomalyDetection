#!/usr/bin/env python3

"""
train_models.py

A script to train and evaluate multiple machine learning models for insider threat detection.
This script loads preprocessed and feature-engineered data, trains Logistic Regression,
Random Forest, and XGBoost classifiers, evaluates their performance using various metrics,
and saves the trained models for future use.

Usage:
    python3 train_models.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------- CONFIGURATION --------------------

# Define directories
FEATURE_ENGINEERED_DATA_DIR = "feature_engineered_data"
MODELS_DIR = "models"
EVALUATION_DIR = "model_evaluation"

# Ensure necessary directories exist
for directory in [MODELS_DIR, EVALUATION_DIR]:
    os.makedirs(directory, exist_ok=True)

# -------------------- LOAD DATA --------------------

def load_data():
    """
    Loads the feature-engineered training and testing data with explicit data types.
    
    Returns:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
    """
    print("\n--- Loading Feature-Engineered Data ---")
    try:
        dtype_spec = {
            'LogID': str,
            # 'UserID': int,
            # Add other columns with expected data types as needed
        }
        
        # Load training data
        df_train = pd.read_csv(
            os.path.join(FEATURE_ENGINEERED_DATA_DIR, "X_train_final.csv"),
            dtype=dtype_spec,
            low_memory=False
        )
        y_train = df_train['Anomalous']
        X_train = df_train.drop(columns=['Anomalous', 'LogID'], errors='ignore')
        
        # Load testing data
        df_test = pd.read_csv(
            os.path.join(FEATURE_ENGINEERED_DATA_DIR, "X_test_final.csv"),
            dtype=dtype_spec,
            low_memory=False
        )
        y_test = df_test['Anomalous']
        X_test = df_test.drop(columns=['Anomalous', 'LogID'], errors='ignore')
        
        print("Successfully loaded training and testing data.")
        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        sys.exit(1)

# -------------------- MODEL TRAINING --------------------

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    
    Returns:
        model (LogisticRegression): Trained Logistic Regression model.
    """
    print("\n--- Training Logistic Regression ---")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    print("Logistic Regression training completed.")
    return model

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Classifier.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    
    Returns:
        model (RandomForestClassifier): Trained Random Forest model.
    """
    print("\n--- Training Random Forest Classifier ---")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Random Forest training completed.")
    return model

def train_xgboost(X_train, y_train):
    """
    Trains an XGBoost Classifier.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    
    Returns:
        model (XGBClassifier): Trained XGBoost model.
    """
    print("\n--- Training XGBoost Classifier ---")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='logloss',  # Required to suppress warnings
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train)  # Handle class imbalance
    )
    model.fit(X_train, y_train)
    print("XGBoost training completed.")
    return model


# -------------------- MODEL EVALUATION --------------------

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates the trained model on the testing set and prints various performance metrics.
    Also saves the classification report and confusion matrix plot.
    
    Args:
        model: Trained machine learning model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        model_name (str): Name of the model for identification.
    
    Returns:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    print(f"\n--- Evaluating {model_name} ---")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Print metrics
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    # Classification Report
    report = classification_report(y_test, y_pred, zero_division=0)
    print("\nClassification Report:")
    print(report)
    
    # Save Classification Report
    with open(os.path.join(EVALUATION_DIR, f"{model_name}_classification_report.txt"), 'w') as f:
        f.write(f"{model_name} Classification Report\n")
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomalous'], yticklabels=['Normal', 'Anomalous'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(EVALUATION_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()
    
    print(f"Saved classification report and confusion matrix for {model_name}.\n")
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0,1], [0,1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(EVALUATION_DIR, f"{model_name}_roc_curve.png"))
    plt.close()
    
    print(f"Saved ROC curve for {model_name}.\n")
    
    # Return metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }
    return metrics

def generate_evaluation_report(metrics_list):
    """
    Generates a summary report of all model evaluations and saves it as a CSV file.
    
    Args:
        metrics_list (list): List of dictionaries containing model evaluation metrics.
    """
    print("\n--- Generating Evaluation Report ---")
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(os.path.join(EVALUATION_DIR, "model_evaluation_report.csv"), index=False)
    print("Saved model evaluation report as 'model_evaluation_report.csv'.")

# -------------------- MAIN FUNCTION --------------------

def main():
    """
    Main function to execute the training and evaluation pipeline.
    """
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Initialize list to store metrics
    metrics_list = []
    
    # Train models
    logistic_model = train_logistic_regression(X_train, y_train)
    random_forest_model = train_random_forest(X_train, y_train)
    xgboost_model = train_xgboost(X_train, y_train)
    
    # Evaluate models
    metrics_logistic = evaluate_model(logistic_model, X_test, y_test, "Logistic_Regression")
    metrics_rf = evaluate_model(random_forest_model, X_test, y_test, "Random_Forest")
    metrics_xgb = evaluate_model(xgboost_model, X_test, y_test, "XGBoost")
    
    # Append metrics to the list
    metrics_list.extend([metrics_logistic, metrics_rf, metrics_xgb])
    
    # Generate summary report
    generate_evaluation_report(metrics_list)
    
    # Save trained models
    print("\n--- Saving Trained Models ---")
    joblib.dump(logistic_model, os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
    joblib.dump(random_forest_model, os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    joblib.dump(xgboost_model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    print("Trained models saved in the 'models/' directory.\n")
    
    print("Model training and evaluation pipeline completed successfully.")

# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    main()
