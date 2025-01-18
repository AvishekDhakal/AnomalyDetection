#!/usr/bin/env python3

"""
model_training.py

This script trains multiple machine learning models on the enriched insider threat detection dataset.
It evaluates each model using various performance metrics and saves the best-performing model for future inference.

Key Steps:
1. Data Loading and Preprocessing
2. Handling Duplicate Columns
3. Feature-Label Separation and Data Splitting
4. Model Training
5. Model Evaluation
6. Visualization of Evaluation Metrics
7. Saving the Best Model

Author: [Your Name]
Date: YYYY-MM-DD
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- CONFIGURATION --------------------

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
DATA_PATH = "data/train_enriched.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "model_outputs"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

# Ensure output directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- FUNCTIONS --------------------

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

def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and removes duplicate columns in the DataFrame.
    """
    duplicated_columns = df.columns[df.columns.duplicated()].unique()
    if len(duplicated_columns) > 0:
        logging.warning(f"Duplicate columns found: {duplicated_columns}. Removing duplicates.")
        df = df.loc[:, ~df.columns.duplicated()]
    else:
        logging.info("No duplicate columns found.")
    return df

def separate_features_labels(df: pd.DataFrame, label_column: str = "Anomalous") -> (pd.DataFrame, pd.Series):
    """
    Separates the DataFrame into features (X) and label (y).
    """
    if label_column not in df.columns:
        logging.error(f"Label column '{label_column}' not found in the dataset.")
        raise ValueError(f"Label column '{label_column}' not found in the dataset.")
    
    X = df.drop(['LogID', 'Timestamp', label_column], axis=1)
    y = df[label_column]
    logging.info(f"Separated features and label. Features shape: {X.shape}, Label shape: {y.shape}")
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the data into training and validation sets.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logging.info(f"Data split into training and validation sets.")
    logging.info(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
    return X_train, X_val, y_train, y_val

def train_models(X_train, y_train):
    """
    Trains multiple machine learning models and returns a dictionary of trained models.
    """
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
    }
    
    trained_models = {}
    
    for name, model in models.items():
        logging.info(f"Training {name}...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        logging.info(f"{name} training completed.")
    
    return trained_models

def evaluate_models(trained_models, X_val, y_val):
    """
    Evaluates trained models on the validation set and returns a DataFrame of metrics.
    """
    metrics = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": [],
        "ROC-AUC": []
    }
    
    for name, model in trained_models.items():
        logging.info(f"Evaluating {name}...")
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_val, y_proba)
        
        metrics["Model"].append(name)
        metrics["Accuracy"].append(acc)
        metrics["Precision"].append(prec)
        metrics["Recall"].append(rec)
        metrics["F1-Score"].append(f1)
        metrics["ROC-AUC"].append(roc_auc)
        
        logging.info(f"{name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

def plot_confusion_matrix(model, X_val, y_val, model_name: str):
    """
    Plots and saves the confusion matrix for a given model.
    """
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name}.png"))
    plt.close()
    logging.info(f"Confusion matrix for {model_name} saved.")

def plot_roc_curve(model, X_val, y_val, model_name: str):
    """
    Plots and saves the ROC curve for a given model.
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:,1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_val)
    else:
        logging.warning(f"{model_name} does not have predict_proba or decision_function. Skipping ROC curve.")
        return
    
    fpr, tpr, thresholds = roc_curve(y_val, y_proba)
    roc_auc = roc_auc_score(y_val, y_proba)
    
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"roc_curve_{model_name}.png"))
    plt.close()
    logging.info(f"ROC curve for {model_name} saved.")

def save_metrics(metrics_df: pd.DataFrame, output_path: str):
    """
    Saves the metrics DataFrame to a CSV file.
    """
    metrics_df.to_csv(output_path, index=False)
    logging.info(f"Model evaluation metrics saved to {output_path}.")

def save_best_model(trained_models, metrics_df: pd.DataFrame, save_path: str):
    """
    Identifies the best model based on ROC-AUC and saves it.
    """
    best_model_name = metrics_df.loc[metrics_df['ROC-AUC'].idxmax()]['Model']
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, save_path)
    logging.info(f"Best model '{best_model_name}' saved to '{save_path}'.")

# -------------------- MAIN FUNCTION --------------------

def main():
    # 1. Load data
    df = load_data(DATA_PATH)
    
    # 2. Remove duplicate columns if any
    df = remove_duplicate_columns(df)
    
    # 3. Separate features and labels
    X, y = separate_features_labels(df, label_column="Anomalous")
    
    # 4. Split data into training and validation sets
    X_train, X_val, y_train, y_val = split_data(X, y)
    
    # 5. Train models
    trained_models = train_models(X_train, y_train)
    
    # 6. Evaluate models
    metrics_df = evaluate_models(trained_models, X_val, y_val)
    
    # 7. Save evaluation metrics
    metrics_output_path = os.path.join(OUTPUT_DIR, "model_evaluation_metrics.csv")
    save_metrics(metrics_df, metrics_output_path)
    
    # 8. Plot confusion matrices and ROC curves
    for name, model in trained_models.items():
        plot_confusion_matrix(model, X_val, y_val, name)
        plot_roc_curve(model, X_val, y_val, name)
    
    # 9. Save the best model based on ROC-AUC
    save_best_model(trained_models, metrics_df, BEST_MODEL_PATH)
    
    # 10. Print a summary of metrics
    print("\nModel Evaluation Metrics:")
    print(metrics_df.sort_values(by="ROC-AUC", ascending=False))
    
    logging.info("Model training and evaluation process completed.")

if __name__ == "__main__":
    main()
