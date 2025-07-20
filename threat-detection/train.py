#!/usr/bin/env python3
"""
enhanced_model_training.py

This script trains two models (Logistic Regression and Random Forest)
on a realistic training dataset (e.g. 100,000 logs with ~10-20% anomalies).
It computes performance metrics and generates key visualizations including
confusion matrix, ROC curve, precision-recall curve, feature importance (for Random Forest),
and logistic regression coefficients (for Logistic Regression).

An explicit ordered list of features used for modeling is saved in final_models/features.pkl.
All extraneous comparisons and fairness evaluations have been removed.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)
import shap
import joblib
import pickle

# Configuration
MODEL_DIR = "final_models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CLASS_WEIGHTS = {0: 1, 1: 5}
PLOT_PARAMS = {
    'figure.figsize': (16, 10),
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
}
plt.rcParams.update(PLOT_PARAMS)

# Define an explicit, ordered list of features for modeling.
# These features will be saved to the feature mapping file.
MODEL_FEATURES = [
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

def create_directory_structure():
    """Create organized output directories under MODEL_DIR."""
    dirs = [
        f"{MODEL_DIR}/Original/Logistic_Regression",
        f"{MODEL_DIR}/Original/Random_Forest"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Directories created.")

def load_and_preprocess_data():
    """
    Load data from CSV, clean column names, and select only the MODEL_FEATURES plus the target 'Anomalous'.
    """
    df = pd.read_csv("data/enhanced_train.csv")
    # Clean column names.
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    
    # Define the required columns: the ordered feature list plus 'Anomalous'.
    required_columns = MODEL_FEATURES + ['Anomalous']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}\nActual columns: {list(df.columns)}")
    
    total = len(df)
    count_anom = df['Anomalous'].sum()
    print(f"Total logs: {total} | Anomalies: {count_anom} | Anomaly ratio: {count_anom/total:.2%}")
    
    # Subset and reorder the DataFrame.
    df_model = df[required_columns].copy()
    return df_model

def prepare_datasets(df):
    """
    Split the dataset into training and test sets using the natural (realistic) distribution.
    Returns (X_train, X_test, y_train, y_test).
    """
    X = df.drop('Anomalous', axis=1)
    y = df['Anomalous']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    print(f"Training set: {len(X_train)} logs; Anomaly ratio: {y_train.mean():.2%}")
    print(f"Test set: {len(X_test)} logs; Anomaly ratio: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test

def save_feature_mapping():
    """
    Save the explicit ordered list of model features to MODEL_DIR/features.pkl.
    """
    mapping = {'feature_columns': MODEL_FEATURES}
    os.makedirs(MODEL_DIR, exist_ok=True)
    mapping_path = os.path.join(MODEL_DIR, "features.pkl")
    with open(mapping_path, "wb") as f:
        pickle.dump(mapping, f)
    print("Saved feature mapping in", mapping_path)

def train_model(model_type, X_train, X_test, y_train, base_dir):
    """
    Train the specified model on the training data.
    Scales numerical features, trains the model, and returns model and predictions.
    """
    scaler = StandardScaler()
    num_cols = ['req_freq_5min', 'inventory_change_rate', 'role_risk']
    
    # Scale numerical features together.
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    if model_type == 'logistic':
        model = LogisticRegression(class_weight=CLASS_WEIGHTS, max_iter=1000, random_state=RANDOM_STATE)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=RANDOM_STATE)
    else:
        raise ValueError("Unknown model type")
    
    start_time = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error training {model_type}: {e}")
        raise e
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Training {model_type} completed in {elapsed:.2f} seconds.")
    
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    
    model_dir = f"{base_dir}/{model_type.replace(' ', '_')}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print(f"Saved {model_type} model and scaler in {model_dir}.")
    
    return model, y_pred, y_proba, scaler

def plot_precision_recall(y_test, y_proba, model_dir):
    """Plot and save the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, marker='.', label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(model_dir, "precision_recall_curve.png"), bbox_inches='tight')
    plt.close()

def generate_visualizations(model, X_test, y_test, y_pred, y_proba, model_dir):
    """Generate visualizations: classification report, confusion matrix, ROC curve, precision-recall curve,
    and model-specific plots."""
    # Classification report.
    report = classification_report(y_test, y_pred)
    with open(os.path.join(model_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    
    # Confusion matrix.
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), bbox_inches='tight')
    plt.close()
    
    # ROC curve.
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.2f}')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(model_dir, "roc_curve.png"), bbox_inches='tight')
    plt.close()
    
    # Precision-Recall curve.
    plot_precision_recall(y_test, y_proba, model_dir)
    
    # Model-specific plots.
    if isinstance(model, RandomForestClassifier):
        # Feature importances.
        plt.figure()
        feature_imp = pd.Series(model.feature_importances_, index=X_test.columns)
        feature_imp.nlargest(15).plot(kind='barh')
        plt.title("Feature Importances")
        plt.savefig(os.path.join(model_dir, "feature_importances.png"), bbox_inches='tight')
        plt.close()
        
        # SHAP summary plot.
        explainer = shap.TreeExplainer(model)
        # Optionally, use a sample to speed up SHAP computation.
        X_shap = X_test.sample(n=min(100, len(X_test)), random_state=RANDOM_STATE)
        shap_values = explainer.shap_values(X_shap)
        plt.figure()
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
        plt.savefig(os.path.join(model_dir, "shap_summary.png"), bbox_inches='tight')
        plt.close('all')
    elif isinstance(model, LogisticRegression):
        # Logistic regression coefficients.
        coefficients = pd.DataFrame({
            'feature': X_test.columns,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        plt.figure()
        sns.barplot(x='coefficient', y='feature', data=coefficients.head(15))
        plt.title("Top Feature Coefficients")
        plt.savefig(os.path.join(model_dir, "feature_coefficients.png"), bbox_inches='tight')
        plt.close()

def main():
    create_directory_structure()
    df = load_and_preprocess_data()
    save_feature_mapping()
    X_train, X_test, y_train, y_test = prepare_datasets(df)
    
    for model_type in ['logistic', 'random_forest']:
        print(f"\nTraining {model_type} on original data...")
        base_dir = f"{MODEL_DIR}/Original"
        try:
            model, y_pred, y_proba, scaler = train_model(model_type, X_train, X_test, y_train, base_dir)
        except Exception as e:
            print(f"Skipping {model_type} due to training error: {e}")
            continue
        
        model_dir = os.path.join(base_dir, model_type.replace(' ', '_'))
        generate_visualizations(model, X_test, y_test, y_pred, y_proba, model_dir)
        
        # Print simple performance metrics.
        accuracy = np.mean(y_pred == y_test)
        auc = roc_auc_score(y_test, y_proba)
        print(f"{model_type} -> Accuracy: {accuracy:.2f}, AUC: {auc:.2f}")
    
    print("\nTraining complete. Models, scaler, feature mapping, and visualizations saved in", MODEL_DIR)


if __name__ == "__main__":
    main()
