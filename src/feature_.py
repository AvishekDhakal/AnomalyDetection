# model_evaluation.py

import os
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
    f1_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier  # If still using XGBoost elsewhere

# Suppress specific warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='xgboost')

def create_output_directories():
    """
    Create necessary directories for saving plots and models if they don't exist.
    """
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    print("Output directories are set up.")

def load_data(filepath):
    """
    Load the final engineered dataset.

    Parameters:
    - filepath (str): Path to the CSV file containing engineered features.

    Returns:
    - DataFrame: Loaded data.
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully with shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

def split_data(data, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    Parameters:
    - data (DataFrame): The dataset to split.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Seed used by the random number generator.

    Returns:
    - X_train, X_test, y_train, y_test: Split datasets.
    """
    if 'anomalous' not in data.columns:
        print("Error: 'anomalous' column not found in the data.")
        exit(1)
    
    X = data.drop(['anomalous'], axis=1)
    y = data['anomalous']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data split into train and test sets with test size = {test_size}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def load_model(model_path, model_type='random_forest'):
    """
    Load a trained model from a file.

    Parameters:
    - model_path (str): Path to the model file.
    - model_type (str): Type of the model ('random_forest', 'xgboost', 'gradient_boosting').

    Returns:
    - Model object: Loaded model.
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from '{model_path}'")
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate the model's performance and generate relevant plots.

    Parameters:
    - model: Trained model to evaluate.
    - X_test (DataFrame): Test features.
    - y_test (Series): True labels.
    - model_name (str): Name of the model (e.g., 'Random Forest').

    Returns:
    - None
    """
    print(f"\nEvaluating {model_name}...")
    
    # Generate predictions and probabilities
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        # For models that do not have predict_proba
        y_proba = model.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())  # Normalize to [0,1]
    
    # Classification Report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"{model_name} ROC-AUC: {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')
    plt.close()
    print(f"Confusion matrix saved as 'plots/{model_name}_confusion_matrix.png'")
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'plots/{model_name}_roc_curve.png')
    plt.close()
    print(f"ROC curve saved as 'plots/{model_name}_roc_curve.png'")
    
    # Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, label=f'{model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='lower left')
    plt.savefig(f'plots/{model_name}_precision_recall_curve.png')
    plt.close()
    print(f"Precision-Recall curve saved as 'plots/{model_name}_precision_recall_curve.png'")
    
    # Feature Importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_test.columns
        feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
        plt.figure(figsize=(10,8))
        sns.barplot(x=feature_importances[:10], y=feature_importances.index[:10])
        plt.title(f'Top 10 Feature Importances - {model_name}')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(f'plots/{model_name}_feature_importances.png')
        plt.close()
        print(f"Feature importances saved as 'plots/{model_name}_feature_importances.png'")
    else:
        print(f"{model_name} does not have 'feature_importances_' attribute.")
    
    # Cross-Validation Scores
    print(f"\nPerforming 5-Fold Cross-Validation for {model_name}...")
    try:
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='f1')
        print(f"Cross-validated F1-Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    except Exception as e:
        print(f"Cross-validation failed for {model_name}: {e}")

def main():
    """
    Main function to execute the evaluation pipeline for all models.
    """
    # Step 1: Create output directories
    create_output_directories()
    
    # Step 2: Load data
    data_path = "data/final_engineered_features.csv"
    data = load_data(data_path)
    
    # Step 3: Split data into train and test
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Step 4: Load models
    rf_model_path = "models/random_forest_classifier.joblib"
    gbc_model_path = "models/gradient_boosting_classifier.joblib"
    
    rf_model = load_model(rf_model_path, model_type='random_forest')
    gbc_model = load_model(gbc_model_path, model_type='gradient_boosting')
    
    # Step 5: Evaluate Random Forest
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Step 6: Evaluate Gradient Boosting
    evaluate_model(gbc_model, X_test, y_test, "Gradient Boosting")
    
    print("\nModel evaluation completed successfully. All plots are saved in the 'plots/' directory.")

if __name__ == "__main__":
    main()
