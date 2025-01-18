import os
import pandas as pd
import numpy as np
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# -------------------- CONFIGURATION --------------------

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
DATA_PATH = "data/enriched_train.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "model_outputs"
FEATURE_IMPORTANCE_DIR = os.path.join(OUTPUT_DIR, "feature_importance")
FEATURE_IMPORTANCE_TXT = os.path.join(FEATURE_IMPORTANCE_DIR, "feature_importance_summary.txt")

# Ensure output directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)

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

def plot_feature_importance(model, feature_names, model_name, X_train, y_train):
    """
    Plots feature importance for tree-based models or computes permutation importance for others.
    """
    importance_data = []

    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        # Tree-based models
        importances = model.named_steps['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance - {model_name}")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(FEATURE_IMPORTANCE_DIR, f"feature_importance_{model_name}.png"))
        plt.close()
        logging.info(f"Feature importance for {model_name} saved as PNG.")

        # Save to text file
        for i in indices:
            importance_data.append((feature_names[i], importances[i]))

    else:
        # Permutation importance for models without native feature_importances_
        result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1)
        importances = result.importances_mean
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f"Permutation Importance - {model_name}")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(FEATURE_IMPORTANCE_DIR, f"permutation_importance_{model_name}.png"))
        plt.close()
        logging.info(f"Permutation importance for {model_name} saved as PNG.")

        # Save to text file
        for i in indices:
            importance_data.append((feature_names[i], importances[i]))

    return importance_data

def analyze_feature_importance(trained_models, X_train, y_train):
    """
    Analyzes and saves feature importance for all models.
    """
    with open(FEATURE_IMPORTANCE_TXT, "w") as f:
        for name, model in trained_models.items():
            logging.info(f"Analyzing feature importance for {name}...")
            importance_data = plot_feature_importance(model, X_train.columns, name, X_train, y_train)

            # Write to summary file
            f.write(f"Feature Importance - {name}\n")
            for feature, importance in importance_data:
                f.write(f"{feature}: {importance:.6f}\n")
            f.write("\n")
            logging.info(f"Feature importance for {name} saved to TXT.")

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
    
    # 6. Analyze feature importance
    analyze_feature_importance(trained_models, X_train, y_train)
    
    logging.info("Feature importance analysis completed.")

if __name__ == "__main__":
    main()
