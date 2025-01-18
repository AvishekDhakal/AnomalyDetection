import os
import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

# -------------------- CONFIGURATION --------------------

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
DATA_PATH = "data/enriched_train.csv"
MODEL_OUTPUT_DIR = "models"
BEST_MODEL_PATH_PRECISION = os.path.join(MODEL_OUTPUT_DIR, "random_forest_high_precision.pkl")
BEST_MODEL_PATH_RECALL = os.path.join(MODEL_OUTPUT_DIR, "random_forest_high_recall.pkl")

# Ensure output directories exist
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# -------------------- FUNCTIONS --------------------

def load_data(filepath):
    """Loads the dataset."""
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Successfully loaded data from {filepath} with shape {df.shape}.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise

def separate_features_labels(df, label_column="Anomalous"):
    """Separates features and labels."""
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in the dataset.")
    
    X = df.drop(['LogID', 'Timestamp', label_column], axis=1)
    y = df[label_column]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the data into training and validation sets."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def tune_random_forest(X_train, y_train, scoring, output_path):
    """Tunes a RandomForestClassifier for a specific objective."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        scoring=scoring,
        cv=5,
        verbose=2,
        n_jobs=-1
    )
    
    logging.info(f"Starting grid search for RandomForest with scoring={scoring}.")
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters for scoring={scoring}: {grid_search.best_params_}")
    
    # Save the best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, output_path)
    logging.info(f"Best RandomForest model for scoring={scoring} saved to {output_path}.")
    
    return best_model

def evaluate_model(model, X_val, y_val):
    """Evaluates a trained model."""
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1-Score": f1_score(y_val, y_pred),
        "ROC-AUC": roc_auc_score(y_val, y_proba) if y_proba is not None else None
    }
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    return metrics

# -------------------- MAIN FUNCTION --------------------

def main():
    # Load and prepare data
    df = load_data(DATA_PATH)
    X, y = separate_features_labels(df)
    X_train, X_val, y_train, y_val = split_data(X, y)

    # Hyperparameter tuning for high precision
    best_rf_precision = tune_random_forest(
        X_train, y_train,
        scoring="precision",
        output_path=BEST_MODEL_PATH_PRECISION
    )
    logging.info("Evaluating high-precision RandomForest model...")
    evaluate_model(best_rf_precision, X_val, y_val)

    # Hyperparameter tuning for high recall
    best_rf_recall = tune_random_forest(
        X_train, y_train,
        scoring="recall",
        output_path=BEST_MODEL_PATH_RECALL
    )
    logging.info("Evaluating high-recall RandomForest model...")
    evaluate_model(best_rf_recall, X_val, y_val)

if __name__ == "__main__":
    main()
