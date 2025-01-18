import os
import pandas as pd
import joblib
import logging
from sklearn.ensemble import VotingClassifier

# Configuration
MODEL_PATH_RECALL = "models/logistic_regression_high_recall.pkl"
MODEL_PATH_PRECISION = "models/random_forest_high_precision.pkl"
INFERENCE_DATA_PATH = "data/test_enriched.csv"
OUTPUT_RESULTS_PATH = "data/inference_results_voting.csv"
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
    # Load the models
    if not os.path.exists(MODEL_PATH_RECALL) or not os.path.exists(MODEL_PATH_PRECISION):
        logging.error("One or more model files are missing.")
        raise FileNotFoundError("Ensure both models are available.")

    model_recall = joblib.load(MODEL_PATH_RECALL)
    model_precision = joblib.load(MODEL_PATH_PRECISION)
    logging.info("Loaded both high-recall and high-precision models.")

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

    # Make predictions using both models
    logging.info("Making predictions with high-recall Logistic Regression...")
    y_pred_recall = model_recall.predict(X_inference)
    logging.info("Making predictions with high-precision Random Forest...")
    y_pred_precision = model_precision.predict(X_inference)

    # Append individual model predictions
    df['Predicted_Label_Recall'] = y_pred_recall
    df['Predicted_Label_Precision'] = y_pred_precision

    # Print logs deemed anomalous by both models
    anomalous_recall = df[df['Predicted_Label_Recall'] == 1]
    anomalous_precision = df[df['Predicted_Label_Precision'] == 1]

    print("Anomalous logs found by high-recall Logistic Regression:")
    print(anomalous_recall)

    print("Anomalous logs found by high-precision Random Forest:")
    print(anomalous_precision)

    # Combine models using voting
    logging.info("Applying voting mechanism...")
    df['Final_Predicted_Label'] = (df['Predicted_Label_Recall'] + df['Predicted_Label_Precision'] >= 1).astype(int)

    # Print final anomalous logs after voting
    final_anomalous_logs = df[df['Final_Predicted_Label'] == 1]
    print("Final anomalous logs after voting:")
    print(final_anomalous_logs)

    # Save the results to a new CSV file
    df.to_csv(OUTPUT_RESULTS_PATH, index=False)
    logging.info(f"Inference results saved to '{OUTPUT_RESULTS_PATH}'.")

if __name__ == "__main__":
    main()
