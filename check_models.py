import os
import pandas as pd
import joblib

# Paths
INFERENCE_DATA_PATH = "data/final.csv"
FEATURE_NAMES_PATH = "models/feature_names.pkl"

def load_feature_names(feature_names_path: str):
    """
    Loads the feature names from the feature_names.pkl file.
    """
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names file not found at '{feature_names_path}'.")
    return joblib.load(feature_names_path)

def load_inference_data(inference_data_path: str):
    """
    Loads the enriched inference dataset.
    """
    if not os.path.exists(inference_data_path):
        raise FileNotFoundError(f"Inference data file not found at '{inference_data_path}'.")
    return pd.read_csv(inference_data_path)

def compare_features(feature_names: list, inference_df: pd.DataFrame):
    """
    Compares the feature names from feature_names.pkl with the features in the inference dataset.
    Prints missing features and extra features.
    """
    inference_features = set(inference_df.columns)
    required_features = set(feature_names)

    # Find missing and extra features
    missing_features = required_features - inference_features
    extra_features = inference_features - required_features

    print("\nComparison of Features:")
    if missing_features:
        print(f"Missing features in inference dataset (present in feature_names.pkl): {missing_features}")
    else:
        print("No missing features in the inference dataset.")

    if extra_features:
        print(f"Extra features in inference dataset (not in feature_names.pkl): {extra_features}")
    else:
        print("No extra features in the inference dataset.")

def main():
    # Load feature names
    feature_names = load_feature_names(FEATURE_NAMES_PATH)

    # Load inference dataset
    inference_df = load_inference_data(INFERENCE_DATA_PATH)

    # Compare features
    compare_features(feature_names, inference_df)

if __name__ == "__main__":
    main()
