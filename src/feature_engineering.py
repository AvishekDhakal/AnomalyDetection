# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class FeatureEngineer:
    def __init__(self, filepath):
        """
        Initialize the FeatureEngineer with the path to the processed CSV log file.

        Parameters:
        - filepath (str): Path to the preprocessed CSV file.
        """
        self.filepath = filepath
        self.data = None
        self.engineered_features = None

    def print_columns(self, step_description):
        """
        Print the current columns of the DataFrame for debugging purposes.

        Parameters:
        - step_description (str): Description of the current feature engineering step.
        """
        print(f"\n--- Columns after {step_description} ---")
        print(self.data.columns.tolist())

    def load_data(self):
        """
        Load preprocessed data from the CSV file into a pandas DataFrame.
        """
        self.data = pd.read_csv(self.filepath)
        print(f"Feature Engineering: Data loaded successfully with shape: {self.data.shape}")

    def drop_unnecessary_columns(self):
        """
        Drop columns that are not needed for feature engineering or model training.
        - Drops 'logid' and 'userid' as they are unique identifiers.
        - Drops 'endpointparameter' as it may introduce noise.
        """
        columns_to_drop = ['logid', 'userid', 'endpointparameter']
        existing_columns_to_drop = [col for col in columns_to_drop if col in self.data.columns]
        self.data.drop(columns=existing_columns_to_drop, inplace=True)
        print(f"Feature Engineering: Dropped columns {existing_columns_to_drop}")
        self.print_columns("dropping unnecessary columns")

    def create_interaction_features(self):
        """
        Create interaction features to capture complex patterns.
        - 'access_per_unique_endpoint': Ratio of access count to unique endpoints.
        - 'access_per_hour': Access count normalized per hour.
        """
        # Avoid division by zero
        self.data['access_per_unique_endpoint'] = self.data.apply(
            lambda row: row['accesscount'] / row['uniqueendpoints'] if row['uniqueendpoints'] > 0 else 0, axis=1
        )
        
        # Assuming 'hour' ranges from 0-23, normalize 'accesscount' per hour
        self.data['access_per_hour'] = self.data['accesscount'] / 24  # Simple normalization
        print("Feature Engineering: Created interaction features 'access_per_unique_endpoint' and 'access_per_hour'")
        self.print_columns("creating interaction features")

    def create_time_based_features(self):
        """
        Create additional time-based features.
        - 'is_peak_hour': Binary feature indicating access during peak hours (8 AM - 8 PM).
        - 'is_weekend': Binary feature indicating if the access was on a weekend.
        """
        self.data['is_peak_hour'] = self.data['hour'].apply(lambda x: 1 if 8 <= x <= 20 else 0)
        self.data['is_weekend'] = self.data['dayofweek'].apply(lambda x: 1 if x >=5 else 0)  # 5=Saturday,6=Sunday
        print("Feature Engineering: Created time-based features 'is_peak_hour' and 'is_weekend'")
        self.print_columns("creating time-based features")

    def encode_interaction_features(self):
        """
        Encode interaction features if necessary.
        - Currently, interaction features are numerical and do not require encoding.
        """
        # Placeholder for any encoding if interaction features are categorical
        print("Feature Engineering: No encoding required for interaction features")
        pass

    def scale_engineered_features(self):
        """
        Scale engineered numerical features to ensure uniformity.
        """
        features_to_scale = ['access_per_unique_endpoint', 'access_per_hour']
        scaler = StandardScaler()
        self.data[features_to_scale] = scaler.fit_transform(self.data[features_to_scale])
        print("Feature Engineering: Scaled engineered numerical features 'access_per_unique_endpoint' and 'access_per_hour'")
        self.print_columns("scaling engineered features")

    def prepare_final_features(self):
        """
        Prepare the final set of features for model training.
        - Retain 'anomalous' as the target variable.
        """
        # Ensure 'anomalous' exists
        if 'anomalous' not in self.data.columns:
            raise KeyError("'anomalous' column is missing from the DataFrame.")

        # No dropping of 'anomalous' here; it will be separated later
        self.engineered_features = self.data.copy()
        print("Feature Engineering: Prepared final feature set for model training")
        self.print_columns("preparing final features")

    def optimize_data_types(self):
        """
        Optimize data types to reduce memory usage.
        """
        for col in self.engineered_features.select_dtypes(include=['float64']).columns:
            self.engineered_features[col] = pd.to_numeric(self.engineered_features[col], downcast='float')
        for col in self.engineered_features.select_dtypes(include=['int64']).columns:
            self.engineered_features[col] = pd.to_numeric(self.engineered_features[col], downcast='integer')
        print("Feature Engineering: Optimized data types for memory efficiency")
        self.print_columns("optimizing data types")

    def preprocess(self):
        """
        Execute all feature engineering steps in sequence.
        """
        self.load_data()
        self.drop_unnecessary_columns()
        self.create_interaction_features()
        self.create_time_based_features()
        self.encode_interaction_features()
        self.scale_engineered_features()
        self.prepare_final_features()
        self.optimize_data_types()
        print("\nFeature engineering completed successfully.")

    def get_engineered_features(self):
        """
        Retrieve the engineered feature set and labels.

        Returns:
        - X (pd.DataFrame): Engineered feature matrix.
        - y (pd.Series): Labels.
        """
        if 'anomalous' not in self.engineered_features.columns:
            raise KeyError("'anomalous' column is missing from the engineered features.")

        X = self.engineered_features.drop(['anomalous'], axis=1)
        y = self.engineered_features['anomalous']
        return X, y

    def save_engineered_data(self, path='data/engineered_features_final.csv'):
        """
        Save the engineered features to a CSV file.

        Parameters:
        - path (str): File path to save the engineered features.
        """
        # Ensure 'anomalous' is included for labels
        self.engineered_features.to_csv(path, index=False)
        print(f"Feature Engineering: Engineered data saved successfully at '{path}'")

# Example usage:
if __name__ == "__main__":
    # Define file paths
    processed_logs_path = "data/processed_logs_final.csv"
    engineered_features_path = "data/engineered_features_final.csv"

    # Ensure the 'data' directory exists
    os.makedirs("data", exist_ok=True)

    # Initialize the FeatureEngineer with the path to the processed CSV file
    feature_engineer = FeatureEngineer(filepath=processed_logs_path)

    try:
        # Execute feature engineering
        feature_engineer.preprocess()

        # Retrieve engineered features and labels
        X, y = feature_engineer.get_engineered_features()

        # Save the engineered features
        feature_engineer.save_engineered_data(path=engineered_features_path)
    except Exception as e:
        print(f"\nAn error occurred during feature engineering: {e}")
