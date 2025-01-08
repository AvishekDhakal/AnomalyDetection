# additional_feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os

class AdditionalFeatureEngineer:
    def __init__(self, filepath):
        """
        Initialize the AdditionalFeatureEngineer with the path to the engineered CSV log file.

        Parameters:
        - filepath (str): Path to the engineered CSV file.
        """
        self.filepath = filepath
        self.data = None
        self.engineered_features = None
        self.encoder = None

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
        Load engineered data from the CSV file into a pandas DataFrame.
        """
        self.data = pd.read_csv(self.filepath)
        print(f"Additional Feature Engineering: Data loaded successfully with shape: {self.data.shape}")

    def create_role_peak_hour_interaction(self):
        """
        Create an interaction feature between role and peak hour.
        """
        # Assuming one-hot encoded roles
        roles = ['role_Doctor', 'role_Nurse', 'role_Staff']
        self.data['role_peak_hour'] = 0

        for role in roles:
            self.data[role + '_peak'] = self.data.apply(
                lambda row: 1 if row[role] == 1 and row['is_peak_hour'] == 1 else 0, axis=1
            )
            self.data[role + '_non_peak'] = self.data.apply(
                lambda row: 1 if row[role] == 1 and row['is_peak_hour'] == 0 else 0, axis=1
            )
        
        # Drop the original 'is_peak_hour' if no longer needed
        # self.data.drop('is_peak_hour', axis=1, inplace=True)
        print("Additional Feature Engineering: Created interaction features between roles and peak hours")
        self.print_columns("creating role-peak hour interaction features")

    def encode_new_categorical_features(self):
        """
        Encode newly created categorical interaction features using One-Hot Encoding.
        """
        interaction_cols = ['role_Doctor_peak', 'role_Doctor_non_peak',
                            'role_Nurse_peak', 'role_Nurse_non_peak',
                            'role_Staff_peak', 'role_Staff_non_peak']

        self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoded_interactions = self.encoder.fit_transform(self.data[interaction_cols])
        encoded_interaction_names = self.encoder.get_feature_names_out(interaction_cols)
        encoded_interaction_df = pd.DataFrame(encoded_interactions, columns=encoded_interaction_names, index=self.data.index)

        # Concatenate with original data
        self.data = pd.concat([self.data.drop(interaction_cols, axis=1), encoded_interaction_df], axis=1)
        print("Additional Feature Engineering: Encoded interaction features using One-Hot Encoding")
        self.print_columns("encoding interaction features")

    def create_http_method_frequencies(self):
        """
        Create frequency features for each HTTP method per user.
        """
        http_methods = ['http_method_GET', 'http_method_POST', 'http_method_PUT']

        for method in http_methods:
            freq_col = f"{method}_freq"
            self.data[freq_col] = self.data.groupby('baseendpoint_/admin/settings')[method].transform('mean')
        
        print("Additional Feature Engineering: Created HTTP method frequency features")
        self.print_columns("creating HTTP method frequency features")

    def handle_class_imbalance(self):
        """
        Placeholder for handling class imbalance techniques like SMOTE.
        """
        # Class imbalance is typically handled during model training, not feature engineering
        print("Additional Feature Engineering: No handling of class imbalance here")
        pass

    def prepare_final_features(self):
        """
        Finalize the feature set by selecting relevant features for modeling.
        """
        # For demonstration, we'll retain all features
        self.engineered_features = self.data.copy()
        print("Additional Feature Engineering: Finalized feature set for modeling")
        self.print_columns("finalizing features")

    def optimize_data_types(self):
        """
        Optimize data types to reduce memory usage.
        """
        for col in self.engineered_features.select_dtypes(include=['float64']).columns:
            self.engineered_features[col] = pd.to_numeric(self.engineered_features[col], downcast='float')
        for col in self.engineered_features.select_dtypes(include=['int64']).columns:
            self.engineered_features[col] = pd.to_numeric(self.engineered_features[col], downcast='integer')
        print("Additional Feature Engineering: Optimized data types for memory efficiency")
        self.print_columns("optimizing data types")

    def preprocess(self):
        """
        Execute all additional feature engineering steps in sequence.
        """
        self.load_data()
        self.create_role_peak_hour_interaction()
        self.encode_new_categorical_features()
        self.create_http_method_frequencies()
        self.prepare_final_features()
        self.optimize_data_types()
        print("\nAdditional feature engineering completed successfully.")

    def get_final_features_and_labels(self):
        """
        Retrieve the final engineered feature set and labels.

        Returns:
        - X (pd.DataFrame): Final feature matrix.
        - y (pd.Series): Labels.
        """
        if 'anomalous' not in self.engineered_features.columns:
            raise KeyError("'anomalous' column is missing from the engineered features.")

        X = self.engineered_features.drop(['anomalous'], axis=1)
        y = self.engineered_features['anomalous']
        return X, y

    def save_final_engineered_data(self, path='data/final_engineered_features.csv'):
        """
        Save the final engineered features to a CSV file.

        Parameters:
        - path (str): File path to save the final engineered features.
        """
        # Ensure 'anomalous' is included for labels
        self.engineered_features.to_csv(path, index=False)
        print(f"Additional Feature Engineering: Final engineered data saved successfully at '{path}'")

# Example usage:
if __name__ == "__main__":
    # Define file paths
    engineered_features_path = "data/engineered_features_final.csv"
    final_engineered_features_path = "data/final_engineered_features.csv"

    # Ensure the 'data' directory exists
    os.makedirs("data", exist_ok=True)

    # Initialize the AdditionalFeatureEngineer with the path to the engineered CSV file
    additional_feature_engineer = AdditionalFeatureEngineer(filepath=engineered_features_path)

    try:
        # Execute additional feature engineering
        additional_feature_engineer.preprocess()

        # Retrieve final engineered features and labels
        X_final, y_final = additional_feature_engineer.get_final_features_and_labels()

        # Save the final engineered features
        additional_feature_engineer.save_final_engineered_data(path=final_engineered_features_path)
    except Exception as e:
        print(f"\nAn error occurred during additional feature engineering: {e}")
