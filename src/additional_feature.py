# additional_feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
import logging

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

        # Setup logging
        os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists
        logging.basicConfig(
            filename='logs/additional_feature_engineering.log',
            filemode='a',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("AdditionalFeatureEngineer initialized.")

    def print_columns(self, step_description):
        """
        Print the current columns of the DataFrame for debugging purposes.

        Parameters:
        - step_description (str): Description of the current feature engineering step.
        """
        print(f"\n--- Columns after {step_description} ---")
        print(self.data.columns.tolist())
        logging.info(f"Columns after {step_description}: {self.data.columns.tolist()}")

    def validate_data(self):
        """
        Validate the presence and format of essential columns.
        """
        try:
            required_columns = [
                'logid',
                'is_peak_hour',
                'role_Doctor',
                'role_Nurse',
                'role_Staff',
                'http_method_GET',
                'http_method_HEAD',
                'http_method_OPTIONS',
                'http_method_PATCH',
                'http_method_POST',
                'http_method_PUT',
                'baseendpoint_/admin/settings',
                'baseendpoint_/billing/invoices',
                'baseendpoint_/inventory/items',
                'baseendpoint_/patient/records'
            ]

            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise KeyError(f"Required columns are missing from the data: {missing_columns}")

            logging.info("Data validation passed.")
            print("Additional Feature Engineering: Data validation passed.")
        except KeyError as ke:
            logging.error(f"Data validation failed: {ke}")
            print(f"Additional Feature Engineering Error: Data validation failed with error: {ke}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during data validation: {e}")
            print(f"Additional Feature Engineering Error: Unexpected error during data validation: {e}")
            raise

    def load_data(self):
        """
        Load engineered data from the CSV file into a pandas DataFrame.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            logging.info(f"Data loaded successfully with shape: {self.data.shape}")
            print(f"Additional Feature Engineering: Data loaded successfully with shape: {self.data.shape}")
        except Exception as e:
            logging.error(f"Failed to load data from {self.filepath}: {e}")
            raise

    def create_role_peak_hour_interaction(self):
        """
        Create interaction features between user roles and peak hours.
        Generates new binary columns indicating if a role was active during peak or non-peak hours.
        """
        try:
            roles = ['role_Doctor', 'role_Nurse', 'role_Staff']
            for role in roles:
                peak_col = f"{role}_peak"
                non_peak_col = f"{role}_non_peak"
                
                self.data[peak_col] = self.data.apply(
                    lambda row: 1 if row[role] == 1 and row['is_peak_hour'] == 1 else 0, axis=1
                )
                self.data[non_peak_col] = self.data.apply(
                    lambda row: 1 if row[role] == 1 and row['is_peak_hour'] == 0 else 0, axis=1
                )
            
            logging.info("Created interaction features between roles and peak hours.")
            print("Additional Feature Engineering: Created interaction features between roles and peak hours.")
            self.print_columns("creating role-peak hour interaction features")
        except KeyError as ke:
            logging.error(f"KeyError in creating role-peak hour interaction features: {ke}")
            print(f"Additional Feature Engineering Error: Creating role-peak hour interaction features failed with error: {ke}")
            raise
        except Exception as e:
            logging.error(f"Error in creating role-peak hour interaction features: {e}")
            print(f"Additional Feature Engineering Error: Creating role-peak hour interaction features failed with error: {e}")
            raise

    def encode_new_categorical_features(self):
        """
        Encode newly created categorical interaction features using One-Hot Encoding.
        Drops the first category to avoid multicollinearity.
        """
        try:
            interaction_cols = [
                'role_Doctor_peak', 'role_Doctor_non_peak',
                'role_Nurse_peak', 'role_Nurse_non_peak',
                'role_Staff_peak', 'role_Staff_non_peak'
            ]

            # Initialize OneHotEncoder
            self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            encoded_interactions = self.encoder.fit_transform(self.data[interaction_cols])

            # Generate new column names
            encoded_interaction_names = self.encoder.get_feature_names_out(interaction_cols)

            # Create a DataFrame with encoded features
            encoded_interaction_df = pd.DataFrame(encoded_interactions, columns=encoded_interaction_names, index=self.data.index)

            # Concatenate with original data and drop original interaction columns
            self.data = pd.concat([self.data.drop(interaction_cols, axis=1), encoded_interaction_df], axis=1)

            logging.info("Encoded interaction features using One-Hot Encoding.")
            print("Additional Feature Engineering: Encoded interaction features using One-Hot Encoding.")
            self.print_columns("encoding interaction features")
        except KeyError as ke:
            logging.error(f"KeyError in encoding interaction features: {ke}")
            print(f"Additional Feature Engineering Error: Encoding interaction features failed with error: {ke}")
            raise
        except Exception as e:
            logging.error(f"Error in encoding interaction features: {e}")
            print(f"Additional Feature Engineering Error: Encoding interaction features failed with error: {e}")
            raise

    def create_http_method_frequencies(self):
        """
        Create frequency features for each HTTP method per base endpoint.
        Calculates the mean usage of each HTTP method per endpoint.
        """
        try:
            http_methods = ['http_method_GET', 'http_method_POST', 'http_method_PUT']
            baseendpoints = [
                'baseendpoint_/admin/settings', 
                'baseendpoint_/billing/invoices', 
                'baseendpoint_/inventory/items', 
                'baseendpoint_/patient/records'
            ]

            for method in http_methods:
                for endpoint in baseendpoints:
                    freq_col = f"{method}_freq_{endpoint.split('/')[-1]}"
                    self.data[freq_col] = self.data.groupby(endpoint)[method].transform('mean')
            
            logging.info("Created HTTP method frequency features per base endpoint.")
            print("Additional Feature Engineering: Created HTTP method frequency features per base endpoint.")
            self.print_columns("creating HTTP method frequency features")
        except KeyError as ke:
            logging.error(f"KeyError in creating HTTP method frequency features: {ke}")
            print(f"Additional Feature Engineering Error: Creating HTTP method frequency features failed with error: {ke}")
            raise
        except Exception as e:
            logging.error(f"Error in creating HTTP method frequency features: {e}")
            print(f"Additional Feature Engineering Error: Creating HTTP method frequency features failed with error: {e}")
            raise

    def handle_class_imbalance(self):
        """
        Placeholder for handling class imbalance techniques like SMOTE.
        Note: Class imbalance is typically handled during model training, not feature engineering.
        """
        print("Additional Feature Engineering: No handling of class imbalance here.")
        logging.info("No handling of class imbalance during feature engineering.")
        pass

    def prepare_final_features(self):
        """
        Finalize the feature set by selecting relevant features for modeling.
        """
        try:
            # For demonstration, we'll retain all features
            self.engineered_features = self.data.copy()
            logging.info("Finalized feature set for modeling.")
            print("Additional Feature Engineering: Finalized feature set for modeling.")
            self.print_columns("finalizing features")
        except Exception as e:
            logging.error(f"Error in finalizing feature set: {e}")
            print(f"Additional Feature Engineering Error: Finalizing feature set failed with error: {e}")
            raise

    def optimize_data_types(self):
        """
        Optimize data types to reduce memory usage.
        """
        try:
            for col in self.engineered_features.select_dtypes(include=['float64']).columns:
                self.engineered_features[col] = pd.to_numeric(self.engineered_features[col], downcast='float')
            for col in self.engineered_features.select_dtypes(include=['int64']).columns:
                self.engineered_features[col] = pd.to_numeric(self.engineered_features[col], downcast='integer')
            logging.info("Optimized data types for memory efficiency.")
            print("Additional Feature Engineering: Optimized data types for memory efficiency.")
            self.print_columns("optimizing data types")
        except Exception as e:
            logging.error(f"Error in optimizing data types: {e}")
            print(f"Additional Feature Engineering Error: Optimizing data types failed with error: {e}")
            raise

    def preprocess(self):
        """
        Execute all additional feature engineering steps in sequence.
        """
        try:
            self.load_data()
            self.validate_data()
            self.create_role_peak_hour_interaction()
            self.encode_new_categorical_features()
            self.create_http_method_frequencies()
            self.handle_class_imbalance()  # Currently does nothing
            self.prepare_final_features()
            self.optimize_data_types()
            logging.info("Additional feature engineering completed successfully.")
            print("\nAdditional feature engineering completed successfully.")
        except Exception as e:
            logging.error(f"Additional feature engineering failed: {e}")
            print(f"\nAdditional feature engineering failed: {e}")
            raise

    def get_final_features_and_labels(self):
        """
        Retrieve the final engineered feature set and labels.

        Returns:
        - X (pd.DataFrame): Final feature matrix.
        - y (pd.Series): Labels.
        """
        try:
            if 'anomalous' not in self.engineered_features.columns:
                raise KeyError("'anomalous' column is missing from the engineered features.")

            X = self.engineered_features.drop(['anomalous', 'logid'], axis=1)
            y = self.engineered_features['anomalous']
            logging.info("Retrieved final engineered features and labels.")
            print("Additional Feature Engineering: Retrieved final engineered features and labels.")
            return X, y
        except KeyError as ke:
            logging.error(f"KeyError in get_final_features_and_labels: {ke}")
            print(f"Additional Feature Engineering Error: {ke}")
            raise
        except Exception as e:
            logging.error(f"Error in get_final_features_and_labels: {e}")
            print(f"Additional Feature Engineering Error: {e}")
            raise

    def save_final_engineered_data(self, path='data/final_engineered_features.csv'):
        """
        Save the final engineered features to a CSV file.

        Parameters:
        - path (str): File path to save the final engineered features.
        """
        try:
            self.engineered_features.to_csv(path, index=False)
            logging.info(f"Final engineered data saved successfully at '{path}'.")
            print(f"Additional Feature Engineering: Final engineered data saved successfully at '{path}'.")
        except Exception as e:
            logging.error(f"Failed to save final engineered data: {e}")
            print(f"Additional Feature Engineering Error: Saving final engineered data failed with error: {e}")
            raise

# Example usage:
if __name__ == "__main__":
    # Define file paths
    engineered_features_path = "data/engineered_features_final.csv"
    final_engineered_features_path = "data/final_engineered_features.csv"

    # Ensure the 'data' and 'logs' directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

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
