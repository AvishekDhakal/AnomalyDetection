# feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import logging
import joblib

class FeatureEngineer:
    def __init__(self, filepath, has_labels=True):
        """
        Initialize the FeatureEngineer with the path to the processed CSV log file.

        Parameters:
        - filepath (str): Path to the preprocessed CSV file.
        - has_labels (bool): Indicates whether the data includes the 'anomalous' label.
        """
        self.filepath = filepath
        self.has_labels = has_labels
        self.data = None
        self.engineered_features = None
        self.scaler = None

        # Setup logging
        os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists
        logging.basicConfig(
            filename='logs/feature_engineering.log',
            filemode='a',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("FeatureEngineer initialized.")

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
            # List of expected one-hot encoded 'role' columns
            role_columns = ['role_Doctor', 'role_Nurse', 'role_Staff']
            http_method_columns = ['http_method_GET', 'http_method_HEAD', 'http_method_OPTIONS', 
                                   'http_method_PATCH', 'http_method_POST', 'http_method_PUT']
            baseendpoint_columns = [
                'baseendpoint_/admin/settings', 
                'baseendpoint_/billing/invoices', 
                'baseendpoint_/inventory/items', 
                'baseendpoint_/patient/records'
            ]

            # Base required columns
            required_columns = [
                'logid',
                'hour',
                'dayofweek',
                'month',
                'day',
                'weekofyear',
                'is_authorized_subnet',
                'accesscount',
                'uniqueendpoints'
            ]

            if self.has_labels:
                required_columns.append('anomalous')

            # Add one-hot encoded categorical columns
            required_columns += role_columns + http_method_columns + baseendpoint_columns

            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise KeyError(f"Required columns are missing from the data: {missing_columns}")

            logging.info("Data validation passed.")
            print("Feature Engineering: Data validation passed.")
        except KeyError as ke:
            logging.error(f"Data validation failed: {ke}")
            print(f"Feature Engineering Error: Data validation failed with error: {ke}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during data validation: {e}")
            print(f"Feature Engineering Error: Unexpected error during data validation: {e}")
            raise

    def load_data(self):
        """
        Load preprocessed data from the CSV file into a pandas DataFrame.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            logging.info(f"Data loaded successfully with shape: {self.data.shape}")
            print(f"Feature Engineering: Data loaded successfully with shape: {self.data.shape}")
        except Exception as e:
            logging.error(f"Failed to load data from {self.filepath}: {e}")
            raise

    def drop_unnecessary_columns(self):
        """
        Drop columns that are not needed for feature engineering or model training.
        - Drops 'userid' and 'endpointparameter' as they may not provide additional predictive value.
        """
        try:
            columns_to_drop = ['userid', 'endpointparameter']
            existing_columns_to_drop = [col for col in columns_to_drop if col in self.data.columns]
            self.data.drop(columns=existing_columns_to_drop, inplace=True)
            logging.info(f"Dropped columns: {existing_columns_to_drop}")
            print(f"Feature Engineering: Dropped columns {existing_columns_to_drop}")
            self.print_columns("dropping unnecessary columns")
        except Exception as e:
            logging.error(f"Error in dropping unnecessary columns: {e}")
            print(f"Feature Engineering Error: Dropping unnecessary columns failed with error: {e}")
            raise

    def create_interaction_features(self):
        """
        Create interaction features to capture complex patterns.
        - 'access_per_unique_endpoint': Ratio of access count to unique endpoints.
        - 'access_per_hour': Access count normalized per hour.
        """
        try:
            # Avoid division by zero
            self.data['access_per_unique_endpoint'] = self.data.apply(
                lambda row: row['accesscount'] / row['uniqueendpoints'] if row['uniqueendpoints'] > 0 else 0, axis=1
            )
            
            # Assuming 'hour' ranges from 0-23, normalize 'accesscount' per hour
            self.data['access_per_hour'] = self.data['accesscount'] / 24  # Simple normalization
            logging.info("Created interaction features 'access_per_unique_endpoint' and 'access_per_hour'")
            print("Feature Engineering: Created interaction features 'access_per_unique_endpoint' and 'access_per_hour'")
            self.print_columns("creating interaction features")
        except Exception as e:
            logging.error(f"Error in creating interaction features: {e}")
            print(f"Feature Engineering Error: Creating interaction features failed with error: {e}")
            raise

    def create_time_based_features(self):
        """
        Create additional time-based features.
        - 'is_peak_hour': Binary feature indicating access during peak hours (8 AM - 8 PM).
        - 'is_weekend': Binary feature indicating if the access was on a weekend.
        """
        try:
            self.data['is_peak_hour'] = self.data['hour'].apply(lambda x: 1 if 8 <= x <= 20 else 0)
            self.data['is_weekend'] = self.data['dayofweek'].apply(lambda x: 1 if x >=5 else 0)  # 5=Saturday,6=Sunday
            logging.info("Created time-based features 'is_peak_hour' and 'is_weekend'")
            print("Feature Engineering: Created time-based features 'is_peak_hour' and 'is_weekend'")
            self.print_columns("creating time-based features")
        except Exception as e:
            logging.error(f"Error in creating time-based features: {e}")
            print(f"Feature Engineering Error: Creating time-based features failed with error: {e}")
            raise

    def create_additional_features(self):
        """
        Create additional features to enhance model performance.
        - 'average_access_per_user': Average access per user.
        - 'endpoint_popularity': Frequency of endpoint access.
        """
        try:
            # Create 'average_access_per_user'
            self.data['average_access_per_user'] = self.data['accesscount'] / (self.data['uniqueendpoints'] + 1)

            # Calculate 'endpoint_popularity' based on one-hot encoded 'baseendpoint_/...' columns
            baseendpoint_columns = [
                'baseendpoint_/admin/settings', 
                'baseendpoint_/billing/invoices', 
                'baseendpoint_/inventory/items', 
                'baseendpoint_/patient/records'
            ]
            # Compute total access counts per endpoint
            endpoint_counts = self.data[baseendpoint_columns].sum()

            # Create a mapping from baseendpoint column to its total count
            endpoint_popularity_map = {col: count for col, count in endpoint_counts.items()}

            # Define a function to get the popularity based on which baseendpoint is 1
            def get_endpoint_popularity(row):
                for col in baseendpoint_columns:
                    if row[col] == 1:
                        return endpoint_popularity_map[col]
                return 0  # If no baseendpoint matched

            self.data['endpoint_popularity'] = self.data.apply(get_endpoint_popularity, axis=1)

            logging.info("Created additional features 'average_access_per_user' and 'endpoint_popularity'")
            print("Feature Engineering: Created additional features 'average_access_per_user' and 'endpoint_popularity'")
            self.print_columns("creating additional features")
        except KeyError as ke:
            logging.error(f"KeyError in creating additional features: {ke}")
            print(f"Feature Engineering Error: Creating additional features failed with error: {ke}")
            raise
        except Exception as e:
            logging.error(f"Error in creating additional features: {e}")
            print(f"Feature Engineering Error: Creating additional features failed with error: {e}")
            raise

    def encode_interaction_features(self):
        """
        Encode interaction features if necessary.
        - Currently, interaction features are numerical and do not require encoding.
        """
        try:
            # Placeholder for any encoding if interaction features become categorical in the future
            logging.info("No encoding required for interaction features")
            print("Feature Engineering: No encoding required for interaction features")
        except Exception as e:
            logging.error(f"Error in encoding interaction features: {e}")
            print(f"Feature Engineering Error: Encoding interaction features failed with error: {e}")
            raise

    def scale_engineered_features(self):
        """
        Scale engineered numerical features to ensure uniformity.
        """
        try:
            features_to_scale = ['access_per_unique_endpoint', 'access_per_hour']
            self.scaler = StandardScaler()
            self.data[features_to_scale] = self.scaler.fit_transform(self.data[features_to_scale])
            logging.info("Scaled engineered numerical features 'access_per_unique_endpoint' and 'access_per_hour'")
            print("Feature Engineering: Scaled engineered numerical features 'access_per_unique_endpoint' and 'access_per_hour'")
            self.print_columns("scaling engineered features")
        except Exception as e:
            logging.error(f"Error in scaling engineered features: {e}")
            print(f"Feature Engineering Error: Scaling engineered features failed with error: {e}")
            raise

    def prepare_final_features(self):
        """
        Prepare the final set of features for model training or inference.
        - Retains 'anomalous' as the target variable if present.
        """
        try:
            if self.has_labels and 'anomalous' in self.data.columns:
                self.engineered_features = self.data.copy()
                logging.info("Prepared final features with labels for training.")
                print("Feature Engineering: Prepared final features with labels for training.")
            else:
                self.engineered_features = self.data.copy()
                logging.info("Prepared final features without labels for inference.")
                print("Feature Engineering: Prepared final features without labels for inference.")
            self.print_columns("preparing final features")
        except Exception as e:
            logging.error(f"Error in preparing final features: {e}")
            print(f"Feature Engineering Error: Preparing final features failed with error: {e}")
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
            print("Feature Engineering: Optimized data types for memory efficiency.")
            self.print_columns("optimizing data types")
        except Exception as e:
            logging.error(f"Error in optimizing data types: {e}")
            print(f"Feature Engineering Error: Optimizing data types failed with error: {e}")
            raise

    def preprocess(self):
        """
        Execute all feature engineering steps in sequence.
        """
        try:
            self.load_data()
            self.validate_data()  # Ensure all required columns are present
            self.drop_unnecessary_columns()
            self.create_interaction_features()
            self.create_time_based_features()
            self.create_additional_features()  # Must be before encoding
            self.encode_interaction_features()
            self.scale_engineered_features()
            self.prepare_final_features()
            self.optimize_data_types()
            logging.info("Feature engineering completed successfully.")
            print("\nFeature engineering completed successfully.")
        except Exception as e:
            logging.error(f"Feature engineering failed: {e}")
            print(f"\nFeature engineering failed: {e}")
            raise

    def get_engineered_features(self):
        """
        Retrieve the engineered feature set and labels.

        Returns:
        - X (pd.DataFrame): Engineered feature matrix.
        - y (pd.Series or None): Labels.
        - log_ids (pd.Series): LogID for traceability.
        """
        try:
            log_ids = self.engineered_features['logid']
            if self.has_labels and 'anomalous' in self.engineered_features.columns:
                y = self.engineered_features['anomalous']
                X = self.engineered_features.drop(['anomalous', 'logid'], axis=1)
                logging.info("Retrieved engineered features and labels.")
                print("Feature Engineering: Retrieved engineered features and labels.")
            else:
                y = None
                X = self.engineered_features.drop(['logid'], axis=1)
                logging.info("Retrieved engineered features without labels for inference.")
                print("Feature Engineering: Retrieved engineered features without labels for inference.")
            return X, y, log_ids
        except KeyError as ke:
            logging.error(f"KeyError in get_engineered_features: {ke}")
            print(f"Feature Engineering Error: {ke}")
            raise
        except Exception as e:
            logging.error(f"Error in get_engineered_features: {e}")
            print(f"Feature Engineering Error: {e}")
            raise

    def save_engineered_data(self, path='data/engineered_features_final.csv'):
        """
        Save the engineered features to a CSV file.

        Parameters:
        - path (str): File path to save the engineered features.
        """
        try:
            self.engineered_features.to_csv(path, index=False)
            logging.info(f"Engineered data saved successfully at '{path}'.")
            print(f"Feature Engineering: Engineered data saved successfully at '{path}'.")
        except Exception as e:
            logging.error(f"Failed to save engineered data: {e}")
            print(f"Feature Engineering Error: Saving engineered data failed with error: {e}")
            raise

    def load_scaler(self, path='data/feature_scaler.joblib'):
        """
        Load a previously saved scaler.

        Parameters:
        - path (str): File path to load the scaler.
        """
        try:
            self.scaler = joblib.load(path)
            logging.info(f"Scaler loaded successfully from '{path}'.")
            print(f"Feature Engineering: Scaler loaded successfully from '{path}'.")
        except Exception as e:
            logging.error(f"Failed to load scaler: {e}")
            print(f"Feature Engineering Error: Loading scaler failed with error: {e}")
            raise

    def save_scaler(self, path='data/feature_scaler.joblib'):
        """
        Save the scaler for future use.

        Parameters:
        - path (str): File path to save the scaler.
        """
        try:
            joblib.dump(self.scaler, path)
            logging.info(f"Scaler saved successfully at '{path}'.")
            print(f"Feature Engineering: Scaler saved successfully at '{path}'.")
        except Exception as e:
            logging.error(f"Failed to save scaler: {e}")
            print(f"Feature Engineering Error: Saving scaler failed with error: {e}")
            raise

# Example usage:
if __name__ == "__main__":
    # Define file paths
    processed_logs_path = "data/processed_logs_final.csv"
    engineered_features_path = "data/engineered_features_final.csv"
    feature_scaler_save_path = "data/feature_scaler.joblib"

    # Ensure the 'data' and 'logs' directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)  # For logging

    # Initialize the FeatureEngineer with the path to the processed CSV file
    # Set has_labels=True for training data and has_labels=False for inference data
    # Example for training data (with labels)
    feature_engineer = FeatureEngineer(filepath=processed_logs_path, has_labels=True)

    try:
        # Execute feature engineering
        feature_engineer.preprocess()

        # Retrieve engineered features and labels
        X_train, y_train, log_ids_train = feature_engineer.get_engineered_features()

        # Save the engineered features
        feature_engineer.save_engineered_data(path=engineered_features_path)

        # Save the scaler for future use
        feature_engineer.save_scaler(path=feature_scaler_save_path)
    except Exception as e:
        print(f"\nAn error occurred during feature engineering: {e}")

    # Example for inference data (without labels)
    # Uncomment and modify the following section to process inference data

    # print("\n--- Preprocessing Inference Data ---")
    # processed_inference_path = "data/processed_inference_logs.csv"
    # engineered_inference_path = "data/engineered_inference_features_final.csv"

    # feature_engineer_infer = FeatureEngineer(filepath=processed_inference_path, has_labels=False)

    # try:
    #     # Load the previously saved scaler
    #     feature_engineer_infer.load_scaler(path=feature_scaler_save_path)

    #     # Execute feature engineering
    #     feature_engineer_infer.preprocess()

    #     # Retrieve engineered features and log IDs
    #     X_infer, _, log_ids_infer = feature_engineer_infer.get_engineered_features()

    #     # Save the engineered inference features
    #     feature_engineer_infer.save_engineered_data(path=engineered_inference_path)
    # except Exception as e:
    #     print(f"\nAn error occurred during inference feature engineering: {e}")
