# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os
import logging

class DataPreprocessor:
    def __init__(self, filepath, has_labels=True):
        """
        Initialize the DataPreprocessor with the path to the CSV log file.

        Parameters:
        - filepath (str): Path to the CSV file containing log data.
        - has_labels (bool): Indicates whether the data includes the 'Anomalous' label.
        """
        self.filepath = filepath
        self.has_labels = has_labels
        self.data = None
        self.features = None
        self.labels = None
        self.log_ids = None
        self.encoder_categorical = None
        self.scaler = None

        # Setup logging
        os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists
        logging.basicConfig(
            filename='logs/preprocessing.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("DataPreprocessor initialized.")

    def print_columns(self, step_description):
        """
        Print the current columns of the DataFrame for debugging purposes.

        Parameters:
        - step_description (str): Description of the current preprocessing step.
        """
        print(f"\n--- Columns after {step_description} ---")
        print(self.data.columns.tolist())
        logging.info(f"Columns after {step_description}: {self.data.columns.tolist()}")

    def load_data(self):
        """
        Load data from the CSV file into a pandas DataFrame and standardize column names.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            self.data.columns = self.data.columns.str.strip().str.lower()
            logging.info(f"Data loaded successfully with shape: {self.data.shape}")
            logging.info(f"Raw columns: {self.data.columns.tolist()}")
            print(f"Data loaded successfully with shape: {self.data.shape}")
        except Exception as e:
            logging.error(f"Failed to load data from {self.filepath}: {e}")
            raise

    def handle_missing_values(self):
        """
        Handle missing values in the dataset.
        - Numerical features: Impute with median.
        - Categorical features: Impute with mode.
        """
        try:
            # Identify numerical and categorical columns
            numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Remove 'anomalous' from numerical_cols if labels are present
            if self.has_labels and 'anomalous' in numerical_cols:
                numerical_cols.remove('anomalous')
            
            # 'logid' is of type object, so it shouldn't be in numerical_cols
            # No need to remove 'logid' from numerical_cols
            
            categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
            categorical_cols.remove('timestamp')  # To be processed separately
            categorical_cols.remove('ip_address')  # Handled separately

            # Impute numerical columns with median
            imputer_num = SimpleImputer(strategy='median')
            self.data[numerical_cols] = imputer_num.fit_transform(self.data[numerical_cols])

            # Impute categorical columns with mode
            imputer_cat = SimpleImputer(strategy='most_frequent')
            self.data[categorical_cols] = imputer_cat.fit_transform(self.data[categorical_cols])

            logging.info("Missing values handled successfully.")
            print("Missing values handled successfully.")
        except ValueError as ve:
            logging.error(f"ValueError in handle_missing_values: {ve}")
            print(f"ValueError in handle_missing_values: {ve}")
            raise
        except Exception as e:
            logging.error(f"Error in handle_missing_values: {e}")
            print(f"Error in handle_missing_values: {e}")
            raise

    def convert_timestamp(self):
        """
        Convert the 'timestamp' column to datetime and extract relevant features.
        """
        try:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data['hour'] = self.data['timestamp'].dt.hour
            self.data['dayofweek'] = self.data['timestamp'].dt.dayofweek
            self.data['month'] = self.data['timestamp'].dt.month
            self.data['day'] = self.data['timestamp'].dt.day
            self.data['weekofyear'] = self.data['timestamp'].dt.isocalendar().week.astype(int)
            self.data.drop('timestamp', axis=1, inplace=True)
            logging.info("Timestamp converted and new time-based features extracted.")
            print("Timestamp converted and new time-based features extracted.")
        except Exception as e:
            logging.error(f"Error in converting timestamp: {e}")
            print(f"Error in converting timestamp: {e}")
            raise

    def separate_endpoint(self):
        """
        Separate the base endpoint from its parameters to reduce cardinality.
        """
        try:
            self.data['baseendpoint'] = self.data['endpoint'].apply(lambda x: x.split('?')[0] if '?' in x else x)
            self.data['endpointparameter'] = self.data['endpoint'].apply(lambda x: x.split('?')[1] if '?' in x else '')
            self.data.drop('endpoint', axis=1, inplace=True)
            logging.info("BaseEndpoint and EndpointParameter extracted successfully.")
            print("BaseEndpoint and EndpointParameter extracted successfully.")
        except Exception as e:
            logging.error(f"Error in separating endpoint: {e}")
            print(f"Error in separating endpoint: {e}")
            raise

    def extract_ip_features(self):
        """
        Extract subnet information from the 'ip_address' and encode it.
        """
        try:
            if 'ip_address' not in self.data.columns:
                raise KeyError("'ip_address' column is missing from the DataFrame.")

            self.data['subnet'] = self.data['ip_address'].apply(lambda x: '.'.join(x.split('.')[:3]))
            authorized_subnet = "10.0.0"
            self.data['is_authorized_subnet'] = self.data['subnet'].apply(lambda x: 1 if x == authorized_subnet else 0)
            self.data.drop(['ip_address', 'subnet'], axis=1, inplace=True)
            logging.info("IP Address features extracted and encoded successfully.")
            print("IP Address features extracted and encoded successfully.")
        except KeyError as ke:
            logging.error(f"KeyError in extracting IP features: {ke}")
            print(f"KeyError in extracting IP features: {ke}")
            raise
        except Exception as e:
            logging.error(f"Error in extracting IP features: {e}")
            print(f"Error in extracting IP features: {e}")
            raise

    def feature_engineering(self):
        """
        Perform feature engineering by extracting additional relevant features.
        """
        try:
            # Access Frequency per User
            self.data['accesscount'] = self.data.groupby('userid')['userid'].transform('count')

            # Unique Endpoints Accessed per User
            self.data['uniqueendpoints'] = self.data.groupby('userid')['baseendpoint'].transform('nunique')

            logging.info("Additional feature engineering completed.")
            print("Additional feature engineering completed.")
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            print(f"Error in feature engineering: {e}")
            raise

    def encode_categorical_features(self):
        """
        Encode remaining categorical features using One-Hot Encoding.
        """
        try:
            # Identify categorical columns
            categorical_cols = ['role', 'http_method', 'baseendpoint']

            self.encoder_categorical = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            encoded_cols = self.encoder_categorical.fit_transform(self.data[categorical_cols])
            encoded_col_names = self.encoder_categorical.get_feature_names_out(categorical_cols)
            encoded_df = pd.DataFrame(encoded_cols, columns=encoded_col_names, index=self.data.index)

            # Concatenate with original data
            self.data = pd.concat([self.data.drop(categorical_cols, axis=1), encoded_df], axis=1)
            logging.info("Categorical features encoded successfully.")
            print("Categorical features encoded successfully.")
        except Exception as e:
            logging.error(f"Error in encoding categorical features: {e}")
            print(f"Error in encoding categorical features: {e}")
            raise

    def scale_features(self):
        """
        Scale numerical features using StandardScaler.
        """
        try:
            numerical_cols = ['hour', 'dayofweek', 'month', 'day', 'weekofyear', 'accesscount', 'uniqueendpoints']
            self.scaler = StandardScaler()
            self.data[numerical_cols] = self.scaler.fit_transform(self.data[numerical_cols])
            logging.info("Numerical features scaled successfully.")
            print("Numerical features scaled successfully.")
        except Exception as e:
            logging.error(f"Error in scaling features: {e}")
            print(f"Error in scaling features: {e}")
            raise

    def prepare_features_and_labels(self):
        """
        Separate features and labels, and retain LogID for traceability.
        """
        try:
            self.log_ids = self.data['logid']
            if self.has_labels and 'anomalous' in self.data.columns:
                self.labels = self.data['anomalous']
                self.features = self.data.drop(['anomalous', 'logid'], axis=1)
            else:
                self.labels = None  # Placeholder if 'anomalous' is missing
                self.features = self.data.drop(['logid'], axis=1)
            logging.info("Features and labels separated successfully.")
            print("Features and labels separated successfully.")
        except Exception as e:
            logging.error(f"Error in preparing features and labels: {e}")
            print(f"Error in preparing features and labels: {e}")
            raise

    def optimize_data_types(self):
        """
        Optimize data types to reduce memory usage.
        """
        try:
            for col in self.features.select_dtypes(include=['float64']).columns:
                self.features[col] = pd.to_numeric(self.features[col], downcast='float')
            for col in self.features.select_dtypes(include=['int64']).columns:
                self.features[col] = pd.to_numeric(self.features[col], downcast='integer')
            logging.info("Data types optimized successfully.")
            print("Data types optimized successfully.")
        except Exception as e:
            logging.error(f"Error in optimizing data types: {e}")
            print(f"Error in optimizing data types: {e}")
            raise

    def preprocess(self):
        """
        Execute all preprocessing steps in sequence with debugging.
        """
        try:
            self.load_data()
            self.print_columns("loading data")

            self.handle_missing_values()
            self.print_columns("handling missing values")

            self.convert_timestamp()
            self.print_columns("converting timestamp")

            self.separate_endpoint()
            self.print_columns("separating endpoint")

            self.extract_ip_features()
            self.print_columns("extracting IP features")

            self.feature_engineering()
            self.print_columns("feature engineering")

            self.encode_categorical_features()
            self.print_columns("encoding categorical features")

            self.scale_features()
            self.print_columns("scaling features")

            self.prepare_features_and_labels()
            self.print_columns("preparing features and labels")

            self.optimize_data_types()
            self.print_columns("optimizing data types")

            logging.info("Data preprocessing completed successfully.")
            print("\nData preprocessing completed successfully.")
        except Exception as e:
            logging.error(f"Preprocessing failed: {e}")
            print(f"\nAn error occurred during preprocessing: {e}")
            raise

    def get_processed_data(self):
        """
        Retrieve the preprocessed features, labels, and LogIDs.

        Returns:
        - X (pd.DataFrame): Feature matrix.
        - y (pd.Series or None): Labels.
        - log_ids (pd.Series): LogID for traceability.
        """
        return self.features, self.labels, self.log_ids

    def save_preprocessor(self, path='data/preprocessor.joblib'):
        """
        Save the fitted encoders and scaler for future use.

        Parameters:
        - path (str): File path to save the preprocessor.
        """
        try:
            joblib.dump({
                'encoder_categorical': self.encoder_categorical,  # Encoder for categorical features
                'scaler': self.scaler
            }, path)
            logging.info(f"Preprocessor saved successfully at '{path}'.")
            print(f"Preprocessor saved successfully at '{path}'.")
        except Exception as e:
            logging.error(f"Failed to save preprocessor: {e}")
            print(f"Failed to save preprocessor: {e}")
            raise

    def load_preprocessor(self, path='data/preprocessor.joblib'):
        """
        Load the fitted encoders and scaler from a file.

        Parameters:
        - path (str): File path to load the preprocessor.
        """
        try:
            preprocessor = joblib.load(path)
            self.encoder_categorical = preprocessor['encoder_categorical']
            self.scaler = preprocessor['scaler']
            logging.info(f"Preprocessor loaded successfully from '{path}'.")
            print(f"Preprocessor loaded successfully from '{path}'.")
        except Exception as e:
            logging.error(f"Failed to load preprocessor: {e}")
            print(f"Failed to load preprocessor: {e}")
            raise

# Example usage:
if __name__ == "__main__":
    # Define file paths
    master_logs_path = "data/master_logs.csv"  # Contains 'anomalous' column
    inference_logs_path = "data/inference_logs.csv"  # Does NOT contain 'anomalous' column
    processed_logs_path = "data/processed_logs_final.csv"
    processed_inference_path = "data/processed_inference_logs.csv"
    preprocessor_save_path = "data/preprocessor.joblib"

    # Ensure the 'data' directory exists
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)  # For logging

    # Preprocessing for Training Data (with labels)
    print("\n--- Preprocessing Training Data ---")
    preprocessor_train = DataPreprocessor(filepath=master_logs_path, has_labels=True)
    try:
        # Execute preprocessing
        preprocessor_train.preprocess()

        # Retrieve processed data
        X_train, y_train, log_ids_train = preprocessor_train.get_processed_data()

        # Save the preprocessor for future use
        preprocessor_train.save_preprocessor(path=preprocessor_save_path)

        # Combine processed features with labels and LogID for traceability
        processed_df_train = X_train.copy()
        processed_df_train['anomalous'] = y_train
        processed_df_train['logid'] = log_ids_train
        processed_df_train.to_csv(processed_logs_path, index=False)
        print(f"\nProcessed training data saved successfully at '{processed_logs_path}'.")
    except Exception as e:
        print(f"\nAn error occurred during training data preprocessing: {e}")

    # Preprocessing for Inference Data (without labels)
    print("\n--- Preprocessing Inference Data ---")
    preprocessor_infer = DataPreprocessor(filepath=inference_logs_path, has_labels=False)
    try:
        # Load the preprocessor
        preprocessor_infer.load_preprocessor(path=preprocessor_save_path)

        # Execute preprocessing
        preprocessor_infer.preprocess()

        # Retrieve processed data
        X_infer, _, log_ids_infer = preprocessor_infer.get_processed_data()

        # Combine processed features with LogID for traceability
        processed_df_infer = X_infer.copy()
        processed_df_infer['logid'] = log_ids_infer
        processed_df_infer.to_csv(processed_inference_path, index=False)
        print(f"\nProcessed inference data saved successfully at '{processed_inference_path}'.")
    except Exception as e:
        print(f"\nAn error occurred during inference data preprocessing: {e}")
