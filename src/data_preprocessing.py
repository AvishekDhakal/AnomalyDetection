# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

class DataPreprocessor:
    def __init__(self, filepath):
        """
        Initialize the DataPreprocessor with the path to the CSV log file.

        Parameters:
        - filepath (str): Path to the CSV file containing log data.
        """
        self.filepath = filepath
        self.data = None
        self.features = None
        self.labels = None
        self.log_ids = None
        self.encoder_categorical = None
        self.scaler = None

    def print_columns(self, step_description):
        """
        Print the current columns of the DataFrame for debugging purposes.

        Parameters:
        - step_description (str): Description of the current preprocessing step.
        """
        print(f"\n--- Columns after {step_description} ---")
        print(self.data.columns.tolist())

    def load_data(self):
        """
        Load data from the CSV file into a pandas DataFrame and standardize column names.
        """
        self.data = pd.read_csv(self.filepath)
        self.data.columns = self.data.columns.str.strip().str.lower()
        print(f"Data loaded successfully with shape: {self.data.shape}")
        print("Raw columns:", self.data.columns.tolist())

    def handle_missing_values(self):
        """
        Handle missing values in the dataset.
        - Numerical features: Impute with median.
        - Categorical features: Impute with mode.
        """
        # Identify numerical and categorical columns
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_cols.remove('logid')  # Exclude LogID
        numerical_cols.remove('anomalous')  # Exclude target variable

        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove('timestamp')  # To be processed separately
        categorical_cols.remove('ip_address')  # Handled separately

        # Impute numerical columns with median
        imputer_num = SimpleImputer(strategy='median')
        self.data[numerical_cols] = imputer_num.fit_transform(self.data[numerical_cols])

        # Impute categorical columns with mode
        imputer_cat = SimpleImputer(strategy='most_frequent')
        self.data[categorical_cols] = imputer_cat.fit_transform(self.data[categorical_cols])

        print("Missing values handled successfully.")

    def convert_timestamp(self):
        """
        Convert the 'timestamp' column to datetime and extract relevant features.
        """
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['dayofweek'] = self.data['timestamp'].dt.dayofweek
        self.data['month'] = self.data['timestamp'].dt.month
        self.data['day'] = self.data['timestamp'].dt.day
        self.data['weekofyear'] = self.data['timestamp'].dt.isocalendar().week.astype(int)
        self.data.drop('timestamp', axis=1, inplace=True)
        print("Timestamp converted and new time-based features extracted.")

    def separate_endpoint(self):
        """
        Separate the base endpoint from its parameters to reduce cardinality.
        """
        self.data['baseendpoint'] = self.data['endpoint'].apply(lambda x: x.split('?')[0] if '?' in x else x)
        self.data['endpointparameter'] = self.data['endpoint'].apply(lambda x: x.split('?')[1] if '?' in x else '')
        self.data.drop('endpoint', axis=1, inplace=True)
        print("BaseEndpoint and EndpointParameter extracted successfully.")

    def extract_ip_features(self):
        """
        Extract subnet information from the 'ip_address' and encode it.
        """
        if 'ip_address' not in self.data.columns:
            raise KeyError("'ip_address' column is missing from the DataFrame.")

        self.data['subnet'] = self.data['ip_address'].apply(lambda x: '.'.join(x.split('.')[:3]))
        authorized_subnet = "10.0.0"
        self.data['is_authorized_subnet'] = self.data['subnet'].apply(lambda x: 1 if x == authorized_subnet else 0)
        self.data.drop(['ip_address', 'subnet'], axis=1, inplace=True)
        print("IP Address features extracted and encoded successfully.")

    def feature_engineering(self):
        """
        Perform feature engineering by extracting additional relevant features.
        """
        # Access Frequency per User
        self.data['accesscount'] = self.data.groupby('userid')['userid'].transform('count')

        # Unique Endpoints Accessed per User
        self.data['uniqueendpoints'] = self.data.groupby('userid')['baseendpoint'].transform('nunique')

        print("Additional feature engineering completed.")

    def encode_categorical_features(self):
        """
        Encode remaining categorical features using One-Hot Encoding.
        """
        # Identify categorical columns
        categorical_cols = ['role', 'http_method', 'baseendpoint']

        self.encoder_categorical = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoded_cols = self.encoder_categorical.fit_transform(self.data[categorical_cols])
        encoded_col_names = self.encoder_categorical.get_feature_names_out(categorical_cols)
        encoded_df = pd.DataFrame(encoded_cols, columns=encoded_col_names, index=self.data.index)

        # Concatenate with original data
        self.data = pd.concat([self.data.drop(categorical_cols, axis=1), encoded_df], axis=1)
        print("Categorical features encoded successfully.")

    def scale_features(self):
        """
        Scale numerical features using StandardScaler.
        """
        numerical_cols = ['hour', 'dayofweek', 'month', 'day', 'weekofyear', 'accesscount', 'uniqueendpoints']
        self.scaler = StandardScaler()
        self.data[numerical_cols] = self.scaler.fit_transform(self.data[numerical_cols])
        print("Numerical features scaled successfully.")

    def prepare_features_and_labels(self):
        """
        Separate features and labels, and retain LogID for traceability.
        """
        self.log_ids = self.data['logid']
        self.labels = self.data['anomalous']
        self.features = self.data.drop(['anomalous', 'logid'], axis=1)
        print("Features and labels separated successfully.")

    def optimize_data_types(self):
        """
        Optimize data types to reduce memory usage.
        """
        for col in self.features.select_dtypes(include=['float64']).columns:
            self.features[col] = pd.to_numeric(self.features[col], downcast='float')
        for col in self.features.select_dtypes(include=['int64']).columns:
            self.features[col] = pd.to_numeric(self.features[col], downcast='integer')
        print("Data types optimized successfully.")

    def preprocess(self):
        """
        Execute all preprocessing steps in sequence with debugging.
        """
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
        
        print("\nData preprocessing completed successfully.")

    def get_processed_data(self):
        """
        Retrieve the preprocessed features, labels, and LogIDs.

        Returns:
        - X (pd.DataFrame): Feature matrix.
        - y (pd.Series): Labels.
        - log_ids (pd.Series): LogID for traceability.
        """
        return self.features, self.labels, self.log_ids

    def save_preprocessor(self, path='data/preprocessor.joblib'):
        """
        Save the fitted encoders and scaler for future use.

        Parameters:
        - path (str): File path to save the preprocessor.
        """
        joblib.dump({
            'encoder_categorical': self.encoder_categorical,  # Encoder for categorical features
            'scaler': self.scaler
        }, path)
        print(f"Preprocessor saved successfully at '{path}'.")

    def load_preprocessor(self, path='data/preprocessor.joblib'):
        """
        Load the fitted encoders and scaler from a file.

        Parameters:
        - path (str): File path to load the preprocessor.
        """
        preprocessor = joblib.load(path)
        self.encoder_categorical = preprocessor['encoder_categorical']
        self.scaler = preprocessor['scaler']
        print(f"Preprocessor loaded successfully from '{path}'.")

# Example usage:
if __name__ == "__main__":
    # Define file paths
    raw_logs_path = "data/raw_logs.csv"
    processed_logs_path = "data/processed_logs_final.csv"
    preprocessor_save_path = "data/preprocessor.joblib"

    # Ensure the 'data' directory exists
    os.makedirs("data", exist_ok=True)

    # Initialize the DataPreprocessor with the path to the raw CSV file
    preprocessor = DataPreprocessor(filepath=raw_logs_path)

    try:
        # Execute preprocessing
        preprocessor.preprocess()

        # Retrieve processed data
        X, y, log_ids = preprocessor.get_processed_data()

        # Save the preprocessor for future use
        preprocessor.save_preprocessor(path=preprocessor_save_path)

        # Combine processed features with labels and LogID for traceability
        processed_df = X.copy()
        processed_df['anomalous'] = y
        processed_df['logid'] = log_ids
        processed_df.to_csv(processed_logs_path, index=False)
        print(f"\nProcessed data saved successfully at '{processed_logs_path}'.")
    except Exception as e:
        print(f"\nAn error occurred during preprocessing: {e}")
