#!/usr/bin/env python3

"""
data_preprocessing.py

This script performs data preprocessing and feature engineering
on the labeled master logs from a healthcare insider-threat environment.

Key Steps:
1. Data Loading & Basic Checks
2. Data Cleaning (optional flagging of inconsistent entries)
3. Feature Engineering (time-based, IP-based, role-endpoint combos, etc.)
4. Categorical Encoding (Role, HTTP_Method, HTTP_Response, User_Agent, Endpoint, etc.)
5. Preservation of LogID for Traceability
6. Export of final processed dataset for model training or inference

Author: [Your Name]
Date: YYYY-MM-DD
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import joblib
import logging

# -------------------- CONFIGURATION --------------------

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration constants
INTERNAL_IP_SUBNETS = ["10.", "192.168."]
TIME_AFTER_HOURS = (20, 6)  # after 8 PM or before 6 AM
MIN_ENDPOINT_FREQUENCY = 10
ENCODERS_DIR_DEFAULT = "encoders"
FEATURE_NAMES_DEFAULT = "models/feature_names.pkl"

# -------------------- FUNCTIONS --------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the CSV file into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(filepath, parse_dates=['Timestamp'])
        logging.info(f"Successfully loaded data from {filepath} with shape {df.shape}.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise

def basic_cleaning(df: pd.DataFrame, mark_inconsistencies: bool = True, mode: str = "train") -> pd.DataFrame:
    """
    Performs basic cleaning steps:
    1. Ensures 'Anomalous' column exists and is of integer type.
    2. Optionally marks method-response inconsistencies as a separate feature.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        mark_inconsistencies (bool): Flag to mark inconsistencies
        mode (str): 'train' or 'inference'
    """
    # Ensure 'Anomalous' is of integer type only in train mode
    if mode == "train":
        if 'Anomalous' in df.columns:
            if df['Anomalous'].dtype not in [int, np.int64, bool]:
                df['Anomalous'] = df['Anomalous'].astype(int)
                logging.info("Converted 'Anomalous' column to integer type.")
        else:
            # If 'Anomalous' is missing, set to 0 (assumed normal)
            df['Anomalous'] = 0
            logging.warning("'Anomalous' column missing. Setting all entries to 0.")
    elif mode == "inference":
        if 'Anomalous' in df.columns:
            # Optionally, drop the 'Anomalous' column if present
            df = df.drop('Anomalous', axis=1)
            logging.warning("Dropping 'Anomalous' column in inference mode.")
    
    # Ensure the column 'method_response_inconsistent' exists
    df['method_response_inconsistent'] = 0  # Default value is 0
    
    if mark_inconsistencies:
        # Define the mask for inconsistencies
        mask_inconsistent = (
            ((df['HTTP_Method'] == 'DELETE') & (df['HTTP_Response'] == 200)) |
            ((df['HTTP_Method'] == 'PUT') & (df['HTTP_Response'] == 404))
        )
        # Update the column where inconsistencies are found
        df.loc[mask_inconsistent, 'method_response_inconsistent'] = 1
        logging.info("Flagged method-response inconsistencies.")
    else:
        logging.info("'method_response_inconsistent' column initialized but inconsistencies not flagged.")
    
    logging.debug(f"Columns after basic_cleaning: {df.columns.tolist()}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers new features:
    1. Time-based: Hour, DayOfWeek, Is_After_Hours
    2. IP-based: Is_Internal_IP
    3. Endpoint-based: Endpoint_Base, Role_Endpoint
    """
    # 1. Time-based features
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    start_night, end_morning = TIME_AFTER_HOURS
    df['Is_After_Hours'] = df['Hour'].apply(lambda h: 1 if (h >= start_night or h < end_morning) else 0)
    logging.info("Engineered time-based features.")
    
    # 2. IP-based feature
    df['Is_Internal_IP'] = df['IP_Address'].apply(
        lambda ip: 1 if any(ip.startswith(subnet) for subnet in INTERNAL_IP_SUBNETS) else 0
    )
    logging.info("Engineered IP-based feature 'Is_Internal_IP'.")
    
    # 3. Endpoint-based features
    df['Endpoint_Base'] = df['Endpoint'].apply(lambda x: x.split('?')[0] if isinstance(x, str) else 'UNKNOWN_ENDPOINT')
    df['Role_Endpoint'] = df['Role'].astype(str) + '_' + df['Endpoint_Base'].astype(str)
    logging.info("Engineered endpoint-based features 'Endpoint_Base' and 'Role_Endpoint'.")
    logging.debug(f"Columns after engineer_features: {df.columns.tolist()}")
    return df

def bucket_rare_endpoints(df: pd.DataFrame, min_freq: int) -> pd.DataFrame:
    """
    Replaces rare endpoints with 'RARE_ENDPOINT' to reduce cardinality.
    """
    freq = df['Endpoint_Base'].value_counts()
    rare_endpoints = freq[freq < min_freq].index
    df['Endpoint_Base'] = df['Endpoint_Base'].apply(lambda e: 'RARE_ENDPOINT' if e in rare_endpoints else e)
    logging.info(f"Bucketed rare endpoints (frequency < {min_freq}) into 'RARE_ENDPOINT'.")
    logging.debug(f"Columns after bucket_rare_endpoints: {df.columns.tolist()}")
    return df

def encode_categorical_columns(df: pd.DataFrame, mode: str = "train", encoders_dir: str = "encoders") -> pd.DataFrame:
    """
    Encodes categorical columns:
    1. One-hot encode 'Role', 'HTTP_Method', 'HTTP_Response'
    2. One-hot or label encode 'Endpoint_Base' based on cardinality
    3. Label encode 'Role_Endpoint'
    4. One-hot encode 'User_Agent'
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns to encode
        mode (str): 'train' to fit encoders, 'inference' to load and apply encoders
        encoders_dir (str): Directory to save/load encoders
    """
    if not os.path.exists(encoders_dir):
        os.makedirs(encoders_dir)
        logging.info(f"Created encoders directory at '{encoders_dir}'.")
    
    df_encoded = df.copy()
    
    # 1. One-hot encode 'Role', 'HTTP_Method', 'HTTP_Response'
    one_hot_columns = ['Role', 'HTTP_Method', 'HTTP_Response']
    df_encoded = pd.get_dummies(df_encoded, columns=one_hot_columns, drop_first=False)
    logging.info("One-hot encoded 'Role', 'HTTP_Method', 'HTTP_Response'.")
    
    # 2. Encode 'Endpoint_Base'
    unique_endpoints = df_encoded['Endpoint_Base'].nunique()
    logging.info(f"'Endpoint_Base' has {unique_endpoints} unique values.")
    if unique_endpoints <= 20:
        # One-hot encode
        df_encoded = pd.get_dummies(df_encoded, columns=['Endpoint_Base'], prefix='Endpoint', drop_first=False)
        logging.info("One-hot encoded 'Endpoint_Base' as its unique values <= 20.")
    else:
        # Label encode
        le_endpoint = LabelEncoder()
        if mode == "train":
            df_encoded['Endpoint_Base'] = le_endpoint.fit_transform(df_encoded['Endpoint_Base'])
            joblib.dump(le_endpoint, os.path.join(encoders_dir, "le_endpoint.pkl"))
            logging.info("Label encoded 'Endpoint_Base' and saved encoder.")
        else:
            le_endpoint_path = os.path.join(encoders_dir, "le_endpoint.pkl")
            if not os.path.exists(le_endpoint_path):
                logging.error(f"Encoder for 'Endpoint_Base' not found at '{le_endpoint_path}'. Cannot proceed.")
                raise FileNotFoundError(f"Encoder for 'Endpoint_Base' not found at '{le_endpoint_path}'.")
            le_endpoint = joblib.load(le_endpoint_path)
            # Handle unseen endpoints by mapping to 'RARE_ENDPOINT'
            df_encoded['Endpoint_Base'] = df_encoded['Endpoint_Base'].apply(
                lambda x: x if x in le_endpoint.classes_ else 'RARE_ENDPOINT'
            )
            # Ensure 'RARE_ENDPOINT' is in classes
            if 'RARE_ENDPOINT' not in le_endpoint.classes_:
                le_endpoint.classes_ = np.append(le_endpoint.classes_, 'RARE_ENDPOINT')
            df_encoded['Endpoint_Base'] = le_endpoint.transform(df_encoded['Endpoint_Base'])
            logging.info("Label encoded 'Endpoint_Base' using loaded encoder.")
    
    # 3. Label encode 'Role_Endpoint'
    le_role_endpoint = LabelEncoder()
    if mode == "train":
        df_encoded['Role_Endpoint'] = le_role_endpoint.fit_transform(df_encoded['Role_Endpoint'])
        joblib.dump(le_role_endpoint, os.path.join(encoders_dir, "le_role_endpoint.pkl"))
        logging.info("Label encoded 'Role_Endpoint' and saved encoder.")
    else:
        le_role_endpoint_path = os.path.join(encoders_dir, "le_role_endpoint.pkl")
        if not os.path.exists(le_role_endpoint_path):
            logging.error(f"Encoder for 'Role_Endpoint' not found at '{le_role_endpoint_path}'. Cannot proceed.")
            raise FileNotFoundError(f"Encoder for 'Role_Endpoint' not found at '{le_role_endpoint_path}'.")
        le_role_endpoint = joblib.load(le_role_endpoint_path)
        # Handle unseen role-endpoint combinations by mapping to 'Unknown'
        df_encoded['Role_Endpoint'] = df_encoded['Role_Endpoint'].apply(
            lambda x: x if x in le_role_endpoint.classes_ else 'Unknown'
        )
        # Ensure 'Unknown' is in classes
        if 'Unknown' not in le_role_endpoint.classes_:
            le_role_endpoint.classes_ = np.append(le_role_endpoint.classes_, 'Unknown')
        df_encoded['Role_Endpoint'] = le_role_endpoint.transform(df_encoded['Role_Endpoint'])
        logging.info("Label encoded 'Role_Endpoint' using loaded encoder.")
    
    # 4. One-hot encode 'User_Agent'
    if 'User_Agent' in df_encoded.columns:
        df_encoded = pd.get_dummies(df_encoded, columns=['User_Agent'], drop_first=False)
        logging.info("One-hot encoded 'User_Agent'.")
    else:
        logging.warning("'User_Agent' column not found for one-hot encoding.")
    logging.debug(f"Columns after encode_categorical_columns: {df.columns.tolist()}")
    return df_encoded

# def finalize_dataset(df_encoded: pd.DataFrame, output_path: str, mode: str = "train", feature_names_path: str = "models/feature_names.pkl") -> None:
#     """
#     Saves the final preprocessed dataset to CSV, preserving columns that might be needed.
    
#     Parameters:
#         df_encoded (pd.DataFrame): The DataFrame post-encoding
#         output_path (str): File path to save the CSV
#         mode (str): 'train' to include 'Anomalous' column, 'inference' to exclude it
#         feature_names_path (str): Path to save/load feature names for consistency
#     """
#     # Define columns to keep regardless of mode
#     base_cols = [
#         'LogID',
#         'Timestamp',
#         'method_response_inconsistent',
#         'Hour',
#         'DayOfWeek',
#         'Is_After_Hours',
#         'Is_Internal_IP',
#         'Role_Endpoint'
#     ]
    
#     if mode == "train":
#         # Include 'Anomalous' for training
#         base_cols.append('Anomalous')
#     elif mode == "inference":
#         # Exclude 'Anomalous' for inference
#         pass  # Do not append 'Anomalous'
    
#     # Include all one-hot encoded and label encoded features
#     encoded_prefixes = ['Endpoint_', 'User_Agent_', 'Role_', 'HTTP_Method_', 'HTTP_Response_']
#     feature_cols = []
#     for prefix in encoded_prefixes:
#         feature_cols += [col for col in df_encoded.columns if col.startswith(prefix)]
    
#     # Combine base columns and feature columns
#     keep_cols = base_cols + feature_cols
    
#     # Filter the DataFrame
#     df_final = df_encoded[keep_cols].copy()
#     logging.info(f"Final dataset has {df_final.shape[1]} columns.")
    
#     # Save or Align Feature Names
#     if mode == "train":
#         # Save feature names excluding 'LogID' and 'Timestamp'
#         feature_names = df_final.drop(['LogID', 'Timestamp'], axis=1).columns.tolist()
#         os.makedirs(os.path.dirname(feature_names_path), exist_ok=True)
#         joblib.dump(feature_names, feature_names_path)
#         logging.info(f"Saved feature names to '{feature_names_path}'.")
#     elif mode == "inference":
#         # Load feature names and align
#         if not os.path.exists(feature_names_path):
#             logging.error(f"Feature names file not found at '{feature_names_path}'. Cannot align features.")
#             raise FileNotFoundError(f"Feature names file not found at '{feature_names_path}'.")
#         feature_names = joblib.load(feature_names_path)
#         # Identify missing features in df_final
#         missing_features = set(feature_names) - set(df_final.columns)
#         for feature in missing_features:
#             df_final[feature] = 0  # or appropriate default value
#             logging.warning(f"Missing feature '{feature}' in inference data. Adding with default value 0.")
#         # Ensure the order of columns matches
#         df_final = df_final[feature_names]
#         logging.info("Aligned inference data features with training features.")
    
#     # Save the processed dataset
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     logging.debug(f"Columns before saving to CSV: {df_encoded.columns.tolist()}")
#     # Save the processed dataset...
#     df_final.to_csv(output_path, index=False)
#     logging.info(f"Final processed dataset saved to: {output_path}")

def finalize_dataset(df_encoded: pd.DataFrame, output_path: str, mode: str = "train", feature_names_path: str = "models/feature_names.pkl") -> None:
    """
    Saves the final preprocessed dataset to CSV, preserving necessary columns.
    """
    # Define columns to keep regardless of mode
    base_cols = [
        'LogID',  # Ensure 'LogID' is always included
        'Timestamp',
        'method_response_inconsistent',
        'Hour',
        'DayOfWeek',
        'Is_After_Hours',
        'Is_Internal_IP',
        'Role_Endpoint'
    ]
    
    if mode == "train":
        # Include 'Anomalous' for training
        base_cols.append('Anomalous')
    
    # Include all encoded feature columns dynamically
    encoded_prefixes = ['Endpoint_', 'User_Agent_', 'Role_', 'HTTP_Method_', 'HTTP_Response_']
    feature_cols = [col for col in df_encoded.columns if any(col.startswith(prefix) for prefix in encoded_prefixes)]
    
    # Combine base and feature columns
    keep_cols = base_cols + feature_cols
    
    if mode == "train":
        # Save feature names excluding 'LogID' and 'Timestamp'
        df_final = df_encoded[keep_cols].copy()
        feature_names = df_final.drop(['LogID', 'Timestamp'], axis=1).columns.tolist()
        os.makedirs(os.path.dirname(feature_names_path), exist_ok=True)
        joblib.dump(feature_names, feature_names_path)
        logging.info(f"Saved feature names to '{feature_names_path}'.")
    elif mode == "inference":
        # Load feature names and align
        if not os.path.exists(feature_names_path):
            logging.error(f"Feature names file not found at '{feature_names_path}'. Cannot align features.")
            raise FileNotFoundError(f"Feature names file not found at '{feature_names_path}'.")
        
        feature_names = joblib.load(feature_names_path)
        
        # Ensure 'LogID' and 'Timestamp' are always included in inference
        if 'LogID' not in feature_names:
            feature_names.insert(0, 'LogID')
        if 'Timestamp' not in feature_names:
            feature_names.insert(1, 'Timestamp')
        
        # Align features and add missing ones with default values
        df_final = df_encoded.copy()
        for feature in feature_names:
            if feature not in df_final.columns:
                df_final[feature] = 0  # Add missing feature with default value
                logging.warning(f"Missing feature '{feature}' in inference data. Adding with default value 0.")
        df_final = df_final[feature_names]  # Reorder columns to match training
    
    # Save the processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.debug(f"Columns before saving to CSV: {df_final.columns.tolist()}")
    df_final.to_csv(output_path, index=False)
    logging.info(f"Final processed dataset saved to: {output_path}")

# -------------------- MAIN FUNCTION --------------------

def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing & Feature Engineering for Insider Threat Logs")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the input CSV containing labeled or unlabeled logs.")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to the output CSV with preprocessed and feature-engineered logs.")
    parser.add_argument("--mark_inconsistent", action='store_true',
                        help="Flag to mark method/response inconsistencies as a separate feature.")
    parser.add_argument("--min_endpoint_freq", type=int, default=MIN_ENDPOINT_FREQUENCY,
                        help="Minimum frequency threshold for endpoints to avoid large cardinality.")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train",
                        help="Mode of operation: 'train' to include 'Anomalous' column, 'inference' to exclude it.")
    parser.add_argument("--encoders_dir", type=str, default=ENCODERS_DIR_DEFAULT,
                        help="Directory to save/load encoders.")
    parser.add_argument("--feature_names_path", type=str, default=FEATURE_NAMES_DEFAULT,
                        help="Path to save/load feature names for consistency.")
    args = parser.parse_args()
    
    # Validate mode and corresponding input data
    if args.mode == "train":
        df_sample = pd.read_csv(args.input_csv, nrows=1)
        if 'Anomalous' not in df_sample.columns:
            logging.warning("Running in 'train' mode but 'Anomalous' column is missing in the input CSV. Setting 'Anomalous' to 0.")
    
    # 1. Load Data
    logging.info("[DEBUG] Loading data...")
    df = load_data(args.input_csv)
    
    # 2. Basic Cleaning
    logging.info("[DEBUG] Performing basic cleaning...")
    df_clean = basic_cleaning(df, mark_inconsistencies=args.mark_inconsistent, mode=args.mode)
    logging.info("[DEBUG] Basic cleaning completed.")
    
    # 3. Feature Engineering
    logging.info("[DEBUG] Engineering features...")
    df_features = engineer_features(df_clean)
    logging.info("[DEBUG] Feature engineering completed.")
    
    # 4. Bucket Rare Endpoints
    logging.info("[DEBUG] Bucketing rare endpoints...")
    df_features = bucket_rare_endpoints(df_features, min_freq=args.min_endpoint_freq)
    logging.info("[DEBUG] Rare endpoints bucketing completed.")
    
    # 5. Encode Categorical Columns
    logging.info("[DEBUG] Encoding categorical columns...")
    df_encoded = encode_categorical_columns(df_features, mode=args.mode, encoders_dir=args.encoders_dir)
    logging.info("[DEBUG] Categorical encoding completed.")
    
    # 6. Finalize & Save
    logging.info("[DEBUG] Finalizing dataset...")
    finalize_dataset(df_encoded, args.output_csv, mode=args.mode, feature_names_path=args.feature_names_path)
    logging.info(f"Columns before saving to CSV: {df_encoded.columns.tolist()}")
    logging.info("[DEBUG] Dataset finalized and saved.")
    logging.info("Data preprocessing & feature engineering complete.")

if __name__ == "__main__":
    main()



# python3 data_preprocessing.py --input_csv data/inference_logs.csv --output_csv data/preprocessed_test.csv --mode inference
# python3 data_preprocessing.py --input_csv data/master_logs.csv --output_csv data/preprocessed_train.csv