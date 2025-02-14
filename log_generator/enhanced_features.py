#enhanced_features.py


#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import json
import warnings
import argparse

warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, config_path='config.json'):
        self.feature_columns = []  # Will hold the list of features used for training (without meta)
        self.encoders = {}
        self.config = self._load_config(config_path)
        self.inventory_endpoints = ['/inventory/items']
        # These are all columns that might appear in the raw data
        self.required_columns = [
            'LogID', 'Timestamp', 'UserID', 'Role', 'Endpoint', 'HTTP_Response',
            'endpoint_level_1', 'endpoint_level_2', 'endpoint_level_3',
            'hour', 'day_of_week', 'is_unusual_time', 'is_internal_ip',
            'is_authorized', 'HTTP_Method_DELETE', 'HTTP_Method_GET',
            'HTTP_Method_HEAD', 'HTTP_Method_OPTIONS', 'HTTP_Method_PATCH',
            'HTTP_Method_POST', 'HTTP_Method_PUT', 'role_risk'
        ]
        
    def _load_config(self, config_path):
        with open(config_path) as f:
            return json.load(f)
    
    def _add_temporal_features(self, df):
        """Add rolling window frequency features with proper index handling"""
        # Create temporary sorted dataframe
        sorted_df = df.sort_values(['UserID', 'Timestamp']).reset_index(drop=True)
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(sorted_df['Timestamp']):
            sorted_df['Timestamp'] = pd.to_datetime(sorted_df['Timestamp'])
        
        # Calculate 5-minute request frequency
        freq_counts = (
            sorted_df.groupby('UserID', group_keys=False)
            .rolling('5T', on='Timestamp', closed='left')['LogID']
            .count()
            .reset_index()
            .rename(columns={'LogID': 'req_freq_5min'})
        )
        
        # Merge back with original data
        sorted_df = sorted_df.merge(
            freq_counts[['UserID', 'Timestamp', 'req_freq_5min']],
            on=['UserID', 'Timestamp'],
            how='left'
        )
        
        # Calculate inventory change rate
        inventory_mask = sorted_df['Endpoint'].str.startswith(tuple(self.inventory_endpoints))
        inv_counts = (
            sorted_df[inventory_mask]
            .groupby('UserID', group_keys=False)
            .rolling('10T', on='Timestamp', closed='left')['LogID']
            .count()
            .reset_index()
            .rename(columns={'LogID': 'inventory_change_rate'})
        )
        
        # Merge inventory rates
        sorted_df = sorted_df.merge(
            inv_counts[['UserID', 'Timestamp', 'inventory_change_rate']],
            on=['UserID', 'Timestamp'],
            how='left'
        )
        
        # Fill NaN values and restore original order
        sorted_df['req_freq_5min'] = sorted_df['req_freq_5min'].fillna(0)
        sorted_df['inventory_change_rate'] = sorted_df['inventory_change_rate'].fillna(0)
        
        # Restore original index and order
        return sorted_df.sort_index().drop(columns=['level_0'], errors='ignore')

    def _encode_categoricals(self, df, training=True):
        """Handle categorical encoding with consistency"""
        # Role encoding
        if training:
            self.encoders['role'] = LabelEncoder().fit(df['Role'])
        df['role_encoded'] = self.encoders['role'].transform(df['Role'])
        
        # Endpoint level 1 encoding
        if training:
            self.encoders['endpoint_l1'] = LabelEncoder().fit(df['endpoint_level_1'])
        df['endpoint_l1_encoded'] = self.encoders['endpoint_l1'].transform(df['endpoint_level_1'])
        
        return df

    def _save_feature_mapping(self):
        """Save feature configuration to pickle file"""
        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)
        
        feature_config = {
            'feature_columns': self.feature_columns,  # These are the training features (used for prediction)
            'encoders': self.encoders,
            'required_columns': self.required_columns
        }
        
        with open(model_dir / 'features.pkl', 'wb') as f:
            pickle.dump(feature_config, f)

    def _ensure_columns(self, df):
        """Add missing columns with default values"""
        for col in self.required_columns:
            if col not in df.columns:
                df[col] = 0 if col != 'Anomalous' else np.nan
        return df

    def fit_transform(self, input_path, output_path):
        """Process training data with Anomalous column"""
        # Load and validate data
        df = pd.read_csv(input_path)
        df = self._ensure_columns(df)
        
        # Add temporal features
        df = self._add_temporal_features(df)
        
        # Encode categoricals
        df = self._encode_categoricals(df, training=True)
        
        # Define feature columns: drop metadata that are not used by the model.
        # We assume that the training features are all columns except 'LogID', 'Timestamp', and 'Anomalous'
        self.feature_columns = [c for c in df.columns if c not in ['LogID', 'Timestamp', 'Anomalous']]
        
        # Save feature mapping so that inference can use the same features
        self._save_feature_mapping()
        
        # Save processed training data
        df.to_csv(output_path, index=False)
        print(f"Training data processed and saved to {output_path}. Feature mapping saved.")
        return df

    # def transform(self, input_path, output_path):
    #     """Process inference data without Anomalous column using the frozen feature mapping"""
    #     # Load and validate inference data
    #     df = pd.read_csv(input_path)
    #     df = self._ensure_columns(df)
    #     df = self._add_temporal_features(df)
        
    #     # Load the saved feature mapping (from training)
    #     mapping_path = 'models/features.pkl'
    #     try:
    #         with open(mapping_path, 'rb') as f:
    #             saved_mapping = pickle.load(f)
    #     except Exception as e:
    #         print(f"Warning: Feature mapping not found ({e}). Proceeding with current features.")
    #         saved_mapping = {'feature_columns': []}
        
    #     # Load the saved encoders into self.encoders
    #     if 'encoders' in saved_mapping:
    #         self.encoders = saved_mapping['encoders']
        
    #     # Now encode categoricals using the loaded encoders
    #     df = self._encode_categoricals(df, training=False)
        
    #     expected_features = saved_mapping.get('feature_columns', [])
    #     if not expected_features:
    #         print("No saved feature columns found. Using all processed columns except metadata.")
    #         expected_features = [c for c in df.columns if c not in ['LogID', 'Timestamp', 'Anomalous']]
    #     else:
    #         # Add any missing expected features with default value 0
    #         for feature in expected_features:
    #             if feature not in df.columns:
    #                 df[feature] = 0
        
    #     # Select exactly the features used during training
    #     df = df[expected_features]
        
    #     # Save the processed inference data
    #     df.to_csv(output_path, index=False)
    #     print(f"Inference data processed and saved to {output_path}.")
    #     return df
    def transform(self, input_path, output_path):
        """Process inference data without Anomalous column using the frozen feature mapping,
        but keep metadata (e.g. LogID, Timestamp) in the saved CSV for traceability."""
        # Load and validate inference data
        df = pd.read_csv(input_path)
        df = self._ensure_columns(df)
        df = self._add_temporal_features(df)
        
        # Load the saved feature mapping (from training)
        mapping_path = 'models/features.pkl'
        try:
            with open(mapping_path, 'rb') as f:
                saved_mapping = pickle.load(f)
        except Exception as e:
            print(f"Warning: Feature mapping not found ({e}). Proceeding with current features.")
            saved_mapping = {'feature_columns': []}
        
        # Load the saved encoders into self.encoders
        if 'encoders' in saved_mapping:
            self.encoders = saved_mapping['encoders']
        
        # Now encode categoricals using the loaded encoders
        df = self._encode_categoricals(df, training=False)
        
        # Define the expected features (these are used for model prediction)
        expected_features = saved_mapping.get('feature_columns', [])
        if not expected_features:
            print("No saved feature columns found. Using all processed columns except metadata.")
            expected_features = [c for c in df.columns if c not in ['LogID', 'Timestamp', 'Anomalous']]
        else:
            # Add any missing expected features with default value 0
            for feature in expected_features:
                if feature not in df.columns:
                    df[feature] = 0
        
        # Create a copy of the full processed data (including metadata)
        df_full = df.copy()
        
        # Extract the feature subset that the model expects
        df_features = df_full[expected_features]
        
        # (At prediction time you would pass df_features to the model.)
        # But for saving the CSV for later analytics, merge back the metadata.
        # Here, we assume that 'LogID' and 'Timestamp' are in the full data.
        meta_cols = ['LogID', 'Timestamp']
        # Save the full DataFrame (which includes both metadata and features)
        df_full.to_csv(output_path, index=False)
        print(f"Inference data processed and saved to {output_path}.")
        
        # Optionally, if you want to return both the feature subset (for prediction)
        # and the full data (for reporting), you could return them as a tuple.
        return df_full  # Full data contains metadata; model predictions should use df_features separately.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced Feature Engineering")
    parser.add_argument('--mode', choices=['train', 'inference'], required=True,
                        help="Mode of operation: 'train' to fit and save feature mapping; 'inference' to load mapping and process data accordingly.")
    parser.add_argument('--input', type=str, default='data/processed_data.csv',
                        help="Path to input CSV file (processed_data.csv)")
    parser.add_argument('--output', type=str, default=None,
                        help="Path to output CSV file. Defaults to 'data/preprocessed_train.csv' for training mode or 'data/preprocessed_inference.csv' for inference mode.")
    args = parser.parse_args()

    # Determine default output path if not provided
    if args.output is None:
        if args.mode == 'train':
            args.output = 'data/preprocessed_train.csv'
        else:
            args.output = 'data/preprocessed_inference.csv'

    fe = FeatureEngineer()
    
    if args.mode == 'train':
        fe.fit_transform(input_path=args.input, output_path=args.output)
    else:
        fe.transform(input_path=args.input, output_path=args.output)
