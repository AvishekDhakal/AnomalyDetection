#!/usr/bin/env python3

"""
Enhanced Anomaly Detection Data Preprocessing Script

Key Improvements:
1. Config-driven feature engineering
2. Proper endpoint parameter handling
3. Authorization checks aligned with anomaly config
4. Improved categorical encoding
5. Temporal features from config
6. Better handling of unseen categories

Usage:
    python3 data_preprocessing.py --mode train --config config.json
    python3 data_preprocessing.py --mode inference --config config.json
"""
#data_preprocessing.py

# ================================
# 1. Importingo Necessary Libraries
# ================================
import pandas as pd
import numpy as np
import os
import json
import ipaddress
import argparse
import sys
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utils import build_parameters  # Import from shared utilities

# ================================
# 2. Configuration Handling
# ================================
class ConfigManager:
    def __init__(self, config_file):
        self.config = self._load_config(config_file)
        self._validate_config()
        
    def _load_config(self, config_file):
        """Load and validate configuration file"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found")
            
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Set default feature engineering parameters
            config.setdefault('FEATURE_ENGINEERING', {
                'unusual_hours': {'start': 21, 'end': 6},
                'endpoint_parse_depth': 3
            })
            
            return config
            
        except Exception as e:
            raise RuntimeError(f"Error loading config: {str(e)}")

    def _validate_config(self):
        """Ensure required configuration sections exist"""
        required_sections = [
            'ENDPOINTS', 'ANOMALOUS_ENDPOINTS', 'PARAMETERS',
            'ROLES', 'FEATURE_ENGINEERING'
        ]
        missing = [section for section in required_sections 
                 if section not in self.config]
        if missing:
            raise ValueError(f"Missing config sections: {missing}")

# ================================
# 3. Feature Engineering Utilities
# ================================
class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.feature_config = config['FEATURE_ENGINEERING']
        
    def clean_endpoint(self, endpoint):
        """Remove parameters and versioning from endpoint"""
        base_endpoint = endpoint.split('?')[0].split('/v')[0]
        return base_endpoint.rstrip('/')

    def parse_endpoint(self, endpoint):
        """Hierarchical endpoint parsing using config-defined depth"""
        depth = self.feature_config.get('endpoint_parse_depth', 3)
        parts = self.clean_endpoint(endpoint).split('/')[1:]  # Skip empty first element
        
        parsed = {}
        for i in range(depth):
            part = parts[i] if i < len(parts) else 'none'
            parsed[f'endpoint_level_{i+1}'] = part
            
        return parsed

    def is_authorized(self, role, endpoint, method):
        """Improved authorization check with path hierarchy awareness"""
        base_endpoint = self.clean_endpoint(endpoint)
        role_anomalies = self.config['ANOMALY_SCENARIOS'].get(role, {})
        # Check all anomaly definitions for this role's scenarios
        for scenario_name in role_anomalies:
            for ep_def in self.config['ANOMALOUS_ENDPOINTS'].get(scenario_name, []):
                config_ep = self.clean_endpoint(ep_def[0])
                config_method = ep_def[1]
                
                # Check if either:
                # 1. Exact endpoint match with method
                # 2. Base endpoint is parent path of request endpoint with method
                if (base_endpoint.startswith(config_ep) or 
                    config_ep.startswith(base_endpoint)) and \
                    method == config_method:
                    return 0  # Unauthorized
                    
        return 1  # Authorized

    def calculate_temporal_features(self, timestamp):
        """Calculate temporal features using config-defined parameters"""
        unusual_start = self.feature_config['unusual_hours']['start']
        unusual_end = self.feature_config['unusual_hours']['end']
        
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_unusual_time': 1 if unusual_start <= timestamp.hour or timestamp.hour < unusual_end else 0
        }

# ================================
# 4. Data Preprocessing Pipeline
# ================================
class DataPreprocessor:
    def __init__(self, config_manager, mode='train'):
        self.config = config_manager.config
        self.mode = mode
        self.feature_engineer = FeatureEngineer(self.config)
        self.encoders = {}
        if self.mode == 'inference':
            encoder_path = 'models/encoders.pkl'
            if os.path.exists(encoder_path):
                try:
                    import pickle
                    with open(encoder_path, 'rb') as f:
                        self.encoders = pickle.load(f)
                    print("Loaded encoders from", encoder_path)
                except Exception as e:
                    raise ValueError(f"Error loading saved encoders: {e}")
            else:
                raise ValueError("No encoder file found. Please run training mode first to save encoders.")  
            
    def load_data(self, file_path):
        """Load and validate raw data"""
        try:
            df = pd.read_csv(file_path)
            required_cols = ['LogID', 'Role', 'Endpoint', 'HTTP_Method', 'IP_Address', 'Timestamp']
            
            if not set(required_cols).issubset(df.columns):
                missing = set(required_cols) - set(df.columns)
                raise ValueError(f"Missing columns: {missing}")
                
            return df
            
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")

    def preprocess(self, df):
        """Main preprocessing pipeline"""
        # 1. Clean and parse endpoints
        df = self._process_endpoints(df)
        
        # 2. Handle temporal features
        df = self._process_temporal(df)
        
        # 3. IP address features
        df = self._process_ips(df)
        
        # 4. Authorization checks
        df = self._process_authorizations(df)
        
        # 5. Encode categorical features
        df = self._encode_features(df)
        
        # 6. Final cleanup
        return self._final_cleanup(df)

    def _process_endpoints(self, df):
        """Parse endpoints into hierarchical components"""
        endpoint_data = df['Endpoint'].apply(
            self.feature_engineer.parse_endpoint
        ).apply(pd.Series)
        
        return pd.concat([df, endpoint_data], axis=1)

    def _process_temporal(self, df):
        """Extract temporal features from timestamp"""
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df.dropna(subset=['Timestamp'])  # Remove invalid timestamps
        
        temporal_data = df['Timestamp'].apply(
            self.feature_engineer.calculate_temporal_features
        ).apply(pd.Series)
        
        return pd.concat([df, temporal_data], axis=1)

    def _process_ips(self, df):
        """Process IP address features"""
        df['is_internal_ip'] = df['IP_Address'].apply(
            lambda ip: 1 if ipaddress.ip_address(ip).is_private else 0
        )
        return df.drop('IP_Address', axis=1)

    def _process_authorizations(self, df):
        """Add authorization features"""
        df['is_authorized'] = df.apply(
            lambda row: self.feature_engineer.is_authorized(
                row['Role'], row['Endpoint'], row['HTTP_Method']
            ), axis=1
        )
        return df

    def _encode_features(self, df):
        """Smart encoding of categorical features"""
        # Label encode hierarchical endpoint components
        for col in [c for c in df.columns if c.startswith('endpoint_level_')]:
            df[col] = self._label_encode(col, df[col])
            
        # One-hot encode HTTP methods
        df = self._onehot_encode('HTTP_Method', df)
        
        # Target encoding for roles
        if self.mode == 'train' and 'Anomalous' in df.columns:
            role_encoding = df.groupby('Role')['Anomalous'].mean().to_dict()
            df['role_risk'] = df['Role'].map(role_encoding)
            
        return df

    def _label_encode(self, col_name, series):
        if self.mode == 'train':
            encoder = LabelEncoder().fit(series)
            self.encoders[col_name] = encoder
            return encoder.transform(series)
        else:
            encoder = self.encoders.get(col_name)
            if not encoder:
                raise ValueError(f"No encoder found for {col_name}")
            return safe_label_encode(encoder, series)

    def _onehot_encode(self, col_name, df):
        """Handle one-hot encoding with column consistency"""
        if self.mode == 'train':
            encoder = OneHotEncoder(handle_unknown='ignore').fit(df[[col_name]])
            self.encoders[col_name] = encoder
            
        encoder = self.encoders[col_name]
        encoded = encoder.transform(df[[col_name]]).toarray()
        encoded_df = pd.DataFrame(
            encoded,
            columns=[f"{col_name}_{cat}" for cat in encoder.categories_[0]]
        )
        
        return pd.concat([df.drop(col_name, axis=1), encoded_df], axis=1)

    def _final_cleanup(self, df):
        """Final cleanup and validation"""
        # Handle target variable
        if self.mode == 'inference' and 'Anomalous' in df.columns:
            df = df.drop('Anomalous', axis=1)
            
        # Ensure consistent column order
        cols = [c for c in df.columns if c not in ['LogID', 'Timestamp']]
        cols = ['LogID', 'Timestamp'] + cols  # Move ID and timestamp to front
        
        return df[cols]

# ================================
# 5. Main Execution
# ================================
def main():
    parser = argparse.ArgumentParser(description='Enhanced Data Preprocessor')
    parser.add_argument('--mode', choices=['train', 'inference'], required=True)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--input', default='data/master_logs.csv')
    parser.add_argument('--output', default='data/processed_data.pkl',
                       help='Output path for .pkl file; .csv will be created with same base name')
    args = parser.parse_args()

    try:
        # Initialize configuration
        config = ConfigManager(args.config)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config, args.mode)
        
        # Load and process data
        raw_data = preprocessor.load_data(args.input)
        processed_data = preprocessor.preprocess(raw_data)
        
        # Save processed data - Dual Format
        # 1. Save to Pickle (for model consumption)
        processed_data.to_pickle(args.output)
        
        # 2. Save to CSV (for human verification)
        base_path = os.path.splitext(args.output)[0]
        csv_path = f"{base_path}.csv"
        processed_data.to_csv(csv_path, index=False)
        
        print(f"Successfully processed data saved to:")
        print(f"- Pickle: {args.output}")
        print(f"- CSV:    {csv_path}")
        
        # Save encoders if in training mode
        if args.mode == 'train':
            with open('models/encoders.pkl', 'wb') as f:
                pickle.dump(preprocessor.encoders, f)
                
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        sys.exit(1)
def safe_label_encode(encoder, series, default_value="unknown"):
        # Get the classes seen during training
        classes = set(encoder.classes_)
        
        # Convert the series to a numpy array for easier processing
        arr = series.to_numpy()
        
        # Identify unseen labels and map them to a default value
        new_arr = np.array([x if x in classes else default_value for x in arr])
        
        # If the default value is not in the encoder, you must add it
        if default_value not in classes:
            encoder.classes_ = np.append(encoder.classes_, default_value)
        
        # Now perform the transformation
        return encoder.transform(new_arr)

if __name__ == '__main__':
    main()