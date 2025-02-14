#!/usr/bin/env python3
"""
data_preprocessing_phase1.py

Phase 1: Minimal Preprocessing
-------------------------------
- Load a CSV of logs.
- Parse Timestamp into basic time features (Hour, DayOfWeek, Is_After_Hours).
- Create a simple IP-based feature (Is_Internal_IP).
- Extract a minimal endpoint base (before any '?' or complicated path splitting).
- Keep 'Anomalous' if present; if not present, we skip it.
- Save the resulting DataFrame to a new CSV.

Usage:
  python data_preprocessing_phase1.py --input_csv data/logs.csv --output_csv data/logs_phase1.csv
"""

import argparse
import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV with timestamps parsed."""
    try:
        df = pd.read_csv(filepath, parse_dates=['Timestamp'])
        logging.info(f"Loaded data from {filepath} with shape {df.shape}.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create minimal features:
      - Hour, DayOfWeek, Is_After_Hours
      - Is_Internal_IP
      - Endpoint_Base
    Keep 'Anomalous' if it exists for analysis, else skip it.
    """
    # 1) Time-based
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Is_After_Hours'] = df['Hour'].apply(lambda h: 1 if (h >= 20 or h < 6) else 0)
    
    # 2) IP-based
    def is_internal(ip_str):
        if isinstance(ip_str, str):
            return 1 if (ip_str.startswith("10.") or ip_str.startswith("192.168.")) else 0
        return 0
    df['Is_Internal_IP'] = df['IP_Address'].apply(is_internal)
    
    # 3) Endpoint base
    def extract_endpoint_base(ep):
        if not isinstance(ep, str):
            return "UNKNOWN"
        # Minimal approach: split at '?'
        return ep.split('?')[0]
    
    df['Endpoint_Base'] = df['Endpoint'].apply(extract_endpoint_base)
    
    # Keep or drop 'Anomalous'?
    # We'll keep it if it exists (for correlation or EDA).
    # If it's missing, we do nothing special.
    if 'Anomalous' not in df.columns:
        logging.info("'Anomalous' column not found. Skipping it.")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Minimal Phase 1 Preprocessing")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV logs.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV after minimal preprocessing.")
    args = parser.parse_args()
    
    # 1) Load
    df = load_data(args.input_csv)
    
    # 2) Create minimal features
    df_phase1 = basic_features(df)
    
    # 3) Save
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    df_phase1.to_csv(args.output_csv, index=False)
    logging.info(f"Saved phase 1 preprocessed CSV to: {args.output_csv} with shape={df_phase1.shape}.")

if __name__ == "__main__":
    main()



# python3 data_preprocessing_phase1.py --input_csv data/master_logs.csv --output_csv data/train.csv