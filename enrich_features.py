#!/usr/bin/env python3

"""
enrich_features.py

A revised script to add or modify features for your current dataset, which lacks UserID and IP_Address.
Focuses on:
1. Possible expansions of 'Role_Endpoint'
2. Global time-window features (optional)
3. A mini EDA on any newly created columns

Author: [Your Name]
Date: YYYY-MM-DD
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_arguments():
    parser = argparse.ArgumentParser(description="Enrich features given the current columns (no UserID, no IP).")
    parser.add_argument("--input_csv", type=str, default="data/processed_logs.csv",
                        help="Path to the processed CSV file (current structure).")
    parser.add_argument("--output_csv", type=str, default="data/train_enriched.csv",
                        help="Path to save the enriched dataset.")
    parser.add_argument("--output_dir", type=str, default="feature_analysis_output",
                        help="Directory to save any analysis plots/summary.")
    return parser.parse_args()

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset, ensuring Timestamp is parsed as datetime if present.
    """
    df = pd.read_csv(filepath, parse_dates=["Timestamp"], low_memory=False)
    return df


def enrich_role_endpoint(df: pd.DataFrame) -> pd.DataFrame:
    """
    If 'Role_Endpoint' is present, we can create some derived features:
    - For instance, one-hot-encoding the numeric 'Role_Endpoint' if it represents categories.
      (But in your dataset, 'Role_Endpoint' might be numeric with many values.)
    - Or splitting it into Role + Endpoint if it was originally a combined field (but it appears numeric in some rows).
    
    We'll demonstrate a simple frequency encoding approach:
      freq_role_endpoint: how often each unique Role_Endpoint value appears in the dataset.
    """
    if "Role_Endpoint" not in df.columns:
        print("[INFO] 'Role_Endpoint' not found; skipping role-endpoint enrichment.")
        return df
    
    # Let's create frequency encoding for Role_Endpoint
    freq_map = df["Role_Endpoint"].value_counts().to_dict()
    df["freq_role_endpoint"] = df["Role_Endpoint"].map(freq_map)
    
    print("[INFO] role_endpoint frequency feature added.")
    return df


def enrich_time_window_global(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a rolling count of logs in the last 1 hour, but globally (not per user), 
    since there's no UserID. This might be less relevant for insider threat, 
    but demonstrates the approach if you want to see sudden spikes in overall log volume.
    
    WARNING: This can be expensive for 100k rows, but we'll do a simplified approach.
    """
    if "Timestamp" not in df.columns:
        print("[INFO] 'Timestamp' not found; skipping global time-window feature.")
        return df
    
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Timestamp_unix"] = df["Timestamp"].astype(np.int64) // 10**9  # seconds
    
    # We'll do a rolling 1-hour window, but globally
    times = df["Timestamp_unix"].values
    rolling_counts = []
    start_idx = 0
    for i in range(len(df)):
        while times[i] - times[start_idx] > 3600:
            start_idx += 1
        rolling_counts.append(i - start_idx + 1)
    
    df["logs_last_1h_global"] = rolling_counts
    df.drop(columns=["Timestamp_unix"], inplace=True)
    print("[INFO] global 1-hour rolling count feature added.")
    return df


def mini_eda_new_features(df: pd.DataFrame, output_dir: str):
    """
    Quick EDA for newly added columns: freq_role_endpoint, logs_last_1h_global, etc.
    Saves histograms & correlation with 'Anomalous'.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify newly added columns
    new_cols = []
    if "freq_role_endpoint" in df.columns:
        new_cols.append("freq_role_endpoint")
    if "logs_last_1h_global" in df.columns:
        new_cols.append("logs_last_1h_global")
    
    if not new_cols:
        print("[INFO] No new features found for EDA.")
        return
    
    # If 'Anomalous' is present, we can do color-coded histograms
    if "Anomalous" in df.columns:
        for col in new_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                plt.figure(figsize=(6,4))
                sns.histplot(df, x=col, bins=30, kde=True, hue="Anomalous", multiple="stack")
                plt.title(f"Distribution of {col} by Anomalous")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"hist_{col}.png"))
                plt.close()
        # Compute correlation with Anomalous
        corrs = {}
        for col in new_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                corrs[col] = df[[col, "Anomalous"]].corr().iloc[0,1]
        sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        with open(os.path.join(output_dir, "new_features_correlation.txt"), "w") as f:
            for col, val in sorted_corrs:
                f.write(f"{col}: {val:.4f}\n")
    else:
        # If 'Anomalous' not found, just do a single distribution
        for col in new_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                plt.figure(figsize=(6,4))
                sns.histplot(df[col], bins=30, kde=True)
                plt.title(f"Distribution of {col}")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"hist_{col}.png"))
                plt.close()

    print("[INFO] Mini-EDA for new features completed. Check your output dir.")


def main():
    args = parse_arguments()

    # 1. Load Data
    df = load_data(args.input_csv)
    print(f"[INFO] Loaded dataset with shape: {df.shape}")

    # 2. Enrich role-endpoint
    df = enrich_role_endpoint(df)

    # 3. Enrich time-window globally (optional)
    df = enrich_time_window_global(df)

    # 4. Mini EDA on new features
    mini_eda_new_features(df, args.output_dir)

    # 5. Save final enriched dataset
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[INFO] Enriched dataset saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
