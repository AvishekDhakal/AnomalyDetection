# #!/usr/bin/env python3

# """
# final_enrich_features.py

# This script builds on your prior enrichments:
# 1. freq_role_endpoint (frequency encoding of Role_Endpoint)
# 2. logs_last_1h_global (global rolling 1-hour log volume)

# and ADDS:
# - Is_MidNight (logs from 0..3 AM)
# - Is_External_After_Hours (1 if Is_Internal_IP=0 & Is_After_Hours=1)
# - Unusual_Global_Spike (1 if logs_last_1h_global > p90)
# - Is_Weekend_Admin (1 if Role_Admin=True AND DayOfWeek in [5,6])

# Author: [Your Name]
# Date: YYYY-MM-DD
# """

# import os
# import argparse
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import logging

# # -------------------- CONFIGURATION --------------------

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # -------------------- FUNCTIONS --------------------

# def load_data(filepath: str) -> pd.DataFrame:
#     """
#     Loads the dataset, parse Timestamp if present.
#     """
#     try:
#         df = pd.read_csv(filepath, parse_dates=["Timestamp"], low_memory=False)
#         logging.info(f"Successfully loaded data from {filepath} with shape {df.shape}.")
#         return df
#     except Exception as e:
#         logging.error(f"Error loading data from {filepath}: {e}")
#         raise

# def enrich_freq_role_endpoint(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Frequency encoding for Role_Endpoint.
#     """
#     if "Role_Endpoint" not in df.columns:
#         logging.warning("'Role_Endpoint' not found, skipping freq_role_endpoint.")
#         return df
#     freq_map = df["Role_Endpoint"].value_counts().to_dict()
#     df["freq_role_endpoint"] = df["Role_Endpoint"].map(freq_map)
#     logging.info("Added 'freq_role_endpoint'.")
#     return df

# def enrich_logs_last_1h_global(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Creates logs_last_1h_global: a rolling 1-hour log volume (globally).
#     """
#     if "Timestamp" not in df.columns:
#         logging.warning("'Timestamp' not found; skipping logs_last_1h_global.")
#         df["logs_last_1h_global"] = 0
#         return df
    
#     df = df.sort_values("Timestamp").reset_index(drop=True)
    
#     # Set Timestamp as index for rolling
#     df.set_index("Timestamp", inplace=True)
#     df["logs_last_1h_global"] = df["LogID"].rolling('1H').count().astype(int)
#     df.reset_index(inplace=True)
    
#     logging.info("Added 'logs_last_1h_global'.")
#     return df

# def add_time_network_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Adds:
#     - Is_MidNight (1 if Hour in [0..3])
#     - Is_External_After_Hours (1 if Is_Internal_IP=0 & Is_After_Hours=1)
#     - Unusual_Global_Spike (1 if logs_last_1h_global > p90)
#     - Is_Weekend_Admin (1 if Role_Admin=True & DayOfWeek in [5,6])
#     """
#     # 1) Is_MidNight
#     if "Hour" in df.columns:
#         df["Is_MidNight"] = df["Hour"].apply(lambda x: 1 if x in [0,1,2,3] else 0)
#         logging.info("Added 'Is_MidNight'.")
#     else:
#         df["Is_MidNight"] = 0
#         logging.warning("'Hour' column not found, defaulting 'Is_MidNight' to 0.")
    
#     # 2) Is_External_After_Hours
#     if "Is_Internal_IP" in df.columns and "Is_After_Hours" in df.columns:
#         df["Is_External_After_Hours"] = df.apply(
#             lambda row: 1 if (row["Is_Internal_IP"] == 0 and row["Is_After_Hours"] == 1) else 0,
#             axis=1
#         )
#         logging.info("Added 'Is_External_After_Hours'.")
#     else:
#         df["Is_External_After_Hours"] = 0
#         logging.warning("'Is_Internal_IP' or 'Is_After_Hours' column missing, defaulting 'Is_External_After_Hours' to 0.")
    
#     # 3) Unusual_Global_Spike
#     if "logs_last_1h_global" in df.columns:
#         p90 = df["logs_last_1h_global"].quantile(0.90)
#         df["Unusual_Global_Spike"] = df["logs_last_1h_global"].apply(lambda x: 1 if x > p90 else 0)
#         logging.info("Added 'Unusual_Global_Spike'.")
#     else:
#         df["Unusual_Global_Spike"] = 0
#         logging.warning("'logs_last_1h_global' column missing, defaulting 'Unusual_Global_Spike' to 0.")
    
#     # 4) Is_Weekend_Admin
#     if "DayOfWeek" in df.columns and "Role_Admin" in df.columns:
#         df["Is_Weekend_Admin"] = df.apply(
#             lambda row: 1 if (row["Role_Admin"] == True and row["DayOfWeek"] in [5,6]) else 0,
#             axis=1
#         )
#         logging.info("Added 'Is_Weekend_Admin'.")
#     else:
#         df["Is_Weekend_Admin"] = 0
#         logging.warning("'DayOfWeek' or 'Role_Admin' column missing, defaulting 'Is_Weekend_Admin' to 0.")
    
#     return df

# def mini_eda_new_features(df: pd.DataFrame, output_dir: str):
#     """
#     Quick EDA for newly added columns, skipping KDE if the data is low-variance or binary.
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     new_cols = [
#         "freq_role_endpoint",
#         "logs_last_1h_global",
#         "Is_MidNight",
#         "Is_External_After_Hours",
#         "Unusual_Global_Spike",
#         "Is_Weekend_Admin"
#     ]
#     # Filter only columns that actually exist
#     new_cols = [c for c in new_cols if c in df.columns]

#     if not new_cols:
#         logging.info("No new features found for EDA.")
#         return
    
#     has_label = "Anomalous" in df.columns
    
#     for col in new_cols:
#         # Skip if it's not numeric
#         if not pd.api.types.is_numeric_dtype(df[col]):
#             continue
        
#         # Drop NaNs, check how many unique values we have
#         col_data = df[col].dropna()
#         unique_vals = col_data.unique()
#         n_unique = len(unique_vals)

#         # If only 1 unique value, skip entirely
#         if n_unique <= 1:
#             logging.warning(f"Column {col} has only one unique value. Skipping plot.")
#             continue

#         # Decide whether to use KDE
#         # Disable KDE for binary or near-binary data (fewer than 5 unique values).
#         use_kde = True
#         if n_unique < 5:
#             use_kde = False

#         plt.figure(figsize=(6,4))
        
#         # Hue by 'Anomalous' if available
#         if has_label:
#             if use_kde:
#                 sns.histplot(data=df, x=col, bins=30, kde=True, hue="Anomalous", multiple="layer")
#             else:
#                 sns.histplot(data=df, x=col, bins=30, kde=False, hue="Anomalous", multiple="stack")
#             plt.title(f"Distribution of {col} by Anomalous")
#         else:
#             # No label, just do a single hist
#             sns.histplot(data=df, x=col, bins=30, kde=use_kde)
#             plt.title(f"Distribution of {col}")
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f"hist_{col}.png"))
#         plt.close()

#     # Correlation with 'Anomalous' if it exists
#     if has_label:
#         corrs = {}
#         for col in new_cols:
#             if pd.api.types.is_numeric_dtype(df[col]):
#                 # Ensure there's >1 unique value
#                 if df[col].nunique() > 1:
#                     corrs[col] = df[[col, "Anomalous"]].corr().iloc[0,1]
#         sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
#         with open(os.path.join(output_dir, "new_features_correlation.txt"), "w") as f:
#             for c, val in sorted_corrs:
#                 f.write(f"{c}: {val:.4f}\n")
    
#     logging.info("Mini-EDA on final new features completed (with safe KDE checks).")

# def main():
#     parser = argparse.ArgumentParser(description="Final Feature Engineering for a 24/7 Hospital setting with Admin weekend checks.")
#     parser.add_argument("--input_csv", type=str, required=True,
#                         help="Path to the initial processed CSV file.")
#     parser.add_argument("--output_csv", type=str, required=True,
#                         help="Path to save the final enriched dataset.")
#     parser.add_argument("--output_dir", type=str, default="feature_analysis_output_final",
#                         help="Directory to save any analysis plots/summary.")
#     return parser.parse_args()

# if __name__ == "__main__":
#     args = main()
    
#     # 1. Load dataset
#     df = load_data(args.input_csv)
#     logging.info(f"Loaded dataset with shape: {df.shape}")
    
#     # 2. Frequency encoding for Role_Endpoint
#     df = enrich_freq_role_endpoint(df)
    
#     # 3. Rolling 1-hour log volume
#     df = enrich_logs_last_1h_global(df)
    
#     # 4. Add time and network features
#     df = add_time_network_features(df)
    
#     # 5. EDA for new features
#     mini_eda_new_features(df, args.output_dir)
    
#     # 6. Save the final enriched dataset
#     os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
#     df.to_csv(args.output_csv, index=False)
#     logging.info(f"Final enriched dataset saved to: {args.output_csv}")



#!/usr/bin/env python3

"""
enrich_final_feature.py

This script enhances the preprocessed insider threat detection logs by adding the following features:
1. Frequency encoding for 'Role_Endpoint' as 'freq_role_endpoint'.
2. Rolling 1-hour global log volume as 'logs_last_1h_global'.
3. Time and network-based features:
   - 'Is_MidNight': Indicates if the log occurred between 0-3 AM.
   - 'Is_External_After_Hours': Indicates external access after hours.
   - 'Is_Weekend_Admin': Indicates admin activities during weekends.
4. 'Unusual_Global_Spike': Flags logs that exceed the 90th percentile of global log volume.

Supports 'train' and 'inference' modes to handle labeling and feature alignment.

Author: [Your Name]
Date: YYYY-MM-DD
"""

import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import joblib

# -------------------- CONFIGURATION --------------------

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default paths for saving enrichment parameters
FREQ_ROLE_ENDPOINT_MAP_DEFAULT = "enrichment_params/freq_role_endpoint_map.pkl"
UNUSUAL_GLOBAL_SPIKE_THRESHOLD_DEFAULT = "enrichment_params/unusual_global_spike_threshold.pkl"
FEATURE_NAMES_DEFAULT = "models/feature_names.pkl"

# -------------------- FUNCTIONS --------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset, parses Timestamp if present.
    """
    try:
        df = pd.read_csv(filepath, parse_dates=["Timestamp"], low_memory=False)
        logging.info(f"Successfully loaded data from {filepath} with shape {df.shape}.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise

def enrich_freq_role_endpoint(df: pd.DataFrame, mode: str = "train", freq_map_path: str = FREQ_ROLE_ENDPOINT_MAP_DEFAULT) -> pd.DataFrame:
    """
    Frequency encoding for Role_Endpoint.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        mode (str): 'train' or 'inference'
        freq_map_path (str): Path to save/load frequency map

    Returns:
        pd.DataFrame: DataFrame with 'freq_role_endpoint' added
    """
    if "Role_Endpoint" not in df.columns:
        logging.warning("'Role_Endpoint' not found, skipping freq_role_endpoint.")
        df["freq_role_endpoint"] = 0
        return df

    if mode == "train":
        os.makedirs(os.path.dirname(freq_map_path), exist_ok=True)
        freq_map = df["Role_Endpoint"].value_counts().to_dict()
        joblib.dump(freq_map, freq_map_path)
        logging.info(f"Saved 'freq_role_endpoint' frequency map to '{freq_map_path}'.")
    elif mode == "inference":
        if not os.path.exists(freq_map_path):
            logging.error(f"Frequency map file not found at '{freq_map_path}'. Cannot proceed.")
            raise FileNotFoundError(f"Frequency map file not found at '{freq_map_path}'.")
        freq_map = joblib.load(freq_map_path)
        logging.info(f"Loaded 'freq_role_endpoint' frequency map from '{freq_map_path}'.")

    df["freq_role_endpoint"] = df["Role_Endpoint"].map(freq_map).fillna(0).astype(int)
    logging.info("Added 'freq_role_endpoint'.")
    return df

def enrich_logs_last_1h_global(df: pd.DataFrame, mode: str = "train") -> pd.DataFrame:
    """
    Creates logs_last_1h_global: a rolling 1-hour log volume (globally).

    Parameters:
        df (pd.DataFrame): Input DataFrame
        mode (str): 'train' or 'inference'

    Returns:
        pd.DataFrame: DataFrame with 'logs_last_1h_global' added
    """
    if "Timestamp" not in df.columns:
        logging.warning("'Timestamp' not found; skipping logs_last_1h_global.")
        df["logs_last_1h_global"] = 0
        return df

    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Set Timestamp as index for rolling
    df.set_index("Timestamp", inplace=True)
    # Updated 'h' to replace deprecated 'H'
    df["logs_last_1h_global"] = df["LogID"].rolling('1h').count().astype(int)
    df.reset_index(inplace=True)

    logging.info("Added 'logs_last_1h_global'.")
    return df

def add_time_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - Is_MidNight (1 if Hour in [0..3])
    - Is_External_After_Hours (1 if Is_Internal_IP=0 & Is_After_Hours=1)
    - Is_Weekend_Admin (1 if Role_Admin=True AND DayOfWeek in [5,6])

    Note: 'Unusual_Global_Spike' is handled separately.
    """
    # 1) Is_MidNight
    if "Hour" in df.columns:
        df["Is_MidNight"] = df["Hour"].apply(lambda x: 1 if x in [0,1,2,3] else 0)
        logging.info("Added 'Is_MidNight'.")
    else:
        df["Is_MidNight"] = 0
        logging.warning("'Hour' column not found, defaulting 'Is_MidNight' to 0.")

    # 2) Is_External_After_Hours
    if "Is_Internal_IP" in df.columns and "Is_After_Hours" in df.columns:
        df["Is_External_After_Hours"] = df.apply(
            lambda row: 1 if (row["Is_Internal_IP"] == 0 and row["Is_After_Hours"] == 1) else 0,
            axis=1
        )
        logging.info("Added 'Is_External_After_Hours'.")
    else:
        df["Is_External_After_Hours"] = 0
        logging.warning("'Is_Internal_IP' or 'Is_After_Hours' column missing, defaulting 'Is_External_After_Hours' to 0.")

    # 3) Is_Weekend_Admin
    if "DayOfWeek" in df.columns and "Role_Admin" in df.columns:
        df["Is_Weekend_Admin"] = df.apply(
            lambda row: 1 if (row["Role_Admin"] == 1 and row["DayOfWeek"] in [5,6]) else 0,
            axis=1
        )
        logging.info("Added 'Is_Weekend_Admin'.")
    else:
        df["Is_Weekend_Admin"] = 0
        logging.warning("'DayOfWeek' or 'Role_Admin' column missing, defaulting 'Is_Weekend_Admin' to 0.")

    return df



def add_unusual_global_spike(df: pd.DataFrame, mode: str = "train", threshold_path: str = UNUSUAL_GLOBAL_SPIKE_THRESHOLD_DEFAULT) -> pd.DataFrame:
    """
    Adds 'Unusual_Global_Spike' feature based on a threshold.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        mode (str): 'train' or 'inference'
        threshold_path (str): Path to save/load the threshold

    Returns:
        pd.DataFrame: DataFrame with 'Unusual_Global_Spike' added
    """
    if "logs_last_1h_global" in df.columns:
        if mode == "train":
            os.makedirs(os.path.dirname(threshold_path), exist_ok=True)
            p90 = df["logs_last_1h_global"].quantile(0.90)
            joblib.dump(p90, threshold_path)
            logging.info(f"Saved 'Unusual_Global_Spike' threshold (p90={p90}) to '{threshold_path}'.")
        elif mode == "inference":
            if not os.path.exists(threshold_path):
                logging.error(f"'Unusual_Global_Spike' threshold file not found at '{threshold_path}'. Cannot proceed.")
                raise FileNotFoundError(f"'Unusual_Global_Spike' threshold file not found at '{threshold_path}'.")
            p90 = joblib.load(threshold_path)
            logging.info(f"Loaded 'Unusual_Global_Spike' threshold (p90={p90}) from '{threshold_path}'.")

        df["Unusual_Global_Spike"] = df["logs_last_1h_global"].apply(lambda x: 1 if x > p90 else 0)
        logging.info("Added 'Unusual_Global_Spike'.")
    else:
        df["Unusual_Global_Spike"] = 0
        logging.warning("'logs_last_1h_global' column missing, defaulting 'Unusual_Global_Spike' to 0.")

    return df

def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and removes duplicate columns in the DataFrame.
    Keeps the first occurrence and removes subsequent duplicates.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with duplicate columns removed
    """
    duplicated_columns = df.columns[df.columns.duplicated()]
    if len(duplicated_columns) > 0:
        logging.warning(f"Duplicate columns found: {duplicated_columns.tolist()}. Removing duplicates.")
        df = df.loc[:, ~df.columns.duplicated()]
    else:
        logging.info("No duplicate columns found.")
    return df

# def align_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
#     """
#     Aligns the DataFrame features with the expected feature names.
#     Adds missing features with default values and ensures correct column order.

#     Parameters:
#         df (pd.DataFrame): The DataFrame to align
#         feature_names (list): List of expected feature names

#     Returns:
#         pd.DataFrame: Aligned DataFrame
#     """
#     missing_features = set(feature_names) - set(df.columns)
#     for feature in missing_features:
#         df[feature] = 0  # Default value
#         logging.warning(f"Missing feature '{feature}' in enriched data. Adding with default value 0.")

#     # Ensure the order matches
#     df = df[feature_names]
#     logging.info("Aligned enriched data features with training features.")
#     return df

def align_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Aligns the DataFrame features with the expected feature names.
    Ensures 'LogID' and 'Timestamp' are included, adds missing features with default values, and aligns column order.

    Parameters:
        df (pd.DataFrame): The DataFrame to align
        feature_names (list): List of expected feature names

    Returns:
        pd.DataFrame: Aligned DataFrame
    """
    # Add missing features with default values
    missing_features = set(feature_names) - set(df.columns)
    for feature in missing_features:
        df[feature] = 0  # Default value
        logging.warning(f"Missing feature '{feature}' in enriched data. Adding with default value 0.")

    # Ensure the order matches the feature names
    aligned_columns = [col for col in feature_names if col in df.columns]
    df = df[aligned_columns]
    logging.info("Aligned enriched data features with training features.")
    return df

def mini_eda_new_features(df: pd.DataFrame, output_dir: str, mode: str = "train"):
    """
    Quick EDA for newly added columns, skipping KDE if the data is low-variance or binary.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        output_dir (str): Directory to save analysis plots and summaries
        mode (str): 'train' or 'inference'
    """
    os.makedirs(output_dir, exist_ok=True)

    new_cols = [
        "freq_role_endpoint",
        "logs_last_1h_global",
        "Is_MidNight",
        "Is_External_After_Hours",
        "Unusual_Global_Spike",
        "Is_Weekend_Admin"
    ]
    # Filter only columns that actually exist
    new_cols = [c for c in new_cols if c in df.columns]

    if not new_cols:
        logging.info("No new features found for EDA.")
        return

    has_label = "Anomalous" in df.columns and mode == "train"

    for col in new_cols:
        # Skip if it's not numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Drop NaNs, check how many unique values we have
        col_data = df[col].dropna()
        unique_vals = col_data.unique()
        n_unique = len(unique_vals)

        # If only 1 unique value, skip entirely
        if n_unique <= 1:
            logging.warning(f"Column {col} has only one unique value. Skipping plot.")
            continue

        # Decide whether to use KDE
        # Disable KDE for binary or near-binary data (fewer than 5 unique values).
        use_kde = True
        if n_unique < 5:
            use_kde = False

        plt.figure(figsize=(6,4))

        # Hue by 'Anomalous' if available
        if has_label:
            if use_kde:
                sns.histplot(data=df, x=col, bins=30, kde=True, hue="Anomalous", multiple="layer")
            else:
                sns.histplot(data=df, x=col, bins=30, kde=False, hue="Anomalous", multiple="stack")
            plt.title(f"Distribution of {col} by Anomalous")
        else:
            # No label, just do a single hist
            sns.histplot(data=df, x=col, bins=30, kde=use_kde)
            plt.title(f"Distribution of {col}")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"hist_{col}.png"))
        plt.close()

    # Correlation with 'Anomalous' if it exists and in train mode
    if has_label:
        corrs = {}
        for col in new_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Ensure there's >1 unique value
                if df[col].nunique() > 1:
                    corrs[col] = df[[col, "Anomalous"]].corr().iloc[0,1]
        sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        with open(os.path.join(output_dir, "new_features_correlation.txt"), "w") as f:
            for c, val in sorted_corrs:
                f.write(f"{c}: {val:.4f}\n")
        logging.info("Computed correlations between new features and 'Anomalous'.")

    logging.info("Mini-EDA on final new features completed (with safe KDE checks).")

# def finalize_dataset(df: pd.DataFrame, output_path: str, mode: str = "train", feature_names_path: str = FEATURE_NAMES_DEFAULT) -> None:
#     """
#     Saves the final enriched dataset to CSV, preserving all original columns and adding new features.

#     Ensures that the 'Anomalous' column appears only once.

#     Parameters:
#         df (pd.DataFrame): The DataFrame post-enrichment
#         output_path (str): File path to save the CSV
#         mode (str): 'train' or 'inference'
#         feature_names_path (str): Path to save/load feature names for consistency
#     """
#     # First, remove any duplicate columns
#     df = remove_duplicate_columns(df)

#     # Handle 'Anomalous' column
#     if mode == "train":
#         if 'Anomalous' in df.columns:
#             # If 'Anomalous' exists, ensure it's only once
#             anomalous_cols = df.columns[df.columns == 'Anomalous']
#             if len(anomalous_cols) > 1:
#                 logging.warning(f"Multiple 'Anomalous' columns found. Keeping the first and removing duplicates.")
#                 # Keep the first 'Anomalous' column and drop the rest
#                 first_anomalous = anomalous_cols[0]
#                 df = df.drop(columns=anomalous_cols[1:])
#         else:
#             # If 'Anomalous' doesn't exist, add it with default value
#             logging.warning("'Anomalous' column not found in train mode. Adding 'Anomalous' with default value 0.")
#             df['Anomalous'] = 0
#     elif mode == "inference":
#         # Ensure 'Anomalous' is not present in inference mode
#         if 'Anomalous' in df.columns:
#             logging.warning("'Anomalous' column found in inference mode. Removing it to prevent data leakage.")
#             df = df.drop(columns=['Anomalous'])

#     # Save feature names for consistency in training
#     if mode == "train":
#         # Exclude non-feature columns (e.g., 'LogID', 'Timestamp', 'Anomalous')
#         non_feature_cols = ['LogID', 'Timestamp', 'Anomalous']
#         feature_names = [col for col in df.columns if col not in non_feature_cols]
#         os.makedirs(os.path.dirname(feature_names_path), exist_ok=True)
#         joblib.dump(feature_names, feature_names_path)
#         logging.info(f"Saved feature names to '{feature_names_path}'.")
#     elif mode == "inference":
#         # Load feature names and align the DataFrame accordingly
#         if not os.path.exists(feature_names_path):
#             logging.error(f"Feature names file not found at '{feature_names_path}'. Cannot align features.")
#             raise FileNotFoundError(f"Feature names file not found at '{feature_names_path}'.")
#         feature_names = joblib.load(feature_names_path)
#         df = align_features(df, feature_names)

#     # Ensure that all features are present
#     # No need to drop any columns; keep all original plus new features
#     # Just ensure 'Anomalous' is handled as above

#     # Save the final enriched dataset
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     df.to_csv(output_path, index=False)
#     logging.info(f"Final enriched dataset saved to: {output_path}")

def finalize_dataset(df: pd.DataFrame, output_path: str, mode: str = "train", feature_names_path: str = FEATURE_NAMES_DEFAULT) -> None:
    """
    Saves the final enriched dataset to CSV, preserving all original columns and adding new features.
    Ensures that 'LogID' and 'Timestamp' are always included in the final output.
    """
    # First, remove any duplicate columns
    df = remove_duplicate_columns(df)

    # Handle 'Anomalous' column
    if mode == "train":
        if 'Anomalous' in df.columns:
            # Ensure 'Anomalous' column appears only once
            anomalous_cols = df.columns[df.columns == 'Anomalous']
            if len(anomalous_cols) > 1:
                logging.warning(f"Multiple 'Anomalous' columns found. Keeping the first and removing duplicates.")
                df = df.drop(columns=anomalous_cols[1:])
        else:
            # Add 'Anomalous' column if missing
            logging.warning("'Anomalous' column not found in train mode. Adding 'Anomalous' with default value 0.")
            df['Anomalous'] = 0
    elif mode == "inference":
        # Ensure 'Anomalous' column is removed in inference mode
        if 'Anomalous' in df.columns:
            logging.warning("'Anomalous' column found in inference mode. Removing it to prevent data leakage.")
            df = df.drop(columns=['Anomalous'])

    if mode == "train":
        # Save feature names for consistency, excluding non-feature columns
        non_feature_cols = ['LogID', 'Timestamp', 'Anomalous']
        feature_names = [col for col in df.columns if col not in non_feature_cols]
        os.makedirs(os.path.dirname(feature_names_path), exist_ok=True)
        joblib.dump(feature_names, feature_names_path)
        logging.info(f"Saved feature names to '{feature_names_path}'.")
    elif mode == "inference":
        # Load feature names and align features
        if not os.path.exists(feature_names_path):
            logging.error(f"Feature names file not found at '{feature_names_path}'. Cannot align features.")
            raise FileNotFoundError(f"Feature names file not found at '{feature_names_path}'.")
        feature_names = joblib.load(feature_names_path)
        # Ensure 'LogID' and 'Timestamp' are included in the final output
        feature_names = ['LogID', 'Timestamp'] + feature_names
        df = align_features(df, feature_names)

    # Save the final enriched dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Final enriched dataset saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Final Feature Engineering for Insider Threat Detection in Healthcare Logs.")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the initial processed CSV file.")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to save the final enriched dataset.")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train",
                        help="Mode of operation: 'train' to include 'Anomalous' column and save parameters, 'inference' to load and apply parameters.")
    parser.add_argument("--output_dir", type=str, default="feature_analysis_output_final",
                        help="Directory to save any analysis plots/summary.")
    parser.add_argument("--freq_map_path", type=str, default=FREQ_ROLE_ENDPOINT_MAP_DEFAULT,
                        help="Path to save/load frequency map for 'Role_Endpoint'.")
    parser.add_argument("--threshold_path", type=str, default=UNUSUAL_GLOBAL_SPIKE_THRESHOLD_DEFAULT,
                        help="Path to save/load threshold for 'Unusual_Global_Spike'.")
    parser.add_argument("--feature_names_path", type=str, default=FEATURE_NAMES_DEFAULT,
                        help="Path to save/load feature names for consistency.")
    args = parser.parse_args()

    # 1. Load dataset
    logging.info("[DEBUG] Loading dataset...")
    df = load_data(args.input_csv)
    logging.info(f"Loaded dataset with shape: {df.shape}")

    # 2. Remove duplicate columns
    df = remove_duplicate_columns(df)

    # 3. Frequency Encoding for Role_Endpoint
    logging.info("[DEBUG] Performing frequency encoding for 'Role_Endpoint'...")
    df = enrich_freq_role_endpoint(df, mode=args.mode, freq_map_path=args.freq_map_path)

    # 4. Rolling 1-hour log volume
    logging.info("[DEBUG] Computing 'logs_last_1h_global'...")
    df = enrich_logs_last_1h_global(df, mode=args.mode)

    # 5. Add additional time and network features
    logging.info("[DEBUG] Adding additional time and network features...")
    df = add_time_network_features(df)

    # 6. Add Unusual_Global_Spike
    logging.info("[DEBUG] Adding 'Unusual_Global_Spike' feature...")
    df = add_unusual_global_spike(df, mode=args.mode, threshold_path=args.threshold_path)

    # 7. EDA for new features (only in training mode)
    if args.mode == "train":
        logging.info("[DEBUG] Performing EDA on new features...")
        mini_eda_new_features(df, args.output_dir, mode=args.mode)

    # 8. Finalize and save the enriched dataset
    logging.info("[DEBUG] Finalizing and saving the enriched dataset...")
    finalize_dataset(df, args.output_csv, mode=args.mode, feature_names_path=args.feature_names_path)
    logging.info("Feature enrichment complete.")

if __name__ == "__main__":
    main()


#  python3 enrich_final_feature.py --input_csv data/preprocessed_train.csv --output_csv data/train_enriched.csv
#  python3 enrich_final_feature.py --input_csv data/preprocessed_test.csv --output_csv data/test_enriched.csv --mode inference 