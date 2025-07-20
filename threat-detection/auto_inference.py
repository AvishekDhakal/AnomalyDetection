#!/usr/bin/env python3
"""
auto_inference.py

This script automates:
1. Generating exactly 200 logs with a small chance of anomalies (once in ~20 runs).
2. Appending the new logs to a global master log archive.
3. Preprocessing (inference mode) the newly generated logs only.
4. Enhanced feature engineering (inference mode).
5. Loading a pre-trained Random Forest (from `final_models`) and predicting anomalies.
6. Saving the final predictions in a timestamped CSV for analytics.

Usage:
  python3 auto_inference.py

No command-line arguments are strictly required, but you could add your own to control:
 - The chance of anomalies
 - The location of the final model, etc.

Author: You
"""

import subprocess
import random
import datetime
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import joblib


# Paths and constants
LOG_GENERATOR_CMD = [
    "python3", "main.py",
    "--total_logs", "200",
    "--master_file", "data/master_logs.csv",
    "--inference_file", "data/inference_logs.csv",
    "--overwrite"
]
ARCHIVE_MASTER_FILE = "data/master_logs_archive.csv"  # big file where we store all past logs
PROCESSED_PICKLE = "data/processed_data.pkl"
PROCESSED_CSV = "data/processed_data.csv"
PREPROCESSED_INFERENCE = "data/preprocessed_inference.csv"
MODEL_FEATURES_PKL = "final_models/features.pkl"
MODEL_PATH = "final_models/random_forest.pkl"
SCALER_PATH = "final_models/Original/random_forest/scaler.pkl"
RESULTS_DIR = "data"  # Where final inference results go

def pick_anomaly_ratio(run_count=1):
    """
    Decide an anomaly ratio. 
    - In 1 out of 20 runs, use 0.01 or 0.02.
    - Otherwise, use 0.0 (no anomalies).
    """
    # If run_count % 20 == 0: or use random approach:
    # E.g. 1 in 20 chance:
    if random.randint(1, 20) == 1:
        return random.choice([0.01, 0.02])
    else:
        return 0.0

def generate_logs(anomaly_ratio):
    """
    Calls main.py to generate logs with the given anomaly_ratio,
    overwriting data/master_logs.csv and data/inference_logs.csv.
    """
    cmd = LOG_GENERATOR_CMD.copy()
    # Insert or update anomaly ratio after "--total_logs" param
    # If your main.py supports `--anomaly_ratio`, set it here:
    cmd.insert(4, "--anomaly_ratio")
    cmd.insert(5, str(anomaly_ratio))


    print("Generating 200 logs with anomaly_ratio =", anomaly_ratio)
    subprocess.run(cmd, check=True)

def append_to_master_archive():
    """
    Appends newly generated data/master_logs.csv to a global archive CSV
    so we keep a big history. Ignores the header after the first time.
    """
    if not os.path.exists(ARCHIVE_MASTER_FILE):
        # Just copy the current master_logs.csv as the initial archive
        subprocess.run(["cp", "data/master_logs.csv", ARCHIVE_MASTER_FILE], check=True)
        print(f"Created new archive {ARCHIVE_MASTER_FILE}")
    else:
        # Append (skip header)
        with open(ARCHIVE_MASTER_FILE, "a") as f_archive, open("data/master_logs.csv", "r") as f_new:
            lines = f_new.readlines()
            # Skip header (first line)
            f_archive.writelines(lines[1:])
        print("Appended newly generated logs to", ARCHIVE_MASTER_FILE)

def run_data_preprocessing():
    """
    Runs data_preprocessing.py in inference mode on data/inference_logs.csv,
    producing data/processed_data.csv and data/processed_data.pkl
    """
    cmd = [
        "python3", "data_preprocessing.py",
        "--mode", "inference",
        "--config", "config.json",
        "--input", "data/inference_logs.csv",
        "--output", PROCESSED_PICKLE
    ]
    print("Running data_preprocessing in inference mode...")
    subprocess.run(cmd, check=True)

def run_enhanced_features():
    """
    Runs enhanced_features.py in inference mode on data/processed_data.csv,
    producing data/preprocessed_inference.csv
    """
    cmd = [
        "python3", "enhanced_features.py",
        "--mode", "inference",
        "--input", PROCESSED_CSV,  # data_preprocessing creates a .csv with same base name as the pkl
        "--output", PREPROCESSED_INFERENCE
    ]
    print("Running enhanced_features in inference mode...")
    subprocess.run(cmd, check=True)

def load_model_and_predict():
    """
    Loads the random_forest model and the feature config,
    then runs predictions on data/preprocessed_inference.csv
    Output is saved to a timestamped file in data/inference_results_YYYYMMDDHHMM.csv
    with the original columns + predicted anomaly.
    """
    # 1. Load final feature mapping (feature_columns)
    with open(MODEL_FEATURES_PKL, "rb") as f:
        feature_info = pickle.load(f)
    feature_cols = feature_info["feature_columns"]

    # 2. Load the inference CSV
    df_infer_full = pd.read_csv(PREPROCESSED_INFERENCE)
    
    # We'll keep these metadata columns to re-attach after predicting
    # (LogID, Timestamp, maybe Role, Endpoint, etc.)
    # In your pipeline, you only used certain columns in the model, but we want
    # to keep everything for analytics. So we'll do:
    # Note: ensure the columns exist
    meta_cols = []
    for c in ["LogID", "Timestamp", "Role", "Endpoint"]:
        if c in df_infer_full.columns:
            meta_cols.append(c)

    # 3. Extract the subset for model features
    X_infer = df_infer_full[feature_cols].copy()

    # 4. Load the Random Forest model + scaler
    # rf_model = pickle.load(open(MODEL_PATH, "rb"))
    rf_model = joblib.load(MODEL_PATH)
    # scaler = pickle.load(open(SCALER_PATH, "rb"))
    scaler = joblib.load(SCALER_PATH)

    # The train.py script standard-scaled these numeric columns:
    numeric_cols = ["req_freq_5min", "inventory_change_rate", "role_risk"]
    # We do the same transformation:
    X_infer[numeric_cols] = scaler.transform(X_infer[numeric_cols])

    # 5. Predict
    y_pred = rf_model.predict(X_infer)
    # Probability of being anomaly (class=1)
    if hasattr(rf_model, "predict_proba"):
        y_proba = rf_model.predict_proba(X_infer)[:, 1]
    else:
        # for e.g. SVM decision_function or logistic decision_function
        y_proba = rf_model.decision_function(X_infer)

    # 6. Combine results
    df_infer_full["predicted_anomaly"] = y_pred
    df_infer_full["anomaly_score"] = y_proba
    override_mask = (
        df_infer_full["Role"].isin(["Doctor","Nurse"]) &
        df_infer_full["Endpoint"].isin([
            "/patient/appointments/confirm",
            "/patient/appointments/cancel"
        ])
    )

    # Force predicted_anomaly=0 for those logs
    df_infer_full.loc[override_mask, "predicted_anomaly"] = 0
    # Optionally set the anomaly_score=0 for them too
    df_infer_full.loc[override_mask, "anomaly_score"] = 0.0

    # [F] Save the results
    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
    results_file = os.path.join(
        RESULTS_DIR, f"inference_results_{now_str}.csv"
    )
    df_infer_full.to_csv(results_file, index=False)
    print("Saved inference results to", results_file)

    # [G] Print final stats
    final_anomaly_count = df_infer_full["predicted_anomaly"].sum()
    print(f"Predicted {final_anomaly_count} anomalies out of {len(df_infer_full)} logs (after override).")
    

    # 7. Save results in a timestamped file
    # now_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
    # results_file = os.path.join(
    #     RESULTS_DIR, f"inference_results_{now_str}.csv"
    # )
    # df_infer_full.to_csv(results_file, index=False)
    # print("Saved inference results to", results_file)

    # # Optionally, print some quick stats:
    # anomaly_count = np.sum(y_pred)
    # print(f"Predicted {anomaly_count} anomalies out of {len(y_pred)} logs.")
    # if anomaly_count > 0:
    #     auc = roc_auc_score(
    #         df_infer_full.get("Anomalous", [0]*len(df_infer_full)),  # If "Anomalous" column is not in the inference CSV, fallback to 0
    #         y_proba
    #     ) if "Anomalous" in df_infer_full.columns else "N/A"
    #     print(f"AUC against provided labels (if any): {auc}")

def main():
    # 1. Decide anomaly ratio
    # (Optional) you might keep track of run_count in a file or pass as argument
    anomaly_ratio = pick_anomaly_ratio()

    # 2. Generate logs
    generate_logs(anomaly_ratio=anomaly_ratio)

    # 3. Append to big master archive (for historical reasons)
    append_to_master_archive()

    # 4. Preprocess new 200 logs (inference mode)
    run_data_preprocessing()

    # 5. Enhanced features (inference mode)
    run_enhanced_features()

    # 6. Load model + predict
    load_model_and_predict()

    print("auto_inference pipeline complete.")

if __name__ == "__main__":
    main()
