import argparse
import os
import random
import csv
import datetime
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from additional_feature import AdditionalFeatureEngineer

def generate_logs(num_normal, num_anomalous):
    logs = []

    # Define roles, operations, and endpoints with proper mapping
    roles = {
        "Doctor": range(1, 21),
        "Nurse": range(21, 46),
        "Staff": range(46, 61),
        "Admin": range(61, 71),
    }
    http_methods = ["GET", "POST", "PUT", "DELETE"]
    endpoints = {
        "Doctor": ["/patient/records", "/billing/invoices", "/inventory/items"],
        "Nurse": ["/patient/records", "/billing/invoices", "/inventory/items"],
        "Staff": ["/inventory/items", "/billing/invoices"],
        "Admin": ["/admin/settings", "/admin/credentials"],
    }

    # Define subnet for all roles
    network_subnet = "10.0.0"

    # Define normal access patterns
    normal_hours = range(8, 20)
    extended_days = 90  # Logs for the last 90 days

    # Generate unique LogID
    log_id = 1

    # Generate normal logs
    for _ in range(num_normal):
        role = random.choice(list(roles.keys()))
        user_id = random.choice(roles[role])
        ip_address = f"{network_subnet}.{random.randint(1, 254)}"
        timestamp = datetime.datetime.now() + datetime.timedelta(
            days=random.randint(-extended_days, 0),
            hours=random.choice(normal_hours),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        http_method = random.choice(["GET", "POST"])
        endpoint = random.choice(endpoints[role])
        parameter = ""
        if "patient" in endpoint:
            parameter = f"?patient_id={random.randint(1000, 2000)}"
        elif "billing" in endpoint:
            parameter = f"?invoice_id={random.randint(2000, 3000)}"
        elif "inventory" in endpoint:
            parameter = f"?item_id={random.randint(4000, 5000)}"
        elif "admin" in endpoint:
            parameter = f"?admin_id={user_id}"  # Ensure admin_id matches UserID

        logs.append({
            "LogID": log_id,
            "UserID": user_id,
            "Role": role,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "HTTP_Method": http_method,
            "Endpoint": endpoint + parameter,
            "IP_Address": ip_address
        })
        log_id += 1

    return logs

def save_logs_to_csv(logs, file_name):
    write_header = not os.path.exists(file_name) or os.stat(file_name).st_size == 0
    with open(file_name, "a", newline="") as csvfile:
        fieldnames = ["LogID", "UserID", "Role", "Timestamp", "HTTP_Method", "Endpoint", "IP_Address"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(logs)


def generate_log_batches(num_anomalies, num_batches, output_file):
    """
    Generate log batches with the specified number of anomalies and append to a file.
    """
    for batch in range(num_batches):
        logs = generate_logs(num_normal=100 - num_anomalies, num_anomalous=num_anomalies)
        save_logs_to_csv(logs, output_file)
        print(f"Batch {batch + 1} saved to {output_file}.")

def preprocess_logs(input_file, output_file):
    """
    Preprocess the logs using data_preprocessing.py.
    """
    preprocessor = DataPreprocessor(filepath=input_file)
    preprocessor.preprocess()
    X, y, log_ids = preprocessor.get_processed_data()
    processed_data = X.copy()
    processed_data['logid'] = log_ids
    processed_data.to_csv(output_file, index=False)
    print(f"Logs preprocessed and saved to {output_file}.")

def engineer_features(input_file, output_file):
    """
    Perform feature engineering using feature_engineering.py.
    """
    engineer = FeatureEngineer(filepath=input_file)
    engineer.preprocess()
    engineer.save_engineered_data(output_file)
    print(f"Features engineered and saved to {output_file}.")

def enhance_features(input_file, output_file):
    """
    Enhance features using additional_feature.py.
    """
    additional_features = AdditionalFeatureEngineer(filepath=input_file)
    additional_features.preprocess()
    additional_features.save_final_engineered_data(output_file)
    print(f"Additional features engineered and saved to {output_file}.")

def main():
    parser = argparse.ArgumentParser(description="Test pipeline for anomaly detection.")
    parser.add_argument("--num_anomalies", type=int, required=True, help="Number of anomalies per batch.")
    parser.add_argument("--num_batches", type=int, required=True, help="Number of batches to generate.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save intermediate and final outputs.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # File paths
    raw_logs_file = os.path.join(args.output_dir, "raw_logs.csv")
    preprocessed_logs_file = os.path.join(args.output_dir, "preprocessed_logs.csv")
    engineered_logs_file = os.path.join(args.output_dir, "engineered_logs.csv")
    final_features_file = os.path.join(args.output_dir, "final_features.csv")

    # Generate logs
    print("Generating logs...")
    generate_log_batches(args.num_anomalies, args.num_batches, raw_logs_file)

    # Preprocess logs
    print("Preprocessing logs...")
    preprocess_logs(raw_logs_file, preprocessed_logs_file)

    # Engineer features
    print("Engineering features...")
    engineer_features(preprocessed_logs_file, engineered_logs_file)

    # Enhance features
    print("Enhancing features...")
    enhance_features(engineered_logs_file, final_features_file)

    print(f"Pipeline completed. Final features saved to {final_features_file}.")

if __name__ == "__main__":
    main()
