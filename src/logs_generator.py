# log_generator.py

import random
import csv
import datetime
import uuid
import os

def generate_unique_log_id():
    """
    Generates a unique LogID using UUID4.
    
    Returns:
        str: A unique LogID.
    """
    return str(uuid.uuid4())

def generate_normal_log(roles, endpoints, network_subnet, normal_hours, extended_days):
    """
    Generates a single normal log entry.
    
    Args:
        roles (dict): Mapping of roles to user IDs.
        endpoints (dict): Mapping of roles to accessible endpoints.
        network_subnet (str): Subnet for generating IP addresses.
        normal_hours (range): Range of normal operational hours.
        extended_days (int): Number of days to simulate.
    
    Returns:
        dict: A dictionary representing a normal log entry.
    """
    role = random.choice(list(roles.keys()))
    user_id = random.choice(list(roles[role]))
    ip_address = f"{network_subnet}.{random.randint(1, 254)}"
    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        hours=random.choice(normal_hours),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    http_method = random.choice(["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
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

    log_entry = {
        "LogID": generate_unique_log_id(),
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": http_method,
        "Endpoint": endpoint + parameter,
        "IP_Address": ip_address,
        "Anomalous": 0,
    }
    return log_entry

def generate_anomalous_log(roles, endpoints, network_subnet, extended_days):
    """
    Generates a single anomalous log entry.
    
    Args:
        roles (dict): Mapping of roles to user IDs.
        endpoints (dict): Mapping of roles to accessible endpoints.
        network_subnet (str): Subnet for generating IP addresses.
        extended_days (int): Number of days to simulate.
    
    Returns:
        dict: A dictionary representing an anomalous log entry.
    """
    role = random.choice(list(roles.keys()))
    user_id = random.choice(list(roles[role]))
    ip_address = f"{network_subnet}.{random.randint(1, 254)}"  # Same network subnet
    # 10% chance of IP outside the subnet
    if random.random() < 0.1:
        ip_address = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"

    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        hours=random.randint(0, 23),  # Any time of the day
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    http_method = random.choice(["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])

    # Define anomaly types based on role
    if role in ["Doctor", "Nurse"]:
        # Unauthorized access to admin endpoints
        endpoint = random.choice(["/admin/settings", "/admin/credentials"])
        parameter = f"?admin_id={random.randint(3000, 4000)}"
    elif role == "Staff":
        # Invalid operations on inventory or billing
        endpoint = random.choice(["/inventory/items", "/billing/invoices"])
        if http_method in ["PUT", "DELETE"]:
            parameter = f"?item_id={random.randint(4000, 5000)}"
        else:
            parameter = ""
    elif role == "Admin":
        # Unauthorized access to patient records or inventory
        endpoint = random.choice(["/patient/records", "/inventory/items"])
        parameter = f"?patient_id={random.randint(1000, 2000)}"

    log_entry = {
        "LogID": generate_unique_log_id(),
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": http_method,
        "Endpoint": endpoint + parameter,
        "IP_Address": ip_address,
        "Anomalous": 1,
    }
    return log_entry

def generate_logs(num_normal, num_anomalous, roles, endpoints, network_subnet, normal_hours, extended_days):
    """
    Generates a list of log entries containing both normal and anomalous logs.
    
    Args:
        num_normal (int): Number of normal logs to generate.
        num_anomalous (int): Number of anomalous logs to generate.
        roles (dict): Mapping of roles to user IDs.
        endpoints (dict): Mapping of roles to accessible endpoints.
        network_subnet (str): Subnet for generating IP addresses.
        normal_hours (range): Range of normal operational hours.
        extended_days (int): Number of days to simulate.
    
    Returns:
        list: A list of log entry dictionaries.
    """
    logs = []

    # Generate normal logs
    for _ in range(num_normal):
        log = generate_normal_log(roles, endpoints, network_subnet, normal_hours, extended_days)
        logs.append(log)

    # Generate anomalous logs
    for _ in range(num_anomalous):
        log = generate_anomalous_log(roles, endpoints, network_subnet, extended_days)
        logs.append(log)

    return logs

def save_logs_to_csv(logs, master_file, inference_file):
    """
    Saves log entries to master and inference CSV files.
    
    Args:
        logs (list): List of log entry dictionaries.
        master_file (str): Path to the master log CSV file with labels.
        inference_file (str): Path to the inference log CSV file without labels.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(master_file), exist_ok=True)
    os.makedirs(os.path.dirname(inference_file), exist_ok=True)

    # Append to master log file with labels
    fieldnames_master = ["LogID", "UserID", "Role", "Timestamp", "HTTP_Method", "Endpoint", "IP_Address", "Anomalous"]
    master_exists = os.path.isfile(master_file)
    with open(master_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames_master)
        if not master_exists:
            writer.writeheader()
        writer.writerows(logs)

    # Append to inference log file without labels
    fieldnames_inference = ["LogID", "UserID", "Role", "Timestamp", "HTTP_Method", "Endpoint", "IP_Address"]
    inference_exists = os.path.isfile(inference_file)
    with open(inference_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames_inference)
        if not inference_exists:
            writer.writeheader()
        for log in logs:
            inference_log = {k: v for k, v in log.items() if k != "Anomalous"}
            writer.writerow(inference_log)

def main():
    """
    Main function to generate logs.
    Generates 7000 logs with 5000 normal and 2000 anomalous logs.
    """
    num_normal = 5000
    num_anomalous = 2000

    master_file = "data/master_logs.csv"
    inference_file = "data/inference_logs.csv"

    # Define roles, operations, and endpoints with proper mapping
    roles = {
        "Doctor": list(range(1, 21)),
        "Nurse": list(range(21, 46)),
        "Staff": list(range(46, 61)),
        "Admin": list(range(61, 71)),
    }
    http_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
    endpoints = {
        "Doctor": ["/patient/records", "/billing/invoices", "/inventory/items"],
        "Nurse": ["/patient/records", "/billing/invoices", "/inventory/items"],
        "Staff": ["/inventory/items", "/billing/invoices"],
        "Admin": ["/admin/settings", "/admin/credentials"],
    }

    network_subnet = "10.0.0"
    normal_hours = range(8, 20)  # 8 AM to 8 PM
    extended_days = 90  # Logs for the last 90 days

    print("Starting log generation...")
    logs = generate_logs(
        num_normal=num_normal,
        num_anomalous=num_anomalous,
        roles=roles,
        endpoints=endpoints,
        network_subnet=network_subnet,
        normal_hours=normal_hours,
        extended_days=extended_days
    )
    print(f"Generated {len(logs)} logs.")

    save_logs_to_csv(logs, master_file, inference_file)
    print(f"Logs saved to '{master_file}' and '{inference_file}'.")

    print("Log generation completed successfully.")

if __name__ == "__main__":
    main()
