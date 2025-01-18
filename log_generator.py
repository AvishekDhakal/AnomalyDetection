#!/usr/bin/env python3

"""
log_generator.py

An enhanced log generator for a 24/7 healthcare environment, distributing both
normal and anomalous logs across multiple endpoints and roles.

Improvements:
1. Externalized configuration parameters for flexibility.
2. Expanded and refined endpoint coverage.
3. Implemented hierarchical and parameterized endpoints.
4. Enhanced anomaly scenarios with chained anomalies.
5. Improved User-Agent diversity.
6. Refined IP address generation logic.
7. Enhanced timestamp logic for realistic patterns.
8. Added validation and testing mechanisms.
"""

import random
import csv
import datetime
import uuid
import os
import argparse
import json
from collections import defaultdict

# -------------------- CONFIG LOADING --------------------

def load_config(config_file):
    """
    Loads configuration parameters from a JSON file.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

# -------------------- CONFIG & WEIGHTING --------------------

# Load configuration
CONFIG_FILE = "config.json"  # Ensure this file exists in the same directory
config = load_config(CONFIG_FILE)

ROLE_WEIGHTS_NORMAL = config["ROLE_WEIGHTS_NORMAL"]
NORMAL_METHODS = config["NORMAL_METHODS"]["methods"]
NORMAL_METHOD_WEIGHTS = config["NORMAL_METHODS"]["weights"]

HTTP_RESPONSE_CODES_NORMAL = config["HTTP_RESPONSE_CODES"]["normal"]["codes"]
HTTP_RESPONSE_CODES_NORMAL_WEIGHTS = config["HTTP_RESPONSE_CODES"]["normal"]["weights"]

HTTP_RESPONSE_CODES_ANOMALOUS = config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"]
HTTP_RESPONSE_CODES_ANOMALOUS_WEIGHTS = config["HTTP_RESPONSE_CODES"]["anomalous"]["weights"]

ANOMALY_SCENARIOS = config["ANOMALY_SCENARIOS"]
ENDPOINTS = config["ENDPOINTS"]
ANOMALOUS_ENDPOINTS = config["ANOMALOUS_ENDPOINTS"]
USER_AGENTS = config["USER_AGENTS"]

# -------------------- FUNCTIONS --------------------

def weighted_hour_choice():
    """
    Returns an hour (0-23) for normal logs based on a weighted distribution,
    simulating a busier daytime environment but still some overnight activity.
    """
    hours = list(range(24))  # 0 through 23
    weights = [
        1, 1, 1, 1, 1, 1, 1,  # 0-6
        8, 8, 8, 8, 8, 8, 8,  # 7-13
        6, 6, 6, 6, 6,        # 14-18
        3, 3, 3, 3, 3         # 19-23
    ]
    return random.choices(hours, weights=weights, k=1)[0]

def generate_http_response_code(is_anomalous=False):
    """
    Returns an HTTP response code with different distributions
    for normal vs. anomalous logs.
    """
    if not is_anomalous:
        return random.choices(
            HTTP_RESPONSE_CODES_NORMAL,
            weights=HTTP_RESPONSE_CODES_NORMAL_WEIGHTS,
            k=1
        )[0]
    else:
        return random.choices(
            HTTP_RESPONSE_CODES_ANOMALOUS,
            weights=HTTP_RESPONSE_CODES_ANOMALOUS_WEIGHTS,
            k=1
        )[0]

def generate_user_agent(role):
    """
    Generates a user agent. Admin has a 5% chance to appear as PostmanRuntime,
    other roles pick from a diverse set of common UAs.
    """
    common_agents = USER_AGENTS["common"]
    admin_extra_agents = USER_AGENTS.get("admin_extra", [])
    
    if role == "Admin":
        pick = random.choices(["common", "extra"], weights=[0.95, 0.05], k=1)[0]
        if pick == "extra" and admin_extra_agents:
            return random.choice(admin_extra_agents)
        else:
            return random.choice(common_agents)
    else:
        return random.choice(common_agents)

def generate_ip_address(is_anomalous, network_subnet):
    """
    Generates an IP address. Internal subnet for normal logs,
    random public IPs for anomalous logs.
    """
    if is_anomalous:
        # Generate a random public IP (avoiding private and reserved ranges)
        first_octet = random.choice([i for i in range(1, 223) if i not in [10, 172, 192]])
        return f"{first_octet}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
    else:
        return f"{network_subnet}.{random.randint(1, 254)}"

def generate_role_switch_anomaly(roles):
    """
    Picks a user from one role but labels them with a conflicting role.
    Ensures user IDs are unique across roles.
    """
    all_users = [(r, uid) for r in roles for uid in roles[r]]
    chosen_role, chosen_user = random.choice(all_users)
    possible_roles = [r for r in roles if r != chosen_role]
    new_role = random.choice(possible_roles)
    return new_role, chosen_user

def pick_anomaly_scenario(role):
    """
    Returns a scenario label that decides which endpoint + method to use for anomalies.
    """
    scenarios = ANOMALY_SCENARIOS.get(role, {})
    labels = list(scenarios.keys())
    weights = list(scenarios.values())
    return random.choices(labels, weights=weights, k=1)[0]

def build_anomaly_endpoint_method(scenario_label):
    """
    Based on the scenario label, produce an (endpoint_base, forced_method).
    We'll fill parameters later.
    """
    candidates = ANOMALOUS_ENDPOINTS.get(scenario_label, [("/unknown/endpoint", "GET")])
    return random.choice(candidates)

def build_anomaly_parameters(endpoint_base):
    """
    Builds parameters based on the endpoint_base.
    Ensures that parameters are correctly generated based on the specific endpoint.
    """
    if "/patient/records" in endpoint_base:
        return f"/{random.randint(1000, 9999)}?export=true&limit=1000"
    elif "/billing/invoices" in endpoint_base:
        return f"/{random.randint(2000, 9999)}?export=true&limit=500"
    elif "/inventory/items" in endpoint_base:
        return f"/{random.randint(4000, 9999)}?export=true&limit=300"
    elif "/admin/settings" in endpoint_base or "/admin/credentials" in endpoint_base:
        return f"/{random.randint(61, 70)}"
    elif "/login/attempts" in endpoint_base:
        return f"/{random.randint(1, 70)}&attempts=5"
    elif "/unknown/endpoint" in endpoint_base:
        return ""
    else:
        return ""  # Default to empty string if endpoint doesn't match any condition

def generate_unique_log_id():
    """
    Generates a unique LogID using UUID4.
    """
    return str(uuid.uuid4())

def generate_multiple_failed_attempts(user_id, role, network_subnet, extended_days, count=4):
    """
    Generates a sequence of anomaly logs that simulate multiple failed attempts (403)
    followed by a suspicious success (200).
    """
    logs = []
    # Decide a base timestamp
    base_timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )

    for i in range(count):
        log_id = generate_unique_log_id()
        ip_address = generate_ip_address(is_anomalous=True, network_subnet=network_subnet)
        # Introduce randomness in seconds apart
        timestamp = base_timestamp - datetime.timedelta(seconds=random.randint(1, 10) + i * 5)
        # For the first few logs, force a 403
        if i < count - 1:
            http_response = 403
        else:
            # final attempt might be 200 or 500
            http_response = random.choices([200, 500], weights=[0.8, 0.2], k=1)[0]

        log_entry = {
            "LogID": log_id,
            "UserID": user_id,
            "Role": role,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "HTTP_Method": "DELETE",
            "Endpoint": f"/login/attempts/{user_id}?attempts=5",
            "IP_Address": ip_address,
            "User_Agent": generate_user_agent(role),
            "HTTP_Response": http_response,
            "Anomalous": 1,
        }
        logs.append(log_entry)
    return logs

def generate_chain_of_anomalies(user_id, role, network_subnet, extended_days):
    """
    Generates a chain of related anomalous logs to simulate complex attack patterns.
    Example: Privilege Escalation followed by Data Exfiltration.
    """
    logs = []
    # Base timestamp
    base_timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )

    # Privilege Escalation
    log_id1 = generate_unique_log_id()
    ip_address1 = generate_ip_address(is_anomalous=True, network_subnet=network_subnet)
    timestamp1 = base_timestamp - datetime.timedelta(seconds=random.randint(1, 60))
    http_response1 = random.choices([200, 500], weights=[0.8, 0.2], k=1)[0]

    log_entry1 = {
        "LogID": log_id1,
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp1.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": "POST",
        "Endpoint": f"/admin/credentials/{random.randint(61,70)}",
        "IP_Address": ip_address1,
        "User_Agent": generate_user_agent(role),
        "HTTP_Response": http_response1,
        "Anomalous": 1,
    }
    logs.append(log_entry1)

    # Data Exfiltration
    log_id2 = generate_unique_log_id()
    ip_address2 = generate_ip_address(is_anomalous=True, network_subnet=network_subnet)
    # Ensure timestamp2 is after timestamp1
    timestamp2 = timestamp1 + datetime.timedelta(seconds=random.randint(1, 60))
    http_response2 = generate_http_response_code(is_anomalous=True)

    log_entry2 = {
        "LogID": log_id2,
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp2.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": "GET",
        "Endpoint": f"/patient/records/{random.randint(1000,9999)}?export=true&limit=1000",
        "IP_Address": ip_address2,
        "User_Agent": generate_user_agent(role),
        "HTTP_Response": http_response2,
        "Anomalous": 1,
    }
    logs.append(log_entry2)

    return logs

def generate_normal_log(roles, endpoints, network_subnet, extended_days):
    """
    Generates a single normal log entry with 24-hour weighting
    and weighted role selection.
    """
    # Weighted role selection
    role_choices = list(roles.keys())  # ["Doctor", "Nurse", "Staff", "Admin"]
    roles_list = list(ROLE_WEIGHTS_NORMAL.keys())
    weights_list = list(ROLE_WEIGHTS_NORMAL.values())
    role = random.choices(role_choices, weights=weights_list, k=1)[0]
    user_id = random.choice(roles[role])

    # Weighted hour for normal
    chosen_hour = weighted_hour_choice()
    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        hours=chosen_hour,
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )

    # Normal logs: IP always internal subnet in this example
    ip_address = generate_ip_address(is_anomalous=False, network_subnet=network_subnet)

    # Weighted HTTP method distribution for normal logs
    http_method = random.choices(NORMAL_METHODS, weights=NORMAL_METHOD_WEIGHTS, k=1)[0]

    endpoint = random.choice(endpoints[role])

    # Add parameters based on endpoint with hierarchical structure
    parameter = ""
    if "/patient/records" in endpoint:
        parameter = f"/{random.randint(1000, 9999)}?export=true&limit=1000"
    elif "/billing/invoices" in endpoint:
        parameter = f"/{random.randint(2000, 9999)}?export=true&limit=500"
    elif "/inventory/items" in endpoint:
        parameter = f"/{random.randint(4000, 9999)}?export=true&limit=300"
    elif "/patient/labs" in endpoint:
        parameter = f"/{random.randint(1000,9999)}"
    elif "/patient/appointments" in endpoint:
        parameter = f"/{random.randint(500,1000)}"
    elif "/patient/scheduling" in endpoint:
        parameter = f"/{random.randint(500,1000)}"
    elif "/admin/settings" in endpoint or "/admin/credentials" in endpoint or "/admin/logs" in endpoint or "/admin/users" in endpoint:
        parameter = f"/{random.randint(61,70)}"
    elif "/claims/status" in endpoint:
        parameter = f"/{random.randint(3000, 9999)}"
    elif "/pharmacy/orders" in endpoint or "/pharmacy/refills" in endpoint:
        parameter = f"/{random.randint(5000,9999)}"
    elif "/lab/results" in endpoint or "/lab/tests" in endpoint:
        parameter = f"/{random.randint(7000,9999)}"
    else:
        parameter = ""

    endpoint_full = endpoint + parameter

    log_entry = {
        "LogID": generate_unique_log_id(),
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": http_method,
        "Endpoint": endpoint_full,
        "IP_Address": ip_address,
        "User_Agent": generate_user_agent(role),
        "HTTP_Response": generate_http_response_code(is_anomalous=False),
        "Anomalous": 0,
    }
    return log_entry

def generate_anomalous_log(
    roles,
    endpoints,
    network_subnet,
    extended_days,
    role_switch_probability=0.05,
    anomaly_ip_probability=0.10,
    after_hours_probability=0.25
):
    """
    Generates a single anomalous log entry using scenario-based approach.
    1. Possibly role-switch (5%).
    2. Possibly out-of-subnet IP (10%).
    3. Possibly forced after-hours (25%).
    4. Weighted scenario selection per role.
    5. Handle chained anomalies.
    """
    # 1. Role-Switch
    if random.random() < role_switch_probability:
        role, user_id = generate_role_switch_anomaly(roles)
    else:
        role = random.choice(list(roles.keys()))
        user_id = random.choice(roles[role])

    # 2. IP Logic
    if random.random() < anomaly_ip_probability:
        # out-of-subnet
        ip_address = generate_ip_address(is_anomalous=True, network_subnet=network_subnet)
    else:
        ip_address = generate_ip_address(is_anomalous=False, network_subnet=network_subnet)

    # 3. Time Logic
    if random.random() < after_hours_probability:
        # Force after hours: pick from [21..23] + [0..5]
        possible_hours = list(range(21, 24)) + list(range(0, 6))
        chosen_hour = random.choice(possible_hours)
        timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, extended_days),
            hours=chosen_hour,
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
    else:
        chosen_hour = random.randint(0, 23)
        timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, extended_days),
            hours=chosen_hour,
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )

    # 4. Weighted scenario approach
    scenario_label = pick_anomaly_scenario(role)

    # 5. Check if scenario requires chained anomalies
    if scenario_label == "multiple_failed_attempts":
        # Generate a sequence of failed attempts
        anomaly_logs = generate_multiple_failed_attempts(user_id, role, network_subnet, extended_days, count=4)
        return anomaly_logs
    elif scenario_label == "chain_of_anomalies":
        # Placeholder for chained anomalies if defined in config
        anomaly_logs = generate_chain_of_anomalies(user_id, role, network_subnet, extended_days)
        return anomaly_logs
    else:
        # Single anomalous log
        endpoint_base, forced_method = build_anomaly_endpoint_method(scenario_label)
        http_method = forced_method
        parameter = build_anomaly_parameters(endpoint_base)
        endpoint_full = endpoint_base + parameter

        log_entry = {
            "LogID": generate_unique_log_id(),
            "UserID": user_id,
            "Role": role,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "HTTP_Method": http_method,
            "Endpoint": endpoint_full,
            "IP_Address": ip_address,
            "User_Agent": generate_user_agent(role),
            "HTTP_Response": generate_http_response_code(is_anomalous=True),
            "Anomalous": 1,
        }
        # Return as a list for consistency
        return [log_entry]

def generate_logs(
    total_logs,
    anomaly_ratio,
    roles,
    endpoints,
    network_subnet,
    extended_days
):
    """
    Generates normal and anomalous logs using the specified ratio.
    """
    num_anomalous = int(total_logs * anomaly_ratio)
    num_normal = total_logs - num_anomalous

    logs = []
    # Generate normal logs
    for _ in range(num_normal):
        log = generate_normal_log(roles, endpoints, network_subnet, extended_days)
        logs.append(log)

    # Generate anomalous logs
    for _ in range(num_anomalous):
        # generate_anomalous_log might return 1 or multiple logs
        anomaly_logs = generate_anomalous_log(roles, endpoints, network_subnet, extended_days)
        logs.extend(anomaly_logs)

    # Shuffle logs to mix normal and anomalous entries
    random.shuffle(logs)
    return logs

def save_logs_to_csv(logs, master_file, inference_file):
    """
    Saves the generated logs to CSV files: master (with 'Anomalous') and inference (without).
    """
    os.makedirs(os.path.dirname(master_file), exist_ok=True)
    os.makedirs(os.path.dirname(inference_file), exist_ok=True)

    fieldnames_master = [
        "LogID",
        "UserID",
        "Role",
        "Timestamp",
        "HTTP_Method",
        "Endpoint",
        "IP_Address",
        "User_Agent",
        "HTTP_Response",
        "Anomalous"
    ]
    master_exists = os.path.isfile(master_file)
    with open(master_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames_master)
        if not master_exists:
            writer.writeheader()
        writer.writerows(logs)

    fieldnames_inference = [
        "LogID",
        "UserID",
        "Role",
        "Timestamp",
        "HTTP_Method",
        "Endpoint",
        "IP_Address",
        "User_Agent",
        "HTTP_Response"
    ]
    inference_exists = os.path.isfile(inference_file)
    with open(inference_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames_inference)
        if not inference_exists:
            writer.writeheader()
        for log in logs:
            inference_log = {k: v for k, v in log.items() if k != "Anomalous"}
            writer.writerow(inference_log)

def validate_logs(logs, expected_anomaly_ratio):
    """
    Validates that the generated logs meet the expected anomaly ratio.
    """
    total_logs = len(logs)
    actual_anomalies = sum(log["Anomalous"] for log in logs)
    actual_ratio = actual_anomalies / total_logs if total_logs > 0 else 0
    print(f"Expected Anomaly Ratio: {expected_anomaly_ratio * 100}%")
    print(f"Actual Anomaly Ratio: {actual_ratio * 100:.2f}%")
    if not abs(actual_ratio - expected_anomaly_ratio) < 0.01:
        print("Warning: Anomaly ratio deviates significantly from expected.")

# -------------------- ARGUMENT PARSING --------------------

def parse_arguments():
    """
    Parses command-line arguments for the log generator.
    """
    parser = argparse.ArgumentParser(description="Enhanced Synthetic Log Generator for Healthcare Environment")
    parser.add_argument(
        "--total_logs",
        type=int,
        default=100000,
        help="Total number of logs to generate (default: 100000)"
    )
    parser.add_argument(
        "--anomaly_ratio",
        type=float,
        default=0.02,
        help="Ratio of anomalous logs (default: 0.02)"
    )
    parser.add_argument(
        "--master_file",
        type=str,
        default="data/master_logs.csv",
        help="Path to the master CSV file (with labels) (default: data/master_logs.csv)"
    )
    parser.add_argument(
        "--inference_file",
        type=str,
        default="data/inference_logs.csv",
        help="Path to the inference CSV file (without labels) (default: data/inference_logs.csv)"
    )
    parser.add_argument(
        "--network_subnet",
        type=str,
        default="10.0.0",
        help="Base subnet for internal IP addresses (default: 10.0.0)"
    )
    parser.add_argument(
        "--extended_days",
        type=int,
        default=90,
        help="Number of days to simulate logs from the past (default: 90)"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
        help="Path to the configuration JSON file (default: config.json)"
    )
    return parser.parse_args()

# -------------------- MAIN GENERATION FUNCTION --------------------

def main():
    """
    Main entry point for generating logs.
    """
    args = parse_arguments()
    total_logs = args.total_logs
    anomaly_ratio = args.anomaly_ratio
    master_file = args.master_file
    inference_file = args.inference_file
    network_subnet = args.network_subnet
    extended_days = args.extended_days
    config_file = args.config_file

    # Reload configuration in case config_file path is different
    global config
    config = load_config(config_file)

    # Roles and user IDs
    # Ensure unique UserIDs across all roles
    all_user_ids = list(range(1, 71))  # 70 unique users
    roles = {
        "Doctor": random.sample(all_user_ids, 20),    # 20 Doctors
        "Nurse": random.sample([uid for uid in all_user_ids if uid not in all_user_ids[:20]], 25),   # 25 Nurses
        "Staff": random.sample([uid for uid in all_user_ids if uid not in all_user_ids[:45]], 15),   # 15 Staff
        "Admin": random.sample([uid for uid in all_user_ids if uid not in all_user_ids[:60]], 10),   # 10 Admin
    }

    # Update endpoints from config
    endpoints = config["ENDPOINTS"]

    print(f"Starting log generation with the following parameters:")
    print(f"Total Logs: {total_logs}")
    print(f"Anomaly Ratio: {anomaly_ratio * 100}%")
    print(f"Master File: {master_file}")
    print(f"Inference File: {inference_file}")
    print(f"Network Subnet: {network_subnet}.x")
    print(f"Extended Days: {extended_days}")
    print(f"Configuration File: {config_file}\n")

    logs = generate_logs(
        total_logs=total_logs,
        anomaly_ratio=anomaly_ratio,
        roles=roles,
        endpoints=endpoints,
        network_subnet=network_subnet,
        extended_days=extended_days
    )

    total_anomalies = sum(log["Anomalous"] for log in logs)
    print(f"Generated {len(logs)} logs. Anomalies: {total_anomalies}")

    validate_logs(logs, anomaly_ratio)

    save_logs_to_csv(logs, master_file, inference_file)
    print(f"Logs saved to '{master_file}' and '{inference_file}'.")
    print("Log generation completed successfully.")

if __name__ == "__main__":
    main()
