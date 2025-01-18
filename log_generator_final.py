#!/usr/bin/env python3

"""
log_generator.py

An enhanced log generator for a 24/7 healthcare environment, distributing both
normal and anomalous logs across multiple endpoints and roles.

Improvements:
1. Weighted role selection for normal logs.
2. Refined HTTP method distribution for normal logs to reflect typical REST usage (mostly GET, POST).
3. Adjusted HTTP response code distribution to better mirror real-world likelihoods.
4. Expanded endpoint coverage beyond /admin/* for more realism.
5. Optional sequential anomaly scenario for multiple failed login attempts.
"""

import random
import csv
import datetime
import uuid
import os
import argparse

# -------------------- CONFIG & WEIGHTING --------------------

# NEW/UPDATED: Weighted role distribution for normal logs
# Matches roles in the order: ["Doctor", "Nurse", "Staff", "Admin"]
ROLE_WEIGHTS_NORMAL = [0.15, 0.40, 0.35, 0.10]  # Adjust as needed

# NEW/UPDATED: Refined HTTP method distribution for normal logs
NORMAL_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
NORMAL_METHOD_WEIGHTS = [0.65, 0.20, 0.05, 0.02, 0.03, 0.03, 0.02]  # Totals 1.0

def weighted_hour_choice():
    """
    Returns an hour (0-23) for normal logs based on a weighted distribution,
    simulating a busier daytime environment but still some overnight activity.
    """
    hours = list(range(24))  # 0 through 23
    # Example weighting:
    # - Heavier weights around 8 AM -> 18 PM
    # - Lighter overnight
    # Weights sum to 100 for convenience.
    weights = [
        2, 2, 2, 2, 2, 2, 2,  # 0-6
        5, 5, 5, 5, 5, 5, 5,  # 7-13
        5, 5, 5, 5, 5,        # 14-18
        2, 2, 2, 2, 2         # 19-23
    ]
    return random.choices(hours, weights=weights, k=1)[0]


def generate_http_response_code(is_anomalous=False):
    """
    Returns an HTTP response code with different distributions
    for normal vs. anomalous logs.
    """
    if not is_anomalous:
        # NEW/UPDATED: Adjust the normal distribution
        return random.choices(
            [200, 201, 302, 304],
            weights=[0.70, 0.15, 0.10, 0.05],  # Adjust as desired
            k=1
        )[0]
    else:
        # NEW/UPDATED: Slightly adjust anomaly distribution
        return random.choices(
            [200, 403, 404, 500],
            weights=[0.2, 0.4, 0.2, 0.2],
            k=1
        )[0]


def generate_user_agent(role):
    """
    Generates a user agent. Admin has a 5% chance to appear as PostmanRuntime,
    other roles only pick from common UAs.
    """
    common_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Mozilla/5.0 (X11; Linux x86_64)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "curl/7.68.0",
        "Python-urllib/3.9",
    ]
    if role == "Admin":
        pick = random.choices(["common", "postman"], weights=[0.95, 0.05], k=1)[0]
        if pick == "postman":
            return "PostmanRuntime/7.26.8"
        else:
            return random.choice(common_agents)
    else:
        return random.choice(common_agents)


# -------------------- ANOMALY SCENARIOS --------------------

def generate_role_switch_anomaly(roles):
    """
    Picks a user from one role but labels them with a conflicting role.
    """
    all_users = [(r, uid) for r in roles for uid in roles[r]]
    chosen_role, chosen_user = random.choice(all_users)
    possible_roles = [r for r in roles if r != chosen_role]
    new_role = random.choice(possible_roles)
    return new_role, chosen_user


def pick_anomaly_scenario(role):
    """
    Returns a scenario label that decides which endpoint + method to use for anomalies.
    Weighted to reduce /admin dominance for non-admin roles and include new anomaly types.
    """
    if role in ["Doctor", "Nurse"]:
        scenarios = [
            ("admin_unauthorized", 0.10),
            ("patient_records_put_del", 0.25),
            ("inventory_put_del", 0.20),
            ("billing_invoices_put_del", 0.15),
            ("data_exfiltration", 0.15),
            ("privilege_escalation", 0.10),
            ("multiple_failed_attempts", 0.05),  # Add a small chance
        ]
    elif role == "Staff":
        scenarios = [
            ("admin_unauthorized", 0.05),
            ("inventory_put_del", 0.30),
            ("billing_invoices_put_del", 0.30),
            ("multiple_failed_attempts", 0.15),
            ("unusual_access_pattern", 0.10),
            ("data_exfiltration", 0.05),
            ("privilege_escalation", 0.05),
        ]
    else:  # Admin
        scenarios = [
            ("admin_suspicious", 0.25),
            ("patient_records_put_del", 0.25),
            ("inventory_put_del", 0.20),
            ("data_exfiltration", 0.10),
            ("privilege_escalation", 0.10),
            ("multiple_failed_attempts", 0.10)
        ]
    labels = [s[0] for s in scenarios]
    weights = [s[1] for s in scenarios]
    return random.choices(labels, weights=weights, k=1)[0]


def build_anomaly_endpoint_method(scenario_label):
    """
    Based on the scenario label, produce an (endpoint_base, forced_method).
    We'll fill parameters later.
    """
    if scenario_label == "admin_unauthorized":
        # Non-admin roles accessing /admin/*
        candidates = [
            ("/admin/settings", "GET"),
            ("/admin/settings", "PUT"),
            ("/admin/credentials", "GET"),
            ("/admin/credentials", "DELETE"),
        ]
        return random.choice(candidates)

    elif scenario_label == "admin_suspicious":
        # Admin messing with settings or credentials in weird ways
        candidates = [
            ("/admin/settings", "DELETE"),
            ("/admin/settings", "PUT"),
            ("/admin/credentials", "PUT"),
            ("/admin/credentials", "DELETE"),
        ]
        return random.choice(candidates)

    elif scenario_label == "patient_records_put_del":
        candidates = [
            ("/patient/records", "PUT"),
            ("/patient/records", "DELETE"),
        ]
        return random.choice(candidates)

    elif scenario_label == "inventory_put_del":
        candidates = [
            ("/inventory/items", "PUT"),
            ("/inventory/items", "DELETE"),
        ]
        return random.choice(candidates)

    elif scenario_label == "billing_invoices_put_del":
        candidates = [
            ("/billing/invoices", "PUT"),
            ("/billing/invoices", "DELETE"),
        ]
        return random.choice(candidates)

    elif scenario_label == "data_exfiltration":
        # Simulate large data downloads
        candidates = [
            ("/patient/records", "GET"),
            ("/billing/invoices", "GET"),
            ("/inventory/items", "GET"),
        ]
        return random.choice(candidates)

    elif scenario_label == "privilege_escalation":
        candidates = [
            ("/admin/settings", "POST"),
            ("/admin/credentials", "PUT"),
        ]
        return random.choice(candidates)

    elif scenario_label == "multiple_failed_attempts":
        # We'll handle the multi-log aspect externally, but let's define a base
        return ("/login/attempts", "DELETE")

    elif scenario_label == "unusual_access_pattern":
        candidates = [
            ("/billing/invoices", "PUT"),
            ("/patient/records", "DELETE"),
            ("/inventory/items", "PATCH"),
        ]
        return random.choice(candidates)

    # Default fallback (shouldn't happen)
    return ("/unknown/endpoint", "GET")


def build_anomaly_parameters(endpoint_base):
    """
    Builds parameters based on the endpoint_base.
    Ensures that parameters are correctly generated based on the specific endpoint.
    """
    if "/patient/records" in endpoint_base:
        return f"?patient_id={random.randint(1000, 2000)}&export=true&limit=1000"
    elif "/billing/invoices" in endpoint_base:
        return f"?invoice_id={random.randint(2000, 3000)}&export=true&limit=500"
    elif "/inventory/items" in endpoint_base:
        return f"?item_id={random.randint(4000, 5000)}&export=true&limit=300"
    elif "/admin/settings" in endpoint_base or "/admin/credentials" in endpoint_base:
        return f"?admin_id={random.randint(61, 70)}"
    elif "/login/attempts" in endpoint_base:
        return f"?user_id={random.randint(1, 70)}&attempts=5"
    elif "/unknown/endpoint" in endpoint_base:
        return ""
    else:
        return ""  # Default to empty string if endpoint doesn't match any condition


def generate_unique_log_id():
    """
    Generates a unique LogID using UUID4.
    """
    return str(uuid.uuid4())

# NEW/UPDATED: Optional multi-log anomaly generator
def generate_multiple_failed_attempts(user_id, role, network_subnet, extended_days, count=4):
    """
    Generates a sequence of anomaly logs that simulate multiple failed attempts (403)
    followed by a suspicious success (200).
    This is an optional approach to produce sequential anomalies in short bursts.
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
        ip_address = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"  # out-of-subnet
        timestamp = base_timestamp - datetime.timedelta(seconds=i*5)  # space them 5s apart
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
            "Endpoint": f"/login/attempts?user_id={user_id}&attempts=5",
            "IP_Address": ip_address,
            "User_Agent": generate_user_agent(role),
            "HTTP_Response": http_response,
            "Anomalous": 1,
        }
        logs.append(log_entry)
    return logs


# -------------------- LOG GENERATION --------------------

def generate_normal_log(roles, endpoints, network_subnet, extended_days):
    """
    Generates a single normal log entry with 24-hour weighting
    and weighted role selection.
    """
    # NEW/UPDATED: Weighted role selection
    role_choices = list(roles.keys())  # ["Doctor", "Nurse", "Staff", "Admin"]
    role = random.choices(role_choices, weights=ROLE_WEIGHTS_NORMAL, k=1)[0]
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
    ip_address = f"{network_subnet}.{random.randint(1, 254)}"

    # NEW/UPDATED: Weighted HTTP method distribution for normal logs
    http_method = random.choices(NORMAL_METHODS, weights=NORMAL_METHOD_WEIGHTS, k=1)[0]

    endpoint = random.choice(endpoints[role])

    # Add parameters based on endpoint
    parameter = ""
    if "/patient/records" in endpoint:
        parameter = f"?patient_id={random.randint(1000, 2000)}"
    elif "/billing/invoices" in endpoint:
        parameter = f"?invoice_id={random.randint(2000, 3000)}"
    elif "/inventory/items" in endpoint:
        parameter = f"?item_id={random.randint(4000, 5000)}"
    elif "/patient/labs" in endpoint:
        parameter = f"?lab_id={random.randint(1000,2000)}"
    elif "/patient/appointments" in endpoint:
        parameter = f"?appt_id={random.randint(500,1000)}"
    elif "/patient/scheduling" in endpoint:
        parameter = f"?appt_id={random.randint(500,1000)}"
    elif "/admin/settings" in endpoint or "/admin/credentials" in endpoint:
        parameter = f"?admin_id={user_id}"

    log_entry = {
        "LogID": generate_unique_log_id(),
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": http_method,
        "Endpoint": endpoint + parameter,
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
        ip_address = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
    else:
        ip_address = f"{network_subnet}.{random.randint(1, 254)}"

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

    # If we want to produce a multi-log anomaly (i.e., sequential) for multiple_failed_attempts,
    # we can handle that separately. For simplicity, let's handle single anomalies here.
    # We'll return an indicator if "multiple_failed_attempts" was chosen.
    if scenario_label == "multiple_failed_attempts":
        # We will just produce a single log here, but you can call a multi-log generator if needed.
        # Return a single 'failed' attempt (403) or suspicious success (200).
        # Or skip generating a single log entirely in favor of multi-log approach.
        # Below is a simple single-log approach. If you want full sequential, comment this out
        # and handle it externally.
        endpoint_base, forced_method = ("/login/attempts", "DELETE")
        endpoint = endpoint_base + f"?user_id={user_id}&attempts=5"
        log_entry = {
            "LogID": generate_unique_log_id(),
            "UserID": user_id,
            "Role": role,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "HTTP_Method": forced_method,
            "Endpoint": endpoint,
            "IP_Address": ip_address,
            "User_Agent": generate_user_agent(role),
            "HTTP_Response": 403,  # forced fail
            "Anomalous": 1,
        }
        return [log_entry]  # return as a list for consistency
    else:
        endpoint_base, forced_method = build_anomaly_endpoint_method(scenario_label)
        http_method = forced_method
        parameter = build_anomaly_parameters(endpoint_base)
        endpoint = endpoint_base + parameter

        log_entry = {
            "LogID": generate_unique_log_id(),
            "UserID": user_id,
            "Role": role,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "HTTP_Method": http_method,
            "Endpoint": endpoint,
            "IP_Address": ip_address,
            "User_Agent": generate_user_agent(role),
            "HTTP_Response": generate_http_response_code(is_anomalous=True),
            "Anomalous": 1,
        }
        # Return a list for consistent handling of single vs multi logs
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

    # Roles and user IDs
    # NEW/UPDATED: More realistic ID ranges for each role
    roles = {
        "Doctor": list(range(1, 21)),    # 20 Doctors
        "Nurse": list(range(21, 46)),   # 25 Nurses
        "Staff": list(range(46, 61)),   # 15 Staff
        "Admin": list(range(61, 71)),   # 10 Admin
    }

    # NEW/UPDATED: Expanded endpoint coverage for each role
    endpoints = {
        "Doctor": [
            "/patient/records",
            "/patient/labs",
            "/patient/appointments",
            "/billing/invoices"
        ],
        "Nurse": [
            "/patient/records",
            "/patient/labs",
            "/patient/appointments",
            "/billing/invoices"
        ],
        "Staff": [
            "/inventory/items",
            "/billing/invoices",
            "/patient/scheduling"
        ],
        "Admin": [
            "/admin/settings",
            "/admin/credentials"
        ],
    }

    print(f"Starting log generation with the following parameters:")
    print(f"Total Logs: {total_logs}")
    print(f"Anomaly Ratio: {anomaly_ratio * 100}%")
    print(f"Master File: {master_file}")
    print(f"Inference File: {inference_file}")
    print(f"Network Subnet: {network_subnet}.x")
    print(f"Extended Days: {extended_days}\n")

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

    save_logs_to_csv(logs, master_file, inference_file)
    print(f"Logs saved to '{master_file}' and '{inference_file}'.")
    print("Log generation completed successfully.")


if __name__ == "__main__":
    main()




# python3 log_generator --total-logs 500 --anonmaly-ratio 0.005