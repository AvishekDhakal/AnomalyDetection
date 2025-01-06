import random
import csv
import datetime

def generate_augmented_logs(num_normal, num_anomalous):
    logs = []

    # Define roles, operations, and endpoints
    roles = {
        "Doctor": range(1, 21),
        "Nurse": range(21, 46),
        "Admin": range(46, 51),
    }
    http_methods = ["GET", "POST", "PUT", "DELETE"]
    normal_endpoints = [
        "/api/patient/records",
        "/api/billing/invoices",
        "/api/inventory/items",
    ]
    sensitive_endpoints = [
        "/api/patient/confidential",
        "/api/admin/credentials",
    ]
    non_parameterized_endpoints = ["/api/admin/settings"]

    # Define normal access patterns
    normal_hours = range(8, 20)

    # Extend time period for logs
    extended_days = 90

    # Generate normal logs
    for _ in range(num_normal):
        role = random.choice(list(roles.keys()))
        user_id = random.choice(roles[role])
        timestamp = datetime.datetime.now() + datetime.timedelta(
            days=random.randint(-extended_days, 0),
            hours=random.choice(normal_hours),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        http_method = random.choice(["GET", "POST"])
        endpoint = random.choice(normal_endpoints)
        parameter = f"?patient_id={random.randint(1000, 2000)}" if "patient" in endpoint else f"?invoice_id={random.randint(2000, 3000)}"
        logs.append(
            {
                "UserID": user_id,
                "Role": role,
                "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "HTTP_Method": http_method,
                "Endpoint": endpoint + parameter,
                "Anomalous": 0,
            }
        )

    # Generate anomalous logs
    for _ in range(num_anomalous):
        role = random.choice(list(roles.keys()))
        user_id = random.choice(roles[role])
        timestamp = datetime.datetime.now() + datetime.timedelta(
            days=random.randint(-extended_days, 0),
            hours=random.randint(0, 23),  # Any time of the day
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        http_method = random.choice(http_methods)

        # Time-based anomaly logic for Admin
        is_unusual_time = timestamp.hour not in normal_hours if role == "Admin" else False

        if random.random() < 0.5 or is_unusual_time:
            # Unusual endpoint access or unusual time for Admin
            endpoint = random.choice(sensitive_endpoints)
            parameter = f"?admin_id={random.randint(3000, 4000)}" if "admin" in endpoint else f"?patient_id={random.randint(1000, 2000)}"
        else:
            # Unusual role behavior
            endpoint = random.choice(normal_endpoints + non_parameterized_endpoints)
            if endpoint in non_parameterized_endpoints:
                parameter = ""  # No parameters for these endpoints
            else:
                parameter = f"?item_id={random.randint(4000, 5000)}"
            if role == "Nurse" and endpoint not in non_parameterized_endpoints:
                endpoint = "/api/admin/settings"  # Unauthorized operation
                parameter = ""  # No parameters for this endpoint
            elif role == "Doctor":
                endpoint = "/api/billing/invoices"  # Unusual access
                parameter = f"?invoice_id={random.randint(2000, 3000)}"
        logs.append(
            {
                "UserID": user_id,
                "Role": role,
                "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "HTTP_Method": http_method,
                "Endpoint": endpoint + parameter,
                "Anomalous": 1,
            }
        )

    return logs

# Save logs to a CSV file
def save_logs_to_csv(logs, file_name):
    with open(file_name, "w", newline="") as csvfile:
        fieldnames = ["UserID", "Role", "Timestamp", "HTTP_Method", "Endpoint", "Anomalous"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)

# Main function to generate logs
def main():
    num_normal = 2000  # Increased log volume
    num_anomalous = 1000

    logs = generate_augmented_logs(num_normal, num_anomalous)
    save_logs_to_csv(logs, "data/enhanced_logs.csv")
    print("Enhanced logs generated and saved to 'data/enhanced_logs.csv'")

if __name__ == "__main__":
    main()
