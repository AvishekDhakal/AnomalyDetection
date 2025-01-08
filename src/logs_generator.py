import random
import csv
import datetime

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
            "IP_Address": ip_address,
            "Anomalous": 0,
        })
        log_id += 1

    # Generate anomalous logs
    for _ in range(num_anomalous):
        role = random.choice(list(roles.keys()))
        user_id = random.choice(roles[role])
        ip_address = f"{network_subnet}.{random.randint(1, 254)}"  # Same network subnet
        if random.random() < 0.1:  # 10% chance of IP outside the subnet
            ip_address = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"

        timestamp = datetime.datetime.now() + datetime.timedelta(
            days=random.randint(-extended_days, 0),
            hours=random.randint(0, 23),  # Any time of the day
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        http_method = random.choice(http_methods)

        # Anomaly types
        if role in ["Doctor", "Nurse"]:
            endpoint = random.choice(["/admin/settings", "/admin/credentials"])  # Unauthorized access
            parameter = f"?admin_id={random.randint(3000, 4000)}"
        elif role == "Staff":
            endpoint = random.choice(["/inventory/items", "/billing/invoices"])
            parameter = ""
            if random.choice(["PUT", "DELETE"]) == http_method:  # Invalid operations
                parameter = f"?item_id={random.randint(4000, 5000)}"
        elif role == "Admin":
            endpoint = random.choice(["/patient/records", "/inventory/items"])  # Unauthorized access
            parameter = f"?patient_id={random.randint(1000, 2000)}"

        logs.append({
            "LogID": log_id,
            "UserID": user_id,
            "Role": role,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "HTTP_Method": http_method,
            "Endpoint": endpoint + parameter,
            "IP_Address": ip_address,
            "Anomalous": 1,
        })
        log_id += 1

    return logs

# Save logs to a CSV file
def save_logs_to_csv(logs, file_name):
    with open(file_name, "w", newline="") as csvfile:
        fieldnames = ["LogID", "UserID", "Role", "Timestamp", "HTTP_Method", "Endpoint", "IP_Address", "Anomalous"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)

# Main function to generate logs
def main():
    num_normal = 5000  # Increased log volume
    num_anomalous = 2000

    logs = generate_logs(num_normal, num_anomalous)
    save_logs_to_csv(logs, "data/enhanced_logs_final.csv")
    print("Enhanced logs generated and saved to 'enhanced_logs_final.csv'")

if __name__ == "__main__":
    main()
