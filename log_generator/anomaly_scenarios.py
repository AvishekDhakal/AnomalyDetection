# anomaly_scenarios.py

import random
import uuid
import datetime
import ipaddress
from utils import generate_ip_address, generate_public_ip, build_parameters
import logging
import sys

def generate_multiple_failed_attempts(user_id, role, network_subnet, config, extended_days):
    """
    Generates multiple DELETE operations on /login/attempts to simulate failed login attempts.
    """
    logs = []
    num_attempts = random.randint(3, 10)
    base_timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    
    for i in range(num_attempts):
        timestamp = base_timestamp + datetime.timedelta(seconds=i * 10)  # 10 seconds apart
        ip_address = generate_ip_address(is_anomalous=True, network_subnet=network_subnet, config=config)
        
        http_method = "DELETE"
        endpoint = "/login/attempts"
        parameter = build_parameters(endpoint, config)
        endpoint_full = endpoint + parameter
        
        http_response = random.choice(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"])
        
        log_entry = {
            "LogID": str(uuid.uuid4()),
            "UserID": user_id,
            "Role": role,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "HTTP_Method": http_method,
            "Endpoint": endpoint_full,
            "IP_Address": ip_address,
            "HTTP_Response": http_response,
            "Anomalous": 1
        }
        
        logs.append(log_entry)
    
    return logs

def generate_rapid_inventory_changes(user_id, role, network_subnet, config, extended_days):
    """
    Generates rapid PUT/DELETE operations on the SAME item
    """
    logs = []
    endpoint = "/inventory/items"
    
    # Generate single item ID for all operations
    item_id = str(random.randint(*config["PARAMETERS"][endpoint]["id_range"]))
    
    base_timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        minutes=random.randint(0, 59)
    )
    
    # Same IP for all operations
    ip_address = generate_ip_address(is_anomalous=True, network_subnet=network_subnet, config=config)
    
    operations = ["PUT", "DELETE", "PUT"]
    for i, method in enumerate(operations):
        timestamp = base_timestamp + datetime.timedelta(seconds=i*10)
        params = build_parameters(endpoint, config, fixed_id=item_id)  # Fixed ID
        
        logs.append({
            "LogID": str(uuid.uuid4()),
            "UserID": user_id,
            "Role": role,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "HTTP_Method": method,
            "Endpoint": f"{endpoint}{params}",
            "IP_Address": ip_address,
            "HTTP_Response": random.choice(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"]),
            "Anomalous": 1
        })
    
    return logs

def generate_chain_of_anomalies(user_id, role, network_subnet, config, extended_days):
    """
    Generates chained anomalies with consistent target ID and IP
    """
    logs = []
    base_timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days)
    )
    
    # Shared parameters for the chain
    target_id = str(random.randint(1000, 9999))
    ip_address = generate_ip_address(is_anomalous=True,  # Add this line
                                    network_subnet=network_subnet, 
                                    config=config)
    
    
    # Phase 1: Initial compromise
    sql_ep = random.choice(config["ANOMALOUS_ENDPOINTS"]["sql_injection"])
    sql_params = build_parameters(sql_ep[0], config, fixed_id=target_id, payload="' OR 1=1 --")
    
    logs.append({
        "LogID": str(uuid.uuid4()),
        "UserID": user_id,
        "Role": role,
        "Timestamp": base_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": sql_ep[1],
        "Endpoint": f"{sql_ep[0]}{sql_params}",
        "IP_Address": ip_address,
        "HTTP_Response": 500,
        "Anomalous": 1
    })
    
    # Phase 2: Follow-up actions
    for exfil_ep in config["ANOMALOUS_ENDPOINTS"]["data_exfiltration"]:
        timestamp = base_timestamp + datetime.timedelta(minutes=5)
        exfil_params = build_parameters(exfil_ep[0], config, fixed_id=target_id)
        
        logs.append({
            "LogID": str(uuid.uuid4()),
            "UserID": user_id,
            "Role": role,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "HTTP_Method": exfil_ep[1],
            "Endpoint": f"{exfil_ep[0]}{exfil_params}",
            "IP_Address": ip_address,
            "HTTP_Response": 200 if exfil_ep[1] == "GET" else 500,
            "Anomalous": 1
        })
    
    return logs

def generate_sql_injection(user_id, role, network_subnet, config, extended_days):
    """
    Generates SQL Injection attempts on specified endpoints.
    """
    logs = []
    sql_endpoints = config["ANOMALOUS_ENDPOINTS"]["sql_injection"]
    num_injections = random.randint(1, len(sql_endpoints))
    selected_endpoints = random.sample(sql_endpoints, num_injections)
    
    for endpoint in selected_endpoints:
        timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, extended_days),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        ip_address = generate_ip_address(is_anomalous=True, network_subnet=network_subnet, config=config)
        
        log_entry = {
            "LogID": str(uuid.uuid4()),
            "UserID": user_id,
            "Role": role,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "HTTP_Method": endpoint[1],
            "Endpoint": endpoint[0] + build_parameters(endpoint[0], config),
            "IP_Address": ip_address,
            "HTTP_Response": random.choice(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"]),
            "Anomalous": 1
        }
        
        logs.append(log_entry)
    
    return logs

def generate_xss(user_id, role, network_subnet, config, extended_days):
    """
    Generates XSS attempts on specified endpoints.
    """
    logs = []
    xss_endpoints = config["ANOMALOUS_ENDPOINTS"]["xss"]
    num_xss = random.randint(1, len(xss_endpoints))
    selected_endpoints = random.sample(xss_endpoints, num_xss)
    
    for endpoint in selected_endpoints:
        timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, extended_days),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        ip_address = generate_ip_address(is_anomalous=True, network_subnet=network_subnet, config=config)
        
        log_entry = {
            "LogID": str(uuid.uuid4()),
            "UserID": user_id,
            "Role": role,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "HTTP_Method": endpoint[1],
            "Endpoint": endpoint[0] + build_parameters(endpoint[0], config),
            "IP_Address": ip_address,
            "HTTP_Response": random.choice(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"]),
            "Anomalous": 1
        }
        
        logs.append(log_entry)
    
    return logs

def generate_external_ip_only(user_id, role, network_subnet, config, extended_days):
    """
    Generates logs where only the IP address is external (anomalous).
    All other attributes are generated normally.
    """
    logs = []
    anomalous_endpoints = config["ANOMALOUS_ENDPOINTS"]["external_ip_only"]
    selected_endpoint = random.choice(anomalous_endpoints)
    
    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    
    # Generate an external IP address
    ip_address = generate_public_ip()
    
    http_method = selected_endpoint[1]
    endpoint = selected_endpoint[0]
    parameter = build_parameters(endpoint, config)
    endpoint_full = endpoint + parameter
    
    http_response = random.choice(config["HTTP_RESPONSE_CODES"]["normal"]["codes"])
    
    log_entry = {
        "LogID": str(uuid.uuid4()),
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": http_method,
        "Endpoint": endpoint_full,
        "IP_Address": ip_address,
        "HTTP_Response": http_response,
        "Anomalous": 1
    }
    
    logs.append(log_entry)
    
    return logs

def generate_unusual_time_access(user_id, role, network_subnet, config, extended_days):
    """
    Generates access to sensitive endpoints during unusual hours.
    """
    logs = []
    anomalous_endpoints = config["ANOMALOUS_ENDPOINTS"]["unusual_ip_access"]
    selected_endpoint = random.choice(anomalous_endpoints)
    
    # Define unusual hours (e.g., 0-5 and 21-23)
    hour = random.choice(list(range(0, 6)) + list(range(21, 24)))
    
    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    
    # Adjust the hour to be within unusual hours
    timestamp = timestamp.replace(hour=hour)
    
    ip_address = generate_ip_address(is_anomalous=False, network_subnet=network_subnet, config=config)
    
    http_method = selected_endpoint[1]
    endpoint = selected_endpoint[0]
    parameter = build_parameters(endpoint, config)
    endpoint_full = endpoint + parameter
    
    http_response = random.choice(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"])
    
    log_entry = {
        "LogID": str(uuid.uuid4()),
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": http_method,
        "Endpoint": endpoint_full,
        "IP_Address": ip_address,
        "HTTP_Response": http_response,
        "Anomalous": 1
    }
    
    logs.append(log_entry)
    
    return logs

def generate_delete_patient_records(user_id, role, network_subnet, config, extended_days):
    """
    Generates a DELETE operation on /patient/records, which is anomalous for Doctors and Nurses.
    """
    logs = []
    endpoint = "/patient/records"
    http_method = "DELETE"
    
    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    
    ip_address = generate_ip_address(is_anomalous=True, network_subnet=network_subnet, config=config)
    
    parameter = build_parameters(endpoint, config)
    endpoint_full = endpoint + parameter
    
    http_response = random.choice(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"])
    
    log_entry = {
        "LogID": str(uuid.uuid4()),
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": http_method,
        "Endpoint": endpoint_full,
        "IP_Address": ip_address,
        "HTTP_Response": http_response,
        "Anomalous": 1
    }
    
    logs.append(log_entry)
    
    return logs

def generate_delete_billing_invoices(user_id, role, network_subnet, config, extended_days):
    """
    Generates a DELETE operation on /billing/invoices, which is anomalous for Doctors and Nurses.
    """
    logs = []
    endpoint = "/billing/invoices"
    http_method = "DELETE"
    
    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    
    ip_address = generate_ip_address(is_anomalous=True, network_subnet=network_subnet, config=config)
    
    parameter = build_parameters(endpoint, config)
    endpoint_full = endpoint + parameter
    
    http_response = random.choice(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"])
    
    log_entry = {
        "LogID": str(uuid.uuid4()),
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": http_method,
        "Endpoint": endpoint_full,
        "IP_Address": ip_address,
        "HTTP_Response": http_response,
        "Anomalous": 1
    }
    
    logs.append(log_entry)
    
    return logs

def generate_patient_records_put_del(user_id, role, network_subnet, config, extended_days):
    """
    Generates PUT and DELETE operations on /patient/records.
    """
    logs = []
    endpoints = config["ANOMALOUS_ENDPOINTS"]["patient_records_put_del"]
    
    for endpoint in endpoints:
        http_method = endpoint[1]
        endpoint_full = endpoint[0] + build_parameters(endpoint[0], config)
        
        timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, extended_days),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        is_anomalous = 1 if http_method == "DELETE" else 0  # DELETE is anomalous
        
        ip_address = generate_ip_address(is_anomalous=is_anomalous, network_subnet=network_subnet, config=config)
        
        http_response = random.choice(
            config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"] if is_anomalous else config["HTTP_RESPONSE_CODES"]["normal"]["codes"]
        )
        
        log_entry = {
            "LogID": str(uuid.uuid4()),
            "UserID": user_id,
            "Role": role,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "HTTP_Method": http_method,
            "Endpoint": endpoint_full,
            "IP_Address": ip_address,
            "HTTP_Response": http_response,
            "Anomalous": is_anomalous
        }
        
        logs.append(log_entry)
    
    return logs


def generate_access_restricted_endpoints(user_id, role, network_subnet, config, extended_days):
    """
    Generates access attempts to restricted endpoints by Staff.
    """
    logs = []
    restricted_endpoints = config["ANOMALOUS_ENDPOINTS"]["access_restricted_endpoints"]
    selected_endpoint = random.choice(restricted_endpoints)
    
    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    
    ip_address = generate_ip_address(is_anomalous=True, network_subnet=network_subnet, config=config)
    
    http_method = selected_endpoint[1]
    endpoint = selected_endpoint[0]
    parameter = build_parameters(endpoint, config)
    endpoint_full = endpoint + parameter
    
    http_response = random.choice(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"])
    
    log_entry = {
        "LogID": str(uuid.uuid4()),
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": http_method,
        "Endpoint": endpoint_full,
        "IP_Address": ip_address,
        "HTTP_Response": http_response,
        "Anomalous": 1
    }
    
    logs.append(log_entry)
    
    return logs

def generate_access_patient_invoices(user_id, role, network_subnet, config, extended_days):
    """
    Generates access to patient invoices by Staff. Considered normal but monitored.
    """
    logs = []
    endpoints = config["ANOMALOUS_ENDPOINTS"]["access_patient_invoices"]
    
    for endpoint in endpoints:
        http_method = endpoint[1]
        endpoint_full = endpoint[0] + build_parameters(endpoint[0], config)
        
        timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, extended_days),
            hours=random.randint(6, 20),  # Normal operational hours
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Accessing patient invoices is normal, but we monitor for any anomalies
        is_anomalous = 0  # Initially normal
        
        # Introduce a small chance of anomaly if needed
        if random.random() < 0.05:
            is_anomalous = 1
            http_response = random.choice(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"])
        else:
            http_response = random.choice(config["HTTP_RESPONSE_CODES"]["normal"]["codes"])
        
        ip_address = generate_ip_address(is_anomalous=is_anomalous, network_subnet=network_subnet, config=config)
        
        log_entry = {
            "LogID": str(uuid.uuid4()),
            "UserID": user_id,
            "Role": role,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "HTTP_Method": http_method,
            "Endpoint": endpoint_full,
            "IP_Address": ip_address,
            "HTTP_Response": http_response,
            "Anomalous": is_anomalous
        }
        
        logs.append(log_entry)
    
    return logs

def generate_delete_login_attempts(user_id, role, network_subnet, config, extended_days):
    """
    Generates DELETE operations on /login/attempts by Admins.
    """
    logs = []
    endpoint = "/login/attempts"
    http_method = "DELETE"
    
    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    
    ip_address = generate_ip_address(is_anomalous=True, network_subnet=network_subnet, config=config)
    
    parameter = build_parameters(endpoint, config)
    endpoint_full = endpoint + parameter
    
    http_response = random.choice(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"])
    
    log_entry = {
        "LogID": str(uuid.uuid4()),
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": http_method,
        "Endpoint": endpoint_full,
        "IP_Address": ip_address,
        "HTTP_Response": http_response,
        "Anomalous": 1
    }
    
    logs.append(log_entry)
    
    return logs

def generate_delete_admin_logs(user_id, role, network_subnet, config, extended_days):
    """
    Generates DELETE operations on /admin/logs by Admins.
    """
    logs = []
    endpoint = "/admin/logs"
    http_method = "DELETE"
    
    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    
    ip_address = generate_ip_address(is_anomalous=True, network_subnet=network_subnet, config=config)
    
    parameter = build_parameters(endpoint, config)
    endpoint_full = endpoint + parameter
    
    http_response = random.choice(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"])
    
    log_entry = {
        "LogID": str(uuid.uuid4()),
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": http_method,
        "Endpoint": endpoint_full,
        "IP_Address": ip_address,
        "HTTP_Response": http_response,
        "Anomalous": 1
    }
    
    logs.append(log_entry)
    
    return logs

def generate_access_patient_records(user_id, role, network_subnet, config, extended_days):
    """
    Generates access to /patient/records by Admins.
    """
    logs = []
    endpoints = config["ANOMALOUS_ENDPOINTS"]["access_patient_records"]
    
    for endpoint in endpoints:
        http_method = endpoint[1]
        endpoint_full = endpoint[0] + build_parameters(endpoint[0], config)
        
        timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, extended_days),
            hours=random.randint(6, 20),  # Normal operational hours
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Accessing patient records is monitored and potentially anomalous
        is_anomalous = 1  # Flag as anomalous since Admins typically shouldn't access patient records
        
        http_response = random.choice(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"])
        
        ip_address = generate_ip_address(is_anomalous=is_anomalous, network_subnet=network_subnet, config=config)
        
        log_entry = {
            "LogID": str(uuid.uuid4()),
            "UserID": user_id,
            "Role": role,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "HTTP_Method": http_method,
            "Endpoint": endpoint_full,
            "IP_Address": ip_address,
            "HTTP_Response": http_response,
            "Anomalous": is_anomalous
        }
        
        logs.append(log_entry)
    
    return logs

def generate_admin_suspicious(user_id, role, network_subnet, config, extended_days):
    """
    Generates suspicious activities related to admin settings and credentials.
    """
    logs = []
    anomalous_endpoints = config["ANOMALOUS_ENDPOINTS"]["admin_suspicious"]
    selected_endpoint = random.choice(anomalous_endpoints)
    
    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        hours=random.randint(6, 20),  # Normal operational hours
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    
    ip_address = generate_ip_address(is_anomalous=True, network_subnet=network_subnet, config=config)
    
    http_method = selected_endpoint[1]
    endpoint = selected_endpoint[0]
    parameter = build_parameters(endpoint, config)
    endpoint_full = endpoint + parameter
    
    http_response = random.choice(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"])
    
    log_entry = {
        "LogID": str(uuid.uuid4()),
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": http_method,
        "Endpoint": endpoint_full,
        "IP_Address": ip_address,
        "HTTP_Response": http_response,
        "Anomalous": 1
    }
    
    logs.append(log_entry)
    
    return logs

def generate_privilege_escalation(user_id, role, network_subnet, config, extended_days):
    """
    Generates attempts to escalate privileges or access higher-level functionalities.
    """
    logs = []
    anomalous_endpoints = config["ANOMALOUS_ENDPOINTS"]["privilege_escalation"]
    selected_endpoint = random.choice(anomalous_endpoints)
    
    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    
    ip_address = generate_ip_address(is_anomalous=True, network_subnet=network_subnet, config=config)
    
    http_method = selected_endpoint[1]
    endpoint = selected_endpoint[0]
    parameter = build_parameters(endpoint, config)
    endpoint_full = endpoint + parameter
    
    http_response = random.choice(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"])
    
    log_entry = {
        "LogID": str(uuid.uuid4()),
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": http_method,
        "Endpoint": endpoint_full,
        "IP_Address": ip_address,
        "HTTP_Response": http_response,
        "Anomalous": 1
    }
    
    logs.append(log_entry)
    
    return logs

def generate_unauthorized_endpoint_access(user_id, role, network_subnet, config, extended_days):
    """
    Generates attempts to access endpoints beyond typical Admin operations.
    """
    logs = []
    anomalous_endpoints = config["ANOMALOUS_ENDPOINTS"]["unauthorized_endpoint_access"]
    selected_endpoint = random.choice(anomalous_endpoints)
    
    timestamp = datetime.datetime.now() - datetime.timedelta(
        days=random.randint(0, extended_days),
        hours=random.randint(6, 20),  # Normal operational hours
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    
    ip_address = generate_ip_address(is_anomalous=True, network_subnet=network_subnet, config=config)
    
    http_method = selected_endpoint[1]
    endpoint = selected_endpoint[0]
    parameter = build_parameters(endpoint, config)
    endpoint_full = endpoint + parameter
    
    http_response = random.choice(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"])
    
    log_entry = {
        "LogID": str(uuid.uuid4()),
        "UserID": user_id,
        "Role": role,
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "HTTP_Method": http_method,
        "Endpoint": endpoint_full,
        "IP_Address": ip_address,
        "HTTP_Response": http_response,
        "Anomalous": 1
    }
    
    logs.append(log_entry)
    
    return logs

# Additional anomaly generation functions can be implemented similarly based on the config.json
