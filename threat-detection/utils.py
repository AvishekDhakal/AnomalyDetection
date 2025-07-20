# utils.py

import random
import ipaddress
import re

def generate_public_ip():
    """
    Generates a random public IP address, ensuring it's not private or reserved.
    """
    while True:
        ip_int = random.randint(1, (2**32) - 1)
        ip = ipaddress.IPv4Address(ip_int)
        if not (ip.is_private or ip.is_reserved or ip.is_multicast or ip.is_loopback or ip.is_unspecified):
            return str(ip)

def generate_ip_address(is_anomalous, network_subnet, config):
    """
    Generates an IP address based on whether the log is anomalous.
    - Anomalous: Higher probability of public IP
    - Normal: Lower probability of public IP
    """
    ip_config = config.get("IP_ADDRESS_GENERATION", {})
    anomalous_prob = ip_config.get("anomalous_external_ip_probability", 0.3)
    normal_prob = ip_config.get("normal_external_ip_probability", 0.01)
    
    if is_anomalous:
        if random.random() < anomalous_prob:
            return generate_public_ip()
        else:
            return f"{network_subnet}.{random.randint(1, 254)}"
    else:
        if random.random() < normal_prob:
            return generate_public_ip()
        else:
            return f"{network_subnet}.{random.randint(1, 254)}"


def extract_item_id(endpoint_full):
    """
    Extracts the item_id from the endpoint using regex.
    Returns the item_id as an integer if found, else None.
    """
    match = re.search(r"/inventory/items/(\d+)", endpoint_full)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            print(f"Warning: Invalid item_id extracted: {match.group(1)}")
            return None
    return None

def build_parameters(endpoint_base, config, is_anomalous=False, fixed_id=None, payload=None):
    """
    Build parameters with optional fixed ID and payload injection
    """
    params = config["PARAMETERS"].get(endpoint_base, {})
    param_str = ""
    
    # Use fixed ID if provided, else random
    resource_id = fixed_id or str(random.randint(*params["id_range"])) if "id_range" in params else ""
    
    if resource_id:
        param_str += f"/{resource_id}"
    
    # Add parameters
    if "export" in params:
        param_str += f"?export=true&limit={params['limit']}"
    
    # Inject payload for anomalies
    if payload:
        param_str += f"&query={payload}" if "?" in param_str else f"?query={payload}"
    
    return param_str



def validate_anomaly_consistency(logs):
    """
    Validates consistency of IDs and IPs in anomaly sequences
    with enhanced error handling
    """
    if not logs:
        return

    try:
        # Get first log's IP as reference
        ref_ip = logs[0]["IP_Address"]
        
        # Check for resource ID consistency only if endpoints contain IDs
        ref_id = None
        if any('/items/' in log['Endpoint'] or '/records/' in log['Endpoint'] for log in logs):
            first_with_id = next(log for log in logs if '/items/' in log['Endpoint'] or '/records/' in log['Endpoint'])
            ref_id = first_with_id['Endpoint'].split("/")[3].split("?")[0]

        for log in logs:
            # Check IP consistency
            if log["IP_Address"] != ref_ip:
                raise ValueError(f"IP mismatch in chain: {ref_ip} vs {log['IP_Address']}")
            
            # Check ID consistency if applicable
            if ref_id:
                current_endpoint = log["Endpoint"]
                if '/items/' in current_endpoint or '/records/' in current_endpoint:
                    current_id = current_endpoint.split("/")[3].split("?")[0]
                    if current_id != ref_id:
                        raise ValueError(f"ID mismatch: {ref_id} vs {current_id}")

    except (IndexError, StopIteration):
        # Skip validation if no ID-based endpoints
        pass