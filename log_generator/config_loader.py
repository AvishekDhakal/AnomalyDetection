import json
import logging
import sys

def load_config(config_file):
    """
    Loads and validates configuration parameters from a JSON file.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON configuration: {e}")
        sys.exit(1)
    
    # Basic validation can be added here
    required_keys = [
        "ROLE_WEIGHTS_NORMAL", "NORMAL_METHODS", "HTTP_RESPONSE_CODES",
        "ANOMALY_SCENARIOS", "ENDPOINTS",
        "ANOMALOUS_ENDPOINTS", "PARAMETERS", "ITEM_OPERATION_TRACKING", "ROLES",
        "IP_ADDRESS_GENERATION"  # Ensure this key is included
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logging.error(f"Missing required configuration keys: {missing_keys}")
        sys.exit(1)
    
    return config
