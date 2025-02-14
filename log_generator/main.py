# main.py

import argparse
import logging
from config_loader import load_config
from log_generator import LogGenerator

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
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="Overwrite existing CSV files instead of appending"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    return parser.parse_args()

def main():
    """
    Main entry point for generating logs.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    args = parse_arguments()
    total_logs = args.total_logs
    anomaly_ratio = args.anomaly_ratio
    master_file = args.master_file
    inference_file = args.inference_file
    network_subnet = args.network_subnet
    extended_days = args.extended_days
    config_file = args.config_file
    overwrite = args.overwrite
    seed = args.seed

    # Set random seed if provided
    if seed is not None:
        import random
        random.seed(seed)
        logging.info(f"Random seed set to {seed}")
    
    # Load configuration
    config = load_config(config_file)
    
    # Initialize LogGenerator
    generator = LogGenerator(config, network_subnet, extended_days)
    
    logging.info(f"Starting log generation with the following parameters:")
    logging.info(f"Total Logs: {total_logs}")
    logging.info(f"Anomaly Ratio: {anomaly_ratio * 100}%")
    logging.info(f"Master File: {master_file}")
    logging.info(f"Inference File: {inference_file}")
    logging.info(f"Network Subnet: {network_subnet}.x")
    logging.info(f"Extended Days: {extended_days}")
    logging.info(f"Configuration File: {config_file}")
    logging.info(f"Overwrite Mode: {'Enabled' if overwrite else 'Disabled'}\n")
    
    # Generate logs
    logs = generator.generate_logs(total_logs, anomaly_ratio)
    
    total_anomalies = sum(log["Anomalous"] for log in logs)
    logging.info(f"Generated {len(logs)} logs. Anomalies: {total_anomalies}")
    
    # Validate logs
    generator.validate_logs(logs, anomaly_ratio)
    
    # Save logs to CSV
    generator.save_logs_to_csv(logs, master_file, inference_file, overwrite=overwrite)
    logging.info(f"Logs saved to '{master_file}' and '{inference_file}'.")
    logging.info("Log generation completed successfully.")

if __name__ == "__main__":
    main()




# python3 main.py --total_logs 10000 --anomaly_ratio 0.0005 --overwrite
# will give just 3