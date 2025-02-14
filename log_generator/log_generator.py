# log_generator.py

import random
import uuid
import datetime
from collections import defaultdict
from utils import generate_ip_address, extract_item_id, build_parameters
from utils import validate_anomaly_consistency
from anomaly_scenarios import (
    generate_multiple_failed_attempts,
    generate_chain_of_anomalies,
    generate_sql_injection,
    generate_xss,
    generate_external_ip_only,
    generate_unusual_time_access,
    generate_delete_patient_records,
    generate_delete_billing_invoices,
    generate_patient_records_put_del,
    generate_rapid_inventory_changes,
    generate_access_restricted_endpoints,
    generate_access_patient_invoices,
    generate_delete_login_attempts,
    generate_delete_admin_logs,
    generate_access_patient_records,
    generate_admin_suspicious,
    generate_privilege_escalation,
    generate_unauthorized_endpoint_access
)
import ipaddress
import math

class LogGenerator:
    def __init__(self, config, network_subnet, extended_days):
        self.config = config
        self.network_subnet = network_subnet
        self.extended_days = extended_days
        self.item_operation_tracker = defaultdict(lambda: {'PUT': 0, 'DELETE': 0})
        self.roles = config["ROLES"]
        self.endpoints = config["ENDPOINTS"]
        self.average_logs_per_scenario = self.calculate_average_logs_per_scenario()
        print(f"Initialized LogGenerator with average_logs_per_scenario: {self.average_logs_per_scenario}")

    def calculate_average_logs_per_scenario(self):
        """
        Calculates the average number of logs generated per anomaly scenario.
        This considers scenario weights, role-based variations, and the actual number of logs each scenario generates.
        """
        scenario_weights = self.config["ANOMALY_SCENARIOS"]
        # Estimate logs per scenario
        logs_per_scenario = {
            "multiple_failed_attempts": 5,
            "chain_of_anomalies": 3,
            "sql_injection": 2,
            "xss": 2,
            "external_ip_only": 1,
            "unusual_time_access": 1,
            "delete_patient_records": 1,
            "delete_billing_invoices": 1,
            "patient_records_put_del": 2,
            "rapid_inventory_changes": 3,
            "access_restricted_endpoints": 1,
            "access_patient_invoices": 1,
            "delete_login_attempts": 1,
            "delete_admin_logs": 1,
            "access_patient_records": 2,
            "admin_suspicious": 2,
            "privilege_escalation": 2,
            "unauthorized_endpoint_access": 2
        }

        total_weighted_logs = 0
        total_weights = 0
        for role, scenarios in scenario_weights.items():
            for scenario, weight in scenarios.items():
                log_count = logs_per_scenario.get(scenario, 1)  # Default to 1 log if undefined
                total_weighted_logs += weight * log_count
                total_weights += weight

        if total_weights == 0:
            print("Warning: Total weights for anomaly scenarios is zero. Defaulting to 1.")
            return 1

        average_logs_per_scenario = total_weighted_logs / total_weights
        print(f"Calculated precise average logs per anomaly scenario: {average_logs_per_scenario}")
        return average_logs_per_scenario

    def weighted_hour_choice(self, is_after_hours=False):
        """
        Returns an hour (0-23) for normal logs based on a weighted distribution,
        simulating a busier daytime environment but still some overnight activity.
        If is_after_hours is True, selects from after-hours.
        """
        if is_after_hours:
            possible_hours = list(range(21, 24)) + list(range(0, 6))
            weights = [1] * len(possible_hours)
            chosen_hour = random.choices(possible_hours, weights=weights, k=1)[0]
            print(f"Selected after-hours hour: {chosen_hour}")
            return chosen_hour
        else:
            hours = list(range(24))  # 0 through 23
            weights = [
                1, 1, 1, 1, 1, 1, 1,  # 0-6
                8, 8, 8, 8, 8, 8, 8,  # 7-13
                6, 6, 6, 6, 6,        # 14-18
                3, 3, 3, 3, 3         # 19-23
            ]
            chosen_hour = random.choices(hours, weights=weights, k=1)[0]
            # print(f"Selected normal hour: {chosen_hour}")
            return chosen_hour

    def generate_http_response_code(self, is_anomalous=False):
        """
        Returns an HTTP response code with different distributions
        for normal vs. anomalous logs.
        """
        if not is_anomalous:
            code = random.choices(
                self.config["HTTP_RESPONSE_CODES"]["normal"]["codes"],
                weights=self.config["HTTP_RESPONSE_CODES"]["normal"]["weights"],
                k=1
            )[0]
            # print(f"Generated normal HTTP response code: {code}")
            return code
        else:
            code = random.choices(
                self.config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"],
                weights=self.config["HTTP_RESPONSE_CODES"]["anomalous"]["weights"],
                k=1
            )[0]
            # print(f"Generated anomalous HTTP response code: {code}")
            return code

    def pick_anomaly_scenario(self, role):
        """
        Returns a scenario label that decides which endpoint + method to use for anomalies.
        """
        scenarios = self.config["ANOMALY_SCENARIOS"].get(role, {})
        if not scenarios:
            print(f"No anomaly scenarios defined for role {role}. Defaulting to 'unknown'.")
            return "unknown"
        labels = list(scenarios.keys())
        weights = list(scenarios.values())
        chosen_scenario = random.choices(labels, weights=weights, k=1)[0]
        # print(f"Picked anomaly scenario '{chosen_scenario}' for role '{role}'.")
        return chosen_scenario
    @staticmethod
    def is_anomalous_combo(endpoint, method, config):
    # Scan all anomaly combos in ANOMALOUS_ENDPOINTS
        for scenario, combo_list in config["ANOMALOUS_ENDPOINTS"].items():
            for (anom_endpoint, anom_method) in combo_list:
                if anom_endpoint == endpoint and anom_method == method:
                    return True
        return False

    # def generate_normal_log(self):
    #     """
    #     Generates a single normal log entry with 24-hour weighting
    #     and weighted role selection.
    #     """
    #     # Weighted role selection
    #     role_choices = list(self.config["ROLE_WEIGHTS_NORMAL"].keys())
    #     weights_list = list(self.config["ROLE_WEIGHTS_NORMAL"].values())
    #     role = random.choices(role_choices, weights=weights_list, k=1)[0]
    #     user_id = random.choice(self.roles[role])

    #     # Weighted hour for normal
    #     chosen_hour = self.weighted_hour_choice()
    #     timestamp = datetime.datetime.now() - datetime.timedelta(
    #         days=random.randint(0, self.extended_days),
    #         hours=chosen_hour,
    #         minutes=random.randint(0, 59),
    #         seconds=random.randint(0, 59),
    #     )

    #     # Normal logs: IP assigned based on probability
    #     ip_address = generate_ip_address(is_anomalous=False, network_subnet=self.network_subnet, config=self.config)

    #     # Weighted HTTP method distribution for normal logs
    #     http_method = random.choices(
    #         self.config["NORMAL_METHODS"]["methods"],
    #         weights=self.config["NORMAL_METHODS"]["weights"],
    #         k=1
    #     )[0]

    #     endpoint = random.choice(self.endpoints[role])

    #     # Add parameters based on endpoint with hierarchical structure
    #     parameter = build_parameters(endpoint, self.config)
    #     endpoint_full = endpoint + parameter

    #     # Select a normal HTTP response code
    #     http_response = self.generate_http_response_code(is_anomalous=False)

    #     log_entry = {
    #         "LogID": str(uuid.uuid4()),
    #         "UserID": user_id,
    #         "Role": role,
    #         "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
    #         "HTTP_Method": http_method,
    #         "Endpoint": endpoint_full,
    #         "IP_Address": ip_address,
    #         "HTTP_Response": http_response,
    #         "Anomalous": 0,
    #     }

    #     # print(f"Generated normal log: {log_entry}")
    #     return log_entry
    def generate_normal_log(self):
        # Weighted role selection
        role_choices = list(self.config["ROLE_WEIGHTS_NORMAL"].keys())
        weights_list = list(self.config["ROLE_WEIGHTS_NORMAL"].values())
        role = random.choices(role_choices, weights=weights_list, k=1)[0]
        user_id = random.choice(self.roles[role])

        # Weighted hour for normal
        chosen_hour = self.weighted_hour_choice()
        timestamp = datetime.datetime.now() - datetime.timedelta(
            days=random.randint(0, self.extended_days),
            hours=chosen_hour,
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )

        # Normal IP generation
        ip_address = generate_ip_address(is_anomalous=False, network_subnet=self.network_subnet, config=self.config)

        # Weighted HTTP method
        http_method = random.choices(
            self.config["NORMAL_METHODS"]["methods"],
            weights=self.config["NORMAL_METHODS"]["weights"],
            k=1
        )[0]

        # Pick an endpoint from the role
        endpoint = random.choice(self.endpoints[role])

        # Check if (endpoint, method) is anomalous
        if self.is_anomalous_combo(endpoint, http_method, self.config):
            # Option (a): re-roll to avoid anomaly
            return self.generate_normal_log()

            # OR Option (b): forcibly mark as anomaly:
            # is_anomalous = 1
            # ip_address = generate_ip_address(is_anomalous=True, ...)
            # ... then build the log entry with "Anomalous": 1
            # return log_entry

        # If we get here, it's safe to label as normal
        parameter = build_parameters(endpoint, self.config)
        endpoint_full = endpoint + parameter

        http_response = self.generate_http_response_code(is_anomalous=False)

        log_entry = {
            "LogID": str(uuid.uuid4()),
            "UserID": user_id,
            "Role": role,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "HTTP_Method": http_method,
            "Endpoint": endpoint_full,
            "IP_Address": ip_address,
            "HTTP_Response": http_response,
            "Anomalous": 0
        }
        return log_entry

    def generate_anomalous_log(self):
        """
        Generates anomalous logs based on the selected scenario.
        Returns a list of log entries.
        """
        logs = []
        role = random.choice(list(self.roles.keys()))
        user_id = random.choice(self.roles[role])

        scenario_label = self.pick_anomaly_scenario(role)

        if scenario_label == "multiple_failed_attempts":
            anomaly_logs = generate_multiple_failed_attempts(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "chain_of_anomalies":
            anomaly_logs = generate_chain_of_anomalies(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "sql_injection":
            anomaly_logs = generate_sql_injection(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "xss":
            anomaly_logs = generate_xss(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "external_ip_only":
            anomaly_logs = generate_external_ip_only(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "unusual_time_access":
            anomaly_logs = generate_unusual_time_access(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "delete_patient_records":
            anomaly_logs = generate_delete_patient_records(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "delete_billing_invoices":
            anomaly_logs = generate_delete_billing_invoices(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "patient_records_put_del":
            anomaly_logs = generate_patient_records_put_del(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "rapid_inventory_changes":
            anomaly_logs = generate_rapid_inventory_changes(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "access_restricted_endpoints":
            anomaly_logs = generate_access_restricted_endpoints(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "access_patient_invoices":
            anomaly_logs = generate_access_patient_invoices(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "delete_login_attempts":
            anomaly_logs = generate_delete_login_attempts(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "delete_admin_logs":
            anomaly_logs = generate_delete_admin_logs(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "access_patient_records":
            anomaly_logs = generate_access_patient_records(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "admin_suspicious":
            anomaly_logs = generate_admin_suspicious(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "privilege_escalation":
            anomaly_logs = generate_privilege_escalation(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        elif scenario_label == "unauthorized_endpoint_access":
            anomaly_logs = generate_unauthorized_endpoint_access(
                user_id, role, self.network_subnet, self.config, self.extended_days
            )
            logs.extend(anomaly_logs)
        else:
            # Handle unknown scenarios if necessary
            pass

        return logs

    def generate_logs(self, total_logs, anomaly_ratio):
        """
        Generates normal and anomalous logs, ensuring the total number of logs matches `total_logs`.
        Dynamically adjusts anomaly and normal logs to meet the desired anomaly ratio.
        """
        desired_anomalous_logs = int(total_logs * anomaly_ratio)
        print(f"Desired Anomalous Logs: {desired_anomalous_logs}")
        
        num_anomaly_calls = math.ceil(desired_anomalous_logs / self.average_logs_per_scenario)
        print(f"Number of Anomaly Calls to Generate: {num_anomaly_calls}")
        
        log_groups = []
        actual_anomalous_logs = 0
        
        # Generate anomalous logs
        for _ in range(num_anomaly_calls):
            anomaly_logs = self.generate_anomalous_log()
            log_groups.append(anomaly_logs)
            actual_anomalous_logs += len(anomaly_logs)
        
        print(f"Actual Anomalous Logs Generated: {actual_anomalous_logs}")
        
        # Calculate the remaining logs needed to meet `total_logs`
        remaining_logs = total_logs - actual_anomalous_logs
        print(f"Remaining Logs to Generate: {remaining_logs}")
        
        # Adjust normal logs based on remaining capacity
        normal_log_groups = []
        for _ in range(remaining_logs):
            normal_log = self.generate_normal_log()
            normal_log_groups.append([normal_log])  # Single-element group
        
        # Check if anomalies need further adjustment
        if actual_anomalous_logs < desired_anomalous_logs:
            additional_anomalous_logs = desired_anomalous_logs - actual_anomalous_logs
            print(f"Generating {additional_anomalous_logs} additional anomaly logs to meet the desired count.")
            
            for _ in range(math.ceil(additional_anomalous_logs / self.average_logs_per_scenario)):
                anomaly_logs = self.generate_anomalous_log()
                log_groups.append(anomaly_logs)
                actual_anomalous_logs += len(anomaly_logs)
                remaining_logs -= len(anomaly_logs)
                if remaining_logs <= 0:
                    break
        
        # Ensure total logs are within limits
        total_generated_logs = sum(len(group) for group in log_groups) + len(normal_log_groups)
        if total_generated_logs > total_logs:
            excess = total_generated_logs - total_logs
            print(f"Trimming {excess} logs to maintain total_logs constraint.")
            # Trim excess from normal logs
            if len(normal_log_groups) >= excess:
                normal_log_groups = normal_log_groups[:-excess]
            else:
                # If excess is larger than normal logs, trim accordingly
                normal_log_groups = []
        
        # Combine and shuffle log groups
        log_groups.extend(normal_log_groups)
        random.shuffle(log_groups)
        
        # Flatten logs into a single list
        logs = [log for group in log_groups for log in group]
        print(f"Total logs after shuffling and flattening: {len(logs)}")
        
        # Final validation
        anomaly_count = sum(log["Anomalous"] for log in logs)
        print(f"Generated {len(logs)} logs. Anomalies: {anomaly_count}")
        # validate_anomaly_consistency(logs)

        
        return logs

    def validate_logs(self, logs, expected_anomaly_ratio):
        """
        Validates that the generated logs meet the expected anomaly ratio
        and that external IP correlation is within acceptable limits.
        """
        total_logs = len(logs)
        actual_anomalies = sum(log["Anomalous"] for log in logs)
        actual_ratio = actual_anomalies / total_logs if total_logs > 0 else 0
        print(f"Expected Anomaly Ratio: {expected_anomaly_ratio * 100}%")
        print(f"Actual Anomaly Ratio: {actual_ratio * 100:.2f}%")
        tolerance = 0.05  # 5% tolerance
        if not abs(actual_ratio - expected_anomaly_ratio) <= tolerance:
            print("WARNING: Anomaly ratio deviates significantly from expected.")
        else:
            print("Anomaly ratio is within the acceptable range.")

        # Calculate correlation between External IP and Anomalous
        external_ips = sum(1 for log in logs if ipaddress.ip_address(log["IP_Address"]).is_global)
        anomalous_external_ips = sum(
            1 for log in logs 
            if log["Anomalous"] == 1 and ipaddress.ip_address(log["IP_Address"]).is_global
        )
        
        external_ip_ratio = external_ips / total_logs if total_logs > 0 else 0
        anomalous_external_ip_ratio = anomalous_external_ips / total_logs if total_logs > 0 else 0
        
        print(f"Total External IPs: {external_ips} ({external_ip_ratio * 100:.2f}%)")
        print(f"Anomalous External IPs: {anomalous_external_ips} ({anomalous_external_ip_ratio * 100:.2f}%)")
        
        # Define acceptable correlation thresholds
        ip_correlation_threshold = self.config["IP_ADDRESS_GENERATION"].get("ip_correlation_threshold", 0.3)
        
        # Calculate the proportion of external IPs that are anomalous
        if external_ips > 0:
            proportion_anomalous_external = anomalous_external_ips / external_ips
        else:
            proportion_anomalous_external = 0
        
        print(f"Proportion of External IPs that are Anomalous: {proportion_anomalous_external * 100:.2f}%")
        
        # Example check: Ensure that the proportion doesn't exceed the threshold
        if proportion_anomalous_external > ip_correlation_threshold:
            print("WARNING: High correlation between external IPs and anomalies detected.")
        else:
            print("Correlation between external IPs and anomalies is within acceptable limits.")

    def save_logs_to_csv(self, logs, master_file, inference_file, overwrite=False):
        """
        Saves the generated logs to CSV files: master (with 'Anomalous') and inference (without).
        """
        import os
        import csv

        mode_master = "w" if overwrite else "a"
        mode_inference = "w" if overwrite else "a"

        # Ensure directories exist
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
            "HTTP_Response",
            "Anomalous"
        ]
        master_exists = os.path.isfile(master_file) and not overwrite
        with open(master_file, mode_master, newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames_master)
            if not master_exists or overwrite:
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
            "HTTP_Response"
        ]
        inference_exists = os.path.isfile(inference_file) and not overwrite
        with open(inference_file, mode_inference, newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames_inference)
            if not inference_exists or overwrite:
                writer.writeheader()
            for log in logs:
                inference_log = {k: v for k, v in log.items() if k != "Anomalous"}
                writer.writerow(inference_log)

        print(f"Saved {len(logs)} logs to '{master_file}' and '{inference_file}'.")

