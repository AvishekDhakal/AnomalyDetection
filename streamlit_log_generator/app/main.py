# app/main.py

import streamlit as st
from log_generator import generate_logs, save_logs_to_csv
import pandas as pd
import os
import json

def main():
    st.title("Healthcare Log Generator")
    st.write("""
    This application allows you to generate synthetic logs for a 24/7 healthcare environment.
    Configure the parameters below and click 'Generate Logs' to create master and inference logs.
    """)

    # Load configuration
    CONFIG_FILE = "app/config.json"  # Adjust path if necessary
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    st.sidebar.header("Configuration")

    # Preset Configurations
    preset_options = {
        "Custom": {
            "total_logs": 100000,
            "anomaly_ratio": 0.02,
            "master_file": "app/data/master_logs.csv",
            "inference_file": "app/data/inference_logs.csv",
            "network_subnet": "10.0.0",
            "extended_days": 90,
            "role_switch_prob": 0.05,
            "anomaly_ip_prob": 0.10,
            "after_hours_prob": 0.25,
        },
        "Small Test": {
            "total_logs": 500,
            "anomaly_ratio": 0.05,
            "master_file": "app/data/small_master_logs.csv",
            "inference_file": "app/data/small_inference_logs.csv",
            "network_subnet": "10.0.1",
            "extended_days": 30,
            "role_switch_prob": 0.05,
            "anomaly_ip_prob": 0.10,
            "after_hours_prob": 0.25,
        },
        "Large Simulation": {
            "total_logs": 1000000,
            "anomaly_ratio": 0.02,
            "master_file": "app/data/large_master_logs.csv",
            "inference_file": "app/data/large_inference_logs.csv",
            "network_subnet": "10.0.2",
            "extended_days": 365,
            "role_switch_prob": 0.05,
            "anomaly_ip_prob": 0.10,
            "after_hours_prob": 0.25,
        }
    }

    preset = st.sidebar.selectbox("Select Preset Configuration", options=list(preset_options.keys()))

    if preset != "Custom":
        preset_values = preset_options[preset]
        total_logs = preset_values["total_logs"]
        anomaly_ratio = preset_values["anomaly_ratio"]
        master_file = preset_values["master_file"]
        inference_file = preset_values["inference_file"]
        network_subnet = preset_values["network_subnet"]
        extended_days = preset_values["extended_days"]
        role_switch_prob = preset_values["role_switch_prob"]
        anomaly_ip_prob = preset_values["anomaly_ip_prob"]
        after_hours_prob = preset_values["after_hours_prob"]
    else:
        # Configuration Inputs
        total_logs = st.sidebar.number_input(
            "Total Logs",
            min_value=1,
            max_value=1000000,
            value=100000,
            step=1000,
            help="Specify the total number of logs to generate."
        )

        anomaly_ratio = st.sidebar.slider(
            "Anomaly Ratio (%)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.1,
            help="Define the percentage of logs that should be anomalous."
        ) / 100.0

        master_file = st.sidebar.text_input(
            "Master Log File Path",
            value="app/data/master_logs.csv",
            help="Destination path for master logs (with anomalies)."
        )

        inference_file = st.sidebar.text_input(
            "Inference Log File Path",
            value="app/data/inference_logs.csv",
            help="Destination path for inference logs (without anomalies)."
        )

        network_subnet = st.sidebar.text_input(
            "Network Subnet",
            value="10.0.0",
            help="Base subnet for internal IP addresses (e.g., '10.0.0')."
        )

        extended_days = st.sidebar.number_input(
            "Extended Days",
            min_value=1,
            max_value=365,
            value=90,
            step=1,
            help="Number of past days to simulate logs from."
        )

        st.sidebar.markdown("---")
        st.sidebar.write("### Advanced Settings")

        # Advanced Configuration
        role_switch_prob = st.sidebar.slider(
            "Role Switch Probability (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
            help="Likelihood of role-switch anomalies."
        ) / 100.0

        anomaly_ip_prob = st.sidebar.slider(
            "Anomaly IP Probability (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.1,
            help="Chance of using out-of-subnet IP addresses for anomalies."
        ) / 100.0

        after_hours_prob = st.sidebar.slider(
            "After Hours Probability (%)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=0.1,
            help="Probability of logs being generated during after-hours."
        ) / 100.0

    # Overwrite Existing Logs Option
    overwrite_logs = st.sidebar.checkbox(
        "Overwrite Existing Logs",
        value=True,
        help="Check to overwrite existing log files. Uncheck to append to existing files."
    )

    # Add Option to View Only Anomalous Logs
    view_anomalous_only = st.sidebar.checkbox(
        "View Only Anomalous Logs",
        value=False,
        help="Check to display only anomalous logs in the sample preview."
    )

    # Generate Logs Button
    if st.button("Generate Logs"):
        with st.spinner("Generating logs..."):
            # Sanitize file paths
            ALLOWED_DIRS = ["app/data"]

            def sanitize_file_path(file_path: str) -> str:
                # Ensure the file is within allowed directories
                for allowed_dir in ALLOWED_DIRS:
                    if file_path.startswith(allowed_dir):
                        return file_path
                # Default to app/data if not allowed
                return os.path.join("app", "data", os.path.basename(file_path))

            master_file = sanitize_file_path(master_file)
            inference_file = sanitize_file_path(inference_file)

            # Generate logs using the refactored log_generator module
            logs = generate_logs(
                total_logs=total_logs,
                anomaly_ratio=anomaly_ratio,
                roles=config["ROLES"],
                endpoints=config["ENDPOINTS"],
                network_subnet=network_subnet,
                extended_days=extended_days
            )

            # Save logs to CSV
            save_logs_to_csv(
                logs=logs,
                master_file=master_file,
                inference_file=inference_file,
                mode='w' if overwrite_logs else 'a'
            )

            # Display summary
            total_anomalies = sum(log["Anomalous"] for log in logs)
            st.success(f"Generated {len(logs)} logs with {total_anomalies} anomalies.")

            # Display a sample of the logs
            st.subheader("Sample Logs")

            # Filter logs based on the 'View Only Anomalous Logs' option
            if view_anomalous_only:
                filtered_logs = [log for log in logs if log["Anomalous"] == 1]
                if not filtered_logs:
                    st.warning("No anomalous logs found in the generated data.")
                else:
                    sample_size = st.number_input(
                        "Number of Anomalous Logs to Preview",
                        min_value=1,
                        max_value=min(1000, len(filtered_logs)),
                        value=min(100, len(filtered_logs)),
                        step=1,
                        help="Specify how many anomalous logs to display."
                    )
                    df = pd.DataFrame(filtered_logs[:sample_size])  # Show first 'sample_size' anomalous logs
            else:
                sample_size = st.number_input(
                    "Number of Sample Logs to Preview",
                    min_value=1,
                    max_value=min(1000, len(logs)),
                    value=min(100, len(logs)),
                    step=1,
                    help="Specify how many sample logs to display."
                )
                df = pd.DataFrame(logs[:sample_size])  # Show first 'sample_size' logs

            st.dataframe(df)

            # Provide download links
            st.subheader("Download Logs")

            # Function to convert DataFrame to CSV
            def convert_df_to_csv(df: pd.DataFrame) -> bytes:
                return df.to_csv(index=False).encode('utf-8')

            # Download Master Logs
            if os.path.exists(master_file):
                master_df = pd.read_csv(master_file)
                master_csv = convert_df_to_csv(master_df)
                st.download_button(
                    label="Download Master Logs",
                    data=master_csv,
                    file_name=os.path.basename(master_file),
                    mime='text/csv',
                )

            # Download Inference Logs
            if os.path.exists(inference_file):
                inference_df = pd.read_csv(inference_file)
                inference_csv = convert_df_to_csv(inference_df)
                st.download_button(
                    label="Download Inference Logs",
                    data=inference_csv,
                    file_name=os.path.basename(inference_file),
                    mime='text/csv',
                )

            # Optionally, provide download for anomalous logs only
            if view_anomalous_only and filtered_logs:
                anomalous_df = pd.DataFrame(filtered_logs)
                anomalous_csv = convert_df_to_csv(anomalous_df)
                st.download_button(
                    label="Download Anomalous Logs",
                    data=anomalous_csv,
                    file_name="anomalous_logs.csv",
                    mime='text/csv',
                )

if __name__ == "__main__":
    main()
