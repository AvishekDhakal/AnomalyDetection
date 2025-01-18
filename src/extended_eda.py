#!/usr/bin/env python3

"""
eda_extended.py

Performs an extended Exploratory Data Analysis (EDA) on master_logs.csv 
to enable data-driven feature engineering and preprocessing.

New/Additional Steps:
1. IP Address Analysis:
   - Internal vs. External IPs
   - Top 10 IP addresses by anomaly status
2. Endpoint Usage:
   - Top endpoints, normal vs. anomalous
3. Time-of-Day Distribution:
   - Hourly distribution, normal vs. anomalous
4. Day-of-Week Distribution:
   - Day-of-week counts, normal vs. anomalous
5. Derived Feature Correlation:
   - Example: is_internal_subnet, hour, day_of_week, anomaly
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------- CONFIG --------------
SAVE_DIR = "eda_extended_plots"
CSV_PATH = "data/master_logs.csv"

# -------------- HELPER FUNCTIONS --------------

def ensure_directory(dir_name):
    """
    Ensures a directory for saving plots exists.
    """
    os.makedirs(dir_name, exist_ok=True)


def load_and_prepare_data(csv_path):
    """
    Loads the CSV, parses Timestamp, creates derived features:
    is_internal_subnet, hour, day_of_week.
    """
    df = pd.read_csv(csv_path)

    # Convert Timestamp to datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Drop rows where Timestamp is NaT (if any)
    df.dropna(subset=["Timestamp"], inplace=True)

    # Create an 'is_internal_subnet' feature
    # Example: Check if IP starts with "10.0.0."
    df["is_internal_subnet"] = df["IP_Address"].apply(
        lambda ip: 1 if ip.startswith("10.0.0.") else 0
    )

    # Extract hour of day (0-23)
    df["hour"] = df["Timestamp"].dt.hour

    # Extract day of week (0=Monday, 6=Sunday)
    df["day_of_week"] = df["Timestamp"].dt.dayofweek

    return df


def plot_ip_distribution(df, save_dir):
    """
    Plots:
    1. Distribution of internal vs. external IP addresses.
    2. Top 10 IP addresses by anomaly status.
    """
    # 1. Internal vs external
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="is_internal_subnet", hue="Anomalous")
    plt.title("Internal vs. External IP Distribution")
    plt.xlabel("is_internal_subnet (0=External, 1=Internal)")
    plt.ylabel("Count")
    plt.legend(title="Anomalous", labels=["Normal (0)", "Anomalous (1)"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ip_internal_vs_external.png"))
    plt.close()

    # 2. Top 10 IP addresses overall
    top_ips = df["IP_Address"].value_counts().nlargest(10).index
    df_top_ips = df[df["IP_Address"].isin(top_ips)]

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_top_ips, y="IP_Address", hue="Anomalous")
    plt.title("Top 10 IP Addresses by Anomaly Status")
    plt.xlabel("Count")
    plt.ylabel("IP Address")
    plt.legend(title="Anomalous", labels=["Normal (0)", "Anomalous (1)"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "top_10_ips_by_anomaly.png"))
    plt.close()


def plot_endpoint_usage(df, save_dir):
    """
    Plots the top 10 endpoints by total usage, split by anomaly status.
    """
    # Identify top 10 endpoints
    top_endpoints = df["Endpoint"].value_counts().nlargest(10).index
    df_top_endpoints = df[df["Endpoint"].isin(top_endpoints)]

    plt.figure(figsize=(9, 6))
    sns.countplot(data=df_top_endpoints, y="Endpoint", hue="Anomalous")
    plt.title("Top 10 Endpoints by Anomaly Status")
    plt.xlabel("Count")
    plt.ylabel("Endpoint")
    plt.legend(title="Anomalous", labels=["Normal (0)", "Anomalous (1)"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "top_10_endpoints_by_anomaly.png"))
    plt.close()


def plot_hourly_distribution(df, save_dir):
    """
    Plots how logs are distributed by hour of the day, split by anomaly.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="hour", hue="Anomalous")
    plt.title("Hourly Distribution (Normal vs. Anomalous)")
    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel("Count")
    plt.legend(title="Anomalous", labels=["Normal (0)", "Anomalous (1)"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hourly_distribution.png"))
    plt.close()


def plot_day_of_week_distribution(df, save_dir):
    """
    Plots how logs are distributed by day of week, split by anomaly.
    0=Monday, 6=Sunday
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="day_of_week", hue="Anomalous")
    plt.title("Day-of-Week Distribution (Normal vs. Anomalous)")
    plt.xlabel("Day of Week (0=Mon, 6=Sun)")
    plt.ylabel("Count")
    plt.legend(title="Anomalous", labels=["Normal (0)", "Anomalous (1)"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "day_of_week_distribution.png"))
    plt.close()


def plot_correlation_of_derived_features(df, save_dir):
    """
    Plots a correlation heatmap of newly derived features + Anomalous.
    Example: is_internal_subnet, hour, day_of_week, HTTP_Response, Anomalous
    """
    derived_cols = ["is_internal_subnet", "hour", "day_of_week", "HTTP_Response", "Anomalous"]
    corr_df = df[derived_cols].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlation of Derived Features + Anomalous")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "derived_features_correlation.png"))
    plt.close()


def print_additional_insights(df):
    """
    Prints additional pivot tables or stats in the terminal to help 
    with data-driven decisions for feature engineering.
    """
    print("\n=== Additional EDA Insights ===")

    # Example pivot: average HTTP_Response by is_internal_subnet and anomaly
    pivot_resp = pd.pivot_table(
        df,
        values="HTTP_Response",
        index="is_internal_subnet",
        columns="Anomalous",
        aggfunc="mean"
    )
    print("\nAverage HTTP_Response by (is_internal_subnet, Anomalous):")
    print(pivot_resp)

    # Example pivot: average hour by role and anomaly
    pivot_hour = pd.pivot_table(
        df,
        values="hour",
        index="Role",
        columns="Anomalous",
        aggfunc="mean"
    )
    print("\nAverage Hour of Access by (Role, Anomalous):")
    print(pivot_hour)

    # Example: Checking how many logs occur on weekends (day_of_week >= 5)
    weekend_logs = df[df["day_of_week"] >= 5].shape[0]
    print(f"\nNumber of Weekend Logs (Sat, Sun): {weekend_logs}")
    total_logs = df.shape[0]
    print(f"Total Logs: {total_logs}")
    print(f"Percentage of Weekend Logs: {weekend_logs / total_logs * 100:.2f}%\n")


# -------------- MAIN --------------

def main():
    # 1. Ensure directory for saving plots
    ensure_directory(SAVE_DIR)

    # 2. Load Data + Prepare Derived Features
    df = load_and_prepare_data(CSV_PATH)

    # 3. Extended EDA Plots
    plot_ip_distribution(df, SAVE_DIR)
    plot_endpoint_usage(df, SAVE_DIR)
    plot_hourly_distribution(df, SAVE_DIR)
    plot_day_of_week_distribution(df, SAVE_DIR)
    plot_correlation_of_derived_features(df, SAVE_DIR)

    # 4. Print Additional Insights to Terminal
    print_additional_insights(df)

    print("\nExtended EDA complete!")
    print(f"Check the '{SAVE_DIR}' folder for generated plots.\n")


if __name__ == "__main__":
    main()
