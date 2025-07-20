
#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import altair as alt
from glob import glob
import os
import ipaddress
import datetime

###########################################
# CONFIGURATIONS
###########################################
ARCHIVE_FILE = "data/master_logs_archive.csv"
INFERENCE_DIR = "data"
INFERENCE_PATTERN = "inference_results_*.csv"
REFRESH_EVERY_SECONDS = 60  # HTML meta refresh
ANOMALY_COUNT_THRESHOLD = 10
ANOMALY_RATIO_THRESHOLD = 0.05

# Define unusual hour range for a hospital:
# We'll consider 10 PM (22) to 6 AM as "unusual" or "after-peak"
UNUSUAL_HOUR_START = 22  # 10 PM
UNUSUAL_HOUR_END = 6     # 6 AM

###########################################
# HELPER FUNCTIONS
###########################################

def is_private_ip(ip_str):
    """
    Return True if ip_str is a valid private (internal) IP, else False.
    """
    import ipaddress
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        return ip_obj.is_private
    except:
        return False

def load_archive():
    """
    Load the master_logs_archive.csv which stores original logs
    (without the 'Anomalous' label, or ignoring it).
    """
    if not os.path.exists(ARCHIVE_FILE):
        return pd.DataFrame()
    return pd.read_csv(ARCHIVE_FILE)

def load_inference_results():
    """
    Load all inference_results_*.csv, keep essential columns only,
    dropping duplicates on LogID.
    """
    files = sorted(glob(os.path.join(INFERENCE_DIR, INFERENCE_PATTERN)))
    if not files:
        return pd.DataFrame()

    df_list = []
    for f in files:
        temp = pd.read_csv(f, low_memory=False)
        df_list.append(temp)
    df_infer = pd.concat(df_list, ignore_index=True)

    # We only really need these columns from inference
    needed_cols = [
        "LogID", "predicted_anomaly", "anomaly_score",
        "Timestamp", "Role", "Endpoint"
    ]
    keep_cols = [c for c in needed_cols if c in df_infer.columns]
    df_infer = df_infer[keep_cols].drop_duplicates(subset=["LogID"], keep="last")

    # Convert Timestamp if present
    if "Timestamp" in df_infer.columns:
        df_infer["Timestamp"] = pd.to_datetime(df_infer["Timestamp"], errors="coerce")

    return df_infer

def load_merged_data():
    """
    Merge the archive logs with the inference logs on LogID.
    Then unify columns named 'Role_x'/'Role_y', 'Endpoint_x'/'Endpoint_y',
    'Timestamp_x'/'Timestamp_y', etc., into single columns.
    """
    df_arch = load_archive()
    df_infer = load_inference_results()

    if df_arch.empty:
        return df_arch
    if df_infer.empty:
        return df_arch

    # Merge on LogID
    df_merged = df_arch.merge(df_infer, on="LogID", how="left")

    # Unify columns if we have _x / _y
    possible_conflicts = ["Role", "Endpoint", "Timestamp"]
    for base_col in possible_conflicts:
        xcol = base_col + "_x"
        ycol = base_col + "_y"
        if xcol in df_merged.columns and ycol in df_merged.columns:
            df_merged[base_col] = df_merged[xcol].fillna(df_merged[ycol])
            df_merged.drop([xcol, ycol], axis=1, inplace=True)
        elif base_col not in df_merged.columns:
            # If truly missing, do nothing
            pass

    # Also unify predicted_anomaly, anomaly_score if needed
    if "predicted_anomaly_x" in df_merged.columns and "predicted_anomaly_y" in df_merged.columns:
        df_merged["predicted_anomaly"] = df_merged["predicted_anomaly_x"].fillna(df_merged["predicted_anomaly_y"])
        df_merged.drop(["predicted_anomaly_x","predicted_anomaly_y"], axis=1, inplace=True)
    elif "predicted_anomaly" not in df_merged.columns:
        df_merged["predicted_anomaly"] = float('nan')

    if "anomaly_score_x" in df_merged.columns and "anomaly_score_y" in df_merged.columns:
        df_merged["anomaly_score"] = df_merged["anomaly_score_x"].fillna(df_merged["anomaly_score_y"])
        df_merged.drop(["anomaly_score_x","anomaly_score_y"], axis=1, inplace=True)
    elif "anomaly_score" not in df_merged.columns:
        df_merged["anomaly_score"] = float('nan')

    # Convert predicted_anomaly to int
    df_merged["predicted_anomaly"] = df_merged["predicted_anomaly"].fillna(0).astype(int)
    df_merged["anomaly_score"] = df_merged["anomaly_score"].fillna(0).astype(float)

    # Convert final Timestamp if we ended up with them
    if "Timestamp" in df_merged.columns:
        df_merged["Timestamp"] = pd.to_datetime(df_merged["Timestamp"], errors="coerce")

    return df_merged

###########################################
# DASHBOARD MAIN
###########################################
def main():
    st.set_page_config(
        page_title="Enhanced Real-Time Anomaly Dashboard",
        layout="wide"
    )

    # Auto-refresh every REFRESH_EVERY_SECONDS
    st.markdown(
        f'<meta http-equiv="refresh" content="{REFRESH_EVERY_SECONDS}">',
        unsafe_allow_html=True
    )

    st.title("Real-Time Anomaly Detection Dashboard")

    df_merged = load_merged_data()
    if df_merged.empty:
        st.warning("No data found. Possibly no logs or no inference results.")
        return

    # If we do not have IP_Address, add a placeholder
    if "IP_Address" not in df_merged.columns:
        df_merged["IP_Address"] = "unknown"

    # Mark internal vs. external IP
    df_merged["is_internal_ip"] = df_merged["IP_Address"].apply(is_private_ip)

    # If there's a Timestamp column, parse hour
    if "Timestamp" in df_merged.columns and df_merged["Timestamp"].notna().sum() > 0:
        df_merged = df_merged.sort_values("Timestamp").reset_index(drop=True)
        df_merged["Hour"] = df_merged["Timestamp"].dt.hour
        # Let's define "unusual_hour" = True if hour < 6 or hour >= 22
        df_merged["is_unusual_hour"] = (df_merged["Hour"] < UNUSUAL_HOUR_END) | (df_merged["Hour"] >= UNUSUAL_HOUR_START)
    else:
        df_merged["Hour"] = -1
        df_merged["is_unusual_hour"] = False

    # Basic stats
    total_logs = len(df_merged)
    anomaly_count = df_merged["predicted_anomaly"].sum()
    anomaly_ratio = anomaly_count / total_logs if total_logs > 0 else 0

    # Show top metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Logs", str(total_logs))
    col2.metric("Anomalies", str(anomaly_count))
    col3.metric("Anomaly Ratio", f"{anomaly_ratio:.2%}")

    # Alert
    if anomaly_count > 0:
        st.error(f"ALERT: High anomaly volume (count={anomaly_count}, ratio={anomaly_ratio:.2%})")
        st.markdown("""
        <audio autoplay>
            <source src="https://www.soundjay.com/buttons/sounds/button-10.mp3" type="audio/mpeg">
        </audio>
        """, unsafe_allow_html=True)

    normal_count = total_logs - anomaly_count
    st.write(f"**Normal Logs**: {normal_count}, **Anomalous Logs**: {anomaly_count}")

    ###############################################################################
    # Existing analysis for Role, Endpoint, IP
    ###############################################################################

    # A) Role-based
    st.subheader("A) Role-based Analysis")
    if "Role" in df_merged.columns:
        # Overall usage
        role_usage = df_merged["Role"].value_counts().reset_index()
        role_usage.columns = ["Role","count_usage"]
        chart_role_usage = (
            alt.Chart(role_usage)
            .mark_bar(size=25)
            .encode(
                x=alt.X("count_usage:Q", title="Total Logs"),
                y=alt.Y("Role:N", sort="-x"),
                tooltip=["Role","count_usage"]
            )
            .properties(height=300, width=350, title="Overall Usage by Role")
        )
        st.altair_chart(chart_role_usage, use_container_width=True)

        # Anomalies by role
        role_anom_df = df_merged[df_merged["predicted_anomaly"]==1]
        if not role_anom_df.empty:
            role_anom = role_anom_df["Role"].value_counts().reset_index()
            role_anom.columns = ["Role","count_anomalies"]
            chart_role_anom = (
                alt.Chart(role_anom)
                .mark_bar(size=25)
                .encode(
                    x=alt.X("count_anomalies:Q", title="Anomaly Count"),
                    y=alt.Y("Role:N", sort="-x"),
                    tooltip=["Role","count_anomalies"]
                )
                .properties(height=300, width=350, title="Anomalies by Role")
            )
            st.altair_chart(chart_role_anom, use_container_width=True)
        else:
            st.write("No anomalies by Role.")
    else:
        st.write("No 'Role' column found after merging. Please confirm pipeline includes it.")

    # B) Endpoint-based
    st.subheader("B) Endpoint-based Analysis")
    if "Endpoint" in df_merged.columns:
        end_usage = df_merged["Endpoint"].value_counts().reset_index()
        end_usage.columns = ["Endpoint","count_usage"]
        end_usage = end_usage.head(10)

        chart_ep_usage = (
            alt.Chart(end_usage)
            .mark_bar(size=25)
            .encode(
                x=alt.X("count_usage:Q", title="Total Logs"),
                y=alt.Y("Endpoint:N", sort="-x"),
                tooltip=["Endpoint","count_usage"]
            )
            .properties(height=300, width=350, title="Top 10 Endpoints (Overall)")
        )
        st.altair_chart(chart_ep_usage, use_container_width=True)

        # Endpoint anomalies
        ep_anom_df = df_merged[df_merged["predicted_anomaly"]==1]
        if not ep_anom_df.empty:
            ep_anom = ep_anom_df["Endpoint"].value_counts().reset_index()
            ep_anom.columns = ["Endpoint","count_anomalies"]
            ep_anom = ep_anom.head(10)
            chart_ep_anom = (
                alt.Chart(ep_anom)
                .mark_bar(size=25)
                .encode(
                    x=alt.X("count_anomalies:Q", title="Anomaly Count"),
                    y=alt.Y("Endpoint:N", sort="-x"),
                    tooltip=["Endpoint","count_anomalies"]
                )
                .properties(height=300, width=350, title="Top 10 Endpoints (Anomalies)")
            )
            st.altair_chart(chart_ep_anom, use_container_width=True)
        else:
            st.write("No endpoint anomalies.")
    else:
        st.write("No 'Endpoint' column found after merging. Confirm pipeline includes it.")

    # C) IP-based
    st.subheader("C) IP-based Analysis (Internal vs External)")
    ip_usage = df_merged["is_internal_ip"].value_counts().reset_index()
    ip_usage.columns = ["is_internal_ip","count_usage"]
    ip_usage["IP_Type"] = ip_usage["is_internal_ip"].apply(lambda x: "Internal" if x else "External")

    chart_ip_usage = (
        alt.Chart(ip_usage)
        .mark_bar(size=25)
        .encode(
            x=alt.X("count_usage:Q", title="Total Logs"),
            y=alt.Y("IP_Type:N", sort="-x"),
            tooltip=["IP_Type","count_usage"]
        )
        .properties(height=300, width=350, title="Logs by IP Type")
    )
    st.altair_chart(chart_ip_usage, use_container_width=True)

    ip_anom_df = df_merged[df_merged["predicted_anomaly"]==1]
    if not ip_anom_df.empty:
        ip_anom_counts = ip_anom_df["is_internal_ip"].value_counts().reset_index()
        ip_anom_counts.columns = ["is_internal_ip","count_anomalies"]
        ip_anom_counts["IP_Type"] = ip_anom_counts["is_internal_ip"].apply(lambda x: "Internal" if x else "External")

        chart_ip_anom = (
            alt.Chart(ip_anom_counts)
            .mark_bar(size=25)
            .encode(
                x=alt.X("count_anomalies:Q", title="Anomaly Count"),
                y=alt.Y("IP_Type:N", sort="-x"),
                tooltip=["IP_Type","count_anomalies"]
            )
            .properties(height=300, width=350, title="Anomalies by IP Type")
        )
        st.altair_chart(chart_ip_anom, use_container_width=True)
    else:
        st.write("No IP-based anomalies found.")

    ###############################################################################
    # D) Time-based Analysis: ONLY 10 PM - 6 AM
    ###############################################################################
# ---------------------------------------------------------------------------
# D) Time-based Analysis: ONLY 10 PM - 6 AM
# Replaces your old daily_unusual_usage and daily_unusual_anom separate charts
# ---------------------------------------------------------------------------

    st.subheader("D) Time-based Analysis (Unusual Hours: 10PM - 6AM only)")

    # Filter logs that occur in the unusual hour window
    df_unusual = df_merged[df_merged["is_unusual_hour"] == True].copy()

    if "Timestamp" in df_unusual.columns and df_unusual["Timestamp"].notna().sum() > 0:
        # Sort by timestamp so we have chronological order
        df_unusual = df_unusual.sort_values("Timestamp").reset_index(drop=True)

        # Create a daily grouping
        df_unusual["date"] = df_unusual["Timestamp"].dt.date

        # 1) Daily total logs (bar chart)
        daily_unusual_usage = (
            df_unusual.groupby("date")["LogID"].count().reset_index(name="count_usage")
        )

        # 2) Daily anomalies among unusual-hour logs (line chart)
        daily_unusual_anom = (
            df_unusual[df_unusual["predicted_anomaly"] == 1]
            .groupby("date")["LogID"].count()
            .reset_index(name="count_anomalies")
        )

        # Build the base domain for the X-axis
        base = alt.Chart().encode(
            x=alt.X("date:T", title="Date")
        )

        # A. The bar chart for total logs
        bar_usage = base.mark_bar(color="#4C78A8").transform_lookup(
            lookup="date",
            from_=alt.LookupData(daily_unusual_usage, "date", ["count_usage"])
        ).encode(
            y=alt.Y("count_usage:Q", title="Logs in [10PM–6AM]"),
            tooltip=[alt.Tooltip("count_usage:Q", title="Logs")]
        )

        # B. The line chart for anomalies
        line_anom = base.mark_line(point=True, color="#F58518").transform_lookup(
            lookup="date",
            from_=alt.LookupData(daily_unusual_anom, "date", ["count_anomalies"])
        ).encode(
            y=alt.Y("count_anomalies:Q", title="Anomalies in [10PM–6AM]", axis=alt.Axis(labels=True)),
            tooltip=[alt.Tooltip("count_anomalies:Q", title="Anomalies")]
        )

        # Combine them with independent y-scales
        combined_chart = alt.layer(bar_usage, line_anom, data=daily_unusual_usage).resolve_scale(
            y='independent'  # so bar usage & line anomalies each have their own scale
        ).properties(
            width=700,
            height=400,
            title="Daily Logs & Anomalies (Unusual Hours Only)"
        )

        st.altair_chart(combined_chart, use_container_width=True)

    else:
        st.write("No logs found in the 10PM–6AM window or missing valid Timestamps.")


    # Summaries for external IP anomalies & unusual-hour anomalies
    external_ip_anomalies = df_merged[
        (df_merged["predicted_anomaly"] == 1) &
        (df_merged["is_internal_ip"] == False)
    ]
    unusual_hour_anomalies = df_merged[
        (df_merged["predicted_anomaly"] == 1) &
        (df_merged["is_unusual_hour"] == True)
    ]
    st.write(f"External IP Anomalies: {len(external_ip_anomalies)}")
    st.write(f"Unusual Hour Anomalies: {len(unusual_hour_anomalies)}")

    # Expanders
    with st.expander("All Logs (Merged)"):
        st.dataframe(df_merged, use_container_width=True)

    anomalies_df = df_merged[df_merged["predicted_anomaly"] == 1]
    with st.expander("Anomalous Logs Only"):
        if not anomalies_df.empty:
            st.dataframe(anomalies_df, use_container_width=True)
        else:
            st.write("No anomalies found.")

    normal_df = df_merged[df_merged["predicted_anomaly"] == 0]
    with st.expander("Normal Logs Only"):
        st.dataframe(normal_df, use_container_width=True)

    if "anomaly_score" in df_merged.columns:
        borderline_df = df_merged[
            (df_merged["anomaly_score"] > 0.5) &
            (df_merged["predicted_anomaly"] == 0)
        ]
        with st.expander("Potentially Suspicious (High Score but Normal)"):
            if not borderline_df.empty:
                st.dataframe(borderline_df, use_container_width=True)
            else:
                st.write("No borderline logs above anomaly_score>0.5.")


if __name__ == "__main__":
    main()
