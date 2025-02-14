#!/usr/bin/env python3
"""
data_preprocessing_phase2.py

Phase 2: Advanced Feature Engineering
-------------------------------------
1. Inherits the basic Phase 1 approach (time features, IP classification).
2. More sophisticated endpoint parsing: extracts resource, ID, query params.
3. Creates role-endpoint mismatch flags or other domain-specific indicators.
4. Optionally, keep 'Anomalous' if present for supervised tasks.

Usage:
  python data_preprocessing_phase2.py --input_csv data/logs_phase1.csv --output_csv data/logs_phase2.csv
"""

import argparse
import pandas as pd
import numpy as np
import os
import logging
from urllib.parse import urlparse, parse_qs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_endpoint_detailed(endpoint):
    """
    Splits 'endpoint' into:
      - Endpoint_Resource (path without final numeric ID)
      - Endpoint_ID (the numeric ID if present)
      - Query parameters: returns a dict of parsed key-value pairs
    Example:
      "/admin/credentials/65?export=true&limit=500" ->
        resource="/admin/credentials"
        id="65"
        query_dict = {"export":["true"], "limit":["500"]}
    """
    if not isinstance(endpoint, str) or endpoint.strip() == "":
        return ("UNKNOWN_RESOURCE", None, {})
    
    # 1) Separate the path from the query
    split_q = endpoint.split("?")
    path = split_q[0]
    query_str = split_q[1] if len(split_q) > 1 else ""
    
    # Parse the path segments
    segments = path.strip("/").split("/")
    
    # Check if last segment is numeric (endpoint ID)
    endpoint_id = None
    if segments and segments[-1].isdigit():
        endpoint_id = segments[-1]
        # Remove the numeric segment from resource
        segments = segments[:-1]
    
    # Re-construct resource
    if segments:
        endpoint_resource = "/" + "/".join(segments)
    else:
        endpoint_resource = "UNKNOWN_RESOURCE"
    
    # 2) Parse query string into a dict
    query_dict = {}
    if query_str:
        # We'll parse with parse_qs from urllib
        # e.g. "export=true&limit=500" -> {"export":["true"], "limit":["500"]}
        query_dict = parse_qs(query_str)
    
    return (endpoint_resource, endpoint_id, query_dict)

def extract_phase2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates advanced features from the endpoint, role, queries, etc.
    """
    df2 = df.copy()
    
    # Parse endpoints in detail
    parsed_info = df2["Endpoint"].apply(parse_endpoint_detailed)
    df2["Endpoint_Resource"] = parsed_info.apply(lambda x: x[0])
    df2["Endpoint_ID"] = parsed_info.apply(lambda x: x[1])
    df2["Query_Dict"] = parsed_info.apply(lambda x: x[2])
    
    # For demonstration, let's pull out a few known queries:
    # e.g. "export", "limit", "attempts", "comment"
    def get_query_bool(qdict, key):
        return 1 if key in qdict else 0
    def get_query_int(qdict, key):
        # if key present, parse first val as int, else None
        if key in qdict:
            try:
                return int(qdict[key][0])
            except:
                return None
        return None
    def get_query_str(qdict, key):
        # if key present, return the string
        if key in qdict:
            return qdict[key][0]
        return None
    
    df2["Query_export"] = df2["Query_Dict"].apply(lambda d: get_query_bool(d, "export"))
    df2["Query_limit"] = df2["Query_Dict"].apply(lambda d: get_query_int(d, "limit"))
    df2["Query_attempts"] = df2["Query_Dict"].apply(lambda d: get_query_int(d, "attempts"))
    df2["Query_comment"] = df2["Query_Dict"].apply(lambda d: get_query_str(d, "comment"))
    
    # Example: flag if comment has 'XSS'
    df2["Query_commentXSS"] = df2["Query_comment"].apply(
        lambda c: 1 if c and "<script>" in c.lower() else 0
    )
    
    # Convert Endpoint_ID to numeric if possible
    df2["Endpoint_ID"] = pd.to_numeric(df2["Endpoint_ID"], errors='coerce')
    
    # Create a role-endpoint mismatch feature for illustration:
    # e.g. if Nurse or Staff hits /admin/..., we flag mismatch=1
    def role_endpoint_mismatch(row):
        role = row["Role"]
        resource = row["Endpoint_Resource"]
        # simple example: if resource starts with "/admin" but role isn't "Admin"
        if resource.startswith("/admin") and role != "Admin":
            return 1
        return 0
    
    df2["RoleEndpointMismatch"] = df2.apply(role_endpoint_mismatch, axis=1)
    
    # (Optional) drop Query_Dict if you no longer need it as a column
    df2.drop(columns=["Query_Dict"], inplace=True)
    
    return df2

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Advanced Feature Engineering")
    parser.add_argument("--input_csv", type=str, required=True, help="Phase 1 CSV file to enrich with advanced features.")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path for Phase 2 features.")
    args = parser.parse_args()
    
    # 1) Load Phase 1 data
    df = pd.read_csv(args.input_csv)
    logging.info(f"Loaded DF with shape {df.shape} from {args.input_csv}")
    
    # 2) Generate advanced features
    df_phase2 = extract_phase2_features(df)
    logging.info(f"Phase 2 features added. New shape {df_phase2.shape}. Columns now: {df_phase2.columns.tolist()}")
    
    # 3) Save result
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_phase2.to_csv(args.output_csv, index=False)
    logging.info(f"Saved Phase 2 dataset to {args.output_csv}")

if __name__ == "__main__":
    main()
