# #!/usr/bin/env python3

# import pandas as pd
# import argparse
# import os
# import sys
# import pickle
# import json
# from sklearn.preprocessing import OneHotEncoder

# def parse_arguments():
#     """
#     Parses command-line arguments.
#     """
#     parser = argparse.ArgumentParser(description='Feature Engineering for Anomaly Detection')
#     parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'],
#                         help='Mode of operation: train or inference')
#     parser.add_argument('--config', type=str, required=True,
#                         help='Path to the configuration JSON file')
#     return parser.parse_args()

# def load_data(file_path):
#     """
#     Loads the CSV data into a pandas DataFrame.
#     """
#     if not os.path.exists(file_path):
#         print(f"Error: File {file_path} does not exist.")
#         sys.exit(1)
#     try:
#         df = pd.read_csv(file_path)
#         print(f"Loaded data from {file_path} with shape {df.shape}.")
#         return df
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         sys.exit(1)

# def save_pickle(obj, filename):
#     """
#     Saves a Python object to a pickle file.
#     """
#     os.makedirs(os.path.dirname(filename), exist_ok=True)
#     with open(filename, 'wb') as f:
#         pickle.dump(obj, f)
#     print(f"Saved object to {filename}")

# def load_pickle(filename):
#     """
#     Loads a Python object from a pickle file.
#     """
#     if not os.path.exists(filename):
#         print(f"Error: Pickle file {filename} does not exist.")
#         sys.exit(1)
#     with open(filename, 'rb') as f:
#         obj = pickle.load(f)
#     print(f"Loaded object from {filename}")
#     return obj

# def load_config(config_path):
#     """
#     Loads the configuration JSON file.
#     """
#     if not os.path.exists(config_path):
#         print(f"Error: Configuration file {config_path} does not exist.")
#         sys.exit(1)
#     try:
#         with open(config_path, 'r') as f:
#             config = json.load(f)
#         print(f"Loaded configuration from {config_path}.")
#         return config
#     except Exception as e:
#         print(f"Error loading configuration file {config_path}: {e}")
#         sys.exit(1)

# def create_anomaly_features(df, config):
#     """
#     Creates binary features indicating specific anomaly scenarios based on configuration.
#     """
#     anomaly_scenarios = config["ANOMALY_SCENARIOS"]
#     anomalous_endpoints = config["ANOMALOUS_ENDPOINTS"]

#     for role, scenarios in anomaly_scenarios.items():
#         for scenario in scenarios.keys():
#             feature_name = f"Anomaly_{role}_{scenario}"
#             df[feature_name] = 0  # Initialize to 0
#             conditions = anomalous_endpoints.get(scenario, [])
#             for endpoint, method in conditions:
#                 condition = (
#                     (df['Role_' + role] == True) &
#                     (df['Endpoint'] == endpoint) &
#                     (df['HTTP_Method_' + method] == True) &
#                     (df['HTTP_Response'].isin(config["HTTP_RESPONSE_CODES"]["anomalous"]["codes"]))
#                 )
#                 df.loc[condition, feature_name] = 1
#     return df

# def compute_frequency_features(df):
#     """
#     Computes frequency-based features such as number of requests per user,
#     number of unique endpoints accessed per user, and role-specific request counts.
#     """
#     # Number of requests per user
#     user_request_counts = df.groupby('UserID').size().rename('User_Request_Count')
#     df = df.merge(user_request_counts, on='UserID', how='left')
    
#     # Number of unique endpoints accessed per user
#     user_unique_endpoints = df.groupby('UserID')['Endpoint'].nunique().rename('User_Unique_Endpoints')
#     df = df.merge(user_unique_endpoints, on='UserID', how='left')
    
#     # Number of requests per role per user
#     roles = ['Doctor', 'Nurse', 'Staff', 'Admin']
#     for role in roles:
#         role_col = f"Role_{role}"
#         user_role_counts = df.groupby('UserID')[role_col].sum().rename(f"User_{role}_Count")
#         df = df.merge(user_role_counts, on='UserID', how='left')
    
#     return df

# def create_interaction_features(df):
#     """
#     Creates interaction features between Role and HTTP_Method.
#     """
#     roles = ['Doctor', 'Nurse', 'Staff', 'Admin']
#     http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
    
#     for role in roles:
#         for method in http_methods:
#             feature_name = f"{role}_{method}"
#             role_col = f"Role_{role}"
#             method_col = f"HTTP_Method_{method}"
#             if role_col in df.columns and method_col in df.columns:
#                 df[feature_name] = ((df[role_col] == True) & (df[method_col] == True)).astype(int)
#             else:
#                 # If either column is missing, assign 0
#                 df[feature_name] = 0
#     return df

# def train_feature_engineering(df, config):
#     """
#     Performs feature engineering on training data.
#     Returns the enriched DataFrame and fitted encoders.
#     """
#     # Create Anomaly Scenario Features
#     df = create_anomaly_features(df, config)
    
#     # Compute Frequency-Based Features
#     df = compute_frequency_features(df)
    
#     # Create Interaction Features
#     df = create_interaction_features(df)
    
#     # Define feature columns (excluding preserved columns and 'Anomalous')
#     preserved_columns = ['LogID', 'Timestamp', 'Anomalous']
    
#     # Define anomaly feature columns
#     anomaly_feature_columns = []
#     anomaly_scenarios = config["ANOMALY_SCENARIOS"]
#     for role, scenarios in anomaly_scenarios.items():
#         for scenario in scenarios.keys():
#             anomaly_feature_columns.append(f"Anomaly_{role}_{scenario}")
    
#     # Define frequency feature columns
#     frequency_feature_columns = ['User_Request_Count', 'User_Unique_Endpoints']
#     roles = ['Doctor', 'Nurse', 'Staff', 'Admin']
#     for role in roles:
#         frequency_feature_columns.append(f"User_{role}_Count")
    
#     # Define interaction feature columns
#     interaction_feature_columns = [f"{role}_{method}" for role in roles for method in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']]
    
#     # Aggregate all feature columns
#     feature_columns = anomaly_feature_columns + frequency_feature_columns + interaction_feature_columns
    
#     # Select only processed columns for encoding (if any)
#     # In this case, since we're not encoding endpoints, no additional encoding is needed
    
#     # Define encoders (if any categorical features need encoding)
#     # Currently, anomaly features are binary, frequency features are numerical, interaction features are binary
#     # No encoding required unless you have additional categorical features
    
#     # Save the feature list
#     save_pickle(feature_columns, 'mappings/features.pkl')
    
#     # Reorder columns: preserved columns first, then features
#     df = df[preserved_columns + feature_columns]
    
#     return df, feature_columns

# def inference_feature_engineering(df, feature_columns, config):
#     """
#     Performs feature engineering on inference data.
#     Ensures consistency with training features.
#     Returns the enriched DataFrame.
#     """
#     # Create Anomaly Scenario Features
#     df = create_anomaly_features(df, config)
    
#     # Compute Frequency-Based Features
#     df = compute_frequency_features(df)
    
#     # Create Interaction Features
#     df = create_interaction_features(df)
    
#     # Define anomaly feature columns
#     anomaly_feature_columns = []
#     anomaly_scenarios = config["ANOMALY_SCENARIOS"]
#     for role, scenarios in anomaly_scenarios.items():
#         for scenario in scenarios.keys():
#             anomaly_feature_columns.append(f"Anomaly_{role}_{scenario}")
    
#     # Define frequency feature columns
#     frequency_feature_columns = ['User_Request_Count', 'User_Unique_Endpoints']
#     roles = ['Doctor', 'Nurse', 'Staff', 'Admin']
#     for role in roles:
#         frequency_feature_columns.append(f"User_{role}_Count")
    
#     # Define interaction feature columns
#     interaction_feature_columns = [f"{role}_{method}" for role in roles for method in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']]
    
#     # Aggregate all feature columns
#     expected_feature_columns = anomaly_feature_columns + frequency_feature_columns + interaction_feature_columns
    
#     # Ensure all expected features are present
#     for feature in expected_feature_columns:
#         if feature not in df.columns:
#             df[feature] = 0  # Assign default value
    
#     # Select only the expected feature columns
#     df = df[['LogID', 'Timestamp'] + expected_feature_columns]
    
#     # Handle 'Anomalous' column if present
#     if 'Anomalous' in df.columns:
#         df = df[['LogID', 'Timestamp'] + expected_feature_columns + ['Anomalous']]
    
#     return df

# def main():
#     args = parse_arguments()
    
#     # Load configuration
#     config = load_config(args.config)
    
#     if args.mode == 'train':
#         input_file = 'data/preprocessed_train.csv'
#         output_file = 'data/enriched_train.csv'
        
#         # Load data
#         df = load_data(input_file)
        
#         # Check necessary columns
#         required_columns = [
#             'LogID', 'UserID', 'Timestamp', 'Endpoint', 'HTTP_Response',
#             'Hour', 'DayOfWeek', 'Is_After_Hours', 'Is_Internal_IP',
#             'Role_Doctor', 'Role_Nurse', 'Role_Staff', 'Role_Admin',
#             'HTTP_Method_GET', 'HTTP_Method_HEAD', 'HTTP_Method_OPTIONS',
#             'HTTP_Method_PATCH', 'HTTP_Method_POST', 'HTTP_Method_PUT', 'Anomalous'
#         ]
#         missing_columns = set(required_columns) - set(df.columns)
#         if missing_columns:
#             print(f"Error: Missing columns in training data: {missing_columns}")
#             sys.exit(1)
        
#         # Handle missing values if any (assuming preprocessing handled most)
#         # Fill missing values with appropriate defaults
#         df.fillna({
#             'Endpoint': 'unknown',
#             'HTTP_Response': 0,
#             'Role_Doctor': False,
#             'Role_Nurse': False,
#             'Role_Staff': False,
#             'Role_Admin': False,
#             'HTTP_Method_GET': False,
#             'HTTP_Method_HEAD': False,
#             'HTTP_Method_OPTIONS': False,
#             'HTTP_Method_PATCH': False,
#             'HTTP_Method_POST': False,
#             'HTTP_Method_PUT': False
#         }, inplace=True)
        
#         # Perform feature engineering
#         df_enriched, feature_columns = train_feature_engineering(df, config)
        
#         # Save enriched data
#         df_enriched.to_csv(output_file, index=False)
#         print(f"Enriched training data saved to {output_file}")
        
#     elif args.mode == 'inference':
#         input_file = 'data/preprocessed_test.csv'
#         output_file = 'data/enriched_test.csv'
        
#         # Load data
#         df = load_data(input_file)
        
#         # Check necessary columns (Anomalous might be absent)
#         required_columns = [
#             'LogID', 'UserID', 'Timestamp', 'Endpoint', 'HTTP_Response',
#             'Hour', 'DayOfWeek', 'Is_After_Hours', 'Is_Internal_IP',
#             'Role_Doctor', 'Role_Nurse', 'Role_Staff', 'Role_Admin',
#             'HTTP_Method_GET', 'HTTP_Method_HEAD', 'HTTP_Method_OPTIONS',
#             'HTTP_Method_PATCH', 'HTTP_Method_POST', 'HTTP_Method_PUT'
#         ]
#         missing_columns = set(required_columns) - set(df.columns)
#         if missing_columns:
#             print(f"Error: Missing columns in inference data: {missing_columns}")
#             sys.exit(1)
        
#         # Handle missing values if any (assuming preprocessing handled most)
#         df.fillna({
#             'Endpoint': 'unknown',
#             'HTTP_Response': 0,
#             'Role_Doctor': False,
#             'Role_Nurse': False,
#             'Role_Staff': False,
#             'Role_Admin': False,
#             'HTTP_Method_GET': False,
#             'HTTP_Method_HEAD': False,
#             'HTTP_Method_OPTIONS': False,
#             'HTTP_Method_PATCH': False,
#             'HTTP_Method_POST': False,
#             'HTTP_Method_PUT': False
#         }, inplace=True)
        
#         # Perform feature engineering
#         df_enriched = inference_feature_engineering(df, config)
        
#         # Save enriched data
#         df_enriched.to_csv(output_file, index=False)
#         print(f"Enriched inference data saved to {output_file}")
    
#     print("Feature engineering completed successfully.")

# if __name__ == "__main__":
#     main()


import pandas as pd
import argparse
import os
import sys
import pickle
import json
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Feature Engineering for Anomaly Detection')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'],
                        help='Mode of operation: train or inference')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration JSON file')
    return parser.parse_args()

def load_data(file_path):
    """
    Loads the CSV data into a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path} with shape {df.shape}.")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)

def save_pickle(obj, filename):
    """
    Saves a Python object to a pickle file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved object to {filename}")

def load_pickle(filename):
    """
    Loads a Python object from a pickle file.
    """
    if not os.path.exists(filename):
        print(f"Error: Pickle file {filename} does not exist.")
        sys.exit(1)
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(f"Loaded object from {filename}")
    return obj

def load_config(config_path):
    """
    Loads the configuration JSON file.
    """
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} does not exist.")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_path}.")
        return config
    except Exception as e:
        print(f"Error loading configuration file {config_path}: {e}")
        sys.exit(1)

def save_features_summary(features, filename):
    """
    Saves the list of features to a text file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")
    print(f"Saved features summary to {filename}")

def save_feature_plot(features, filename):
    """
    Saves a bar plot of feature counts to a PNG file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(features, range(len(features)))
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved feature plot to {filename}")

def drop_irrelevant_anomaly_features(df):
    """
    Drops anomaly features with no predictive importance.
    """
    irrelevant_features = [
        'Anomaly_Doctor_sql_injection', 'Anomaly_Doctor_xss', 
        'Anomaly_Admin_xss', 'Anomaly_Nurse_sql_injection'
    ]
    df.drop(columns=irrelevant_features, inplace=True, errors='ignore')
    return df

def compute_frequency_features(df):
    """
    Computes frequency-based features such as user request counts and unique endpoints.
    """
    df['User_Request_Count'] = df.groupby('UserID')['LogID'].transform('count')
    df['User_Unique_Endpoints'] = df.groupby('UserID')['Endpoint'].transform('nunique')

    roles = ['Doctor', 'Nurse', 'Staff', 'Admin']
    for role in roles:
        role_col = f"Role_{role}"
        df[f"User_{role}_Count"] = df.groupby('UserID')[role_col].transform('sum')

    return df

def enhance_frequency_features(df):
    """
    Adds time-based frequency features for user requests.
    """
    df['Hour_Request_Count'] = df.groupby(['UserID', 'Hour'])['LogID'].transform('count')
    df['Day_Request_Count'] = df.groupby(['UserID', 'DayOfWeek'])['LogID'].transform('count')
    return df

def create_ratio_features(df):
    """
    Creates ratio features to capture relationships between frequency-based metrics.
    """
    df['Endpoints_Per_Request'] = df['User_Unique_Endpoints'] / df['User_Request_Count'].replace(0, 1)
    df['Admin_Actions_Ratio'] = df['User_Admin_Count'] / df['User_Request_Count'].replace(0, 1)
    return df

def aggregate_role_action_features(df):
    """
    Aggregates role-action features into total action counts for each role.
    """
    roles = ['Doctor', 'Nurse', 'Staff', 'Admin']
    http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']

    for role in roles:
        df[f'{role}_Total_Actions'] = df[[f'{role}_{method}' for method in http_methods if f'{role}_{method}' in df.columns]].sum(axis=1)

    return df

def filter_interaction_features(df):
    """
    Retains only the most relevant interaction features based on EDA.
    """
    important_interactions = ['Admin_DELETE', 'Nurse_DELETE', 'Doctor_DELETE', 'Staff_DELETE']

    for col in df.columns:
        if col.startswith(('Doctor_', 'Nurse_', 'Staff_', 'Admin_')) and col not in important_interactions:
            df.drop(columns=[col], inplace=True, errors='ignore')

    return df

def identify_continuous_requests(df):
    """
    Identifies if a user makes continuous requests to the same endpoint.
    """
    # Ensure the Timestamp column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # Check for continuous requests within 10 seconds
    df['Continuous_Requests'] = df.groupby(['UserID', 'Endpoint'])['Timestamp'].transform(
        lambda x: x.diff().dt.total_seconds().fillna(float('inf')) < 10
    )
    df['Continuous_Requests'] = df['Continuous_Requests'].astype(int)
    return df


def classify_endpoint_type(df):
    """
    Classifies endpoints as "XSS", "SQL", or "Other" based on patterns.
    """
    def classify(endpoint):
        # Ensure the endpoint is a string
        endpoint = str(endpoint)
        if "<script>" in endpoint or "onerror=" in endpoint:
            return "XSS"
        elif "SELECT" in endpoint or "DROP" in endpoint or "--" in endpoint:
            return "SQL"
        else:
            return "Other"

    df['Endpoint_Type'] = df['Endpoint'].apply(classify)
    return df


def scale_features(df, feature_columns):
    """
    Scales continuous features using StandardScaler.
    """
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    save_pickle(scaler, 'mappings/feature_scaler.pkl')
    return df

def train_feature_engineering(df, config):
    """
    Performs feature engineering on training data.
    """
    # Step 1: Drop Irrelevant Features
    df = drop_irrelevant_anomaly_features(df)

    # Step 2: Compute Frequency-Based Features
    df = compute_frequency_features(df)
    df = enhance_frequency_features(df)

    # Step 3: Create Role-Action Features
    df = aggregate_role_action_features(df)
    df = filter_interaction_features(df)

    # Step 4: Create Ratio Features
    df = create_ratio_features(df)

    # Step 5: Identify Continuous Requests
    df = identify_continuous_requests(df)

    # Step 6: Classify Endpoint Type
    df = classify_endpoint_type(df)

    # Step 7: Scale Continuous Features
    continuous_features = ['User_Request_Count', 'User_Unique_Endpoints', 'Endpoints_Per_Request', 'Admin_Actions_Ratio']
    df = scale_features(df, continuous_features)

    # Save feature columns
    feature_columns = list(df.columns.difference(['LogID', 'Timestamp', 'Anomalous']))
    save_pickle(feature_columns, 'mappings/features.pkl')
    save_features_summary(feature_columns, 'mappings/feature_summary.txt')
    save_feature_plot(feature_columns, 'mappings/feature_plot.png')

    return df, feature_columns

def inference_feature_engineering(df, feature_columns, config):
    """
    Performs feature engineering on inference data and ensures consistency with training features.
    """
    # Step 1: Drop Irrelevant Features
    df = drop_irrelevant_anomaly_features(df)

    # Step 2: Compute Frequency-Based Features
    df = compute_frequency_features(df)
    df = enhance_frequency_features(df)

    # Step 3: Create Role-Action Features
    df = aggregate_role_action_features(df)
    df = filter_interaction_features(df)

    # Step 4: Create Ratio Features
    df = create_ratio_features(df)

    # Step 5: Identify Continuous Requests
    df = identify_continuous_requests(df)

    # Step 6: Classify Endpoint Type
    df = classify_endpoint_type(df)

    # Ensure all expected features are present
    for feature in feature_columns:
        if feature not in df.columns:
            df[feature] = 0  # Assign default value

    return df

def main():
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    if args.mode == 'train':
        input_file = 'data/preprocessed_train.csv'
        output_file = 'data/enriched_train.csv'

        # Load data
        df = load_data(input_file)

        # Perform feature engineering
        df_enriched, feature_columns = train_feature_engineering(df, config)

        # Save enriched data
        df_enriched.to_csv(output_file, index=False)
        print(f"Enriched training data saved to {output_file}")

    elif args.mode == 'inference':
        input_file = 'data/preprocessed_test.csv'
        output_file = 'data/enriched_test.csv'

        # Load data
        df = load_data(input_file)

        # Load feature columns from training
        feature_columns = load_pickle('mappings/features.pkl')

        # Perform feature engineering
        df_enriched = inference_feature_engineering(df, feature_columns, config)

        # Save enriched data
        df_enriched.to_csv(output_file, index=False)
        print(f"Enriched inference data saved to {output_file}")

    print("Feature engineering completed successfully.")

if __name__ == "__main__":
    main()
