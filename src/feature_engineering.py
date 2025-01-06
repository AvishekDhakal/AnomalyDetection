import pandas as pd
import numpy as np

# Load the logs
def load_logs(file_path):
    """Load the raw logs from a CSV file."""
    return pd.read_csv(file_path)

# Time-Based Features
def add_time_features(df):
    """Add features based on the timestamp."""
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['IsBusinessHour'] = df['Hour'].between(8, 20).astype(int)
    return df

# Role-Endpoint Features
def add_role_endpoint_features(df):
    """Add features that encode the role and endpoint behavior."""
    sensitive_endpoints = ['/api/admin/credentials', '/api/patient/confidential']
    df['IsSensitiveEndpoint'] = df['Endpoint'].apply(lambda x: int(any(ep in x for ep in sensitive_endpoints)))

    unauthorized_roles = {
        'Nurse': ['/api/admin/settings', '/api/admin/credentials'],
        'Doctor': ['/api/admin/credentials']
    }

    def is_unauthorized_access(row):
        role = row['Role']
        endpoint = row['Endpoint']
        if role in unauthorized_roles:
            return int(any(ep in endpoint for ep in unauthorized_roles[role]))
        return 0

    df['IsUnauthorizedRoleAction'] = df.apply(is_unauthorized_access, axis=1)
    return df

# Behavioral Features
def add_behavioral_features(df):
    """Add features based on user behavior."""
    df['RequestCount'] = df.groupby('UserID')['Timestamp'].transform('count')
    df['UniqueEndpointCount'] = df.groupby('UserID')['Endpoint'].transform('nunique')
    return df

# Feature Engineering Pipeline
def feature_engineering_pipeline(input_file, output_file):
    """Run the feature engineering pipeline."""
    print("Loading logs...")
    logs = load_logs(input_file)

    print("Adding time-based features...")
    logs = add_time_features(logs)

    print("Adding role-endpoint features...")
    logs = add_role_endpoint_features(logs)

    print("Adding behavioral features...")
    logs = add_behavioral_features(logs)

    print("Saving processed logs...")
    logs.to_csv(output_file, index=False)
    print(f"Processed logs saved to {output_file}")

# Main function
def main():
    input_file = 'data/enhanced_logs.csv'
    output_file = 'data/processed_logs.csv'
    feature_engineering_pipeline(input_file, output_file)

if __name__ == "__main__":
    main()
