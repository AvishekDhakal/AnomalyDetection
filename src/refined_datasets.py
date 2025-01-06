import pandas as pd

def refine_dataset(input_file, output_file):
    """Refine the dataset by retaining high-contributing features."""
    print("Loading dataset...")
    data = pd.read_csv(input_file)

    print("Retaining high-contributing features...")
    # Retain only high-contributing features based on the feature importance analysis
    selected_features = [
        'IsSensitiveEndpoint',  # High importance
        'Hour',                 # High importance
        'UniqueEndpointCount',  # High importance
        'RequestCount',         # High importance
        'HTTP_Method',          # Moderate importance
        'IsUnauthorizedRoleAction',  # Relevant for access violations
        'DayOfWeek',            # Moderate contribution, useful for patterns
        'Anomalous'             # Target label
    ]

    refined_data = data[selected_features]

    print("Saving refined dataset...")
    refined_data.to_csv(output_file, index=False)
    print(f"Refined dataset saved to {output_file}")

# Main function
def main():
    input_file = 'data/preprocessed_logs.csv'
    output_file = 'data/refined_logs.csv'

    refine_dataset(input_file, output_file)

if __name__ == "__main__":
    main()
