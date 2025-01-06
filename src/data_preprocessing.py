import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """Preprocess the dataset to ensure all values are numeric and properly formatted."""
    print("Converting Timestamp to numerical features...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour.astype(int)  # Explicitly cast to int
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek.astype(int)  # Explicitly cast to int
    df['IsBusinessHour'] = df['Hour'].between(8, 20).astype(int)  # Explicitly cast to int

    print("Encoding categorical features...")
    label_encoders = {}
    for column in ['Role', 'HTTP_Method', 'Endpoint']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    print("Dropping unnecessary columns...")
    df = df.drop(columns=['Timestamp'])  # Drop the original Timestamp as it's no longer needed

    print("Ensuring all data is numeric...")
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column {column} is not numeric after preprocessing!")

    return df, label_encoders


def preprocess_and_save(input_file, output_file):
    """Preprocess data from input_file and save to output_file."""
    print("Loading dataset...")
    data = pd.read_csv(input_file)

    print("Preprocessing data...")
    data, _ = preprocess_data(data)

    print("Saving preprocessed dataset...")
    data.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

# Main function
def main():
    input_file = 'data/processed_logs.csv'
    output_file = 'data/preprocessed_logs.csv'

    preprocess_and_save(input_file, output_file)

if __name__ == "__main__":
    main()
