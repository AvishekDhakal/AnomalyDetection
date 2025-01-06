import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_file, train_file, test_file, test_size=0.2, random_state=42):
    """Split the processed dataset into training and testing sets."""
    print("Loading processed dataset...")
    data = pd.read_csv(input_file)

    print("Separating features and labels...")
    X = data.drop(columns=['Anomalous'])  # Features
    y = data['Anomalous']  # Labels

    print("Splitting the dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print("Saving training and testing sets...")
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    print(f"Training data saved to {train_file}")
    print(f"Testing data saved to {test_file}")

# Main function
def main():
    input_file = 'data/refined_logs.csv'
    train_file = 'data/train_logs.csv'
    test_file = 'data/test_logs.csv'

    split_data(input_file, train_file, test_file)

if __name__ == "__main__":
    main()
