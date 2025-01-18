import pandas as pd

# Load the CSV file
file_path = 'data/test_enriched_logs.csv'
data = pd.read_csv(file_path)

# Display the headers (column names)
print("Column names in the CSV file:")
print(data.columns.tolist())
