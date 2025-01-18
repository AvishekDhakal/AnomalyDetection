import pandas as pd

# Load preprocessed training data
df_train = pd.read_csv("processed_data/X_train.csv")
print("Columns in X_train:", df_train.columns.tolist())

# Check if 'Time_Since_Last_Access' exists
if 'Time_Since_Last_Access' in df_train.columns:
    print("'Time_Since_Last_Access' is present in X_train.")
else:
    print("'Time_Since_Last_Access' is missing in X_train.")
