import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load engineered logs
file_path = 'data/engineered_logs.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Separate features and labels
X = df.drop(columns=['LogID', 'Timestamp', 'Endpoint', 'IP_Address', 'Anomalous'])
y = df['Anomalous']

# Ensure all features are numeric
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
X = X[numeric_columns]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply scaling to numeric columns
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Ensure numeric_columns in X_train_scaled and X_test_scaled are float64 before assignment
X_train_scaled[numeric_columns] = X_train[numeric_columns].astype('float64')
X_test_scaled[numeric_columns] = X_test[numeric_columns].astype('float64')

# Scale the numeric columns
X_train_scaled[numeric_columns] = scaler.fit_transform(X_train_scaled[numeric_columns])
X_test_scaled[numeric_columns] = scaler.transform(X_test_scaled[numeric_columns])


# Fit scaler on training data and transform both training and testing sets
# Fit scaler on training data and transform both training and testing sets
# X_train_scaled.loc[:, numeric_columns] = scaler.fit_transform(X_train[numeric_columns]).astype('float64')
# X_test_scaled.loc[:, numeric_columns] = scaler.transform(X_test[numeric_columns]).astype('float64')


# Save the scaler for reuse in model deployment
import joblib
joblib.dump(scaler, 'scaler.joblib')

import joblib
joblib.dump(X_train_scaled, "X_train_scaled.pkl")
joblib.dump(X_test_scaled, "X_test_scaled.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")

# Output shapes for confirmation
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
