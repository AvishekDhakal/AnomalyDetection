import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Define dataset path (modify if needed)
data_path = 'data/processed_data.csv'

# Load the dataset
df = pd.read_csv(data_path)

# Display basic info
print("\n=== Dataset Overview ===")
print(df.info())

# Check for missing values
print("\n=== Missing Values ===")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Display summary statistics for numeric columns
print("\n=== Summary Statistics ===")
print(df.describe())

# Distribution of categorical variables
print("\n=== Role Distribution ===")
print(df['Role'].value_counts())

print("\n=== Endpoint Level 1 Distribution ===")
print(df['endpoint_level_1'].value_counts())

print("\n=== HTTP Response Code Distribution ===")
print(df['HTTP_Response'].value_counts())

# Anomaly Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df['Anomalous'])
plt.title("Anomaly Class Distribution")
plt.xlabel("Class (0 = Normal, 1 = Anomalous)")
plt.ylabel("Count")
plt.show()

# Correlation Matrix for Numerical Features
numeric_cols = [
    'hour', 'day_of_week', 'is_unusual_time', 'is_internal_ip', 'is_authorized',
    'HTTP_Method_DELETE', 'HTTP_Method_GET', 'HTTP_Method_HEAD', 'HTTP_Method_OPTIONS',
    'HTTP_Method_PATCH', 'HTTP_Method_POST', 'HTTP_Method_PUT', 'role_risk'
]

plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Plot Hourly Activity
plt.figure(figsize=(8, 5))
sns.countplot(x=df['hour'], hue=df['Anomalous'])
plt.title("User Activity by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Logs")
plt.show()

# Feature Distributions (Histograms)
df[numeric_cols].hist(figsize=(15, 10), bins=30)
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Boxplots for Outliers
df[numeric_cols].plot(kind='box', subplots=True, layout=(4, 4), figsize=(14, 12))
plt.suptitle("Boxplots of Selected Features (Outliers Detection)")
plt.show()

print("\n=== EDA Completed ===")
