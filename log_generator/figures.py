import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the feature-engineered dataset
file_path = "data/preprocessed_train.csv"



df = pd.read_csv(file_path)

# Set plot styles
sns.set(style="whitegrid")

# ---- FIGURE 2.1: Modified Feature Importance Graph (Grouped by Category) ----
# Define feature categories
access_features = ["HTTP_Method_DELETE", "HTTP_Method_GET", "HTTP_Method_POST", "HTTP_Method_PUT"]
role_features = ["role_risk", "is_authorized"]
time_features = ["hour", "day_of_week", "is_unusual_time"]

# Compute mean values for each category
feature_importance = {
    "Access-Based": df[access_features].mean().mean(),
    "Role-Based": df[role_features].mean().mean(),
    "Time-Based": df[time_features].mean().mean(),
}

# Convert to DataFrame for visualization
importance_df = pd.DataFrame(list(feature_importance.items()), columns=["Category", "Mean Importance"])

# Plot feature importance by category
plt.figure(figsize=(8, 5))
sns.barplot(x="Category", y="Mean Importance", data=importance_df, palette="Blues_r")
plt.title("Figure 2.1 - Modified Feature Importance (Grouped by Category)")
plt.ylabel("Average Feature Contribution")
plt.xlabel("Feature Category")
plt.show()

# ---- FIGURE 2.2: Hourly Anomaly Distribution Plot ----
plt.figure(figsize=(10, 5))
sns.histplot(df[df["Anomalous"] == 1]["hour"], bins=24, kde=True, color="red", label="Anomalous")
sns.histplot(df[df["Anomalous"] == 0]["hour"], bins=24, kde=True, color="blue", label="Normal")
plt.legend()
plt.title("Figure 2.2 - Hourly Anomaly Distribution")
plt.xlabel("Hour of the Day")
plt.ylabel("Frequency")
plt.show()

# ---- FIGURE 2.3: Role-Based Anomaly Rate Graph ----
role_anomaly_rate = df.groupby("Role")["Anomalous"].mean().sort_values()

plt.figure(figsize=(8, 5))
sns.barplot(x=role_anomaly_rate.index, y=role_anomaly_rate.values, palette="coolwarm")
plt.title("Figure 2.3 - Role-Based Anomaly Rate")
plt.xlabel("Role")
plt.ylabel("Anomaly Rate")
plt.xticks(rotation=45)
plt.show()

# ---- FIGURE 2.4: Conceptual Diagram of Behavioral Feature Processing ----
# Creating a simple conceptual diagram
fig, ax = plt.subplots(figsize=(8, 5))
ax.text(0.5, 0.9, "Behavioral Feature Processing", fontsize=14, ha="center", fontweight="bold")

# Components of Behavioral Feature Processing
components = [
    "Access Features (HTTP Methods)",
    "Role Features (User Privileges)",
    "Time-Based Features (Login Hours)",
    "Anomaly Detection Model",
    "Insider Threat Flagging"
]

for i, text in enumerate(components):
    ax.text(0.5, 0.75 - i * 0.15, text, fontsize=12, ha="center", bbox=dict(boxstyle="round", facecolor="lightblue"))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

plt.title("Figure 2.4 - Conceptual Diagram of Behavioral Feature Processing")
plt.show()

# ---- FIGURE 2.5: Risky HTTP Methods (DELETE, POST) ----
plt.figure(figsize=(8, 5))
df_risky_http = df.groupby("Anomalous")[["HTTP_Method_DELETE", "HTTP_Method_POST"]].sum()
df_risky_http.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="coolwarm")
plt.title("Figure 2.5 - Risky HTTP Methods (DELETE, POST)")
plt.xlabel("Anomaly Status (0=Normal, 1=Anomalous)")
plt.ylabel("Count")
plt.legend(title="HTTP Methods")
plt.show()

# ---- FIGURE 2.6: Authorization Status ----
plt.figure(figsize=(8, 5))
sns.countplot(x=df["is_authorized"], hue=df["Anomalous"], palette="Set1")
plt.title("Figure 2.6 - Authorization Status")
plt.xlabel("Is Authorized (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.legend(title="Anomalous")
plt.show()

print("EDA and visualizations completed successfully!")