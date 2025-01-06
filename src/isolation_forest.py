import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib

def train_isolation_forest(train_file, test_file, model_file):
    """Train and evaluate an Isolation Forest model."""

    print("Loading training and testing datasets...")
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Separate features and labels
    X_train = train_data.drop(columns=['Anomalous'])
    y_train = train_data['Anomalous']
    X_test = test_data.drop(columns=['Anomalous'])
    y_test = test_data['Anomalous']

    print("Training Isolation Forest model...")
    model = IsolationForest(n_estimators=200, contamination=0.20, random_state=42)
    model.fit(X_train)

    print("Making predictions on the testing set...")
    y_pred = model.predict(X_test)

    # Convert Isolation Forest predictions to match binary labels (1 = anomaly, 0 = normal)
    y_pred = [1 if p == -1 else 0 for p in y_pred]

    print("Evaluating model performance...")
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")

    print("Saving the trained model...")
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

# Main function
def main():
    train_file = 'data/train_logs.csv'
    test_file = 'data/test_logs.csv'
    model_file = 'models/isolation_forest_model.pkl'

    train_isolation_forest(train_file, test_file, model_file)

if __name__ == "__main__":
    main()
