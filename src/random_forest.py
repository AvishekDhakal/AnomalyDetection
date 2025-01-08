# train_random_forest.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_random_forest():
    # Define file paths
    final_engineered_features_path = "data/final_engineered_features.csv"
    model_save_path = "models/random_forest_classifier.joblib"

    # Ensure the 'models' directory exists
    os.makedirs("models", exist_ok=True)

    # Load engineered data
    engineered_data = pd.read_csv(final_engineered_features_path)
    X = engineered_data.drop(['anomalous'], axis=1)
    y = engineered_data['anomalous']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Initialize the classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced'  # Alternative to SMOTE
    )

    # Train the model
    rf_classifier.fit(X_train_resampled, y_train_resampled)

    # Predict
    y_pred = rf_classifier.predict(X_test)
    y_proba = rf_classifier.predict_proba(X_test)[:,1]

    # Evaluate
    print("Random Forest Classifier Performance:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # Save the trained model
    joblib.dump(rf_classifier, model_save_path)
    print(f"Random Forest model saved at '{model_save_path}'")

if __name__ == "__main__":
    train_random_forest()
