# train_xgboost.py

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_xgboost():
    # Define file paths
    final_engineered_features_path = "data/final_engineered_features.csv"
    model_save_path = "models/xgboost_classifier.joblib"

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
    xgb_classifier = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=42,
        scale_pos_weight=y_train_resampled.value_counts()[0] / y_train_resampled.value_counts()[1],
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Train the model
    xgb_classifier.fit(X_train_resampled, y_train_resampled)

    # Predict
    y_pred = xgb_classifier.predict(X_test)
    y_proba = xgb_classifier.predict_proba(X_test)[:,1]

    # Evaluate
    print("XGBoost Classifier Performance:")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # Save the trained model
    joblib.dump(xgb_classifier, model_save_path)
    print(f"XGBoost model saved at '{model_save_path}'")

if __name__ == "__main__":
    train_xgboost()
