# hybrid_model.py

import pandas as pd
import numpy as np
import os
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(log_path='logs/hybrid_model.log'):
    """Setup logging configuration."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging is set up.")

def load_data(data_path):
    """Load dataset from a CSV file."""
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully with shape: {data.shape}")
        print(f"Data loaded successfully with shape: {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Failed to load data from {data_path}: {e}")
        print(f"Error: Failed to load data with error: {e}")
        raise

def preprocess_data(data):
    """Prepare features and target variable."""
    try:
        X = data.drop(['anomalous', 'logid'], axis=1)
        y = data['anomalous']
        logging.info("Prepared feature matrix X and target vector y.")
        print("Prepared feature matrix X and target vector y.")
        return X, y
    except KeyError as ke:
        logging.error(f"KeyError in preparing features: {ke}")
        print(f"Error: Preparing features failed with error: {ke}")
        raise
    except Exception as e:
        logging.error(f"Error in preparing features: {e}")
        print(f"Error: Preparing features failed with error: {e}")
        raise

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logging.info(f"Data split into train and test sets with test size {test_size}.")
        print(f"Data split into train and test sets with test size {test_size}.")
        print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in splitting data: {e}")
        print(f"Error: Splitting data failed with error: {e}")
        raise

def balance_data_SMOTE(X_train, y_train, random_state=42):
    """Apply SMOTE to balance the training data."""
    try:
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logging.info("Applied SMOTE to balance the training data.")
        print("Applied SMOTE to balance the training data.")
        # Visualize the class distribution after SMOTE
        sns.countplot(x=y_res)
        plt.title('Class Distribution After SMOTE')
        plt.savefig('models/hybrid_model_smote_distribution.png')
        plt.close()
        logging.info("SMOTE distribution plot saved at 'models/hybrid_model_smote_distribution.png'.")
        print("SMOTE distribution plot saved at 'models/hybrid_model_smote_distribution.png'.")
        return X_res, y_res
    except Exception as e:
        logging.error(f"Error in applying SMOTE: {e}")
        print(f"Error: Applying SMOTE failed with error: {e}")
        raise

def train_models(X_train, y_train):
    """Train Random Forest and Gradient Boosting models."""
    try:
        # Model A: Random Forest optimized for Accuracy
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        logging.info("Random Forest model trained successfully.")
        print("Random Forest model trained successfully.")

        # Model B: Gradient Boosting optimized for Recall
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=1,
            random_state=42
        )
        gb.fit(X_train, y_train)
        logging.info("Gradient Boosting model trained successfully.")
        print("Gradient Boosting model trained successfully.")

        return rf, gb
    except Exception as e:
        logging.error(f"Error in training individual models: {e}")
        print(f"Error: Training individual models failed with error: {e}")
        raise

def create_voting_classifier(rf, gb):
    """Create a Voting Classifier that combines Random Forest and Gradient Boosting."""
    try:
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb)
            ],
            voting='soft',  # Use soft voting to leverage predicted probabilities
            n_jobs=-1
        )
        ensemble.fit(rf.estimators_[0].tree_.value, gb.estimators_[0].tree_.value)  # Placeholder for fitting
        logging.info("Voting Classifier created successfully.")
        print("Voting Classifier created successfully.")
        return ensemble
    except Exception as e:
        logging.error(f"Error in creating Voting Classifier: {e}")
        print(f"Error: Creating Voting Classifier failed with error: {e}")
        raise

def evaluate_ensemble(ensemble, X_test, y_test, model_name='Ensemble'):
    """Evaluate the ensemble model and print/save metrics."""
    try:
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        # Print metrics
        print(f"\n--- {model_name} Model Evaluation Metrics ---")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)

        # Log metrics
        logging.info(f"{model_name} Accuracy: {accuracy:.4f}")
        logging.info(f"{model_name} Precision: {precision:.4f}")
        logging.info(f"{model_name} Recall: {recall:.4f}")
        logging.info(f"{model_name} F1-Score: {f1:.4f}")
        logging.info(f"{model_name} ROC-AUC: {roc_auc:.4f}")
        logging.info(f"{model_name} Confusion Matrix:\n{conf_matrix}")
        logging.info(f"{model_name} Classification Report:\n{class_report}")

        # Plot Confusion Matrix
        plt.figure(figsize=(6,4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'models/hybrid_{model_name.lower()}_confusion_matrix.png')
        plt.close()
        logging.info(f"{model_name} confusion matrix plot saved at 'models/hybrid_{model_name.lower()}_confusion_matrix.png'.")
        print(f"{model_name} confusion matrix plot saved at 'models/hybrid_{model_name.lower()}_confusion_matrix.png'.")

        # Plot Precision-Recall Curve
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)
        plt.figure(figsize=(8,6))
        plt.plot(recall_vals, precision_vals, marker='.', label=model_name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'models/hybrid_{model_name.lower()}_precision_recall_curve.png')
        plt.close()
        logging.info(f"{model_name} Precision-Recall curve plot saved at 'models/hybrid_{model_name.lower()}_precision_recall_curve.png'.")
        print(f"{model_name} Precision-Recall curve plot saved at 'models/hybrid_{model_name.lower()}_precision_recall_curve.png'.")

    except Exception as e:
        logging.error(f"Error in evaluating {model_name} model: {e}")
        print(f"Error: Evaluating {model_name} model failed with error: {e}")
        raise

def adjust_threshold(ensemble, X_test, y_test, desired_recall=0.90, model_name='Ensemble'):
    """
    Adjusts the decision threshold to achieve the desired recall while maintaining precision.
    
    Parameters:
    - ensemble: Trained Voting Classifier.
    - X_test: Test features.
    - y_test: Test labels.
    - desired_recall: The minimum recall desired.
    - model_name: Name of the model for logging.
    """
    try:
        y_proba = ensemble.predict_proba(X_test)[:, 1]
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)
        
        # Find the threshold where recall is at least desired_recall
        indices = np.where(recall_vals >= desired_recall)[0]
        if len(indices) == 0:
            optimal_threshold = 0.5  # Default threshold
            print(f"{model_name} Warning: Desired recall of {desired_recall} not achievable. Using default threshold of 0.5.")
            logging.warning(f"{model_name} Desired recall of {desired_recall} not achievable. Using default threshold of 0.5.")
        else:
            optimal_threshold = thresholds[indices[-1]]  # Choose the lowest threshold meeting recall
            print(f"{model_name} Info: Optimal threshold determined: {optimal_threshold:.4f}")
            logging.info(f"{model_name} Optimal threshold determined: {optimal_threshold:.4f}")
        
        # Apply the new threshold
        y_pred_adjusted = (y_proba >= optimal_threshold).astype(int)
        
        # Recalculate metrics
        accuracy = accuracy_score(y_test, y_pred_adjusted)
        precision_val = precision_score(y_test, y_pred_adjusted)
        recall_val = recall_score(y_test, y_pred_adjusted)
        f1 = f1_score(y_test, y_pred_adjusted)
        roc_auc = roc_auc_score(y_test, y_proba)
        conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
        class_report = classification_report(y_test, y_pred_adjusted)

        # Print metrics
        print(f"\n--- {model_name} Model Evaluation Metrics (Adjusted Threshold) ---")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision_val:.4f}")
        print(f"Recall   : {recall_val:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)

        # Log metrics
        logging.info(f"{model_name} Adjusted Threshold Accuracy: {accuracy:.4f}")
        logging.info(f"{model_name} Adjusted Threshold Precision: {precision_val:.4f}")
        logging.info(f"{model_name} Adjusted Threshold Recall: {recall_val:.4f}")
        logging.info(f"{model_name} Adjusted Threshold F1-Score: {f1:.4f}")
        logging.info(f"{model_name} Adjusted Threshold ROC-AUC: {roc_auc:.4f}")
        logging.info(f"{model_name} Adjusted Threshold Confusion Matrix:\n{conf_matrix}")
        logging.info(f"{model_name} Adjusted Threshold Classification Report:\n{class_report}")

        # Save adjusted predictions
        predictions_df = pd.DataFrame({
            'logid': X_test.index,  # Ensure 'logid' is the index or adjust accordingly
            'actual': y_test,
            'predicted': y_pred_adjusted
        })
        predictions_df.to_csv('data/hybrid_model_predictions_adjusted_threshold.csv', index=False)
        print(f"{model_name} Adjusted threshold predictions saved successfully at 'data/hybrid_model_predictions_adjusted_threshold.csv'.")
        logging.info(f"{model_name} Adjusted threshold predictions saved successfully at 'data/hybrid_model_predictions_adjusted_threshold.csv'.")

    except Exception as e:
        logging.error(f"Error in adjusting threshold for {model_name}: {e}")
        print(f"Error: Adjusting threshold for {model_name} failed with error: {e}")
        raise

def save_model(model, model_path='models/hybrid_model.joblib'):
    """Save the trained ensemble model to disk."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logging.info(f"Model saved successfully at '{model_path}'.")
        print(f"Model saved successfully at '{model_path}'.")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        print(f"Error: Saving model failed with error: {e}")
        raise

def save_predictions(model, X_test, y_test, predictions_save_path='data/hybrid_model_predictions.csv'):
    """Save the initial predictions to a CSV file."""
    try:
        y_pred_initial = model.predict(X_test)
        predictions_df = pd.DataFrame({
            'logid': X_test.index,  # Ensure 'logid' is the index or adjust accordingly
            'actual': y_test,
            'predicted': y_pred_initial
        })
        predictions_df.to_csv(predictions_save_path, index=False)
        logging.info(f"Initial predictions saved successfully at '{predictions_save_path}'.")
        print(f"Initial predictions saved successfully at '{predictions_save_path}'.")
    except Exception as e:
        logging.error(f"Failed to save initial predictions: {e}")
        print(f"Error: Saving initial predictions failed with error: {e}")

def main():
    # Setup logging
    setup_logging()

    # Define file paths
    data_path = "data/final_engineered_features.csv"
    model_save_path = "models/hybrid_model.joblib"
    predictions_save_path = "data/hybrid_model_predictions.csv"

    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Load data
    data = load_data(data_path)

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Balance data using SMOTE
    X_train_res, y_train_res = balance_data_SMOTE(X_train, y_train)

    # Train individual models
    rf_model, gb_model = train_models(X_train_res, y_train_res)

    # Create Voting Classifier
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        voting='soft',  # Use soft voting to leverage predicted probabilities
        n_jobs=-1
    )
    ensemble.fit(X_train_res, y_train_res)
    logging.info("Ensemble model trained successfully.")
    print("Ensemble model trained successfully.")

    # Evaluate ensemble model
    evaluate_ensemble(ensemble, X_test, y_test, model_name='Ensemble')

    # Adjust threshold for ensemble
    adjust_threshold(ensemble, X_test, y_test, desired_recall=0.90, model_name='Ensemble')

    # Save the trained ensemble model
    save_model(ensemble, model_path=model_save_path)

    # Save initial predictions
    save_predictions(ensemble, X_test, y_test, predictions_save_path=predictions_save_path)

if __name__ == "__main__":
    main()
