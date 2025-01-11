# simple_random_forest.py

import pandas as pd
import numpy as np
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(log_path='logs/simple_random_forest.log'):
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
        plt.savefig('models/simple_random_forest_smote_distribution.png')
        plt.close()
        logging.info("SMOTE distribution plot saved at 'models/simple_random_forest_smote_distribution.png'.")
        print("SMOTE distribution plot saved at 'models/simple_random_forest_smote_distribution.png'.")
        return X_res, y_res
    except Exception as e:
        logging.error(f"Error in applying SMOTE: {e}")
        print(f"Error: Applying SMOTE failed with error: {e}")
        raise

def hyperparameter_tuning(X_train, y_train, model_name='Random Forest'):
    """
    Performs simple hyperparameter tuning using GridSearchCV.
    
    Parameters:
    - X_train: Training features.
    - y_train: Training labels.
    - model_name: Name of the model for logging.
    
    Returns:
    - best_model: Model with the best found parameters.
    """
    try:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring='recall',  # Focus on improving recall
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"{model_name} Info: Best parameters found: {grid_search.best_params_}")
        logging.info(f"{model_name} Best parameters found: {grid_search.best_params_}")
        
        return best_model
    except Exception as e:
        logging.error(f"Error in hyperparameter tuning for {model_name}: {e}")
        print(f"Error: Hyperparameter tuning for {model_name} failed with error: {e}")
        raise

def train_random_forest(X_train, y_train, random_state=42):
    """Train the Random Forest classifier."""
    try:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=random_state,
            class_weight='balanced',  # To handle class imbalance
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        logging.info("Random Forest model trained successfully.")
        print("Random Forest model trained successfully.")
        return rf
    except Exception as e:
        logging.error(f"Error in training Random Forest model: {e}")
        print(f"Error: Training Random Forest model failed with error: {e}")
        raise

def evaluate_model(model, X_test, y_test, model_name='Random Forest'):
    """Evaluate the trained model and print/save metrics."""
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

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
        plt.savefig(f'models/simple_random_forest_confusion_matrix.png')
        plt.close()
        logging.info(f"{model_name} confusion matrix plot saved at 'models/simple_random_forest_confusion_matrix.png'.")
        print(f"{model_name} confusion matrix plot saved at 'models/simple_random_forest_confusion_matrix.png'.")

        # Plot Precision-Recall Curve
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)
        plt.figure(figsize=(8,6))
        plt.plot(recall_vals, precision_vals, marker='.', label=model_name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'models/simple_random_forest_precision_recall_curve.png')
        plt.close()
        logging.info(f"{model_name} Precision-Recall curve plot saved at 'models/simple_random_forest_precision_recall_curve.png'.")
        print(f"{model_name} Precision-Recall curve plot saved at 'models/simple_random_forest_precision_recall_curve.png'.")

    except Exception as e:
        logging.error(f"Error in evaluating {model_name} model: {e}")
        print(f"Error: Evaluating {model_name} model failed with error: {e}")
        raise

def adjust_threshold(model, X_test, y_test, desired_recall=0.90, model_name='Random Forest'):
    """
    Adjusts the decision threshold to achieve the desired recall while maintaining precision.
    
    Parameters:
    - model: Trained classifier.
    - X_test: Test features.
    - y_test: Test labels.
    - desired_recall: The minimum recall desired.
    - model_name: Name of the model for logging.
    """
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
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
        predictions_df.to_csv('data/simple_random_forest_predictions_adjusted_threshold.csv', index=False)
        print(f"{model_name} Adjusted threshold predictions saved successfully at 'data/simple_random_forest_predictions_adjusted_threshold.csv'.")
        logging.info(f"{model_name} Adjusted threshold predictions saved successfully at 'data/simple_random_forest_predictions_adjusted_threshold.csv'.")
        
    except Exception as e:
        logging.error(f"Error in adjusting threshold for {model_name}: {e}")
        print(f"Error: Adjusting threshold for {model_name} failed with error: {e}")
        raise

def plot_feature_importance(model, X, model_name='Random Forest'):
    """
    Plots and saves the feature importances of the model.
    
    Parameters:
    - model: Trained classifier.
    - X: Feature dataframe.
    - model_name: Name of the model for logging.
    """
    try:
        importances = model.feature_importances_
        feature_names = X.columns
        feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
        plt.figure(figsize=(10,8))
        sns.barplot(x=feature_importances[:20], y=feature_importances.index[:20])
        plt.title(f'{model_name} Top 20 Feature Importances')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(f'models/simple_random_forest_feature_importances.png')
        plt.close()
        
        logging.info(f"{model_name} feature importance plot saved at 'models/simple_random_forest_feature_importances.png'.")
        print(f"{model_name} feature importance plot saved at 'models/simple_random_forest_feature_importances.png'.")
    except Exception as e:
        logging.error(f"Error in plotting feature importance for {model_name}: {e}")
        print(f"Error: Plotting feature importance for {model_name} failed with error: {e}")
        raise

def save_model(model, model_path='models/simple_random_forest_model.joblib'):
    """Save the trained model to disk."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logging.info(f"Model saved successfully at '{model_path}'.")
        print(f"Model saved successfully at '{model_path}'.")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        print(f"Error: Saving model failed with error: {e}")
        raise

def save_initial_predictions(model, X_test, y_test, predictions_save_path='data/simple_random_forest_predictions.csv'):
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
    model_save_path = "models/simple_random_forest_model.joblib"
    predictions_save_path = "data/simple_random_forest_predictions.csv"

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

    # Perform hyperparameter tuning
    best_rf_model = hyperparameter_tuning(X_train_res, y_train_res, model_name='Random Forest')

    # Train the model with the best parameters
    rf_model = best_rf_model.fit(X_train_res, y_train_res)
    logging.info("Random Forest model trained with best parameters.")
    print("Random Forest model trained with best parameters.")

    # Evaluate model
    evaluate_model(rf_model, X_test, y_test, model_name='Random Forest')

    # Adjust threshold
    adjust_threshold(rf_model, X_test, y_test, desired_recall=0.90, model_name='Random Forest')

    # Plot feature importances
    plot_feature_importance(rf_model, X_train_res, model_name='Random Forest')

    # Save the trained model
    save_model(rf_model, model_path=model_save_path)

    # Save initial predictions
    save_initial_predictions(rf_model, X_test, y_test, predictions_save_path=predictions_save_path)

if __name__ == "__main__":
    main()
