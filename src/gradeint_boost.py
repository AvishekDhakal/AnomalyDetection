# train_gradient_boosting_enhanced.py

import pandas as pd
import numpy as np
import os
import logging
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform

class GradientBoostingEnhancedTrainer:
    def __init__(self, data_path, model_save_path='models/gradient_boosting_model_enhanced.joblib', 
                 predictions_save_path='data/gradient_boosting_predictions_enhanced.csv'):
        """
        Initialize the GradientBoostingEnhancedTrainer with paths to data and where to save the model and predictions.

        Parameters:
        - data_path (str): Path to the final engineered CSV file.
        - model_save_path (str): Path to save the trained Gradient Boosting model.
        - predictions_save_path (str): Path to save the predictions along with logid.
        """
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.predictions_save_path = predictions_save_path
        self.model = None
        self.X = None
        self.y = None
        self.log_ids = None
        self.X_train_res = None
        self.y_train_res = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.log_ids_train = None
        self.log_ids_test = None

        # Setup logging
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            filename='logs/train_gradient_boosting_enhanced.log',
            filemode='a',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("GradientBoostingEnhancedTrainer initialized.")

    def print_columns(self, df, step_description):
        """
        Print and log the current columns of the DataFrame for debugging purposes.

        Parameters:
        - df (pd.DataFrame): The DataFrame to inspect.
        - step_description (str): Description of the current step.
        """
        print(f"\n--- Columns after {step_description} ---")
        print(df.columns.tolist())
        logging.info(f"Columns after {step_description}: {df.columns.tolist()}")

    def load_data(self):
        """
        Load the final engineered data from the CSV file into a pandas DataFrame.
        """
        try:
            self.data = pd.read_csv(self.data_path)
            logging.info(f"Data loaded successfully with shape: {self.data.shape}")
            print(f"Gradient Boosting Training: Data loaded successfully with shape: {self.data.shape}")
            self.print_columns(self.data, "loading data")
        except Exception as e:
            logging.error(f"Failed to load data from {self.data_path}: {e}")
            print(f"Gradient Boosting Training Error: Failed to load data with error: {e}")
            raise

    def validate_data(self):
        """
        Validate the presence of essential columns.
        """
        try:
            required_columns = [
                'anomalous', 'logid'
            ]
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise KeyError(f"Required columns are missing from the data: {missing_columns}")
            logging.info("Data validation passed.")
            print("Gradient Boosting Training: Data validation passed.")
        except KeyError as ke:
            logging.error(f"Data validation failed: {ke}")
            print(f"Gradient Boosting Training Error: Data validation failed with error: {ke}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during data validation: {e}")
            print(f"Gradient Boosting Training Error: Unexpected error during data validation: {e}")
            raise

    def prepare_features(self):
        """
        Prepare the feature matrix X and target vector y.
        """
        try:
            self.X = self.data.drop(['anomalous', 'logid'], axis=1)
            self.y = self.data['anomalous']
            self.log_ids = self.data['logid']
            logging.info("Prepared feature matrix X and target vector y.")
            print("Gradient Boosting Training: Prepared feature matrix X and target vector y.")
            self.print_columns(self.X, "preparing features")
        except KeyError as ke:
            logging.error(f"KeyError in preparing features: {ke}")
            print(f"Gradient Boosting Training Error: Preparing features failed with error: {ke}")
            raise
        except Exception as e:
            logging.error(f"Error in preparing features: {e}")
            print(f"Gradient Boosting Training Error: Preparing features failed with error: {e}")
            raise

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.

        Parameters:
        - test_size (float): Proportion of the dataset to include in the test split.
        - random_state (int): Random seed.
        """
        try:
            self.X_train, self.X_test, self.y_train, self.y_test, self.log_ids_train, self.log_ids_test = train_test_split(
                self.X, self.y, self.log_ids, test_size=test_size, random_state=random_state, stratify=self.y
            )
            logging.info(f"Data split into train and test sets with test size {test_size}.")
            print(f"Gradient Boosting Training: Data split into train and test sets with test size {test_size}.")
            print(f"Training set shape: {self.X_train.shape}, Testing set shape: {self.X_test.shape}")
        except Exception as e:
            logging.error(f"Error in splitting data: {e}")
            print(f"Gradient Boosting Training Error: Splitting data failed with error: {e}")
            raise

    def resample_data(self):
        """
        Apply SMOTE to balance the training data.
        """
        try:
            smote = SMOTE(random_state=42)
            self.X_train_res, self.y_train_res = smote.fit_resample(self.X_train, self.y_train)
            logging.info("Applied SMOTE to balance the training data.")
            print("Gradient Boosting Training: Applied SMOTE to balance the training data.")
            # Optionally, visualize the resampled data distribution
            sns.countplot(x=self.y_train_res)
            plt.title('Distribution of Classes After SMOTE')
            plt.savefig('models/gradient_boosting_smote_distribution.png')
            plt.close()
            logging.info("SMOTE distribution plot saved at 'models/gradient_boosting_smote_distribution.png'.")
            print("Gradient Boosting Training: SMOTE distribution plot saved at 'models/gradient_boosting_smote_distribution.png'.")
        except Exception as e:
            logging.error(f"Error in resampling data: {e}")
            print(f"Gradient Boosting Training Error: Resampling data failed with error: {e}")
            raise

    def hyperparameter_tuning(self):
        """
        Perform hyperparameter tuning using RandomizedSearchCV to find the best parameters.
        """
        try:
            param_dist = {
                'n_estimators': randint(100, 500),
                'learning_rate': uniform(0.01, 0.29),  # 0.01 to 0.3
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 11),
                'min_samples_leaf': randint(1, 5),
                'subsample': uniform(0.5, 0.5)  # 0.5 to 1.0
            }

            gb = GradientBoostingClassifier(random_state=42)

            random_search = RandomizedSearchCV(
                estimator=gb,
                param_distributions=param_dist,
                n_iter=50,
                cv=3,
                scoring='f1',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )

            random_search.fit(self.X_train_res, self.y_train_res)

            self.model = random_search.best_estimator_
            logging.info(f"Gradient Boosting best parameters: {random_search.best_params_}")
            print(f"Gradient Boosting Training: Best parameters found: {random_search.best_params_}")
        except Exception as e:
            logging.error(f"Error in hyperparameter tuning: {e}")
            print(f"Gradient Boosting Training Error: Hyperparameter tuning failed with error: {e}")
            raise

    def train_model(self):
        """
        Train the Gradient Boosting model with optimized hyperparameters.
        """
        try:
            if self.model is None:
                # If hyperparameter tuning wasn't done, train with default parameters
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
                self.model.fit(self.X_train_res, self.y_train_res)
                logging.info("Gradient Boosting model trained with default parameters.")
                print("Gradient Boosting Training: Model trained with default parameters.")
            else:
                # Model is already trained via hyperparameter tuning
                logging.info("Gradient Boosting model trained via hyperparameter tuning.")
                print("Gradient Boosting Training: Model trained via hyperparameter tuning.")
        except Exception as e:
            logging.error(f"Error in training Gradient Boosting model: {e}")
            print(f"Gradient Boosting Training Error: Training model failed with error: {e}")
            raise

    def evaluate_model(self):
        """
        Evaluate the trained model on the test set and print metrics.
        """
        try:
            y_pred = self.model.predict(self.X_test)
            y_proba = self.model.predict_proba(self.X_test)[:, 1]

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_proba)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            class_report = classification_report(self.y_test, y_pred)

            # Print metrics
            print("\n--- Gradient Boosting Model Evaluation Metrics ---")
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
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"F1-Score: {f1:.4f}")
            logging.info(f"ROC-AUC: {roc_auc:.4f}")
            logging.info(f"Confusion Matrix:\n{conf_matrix}")
            logging.info(f"Classification Report:\n{class_report}")

            # Plot and save the confusion matrix
            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Gradient Boosting Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig('models/gradient_boosting_confusion_matrix.png')
            plt.close()
            logging.info("Confusion matrix plot saved at 'models/gradient_boosting_confusion_matrix.png'.")
            print("Gradient Boosting Training: Confusion matrix plot saved at 'models/gradient_boosting_confusion_matrix.png'.")

            # Plot Precision-Recall Curve
            precision_vals, recall_vals, thresholds = precision_recall_curve(self.y_test, y_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(recall_vals, precision_vals, marker='.', label='Gradient Boosting')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Gradient Boosting Precision-Recall Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig('models/gradient_boosting_precision_recall_curve.png')
            plt.close()
            logging.info("Precision-Recall curve plot saved at 'models/gradient_boosting_precision_recall_curve.png'.")
            print("Gradient Boosting Training: Precision-Recall curve plot saved at 'models/gradient_boosting_precision_recall_curve.png'.")
        except Exception as e:
            logging.error(f"Error in evaluating model: {e}")
            print(f"Gradient Boosting Training Error: Evaluating model failed with error: {e}")
            raise

    def adjust_threshold(self):
        """
        Adjust the decision threshold to improve recall based on the Precision-Recall curve.
        """
        try:
            y_proba = self.model.predict_proba(self.X_test)[:, 1]
            precision, recall, thresholds = precision_recall_curve(self.y_test, y_proba)
            
            # Select threshold where recall is maximized without severely impacting precision
            # Example: Choose the threshold with the highest recall where precision is above 90%
            desired_precision = 0.90
            indices = np.where(precision >= desired_precision)[0]
            if len(indices) > 0:
                optimal_idx = indices[-1]  # Choose the threshold with the highest recall
                optimal_threshold = thresholds[optimal_idx]
            else:
                optimal_threshold = 0.5  # Fallback to default

            logging.info(f"Optimal threshold determined: {optimal_threshold:.4f}")
            print(f"Gradient Boosting Training: Optimal threshold determined: {optimal_threshold:.4f}")

            # Apply the new threshold
            y_pred_adjusted = (y_proba >= optimal_threshold).astype(int)

            # Recalculate metrics
            accuracy = accuracy_score(self.y_test, y_pred_adjusted)
            precision = precision_score(self.y_test, y_pred_adjusted)
            recall = recall_score(self.y_test, y_pred_adjusted)
            f1 = f1_score(self.y_test, y_pred_adjusted)
            roc_auc = roc_auc_score(self.y_test, y_proba)
            conf_matrix = confusion_matrix(self.y_test, y_pred_adjusted)
            class_report = classification_report(self.y_test, y_pred_adjusted)

            # Print metrics
            print("\n--- Gradient Boosting Model Evaluation Metrics (Adjusted Threshold) ---")
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
            logging.info(f"Adjusted Threshold Metrics:")
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"F1-Score: {f1:.4f}")
            logging.info(f"ROC-AUC: {roc_auc:.4f}")
            logging.info(f"Confusion Matrix:\n{conf_matrix}")
            logging.info(f"Classification Report:\n{class_report}")

            # Save adjusted predictions
            adjusted_predictions_df = pd.DataFrame({
                'logid': self.log_ids_test,
                'actual': self.y_test,
                'predicted': y_pred_adjusted
            })
            adjusted_predictions_df.to_csv('data/gradient_boosting_predictions_adjusted_threshold.csv', index=False)
            logging.info(f"Adjusted threshold predictions saved successfully at 'data/gradient_boosting_predictions_adjusted_threshold.csv'.")
            print("Gradient Boosting Training: Adjusted threshold predictions saved successfully at 'data/gradient_boosting_predictions_adjusted_threshold.csv'.")
        except Exception as e:
            logging.error(f"Error in adjusting threshold: {e}")
            print(f"Gradient Boosting Training Error: Adjusting threshold failed with error: {e}")
            raise

    def plot_feature_importance(self):
        """
        Plot and save the feature importances.
        """
        try:
            feature_importances = pd.Series(self.model.feature_importances_, index=self.X.columns)
            top_features = feature_importances.sort_values(ascending=False).head(20)

            plt.figure(figsize=(10, 8))
            sns.barplot(x=top_features.values, y=top_features.index)
            plt.title('Gradient Boosting Top 20 Feature Importances')
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig('models/gradient_boosting_feature_importances.png')
            plt.close()

            logging.info("Feature importance plot saved at 'models/gradient_boosting_feature_importances.png'.")
            print("Gradient Boosting Training: Feature importance plot saved at 'models/gradient_boosting_feature_importances.png'.")
        except Exception as e:
            logging.error(f"Error in plotting feature importances: {e}")
            print(f"Gradient Boosting Training Error: Plotting feature importances failed with error: {e}")
            raise

    def save_model(self):
        """
        Save the trained model to disk.
        """
        try:
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            joblib.dump(self.model, self.model_save_path)
            logging.info(f"Gradient Boosting model saved successfully at '{self.model_save_path}'.")
            print(f"Gradient Boosting Training: Model saved successfully at '{self.model_save_path}'.")
        except Exception as e:
            logging.error(f"Failed to save Gradient Boosting model: {e}")
            print(f"Gradient Boosting Training Error: Saving model failed with error: {e}")
            raise

    def save_predictions(self):
        """
        Save the predictions along with logid to a CSV file for traceability.
        """
        try:
            y_pred = self.model.predict(self.X_test)
            predictions_df = pd.DataFrame({
                'logid': self.log_ids_test,
                'actual': self.y_test,
                'predicted': y_pred
            })
            predictions_df.to_csv(self.predictions_save_path, index=False)
            logging.info(f"Predictions saved successfully at '{self.predictions_save_path}'.")
            print(f"Gradient Boosting Training: Predictions saved successfully at '{self.predictions_save_path}'.")
        except Exception as e:
            logging.error(f"Failed to save predictions: {e}")
            print(f"Gradient Boosting Training Error: Saving predictions failed with error: {e}")
            raise

    def run(self):
        """
        Execute the entire training pipeline.
        """
        try:
            self.load_data()
            self.validate_data()
            self.prepare_features()
            self.split_data()
            self.resample_data()
            self.hyperparameter_tuning()
            self.train_model()
            self.evaluate_model()
            self.adjust_threshold()
            self.plot_feature_importance()
            self.save_model()
            self.save_predictions()
            logging.info("Gradient Boosting Enhanced training pipeline completed successfully.")
            print("\nGradient Boosting Enhanced training pipeline completed successfully.")
        except Exception as e:
            logging.error(f"Training pipeline failed: {e}")
            print(f"\nGradient Boosting Training Error: {e}")

if __name__ == "__main__":
    # Define file paths
    final_engineered_features_path = "data/final_engineered_features.csv"
    gradient_boosting_model_path = "models/gradient_boosting_model_enhanced.joblib"
    gradient_boosting_predictions_path = "data/gradient_boosting_predictions_enhanced.csv"

    # Ensure the 'data' and 'models' directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Initialize the enhanced trainer
    gb_trainer = GradientBoostingEnhancedTrainer(
        data_path=final_engineered_features_path,
        model_save_path=gradient_boosting_model_path,
        predictions_save_path=gradient_boosting_predictions_path
    )

    # Run the training pipeline
    gb_trainer.run()
