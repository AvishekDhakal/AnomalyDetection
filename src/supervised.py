# supervised_anomaly_detection.py

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import roc_curve, precision_recall_curve
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data():
    """
    Load pre-split and preprocessed data from CSV files.
    """
    X_train = pd.read_csv('data/X_train_preprocessed.csv')
    X_test = pd.read_csv('data/X_test_preprocessed.csv')
    y_train = pd.read_csv('data/y_train.csv')
    y_test = pd.read_csv('data/y_test.csv')
    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    """
    Further preprocessing if necessary (e.g., handling categorical variables).
    Since you've already handled non-numeric columns, this may be minimal.
    """
    # If there are any categorical features, handle them here.
    # For this example, assuming all features are numeric.
    return X_train, X_test

def build_model_pipeline(model):
    """
    Build a machine learning pipeline with preprocessing and the given model.
    """
    pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    return pipeline

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """
    Plot the confusion matrix.
    """
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()

def plot_feature_importance(model, feature_names, title='Feature Importances'):
    """
    Plot feature importances for tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10,6))
        sns.barplot(x=importances[indices][:20], y=np.array(feature_names)[indices][:20], palette='viridis')
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()

def plot_roc_curve(y_true, y_scores, model_name='Model'):
    """
    Plot ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.show()

def plot_precision_recall(y_true, y_scores, model_name='Model'):
    """
    Plot Precision-Recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='lower left')
    plt.show()

def main():
    # Step 1: Load Data
    X_train, X_test, y_train, y_test = load_data()
    
    # Step 2: Further Preprocessing (if needed)
    X_train, X_test = preprocess_data(X_train, X_test)
    
    # Optional: Feature Selection (e.g., removing low-variance features)
    # This can be implemented if necessary.
    
    # Step 3: Define Models
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    xgb = XGBClassifier(random_state=42, scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train),
                        use_label_encoder=False, eval_metric='logloss')
    
    # Step 4: Build Pipelines
    rf_pipeline = build_model_pipeline(rf)
    xgb_pipeline = build_model_pipeline(xgb)
    
    # Step 5: Define Hyperparameter Grids
    rf_param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__max_features': ['auto', 'sqrt']
    }
    
    xgb_param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 6, 10],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0]
    }
    
    # Step 6: Setup GridSearchCV
    rf_grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_param_grid,
        scoring='f1',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=2,
        n_jobs=-1
    )
    
    xgb_grid_search = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=xgb_param_grid,
        scoring='f1',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=2,
        n_jobs=-1
    )
    
    # Step 7: Train Models
    print("Starting Grid Search for Random Forest...")
    rf_grid_search.fit(X_train, y_train)
    print("Random Forest Grid Search Completed.")
    print(f"Best Parameters: {rf_grid_search.best_params_}")
    print(f"Best F1-Score: {rf_grid_search.best_score_:.4f}")
    
    print("\nStarting Grid Search for XGBoost...")
    xgb_grid_search.fit(X_train, y_train)
    print("XGBoost Grid Search Completed.")
    print(f"Best Parameters: {xgb_grid_search.best_params_}")
    print(f"Best F1-Score: {xgb_grid_search.best_score_:.4f}")
    
    # Step 8: Evaluate Models on Test Data
    best_rf = rf_grid_search.best_estimator_
    best_xgb = xgb_grid_search.best_estimator_
    
    rf_preds = best_rf.predict(X_test)
    xgb_preds = best_xgb.predict(X_test)
    
    # Confusion Matrix and Classification Report for Random Forest
    cm_rf = confusion_matrix(y_test, rf_preds)
    cr_rf = classification_report(y_test, rf_preds)
    print("\nRandom Forest Confusion Matrix:")
    print(cm_rf)
    print("\nRandom Forest Classification Report:")
    print(cr_rf)
    
    # Confusion Matrix and Classification Report for XGBoost
    cm_xgb = confusion_matrix(y_test, xgb_preds)
    cr_xgb = classification_report(y_test, xgb_preds)
    print("\nXGBoost Confusion Matrix:")
    print(cm_xgb)
    print("\nXGBoost Classification Report:")
    print(cr_xgb)
    
    # Step 9: Visualize Confusion Matrices
    plot_confusion_matrix(cm_rf, classes=['Normal', 'Anomalous'], title='Random Forest Confusion Matrix')
    plot_confusion_matrix(cm_xgb, classes=['Normal', 'Anomalous'], title='XGBoost Confusion Matrix')
    
    # Step 10: Feature Importance for Random Forest
    # Assuming all features are numeric; adjust if there are categorical features
    feature_names = X_train.columns.tolist()
    plot_feature_importance(best_rf.named_steps['classifier'], feature_names, title='Top 20 Feature Importances - Random Forest')
    
    # Feature Importance for XGBoost
    plot_feature_importance(best_xgb.named_steps['classifier'], feature_names, title='Top 20 Feature Importances - XGBoost')
    
    # Step 11: Save the Best Models and Preprocessor
    joblib.dump(best_rf, 'models/random_forest_best.pkl')
    joblib.dump(best_xgb, 'models/xgboost_best.pkl')
    # joblib.dump(preprocessor, 'models/preprocessor.pkl')  # If preprocessor exists
    print("\nBest models saved successfully.")
    
    # Step 12: ROC-AUC and Precision-Recall Curves
 
    
    def plot_roc(y_true, y_scores, model_name='Model'):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.show()
    
    def plot_precision_recall_curve_func(y_true, y_scores, model_name='Model'):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(8,6))
        plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc='lower left')
        plt.show()
    
    # ROC and Precision-Recall for Random Forest
    rf_probs = best_rf.predict_proba(X_test)[:,1]
    plot_roc(y_test, rf_probs, model_name='Random Forest')
    plot_precision_recall_curve_func(y_test, rf_probs, model_name='Random Forest')
    
    # ROC and Precision-Recall for XGBoost
    xgb_probs = best_xgb.predict_proba(X_test)[:,1]
    plot_roc(y_test, xgb_probs, model_name='XGBoost')
    plot_precision_recall_curve_func(y_test, xgb_probs, model_name='XGBoost')
    
    # Step 13: Save the Preprocessor (if any additional steps were added)
    # If you have a preprocessor pipeline, save it for deployment.
    # joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    print("Supervised learning pipeline completed successfully.")

if __name__ == "__main__":
    main()
