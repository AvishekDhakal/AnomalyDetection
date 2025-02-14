import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc)
def load_data(data_path):
    """
    Loads the dataset from the specified CSV file.
    
    Parameters:
    - data_path (str): Path to the CSV file.
    
    Returns:
    - df (DataFrame): Loaded DataFrame.
    """
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data from {data_path}")
        print(f"Dataset Shape: {df.shape}\n")
        return df
    except FileNotFoundError:
        print(f"Error: The file {data_path} does not exist.")
        exit()

def clean_data(df):
    """
    Cleans the DataFrame by handling missing and infinite values in numeric columns.
    
    Parameters:
    - df (DataFrame): Original DataFrame.
    
    Returns:
    - df_clean (DataFrame): Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # Select numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    print(f"Number of numeric columns: {len(numeric_cols)}")
    print(f"Number of non-numeric columns: {len(non_numeric_cols)}\n")
    
    # Handle missing values in numeric columns
    missing = df_clean[numeric_cols].isnull().sum().sum()
    if missing > 0:
        print(f"Found {missing} missing values in numeric columns. Filling with column means.")
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    else:
        print("No missing values found in numeric columns.")
    
    # Handle infinite values in numeric columns
    try:
        inf = np.isinf(df_clean[numeric_cols].values).sum()
    except TypeError as e:
        print(f"Error during np.isinf: {e}")
        inf = 0  # Assume no infinite values if error occurs
    
    if inf > 0:
        print(f"Found {inf} infinite values in numeric columns. Replacing with NaN and filling with column means.")
        df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    else:
        print("No infinite values found in numeric columns.")
    
    print("Data cleaning completed.\n")
    return df_clean

def compute_correlation(df):
    """
    Computes the Pearson correlation matrix for numeric columns.
    
    Parameters:
    - df (DataFrame): Cleaned DataFrame.
    
    Returns:
    - corr_matrix (DataFrame): Correlation matrix.
    """
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    print("Correlation matrix computed.\n")
    return corr_matrix

def identify_high_correlations(corr_matrix, threshold=0.6):
    """
    Identifies feature pairs with correlation above the specified threshold.
    
    Parameters:
    - corr_matrix (DataFrame): Correlation matrix.
    - threshold (float): Correlation coefficient threshold.
    
    Returns:
    - high_corr_df (DataFrame): DataFrame of highly correlated feature pairs.
    """
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find feature pairs with correlation above threshold
    high_corr = upper.stack().reset_index()
    high_corr.columns = ['Feature1', 'Feature2', 'Correlation']
    high_corr_df = high_corr[high_corr['Correlation'].abs() > threshold]
    
    print(f"Found {high_corr_df.shape[0]} feature pairs with correlation above {threshold}.\n")
    return high_corr_df

def plot_correlation_heatmap(corr_matrix, high_corr_df=None, output_path='outputs/correlation_heatmap.png'):
    """
    Plots and saves the correlation heatmap. Optionally highlights highly correlated pairs.
    
    Parameters:
    - corr_matrix (DataFrame): Correlation matrix.
    - high_corr_df (DataFrame, optional): DataFrame of highly correlated feature pairs.
    - output_path (str): Path to save the heatmap image.
    """
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=16)
    
    if high_corr_df is not None and not high_corr_df.empty:
        # Highlight highly correlated pairs
        for _, row in high_corr_df.iterrows():
            feature1 = row['Feature1']
            feature2 = row['Feature2']
            # Get the positions
            x = corr_matrix.columns.get_loc(feature1)
            y = corr_matrix.index.get_loc(feature1)
            x2 = corr_matrix.columns.get_loc(feature2)
            y2 = corr_matrix.index.get_loc(feature2)
            plt.plot([x + 0.5, x2 + 0.5], [y + 0.5, y2 + 0.5], marker='o', color='yellow', markersize=5, linewidth=2)
        plt.legend(['Highly Correlated Pairs'], loc='upper right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Correlation heatmap saved to '{output_path}'.\n")

def save_results(corr_matrix, high_corr_df, output_dir='outputs'):
    """
    Saves the correlation matrix and highly correlated feature pairs to CSV files.
    
    Parameters:
    - corr_matrix (DataFrame): Correlation matrix.
    - high_corr_df (DataFrame): DataFrame of highly correlated feature pairs.
    - output_dir (str): Directory to save the CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
    print(f"Correlation matrix saved to '{os.path.join(output_dir, 'correlation_matrix.csv')}'.")
    
    high_corr_df.to_csv(os.path.join(output_dir, 'high_correlations.csv'), index=False)
    print(f"High correlations saved to '{os.path.join(output_dir, 'high_correlations.csv')}'.\n")

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Classifier.
    
    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training target.
    
    Returns:
    - rf_model (RandomForestClassifier): Trained Random Forest model.
    """
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("Random Forest Classifier Trained Successfully.\n")
    return rf_model

def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model.
    
    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training target.
    
    Returns:
    - lr_model (LogisticRegression): Trained Logistic Regression model.
    """
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train, y_train)
    print("Logistic Regression Trained Successfully.\n")
    return lr_model

def evaluate_model(model, X_test, y_test, model_name='Model', output_dir='outputs'):
    """
    Evaluates the model and saves the classification report and confusion matrix.
    
    Parameters:
    - model: Trained model.
    - X_test (DataFrame): Testing features.
    - y_test (Series): Testing target.
    - model_name (str): Name of the model (for labeling).
    - output_dir (str): Directory to save evaluation reports.
    
    Returns:
    - metrics (dict): Dictionary of evaluation metrics.
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc_val = roc_auc_score(y_test, y_proba)
    
    print(f"=== {model_name} Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc_val:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save Classification Report
    report_path = os.path.join(output_dir, f'{model_name}_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"=== {model_name} Evaluation ===\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc_val:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, zero_division=0))
    print(f"Classification report saved to '{report_path}'.\n")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to '{cm_plot_path}'.\n")
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_plot_path = os.path.join(output_dir, f'{model_name}_roc_curve.png')
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to '{roc_plot_path}'.\n")
    
    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc_val
    }
    
    return metrics

def extract_feature_importances(model, feature_names, model_name='Model'):
    """
    Extracts feature importances from the model.
    
    Parameters:
    - model: Trained model.
    - feature_names (list): List of feature names.
    - model_name (str): Name of the model.
    
    Returns:
    - importances_df (DataFrame): DataFrame of features and their importances.
    """
    if isinstance(model, RandomForestClassifier):
        importances = model.feature_importances_
    elif isinstance(model, LogisticRegression):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model not supported for feature importance extraction.")
    
    importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    importances_df = importances_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    return importances_df

def save_feature_importances(rf_importances, lr_importances, output_path='outputs/feature_importances.txt'):
    """
    Saves feature importances for both models to a text file.
    
    Parameters:
    - rf_importances (DataFrame): Random Forest feature importances.
    - lr_importances (DataFrame): Logistic Regression feature importances.
    - output_path (str): Path to save the TXT file.
    """
    with open(output_path, 'w') as f:
        f.write("=== Random Forest Feature Importances ===\n")
        f.write(rf_importances.to_string(index=False))
        f.write("\n\n=== Logistic Regression Feature Importances ===\n")
        f.write(lr_importances.to_string(index=False))
    print(f"Feature importances saved to '{output_path}'.\n")

def plot_top_features(rf_importances, lr_importances, output_path='outputs/top10_features.png'):
    """
    Plots the top ten features for both Random Forest and Logistic Regression.
    
    Parameters:
    - rf_importances (DataFrame): Random Forest feature importances.
    - lr_importances (DataFrame): Logistic Regression feature importances.
    - output_path (str): Path to save the PNG file.
    """
    top10_rf = rf_importances.head(10)
    top10_lr = lr_importances.head(10)
    
    plt.figure(figsize=(20, 8))
    
    # Random Forest Top 10
    plt.subplot(1, 2, 1)
    sns.barplot(x='Importance', y='Feature', data=top10_rf, palette='viridis')
    plt.title('Top 10 Features - Random Forest', fontsize=16)
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    
    # Logistic Regression Top 10
    plt.subplot(1, 2, 2)
    sns.barplot(x='Importance', y='Feature', data=top10_lr, palette='magma')
    plt.title('Top 10 Features - Logistic Regression', fontsize=16)
    plt.xlabel('Coefficient Magnitude')
    plt.ylabel('Features')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Top 10 feature importances plot saved to '{output_path}'.\n")
    
def plot_top_features_correlation(top_features, df_clean, output_path='outputs/top_features_correlation_heatmap.png'):
    """
    Plots the correlation matrix of the top features.
    
    Parameters:
    - top_features (list): List of top feature names.
    - df_clean (DataFrame): Cleaned DataFrame.
    - output_path (str): Path to save the heatmap image.
    """
    # Extract the top features data
    top_features_data = df_clean[top_features]
    
    # Compute the correlation matrix
    corr_matrix = top_features_data.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Top Features', fontsize=16)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Top features correlation heatmap saved to '{output_path}'.\n")

def main():
    # Define paths
    data_path = 'data/enriched_data.csv'  # Update this path to your dataset
    output_dir = 'outputs'
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    feature_importances_path = os.path.join(output_dir, 'feature_importances.txt')
    top_features_plot_path = os.path.join(output_dir, 'top10_features.png')
    top_features_corr_heatmap_path = os.path.join(output_dir, 'top_features_correlation_heatmap.png')
    
    # Load data
    df = load_data(data_path)
    
    # Data cleaning
    df_clean = clean_data(df)
    
    # Compute correlation matrix
    corr_matrix = compute_correlation(df_clean)
    
    # Identify high correlations
    high_corr_df = identify_high_correlations(corr_matrix, threshold=0.6)  # Adjust threshold as needed
    
    # Plot and save correlation heatmap
    plot_correlation_heatmap(corr_matrix, high_corr_df, output_path=heatmap_path)
    
    # Save correlation results
    save_results(corr_matrix, high_corr_df, output_dir=output_dir)
    
    # Define target and features
    target = 'Anomalous'  # Update this if your target column has a different name
    if target not in df_clean.columns:
        print(f"Error: Target column '{target}' not found in the dataset.")
        exit()
        
    features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if target in features:
        features.remove(target)
    
    X = df_clean[features]
    y = df_clean[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}\n")
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    
    # Evaluate Random Forest
    rf_metrics = evaluate_model(rf_model, X_test, y_test, model_name='Random_Forest', output_dir=output_dir)
    
    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    
    # Evaluate Logistic Regression
    lr_metrics = evaluate_model(lr_model, X_test, y_test, model_name='Logistic_Regression', output_dir=output_dir)
    
    # Extract feature importances
    rf_importances = extract_feature_importances(rf_model, features, model_name='Random_Forest')
    lr_importances = extract_feature_importances(lr_model, features, model_name='Logistic_Regression')
    
    # Save feature importances to TXT file
    save_feature_importances(rf_importances, lr_importances, output_path=feature_importances_path)
    
    # Plot and save top 10 features for both models
    plot_top_features(rf_importances, lr_importances, output_path=top_features_plot_path)
    
    # Select top 10 features from both models
    top10_rf = rf_importances.head(10)['Feature'].tolist()
    top10_lr = lr_importances.head(10)['Feature'].tolist()
    
    # Combine top features (remove duplicates)
    top_features = list(set(top10_rf + top10_lr))
    print(f"Combined Top Features: {top_features}\n")
    
    # Plot and save correlation matrix of top features
    plot_top_features_correlation(top_features, df_clean, output_path=top_features_corr_heatmap_path)
    
    print("Feature importance analysis and visualization completed successfully.")

if __name__ == "__main__":
    main()
