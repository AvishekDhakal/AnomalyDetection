#!/usr/bin/env python3

"""
feature_assessment.py

A comprehensive script to assess and analyze features in the insider threat detection dataset.
This script performs Exploratory Data Analysis (EDA) to understand feature distributions,
correlations, multicollinearity, and feature importance, aiding in diagnosing model performance issues.

Outputs:
- Visualizations saved in the 'feature_assessment_plots/' directory.
- Correlation matrices and feature importance reports saved as CSV files.

Usage:
    python3 feature_assessment.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
import warnings

# -------------------- CONFIGURATION --------------------

# Define directories
FEATURE_ENGINEERED_DATA_DIR = "feature_engineered_data"
FEATURE_ASSESSMENT_DIR = "feature_assessment_plots"
MODELS_DIR = "models"

# Ensure necessary directories exist
os.makedirs(FEATURE_ASSESSMENT_DIR, exist_ok=True)

# -------------------- LOAD DATA --------------------

def load_data():
    """
    Loads the feature-engineered training and testing data with explicit data types.
    
    Returns:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
    """
    print("\n--- Loading Feature-Engineered Data ---")
    try:
        # Define data types to prevent DtypeWarnings
        dtype_spec = {
            'LogID': str,
            'UserID': int,
            'Time_Since_Last_Access': float,
            'PCA_9': float,
            'PCA_8': float,
            'Rolling_Access_24': float,
            'PCA_6': float,
            'PCA_10': float,
            'PCA_7': float,
            'PCA_5': float,
            'PCA_3': float,
            'PCA_4': float,
            'PCA_2': float,
            'Hour': float,  # Changed to float to match data types
            'PCA_1': float,
            'DayOfWeek': float,  # Changed to float to match data types
            'Anomaly_Score': float,
            'User_Access_Freq': float,
            'User_Anomalous_Freq': float,
            'Endpoint_Freq': int,
            'User_Agent_Mozilla/5.0 (Windows NT 10.0; Win64; x64)': bool,
            'User_Agent_Mozilla/5.0 (X11; Linux x86_64)': bool,
            # Add other User_Agent columns if present
        }
        
        # Attempt to load training data with specified dtypes
        try:
            df_train = pd.read_csv(
                os.path.join(FEATURE_ENGINEERED_DATA_DIR, "X_train_final.csv"),
                dtype=dtype_spec,
                low_memory=False
            )
        except ValueError as ve:
            print(f"ValueError during training data loading: {ve}")
            print("Attempting to load training data without specifying dtype_spec.")
            df_train = pd.read_csv(
                os.path.join(FEATURE_ENGINEERED_DATA_DIR, "X_train_final.csv"),
                low_memory=False
            )
        
        y_train = df_train['Anomalous']
        X_train = df_train.drop(columns=['Anomalous', 'LogID'], errors='ignore')
        
        # Attempt to load testing data with specified dtypes
        try:
            df_test = pd.read_csv(
                os.path.join(FEATURE_ENGINEERED_DATA_DIR, "X_test_final.csv"),
                dtype=dtype_spec,
                low_memory=False
            )
        except ValueError as ve:
            print(f"ValueError during testing data loading: {ve}")
            print("Attempting to load testing data without specifying dtype_spec.")
            df_test = pd.read_csv(
                os.path.join(FEATURE_ENGINEERED_DATA_DIR, "X_test_final.csv"),
                low_memory=False
            )
        
        y_test = df_test['Anomalous']
        X_test = df_test.drop(columns=['Anomalous', 'LogID'], errors='ignore')
        
        print("Successfully loaded training and testing data.")
        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except pd.errors.DtypeWarning as e:
        print(f"DtypeWarning: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        sys.exit(1)

# -------------------- DATA QUALITY CHECKS --------------------

def data_quality_checks(X_train, X_test):
    """
    Performs data quality checks on the dataset, including missing values and data types.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
    """
    print("\n--- Performing Data Quality Checks ---")
    
    # Combine training and testing data for uniform analysis
    df = pd.concat([X_train, X_test], axis=0)
    
    # 1. Check for missing values
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        print("\nMissing Values Detected:")
        print(missing_values)
        missing_values.to_frame('Missing_Count').to_csv(
            os.path.join(FEATURE_ASSESSMENT_DIR, "missing_values.csv")
        )
        print("Saved missing values report as 'missing_values.csv'.")
    else:
        print("\nNo missing values detected.")
    
    # 2. Check data types
    print("\nData Types:")
    print(df.dtypes)
    df.dtypes.to_frame('DataType').to_csv(
        os.path.join(FEATURE_ASSESSMENT_DIR, "data_types.csv")
    )
    print("Saved data types report as 'data_types.csv'.")

# -------------------- STATISTICAL SUMMARY --------------------

def statistical_summary(X_train, X_test):
    """
    Generates a statistical summary of the dataset.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
    """
    print("\n--- Generating Statistical Summary ---")
    
    # Combine training and testing data
    df = pd.concat([X_train, X_test], axis=0)
    
    summary = df.describe()
    summary.to_csv(os.path.join(FEATURE_ASSESSMENT_DIR, "statistical_summary.csv"))
    print("Saved statistical summary as 'statistical_summary.csv'.")
    
    # Visualize distributions for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numerical_cols].hist(bins=30, figsize=(20, 15))
    plt.tight_layout()
    plt.savefig(os.path.join(FEATURE_ASSESSMENT_DIR, "numerical_feature_distributions.png"))
    plt.close()
    print("Saved numerical feature distributions as 'numerical_feature_distributions.png'.")

# -------------------- CLASS DISTRIBUTION --------------------

def class_distribution(y_train, y_test):
    """
    Plots and saves the class distribution in training and testing sets.
    
    Args:
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
    """
    print("\n--- Analyzing Class Distribution ---")
    
    # Training set
    plt.figure(figsize=(6,4))
    sns.countplot(x=y_train)
    plt.title('Training Set Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks([0,1], ['Normal', 'Anomalous'])
    plt.tight_layout()
    plt.savefig(os.path.join(FEATURE_ASSESSMENT_DIR, "training_class_distribution.png"))
    plt.close()
    
    # Testing set
    plt.figure(figsize=(6,4))
    sns.countplot(x=y_test)
    plt.title('Testing Set Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks([0,1], ['Normal', 'Anomalous'])
    plt.tight_layout()
    plt.savefig(os.path.join(FEATURE_ASSESSMENT_DIR, "testing_class_distribution.png"))
    plt.close()
    
    # Save counts as CSV
    class_counts = pd.DataFrame({
        'Training': y_train.value_counts(),
        'Testing': y_test.value_counts()
    })
    class_counts.to_csv(os.path.join(FEATURE_ASSESSMENT_DIR, "class_distribution.csv"))
    print("Saved class distribution plots and report.")

# -------------------- CORRELATION ANALYSIS --------------------

def correlation_analysis(X_train, y_train):
    """
    Computes and visualizes the correlation matrix, and identifies highly correlated features.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    """
    print("\n--- Performing Correlation Analysis ---")
    
    # Combine features with target
    df = X_train.copy()
    df['Anomalous'] = y_train
    
    # Compute correlation matrix
    corr_matrix = df.corr()
    
    # Save correlation matrix
    corr_matrix.to_csv(os.path.join(FEATURE_ASSESSMENT_DIR, "correlation_matrix.csv"))
    print("Saved correlation matrix as 'correlation_matrix.csv'.")
    
    # Plot heatmap
    plt.figure(figsize=(20, 18))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(FEATURE_ASSESSMENT_DIR, "correlation_matrix_heatmap.png"))
    plt.close()
    print("Saved correlation matrix heatmap as 'correlation_matrix_heatmap.png'.")
    
    # Identify highly correlated features (absolute correlation > 0.8)
    high_corr = corr_matrix.abs().unstack().sort_values(kind="quicksort", ascending=False)
    high_corr = high_corr[high_corr < 1]  # Remove self-correlation
    high_corr = high_corr[high_corr > 0.8]
    high_corr = high_corr.drop_duplicates()
    
    if not high_corr.empty:
        high_corr.to_frame('Correlation').to_csv(os.path.join(FEATURE_ASSESSMENT_DIR, "highly_correlated_features.csv"))
        print("Saved highly correlated feature pairs as 'highly_correlated_features.csv'.")
    else:
        print("No highly correlated feature pairs found.")

# -------------------- MULTICOLLINEARITY CHECK --------------------

def multicollinearity_check(X):
    """
    Calculates Variance Inflation Factor (VIF) for each feature to detect multicollinearity.
    
    Args:
        X (pd.DataFrame): Feature matrix.
    """
    print("\n--- Checking for Multicollinearity using VIF ---")
    
    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include=['bool']).columns
    if not bool_cols.empty:
        X_numeric = X.copy()
        X_numeric[bool_cols] = X_numeric[bool_cols].astype(int)
        print(f"Converted boolean columns {list(bool_cols)} to integers.")
    else:
        X_numeric = X.copy()
    
    # Add intercept for VIF calculation
    X_with_intercept = X_numeric.copy()
    X_with_intercept['Intercept'] = 1
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_intercept.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_intercept.values, i) 
                       for i in range(X_with_intercept.shape[1])]
    
    # Remove the intercept from VIF data
    vif_data = vif_data[vif_data['Feature'] != 'Intercept']
    
    # Save VIF results
    vif_data.to_csv(os.path.join(FEATURE_ASSESSMENT_DIR, "vif_scores.csv"), index=False)
    print("Saved VIF scores as 'vif_scores.csv'.")
    
    # Identify features with VIF > 5
    high_vif = vif_data[vif_data['VIF'] > 5]
    if not high_vif.empty:
        high_vif.to_csv(os.path.join(FEATURE_ASSESSMENT_DIR, "high_vif_features.csv"), index=False)
        print("Features with VIF > 5 saved as 'high_vif_features.csv'.")
    else:
        print("No features with VIF > 5 detected.")

# -------------------- FEATURE IMPORTANCE --------------------

def feature_importance_analysis(X_train, y_train):
    """
    Trains a Random Forest classifier to determine feature importances.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    """
    print("\n--- Analyzing Feature Importances with Random Forest ---")
    
    # Convert boolean columns to int
    bool_cols = X_train.select_dtypes(include=['bool']).columns
    if not bool_cols.empty:
        X_train_numeric = X_train.copy()
        X_train_numeric[bool_cols] = X_train_numeric[bool_cols].astype(int)
        print(f"Converted boolean columns {list(bool_cols)} to integers for feature importance.")
    else:
        X_train_numeric = X_train.copy()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train_numeric, y_train)
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X_train_numeric.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Save feature importances
    feature_importance_df.to_csv(os.path.join(FEATURE_ASSESSMENT_DIR, "random_forest_feature_importances.csv"), index=False)
    print("Saved feature importances as 'random_forest_feature_importances.csv'.")
    
    # Plot feature importances
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
    plt.title('Random Forest Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(FEATURE_ASSESSMENT_DIR, "random_forest_feature_importances.png"))
    plt.close()
    print("Saved feature importances plot as 'random_forest_feature_importances.png'.")

# -------------------- FEATURE DISTRIBUTION --------------------

def feature_distribution_visualization(X_train, X_test, y_train, y_test):
    """
    Visualizes the distribution of features, distinguishing between classes.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
    """
    print("\n--- Visualizing Feature Distributions by Class ---")
    
    # Combine training and testing data
    df_train = X_train.copy()
    df_train['Anomalous'] = y_train
    df_test = X_test.copy()
    df_test['Anomalous'] = y_test
    df = pd.concat([df_train, df_test], axis=0)
    
    # Select numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove('Anomalous')  # Exclude target variable
    
    # Plot distributions for each numerical feature
    for feature in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x=feature, hue='Anomalous', kde=True, stat="density", common_norm=False)
        plt.title(f'Distribution of {feature} by Class')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(os.path.join(FEATURE_ASSESSMENT_DIR, f"{feature}_distribution.png"))
        plt.close()
    
    print("Saved feature distribution plots.")

# -------------------- MAIN ASSESSMENT FUNCTION --------------------

def main():
    """
    Main function to execute the feature assessment pipeline.
    """
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Data Quality Checks
    data_quality_checks(X_train, X_test)
    
    # Statistical Summary
    statistical_summary(X_train, X_test)
    
    # Class Distribution
    class_distribution(y_train, y_test)
    
    # Correlation Analysis
    correlation_analysis(X_train, y_train)
    
    # Multicollinearity Check
    multicollinearity_check(X_train)
    
    # Feature Importance
    feature_importance_analysis(X_train, y_train)
    
    # Feature Distribution Visualization
    feature_distribution_visualization(X_train, X_test, y_train, y_test)
    
    print("\nFeature assessment pipeline completed successfully.")
    print(f"All plots and reports are saved in the '{FEATURE_ASSESSMENT_DIR}/' directory.")

# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    main()
