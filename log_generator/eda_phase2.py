#!/usr/bin/env python3

"""
Anomaly Detection EDA Script
Generates textual summaries and visualizations of preprocessed data
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats

# Configure settings
sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
pd.set_option('display.max_columns', 50)

def create_eda_dir():
    """Create directory structure for EDA outputs"""
    dir_name = "preprocessed_eda"
    subdirs = ['correlation', 'distributions', 'temporal', 'text_summaries']
    
    os.makedirs(dir_name, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(dir_name, subdir), exist_ok=True)
    return dir_name

def save_text_summary(content, filename, eda_dir):
    """Save textual summary to file"""
    path = os.path.join(eda_dir, 'text_summaries', filename)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Saved text summary: {path}")

def basic_data_summary(df, eda_dir):
    """Generate basic data statistics"""
    summary = f"""Data Summary:
- Total Records: {len(df):,}
- Features: {len(df.columns)}
- Time Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}
- Missing Values: {df.isnull().sum().sum()}
- Duplicate Records: {df.duplicated().sum()}

Class Distribution:
{df['Anomalous'].value_counts().to_string()}
Anomaly Rate: {df['Anomalous'].mean():.2%}

Numerical Features Summary:
{df.describe().T.to_string()}

Categorical Features Summary:
{df.describe(include='object').T.to_string()}
"""
    save_text_summary(summary, 'basic_summary.txt', eda_dir)

def plot_feature_distributions(df, eda_dir):
    """Visualize distributions of key features"""
    features = ['hour', 'day_of_week', 'is_unusual_time', 
               'is_internal_ip', 'is_authorized', 'role_risk']
    
    for feature in features:
        plt.figure()
        if df[feature].nunique() < 10:
            sns.countplot(x=feature, hue='Anomalous', data=df)
        else:
            sns.histplot(df[feature], kde=True, bins=30)
            
        plt.title(f'{feature} Distribution')
        plt.savefig(os.path.join(eda_dir, 'distributions', f'{feature}_dist.png'))
        plt.close()

def plot_correlations(df, eda_dir):
    """Generate correlation analysis"""
    # Numerical feature correlation
    num_features = df.select_dtypes(include=np.number).columns
    corr_matrix = df[num_features].corr()
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig(os.path.join(eda_dir, 'correlation', 'correlation_matrix.png'))
    plt.close()
    
    # Top correlations with target
    target_corr = corr_matrix['Anomalous'].abs().sort_values(ascending=False)
    save_text_summary(
        target_corr.to_string(), 
        'target_correlations.txt', 
        eda_dir
    )

def temporal_analysis(df, eda_dir):
    """Analyze temporal patterns of anomalies"""
    # Hourly anomaly rate
    plt.figure()
    hourly = df.groupby('hour')['Anomalous'].mean()
    sns.lineplot(x=hourly.index, y=hourly.values)
    plt.title('Anomaly Rate by Hour')
    plt.ylabel('Anomaly Rate')
    plt.savefig(os.path.join(eda_dir, 'temporal', 'hourly_anomaly_rate.png'))
    plt.close()
    
    # Weekly pattern
    plt.figure()
    daily = df.groupby('day_of_week')['Anomalous'].mean()
    sns.barplot(x=daily.index, y=daily.values)
    plt.title('Anomaly Rate by Day of Week')
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.ylabel('Anomaly Rate')
    plt.savefig(os.path.join(eda_dir, 'temporal', 'daily_anomaly_rate.png'))
    plt.close()

def advanced_analysis(df, eda_dir):
    """Advanced statistical analysis"""
    # Statistical tests
    normal_mask = df['Anomalous'] == 0
    results = []
    
    for feature in df.select_dtypes(include=np.number).columns:
        if feature == 'Anomalous':
            continue
            
        stat, p = stats.ttest_ind(
            df[feature][normal_mask],
            df[feature][~normal_mask],
            equal_var=False
        )
        results.append({
            'feature': feature,
            't-statistic': stat,
            'p-value': p
        })
    
    stats_df = pd.DataFrame(results)
    save_text_summary(
        stats_df.to_string(), 
        'statistical_tests.txt', 
        eda_dir
    )

def main():
    eda_dir = create_eda_dir()
    
    # Load preprocessed data
    df = pd.read_csv('data/processed_data.csv', parse_dates=['Timestamp'])
    
    # Generate basic summaries
    basic_data_summary(df, eda_dir)
    
    # Visual analysis
    plot_feature_distributions(df, eda_dir)
    plot_correlations(df, eda_dir)
    temporal_analysis(df, eda_dir)
    
    # Advanced analysis
    advanced_analysis(df, eda_dir)
    
    print(f"\nEDA completed. Results saved in: {os.path.abspath(eda_dir)}")

if __name__ == '__main__':
    main()