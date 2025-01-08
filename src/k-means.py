# dbscan_anomaly_detection_optimized.py

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data():
    """
    Load preprocessed training and testing data.
    """
    X_train = pd.read_csv('data/X_train_preprocessed.csv')
    X_test = pd.read_csv('data/X_test_preprocessed.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test

def plot_k_distance(X, k):
    """
    Plot the k-distance graph to help determine the optimal eps parameter for DBSCAN.
    """
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, k-1], axis=0)
    plt.figure(figsize=(10,6))
    plt.plot(distances)
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel(f'{k}-NN Distance')
    plt.title(f'k-distance Graph for k={k}')
    plt.grid(True)
    plt.show()

def main():
    # Step 1: Load Data
    X_train, X_test, y_train, y_test = load_data()
    
    # Step 2: Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 3: Dimensionality Reduction with PCA
    # Adjust n_components based on explained variance
    pca = PCA(n_components=10, random_state=42)  # Example: retain 95% variance
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Step 4: Determine Optimal eps using k-distance graph
    k = 5  # Typically set to min_samples
    print("Plotting k-distance graph to determine optimal eps...")
    plot_k_distance(X_train_pca, k)
    
    # After inspecting the plot, set optimal_eps
    optimal_eps = 1.5  # Example value; adjust based on k-distance graph
    print(f"Training DBSCAN with eps={optimal_eps} and min_samples={k}...")
    
    # Step 5: Initialize and Train DBSCAN
    dbscan = DBSCAN(eps=optimal_eps, min_samples=k, metric='euclidean', n_jobs=-1)
    dbscan.fit(X_train_pca)
    
    # Step 6: Predict on Test Data
    print("Predicting on test data...")
    dbscan_preds = dbscan.fit_predict(X_test_pca)
    
    # Convert predictions to binary labels
    dbscan_anomalies = np.where(dbscan_preds == -1, 1, 0)
    
    # Step 7: Evaluate Model Performance
    cm = confusion_matrix(y_test, dbscan_anomalies)
    cr = classification_report(y_test, dbscan_anomalies)
    
    print("DBSCAN Confusion Matrix:")
    print(cm)
    print("\nDBSCAN Classification Report:")
    print(cr)
    
    # Visualize Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=['Normal', 'Anomalous'], 
                yticklabels=['Normal', 'Anomalous'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('DBSCAN Confusion Matrix')
    plt.show()
    
    # Step 8: Save the DBSCAN Model, Scaler, and PCA
    joblib.dump(dbscan, 'models/dbscan_anomaly_detector_optimized.pkl')
    joblib.dump(scaler, 'scalers/dbscan_scaler_optimized.pkl')
    joblib.dump(pca, 'models/dbscan_pca_optimized.pkl')
    print("DBSCAN model, scaler, and PCA saved successfully.")
    
    # Optional: Visualize the clustering results (2D PCA Projection)
    pca_2d = PCA(n_components=2, random_state=42)
    X_test_pca_2d = pca_2d.fit_transform(X_test_scaled)
    
    plt.figure(figsize=(10,6))
    unique_labels = set(dbscan_preds)
    colors = sns.color_palette(n_colors=len(unique_labels))
    
    for label, color in zip(unique_labels, colors):
        class_member_mask = (dbscan_preds == label)
        if label == -1:
            # Black used for noise.
            color = 'k'
            marker = 'x'
            label_name = 'Anomalies'
        else:
            marker = 'o'
            label_name = f'Cluster {label}'
        plt.scatter(X_test_pca_2d[class_member_mask, 0], X_test_pca_2d[class_member_mask, 1],
                    c=[color], marker=marker, label=label_name, alpha=0.5)
    
    plt.title('DBSCAN Clustering Results (2D PCA Projection)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
