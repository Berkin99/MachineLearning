# Part 1: Build a classifier based on KNN (K=3 for testing) using Euclidean distance.
# o You are expected to code the KNN classifier (including the distance function).
# o Report performance using an appropriate k-fold cross validation using confusion
# matrices on the given dataset.
# o Report the run time performance of your above tests.

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
import time

# Load dataset
def load_data(filename):
    columns = ["ID", "Diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
    audit_data = pd.read_csv(filename, header=None, names=columns)
    audit_data.drop(columns=["ID"], inplace=True)  # Remove ID column as it is not useful
    audit_data["Diagnosis"] = audit_data["Diagnosis"].map({"M": 1, "B": 0})  # Convert M = 1 (Malignant), B = 0 (Benign)
    return audit_data

# Compute Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

# KNN classifier
def knn_classify(train_X, train_y, test_X, k=3):
    predictions = []
    for test_point in test_X:
        distances = euclidean_distance(train_X, test_point)  # Compute distances to all training points
        k_indices = np.argsort(distances)[:k]  # Get indices of K nearest neighbors
        k_labels = train_y[k_indices]  # Retrieve labels of these neighbors
        prediction = np.bincount(k_labels).argmax()  # Choose the most frequent label
        predictions.append(prediction)
    return np.array(predictions)

# Evaluate KNN using 6-fold cross-validation
def evaluate_knn(data, k=3):
    X = data.iloc[:, 1:].values  # Extract features
    y = data.iloc[:, 0].values  # Extract labels
    kf = KFold(n_splits=6, shuffle=True, random_state=40)  # 6-fold cross-validation
    
    total_cm = np.zeros((2, 2), dtype=int)
    accuracies = []
    start_time = time.time()
    
    # Loop through each fold
    fold_count = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Perform classification and evaluate performance
        y_pred = knn_classify(X_train, y_train, X_test, k)
        
        # Calculate confusion matrix and accuracy
        cm = confusion_matrix(y_test, y_pred)
        total_cm += cm
        fold_accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        
        # Print the results for this fold
        print(f"\nFold {fold_count} Accuracy: {fold_accuracy:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        fold_count += 1
    
    end_time = time.time()
    avg_accuracy = np.mean(accuracies)
    runtime = end_time - start_time
    
    # Print overall performance
    print("\nOverall Performance (after 6 folds):")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Total Runtime: {runtime:.4f} seconds")
    print(f"Total Confusion Matrix:\n{total_cm}")

if __name__ == "__main__":
    data = load_data("Homework1/dataset/wdbc.data")  # Change to your dataset location
    evaluate_knn(data, k=3)  # Perform 6-fold evaluation
