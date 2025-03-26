import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
import time

# Load dataset
def load_data(filename):
    columns = ["ID", "Diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
    data = pd.read_csv(filename, header=None, names=columns)
    data.drop(columns=["ID"], inplace=True)  # Remove ID column as it is not useful
    data["Diagnosis"] = data["Diagnosis"].map({"M": 1, "B": 0})  # Convert M = 1 (Malignant), B = 0 (Benign)
    return data

# Compute Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2), axis=1)

def minkowski_distance(x1, x2, p=3):
    return np.sum(np.abs(x1 - x2) ** p, axis=1) ** (1 / p)

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

# Evaluate KNN using 10-fold cross-validation
def evaluate_knn(data, k=3):
    X = data.iloc[:, 1:].values  # Extract features
    y = data.iloc[:, 0].values  # Extract labels
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    total_cm = np.zeros((2, 2), dtype=int)
    accuracies = []
    start_time = time.time()
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_pred = knn_classify(X_train, y_train, X_test, k)
        cm = confusion_matrix(y_test, y_pred)
        total_cm += cm
        accuracies.append(accuracy_score(y_test, y_pred))
    
    end_time = time.time()
    avg_accuracy = np.mean(accuracies)
    runtime = end_time - start_time
    
    return total_cm, avg_accuracy, runtime

if __name__ == "__main__":
    data = load_data("C:/Users/berki/Desktop/YeniaySrc/WorkspaceAI/MachineLearning/Homework1/src/wdbc.data")
    cm, acc, time_taken = evaluate_knn(data, k=3)
    print("Confusion Matrix:")
    print(cm)
    print(f"Average Accuracy: {acc:.4f}")
    print(f"Total Runtime: {time_taken:.4f} seconds")