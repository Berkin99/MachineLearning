# Part 2: Build a regressor based on KNN (K=3 for testing) using Manhattan distance.
# o You are expected to code the KNN classifier (including the distance function).
# o Report performance using an appropriate k-fold cross validation on the given dataset.
# o Report the run time performance of your above tests

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import time

# Load dataset with necessary preprocessing
def load_data(filename):
    # Load data from the given filename and process it
    bs_data = pd.read_csv(filename)
    
    # Drop unnecessary columns: "instant", "casual", "registered"
    bs_data.drop(columns=["instant", "casual", "registered"], inplace=True)
    
    # Extract day information from "dteday"
    bs_data["day"] = pd.to_datetime(bs_data["dteday"]).dt.day
    bs_data.drop(columns=["dteday"], inplace=True)
    
    # One-Hot Encoding for categorical variables
    bs_data = pd.get_dummies(bs_data, columns=["season", "weekday", "weathersit", "mnth"])

    # Convert all columns to numeric, and handle errors by coercing invalid entries
    bs_data = bs_data.apply(pd.to_numeric, errors="coerce")
    
    # Drop rows with missing values
    bs_data.dropna(inplace=True)

    return bs_data

# Compute Manhattan distance
def manhattan_distance(x1, x2):
    # Compute Manhattan distance between two points
    return np.sum(np.abs(x1 - x2), axis=1)

# KNN Regressor (using Manhattan distance)
def knn_regressor(train_X, train_y, test_X, k=3):
    predictions = []
    for test_point in test_X:
        # Calculate distances from test point to all training points
        distances = manhattan_distance(train_X, test_point)
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:k]
        
        # Get the target values (y) for those k nearest neighbors
        k_values = train_y[k_indices]
        
        # Predict the average of k nearest neighbors (regression)
        predictions.append(np.mean(k_values)) 
    
    return np.array(predictions)

# Evaluate KNN regressor using k-fold cross-validation
def evaluate_knn(data, target_column, k=3, folds=5):
    # Split data into features and target variable
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values

    # Initialize KFold cross-validation with specified splits
    kf = KFold(n_splits=folds, shuffle=True, random_state=40)

    total_train_mae = []  # To store training Mean Absolute Error for each fold
    total_test_mae = []  # To store testing Mean Absolute Error for each fold
    start_time = time.time()  # Record start time for performance measurement

    # Loop through each fold
    fold_count = 1
    for train_index, test_index in kf.split(X):
        # Split the data into training and test sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Make predictions using the KNN regressor for training and test sets
        y_train_pred = knn_regressor(X_train, y_train, X_train, k)
        y_test_pred = knn_regressor(X_train, y_train, X_test, k)
        
        # Calculate Mean Absolute Error for both training and testing data
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        total_train_mae.append(train_mae)
        total_test_mae.append(test_mae)

        # Print the results for this fold
        print(f"Fold {fold_count} Performance:")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Testing MAE: {test_mae:.4f}")
        
        fold_count += 1

    end_time = time.time()  # Record end time
    avg_train_mae = np.mean(total_train_mae)  # Average training MAE across all folds
    avg_test_mae = np.mean(total_test_mae)  # Average testing MAE across all folds
    runtime = end_time - start_time  # Total runtime for cross-validation

    # Print overall performance
    print(f"\nOverall Performance (after {folds} folds):")
    print(f"Average Training MAE: {avg_train_mae:.4f}")
    print(f"Average Testing MAE: {avg_test_mae:.4f}")
    print(f"Total Runtime: {runtime:.4f} seconds")

if __name__ == "__main__":
    # Load the dataset
    data = load_data("Homework1/dataset/day.csv")  # Change the path to your dataset location
    target = "cnt"  # The target column to predict
    evaluate_knn(data, target, k=3, folds=6)