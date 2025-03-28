# Part 4: Model a regressor based on the linear SVM.
# o You may use an available implementation of SVM in Python.
# o Report performance using an appropriate k-fold cross validation.
# o Report the run time performance of your above tests.

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, roc_curve, auc
from sklearn.svm import SVC, SVR
import time
import os
import time
import matplotlib.pyplot as plt

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

# Linear SVM Regressor
def svm_regressor(train_X, train_y, test_X):
    # Initialize the Support Vector Regressor with a linear kernel
    svr = SVR(kernel='linear')
    
    # Train the model on the training data
    svr.fit(train_X, train_y)
    
    # Make predictions on the test data
    predictions = svr.predict(test_X)
    
    return predictions, svr

# Evaluate SVM Regressor using k-fold cross-validation
def evaluate_svm(data, target_column, folds=6):
    # Split data into features and target variable
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values

    # Initialize KFold cross-validation with 6 splits
    kf = KFold(n_splits=folds, shuffle=True, random_state=40)

    total_mae = []  # To store Mean Absolute Error for each fold
    start_time = time.time()  # Record start time for performance measurement

    fold_count = 1
    for train_index, test_index in kf.split(X):
        # Split the data into training and test sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Make predictions using the SVM regressor
        y_pred, model = svm_regressor(X_train, y_train, X_test)
        
        # Calculate Mean Absolute Error for both training and testing sets
        train_pred = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        
        # Append the test MAE for the fold
        total_mae.append(test_mae)

        # Print performance for the current fold
        print(f"\nFold {fold_count} Performance:")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Testing MAE: {test_mae:.4f}")
        
        fold_count += 1

    end_time = time.time()  # Record end time
    avg_mae = np.mean(total_mae)  # Average MAE across all folds
    runtime = end_time - start_time  # Total runtime for cross-validation

    return avg_mae, runtime

if __name__ == "__main__":
    # Load the dataset
    data = load_data("Homework1/dataset/day.csv")  # Change the path to your dataset location
    target = "cnt"  # The target column to predict

    # Evaluate the SVM regressor and report results for the 6-fold cross-validation
    mae, time_taken = evaluate_svm(data, target, folds=6)
    
    # Print overall results: Average MAE and Total Runtime
    print(f"\nOverall Performance (after 6 folds):")
    print(f"Average Testing MAE: {mae:.4f}")
    print(f"Total Runtime: {time_taken:.4f} seconds")
