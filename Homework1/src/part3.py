# Part 3: Model a classifier based on the linear SVM.
# o You may use any available implementation of SVM in Python.
# o Report performance using an appropriate k-fold cross validation using ROC curves and
# confusion matrices. Find the best threshold for the SVM output as described in the
# note by Fawcett.
# o Report the run time performance of your above tests.

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# Load dataset
def load_data(filename):
    columns = ["ID", "Diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
    audit_data = pd.read_csv(filename, header=None, names=columns)
    audit_data.drop(columns=["ID"], inplace=True)  # Remove ID column as it is not useful
    audit_data["Diagnosis"] = audit_data["Diagnosis"].map({"M": 1, "B": 0})  # Convert M = 1 (Malignant), B = 0 (Benign)
    return audit_data
# Function to train the SVM model
def train_svm(X_train, y_train):
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    return model

# Function to make predictions with the SVM model
def make_predictions(model, X_test):
    return model.predict(X_test), model.predict_proba(X_test)[:, 1]

# Function to calculate metrics: Confusion Matrix, Accuracy, and ROC AUC
def evaluate_metrics(y_test, y_pred, y_prob):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    return cm, accuracy, fpr, tpr, roc_auc

# Function to print fold performance
def print_fold_performance(fold_count, cm, accuracy, roc_auc):
    print(f"Fold {fold_count} Performance:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

# Function to calculate average TPR and FPR for the ROC curve
def calculate_mean_roc(tpr_list, fpr_list):
    mean_tpr = np.mean(tpr_list, axis=0)
    mean_fpr = np.mean(fpr_list, axis=0)
    return mean_fpr, mean_tpr

# Main evaluation function for SVM using k-fold cross-validation
def evaluate_svm(data, folds=6):
    X = data.iloc[:, 1:].values  # Extract features
    y = data.iloc[:, 0].values  # Extract labels
    kf = KFold(n_splits=folds, shuffle=True, random_state=40)
    
    accuracies = []
    tpr_list = []
    fpr_list = []
    roc_auc_list = []
    start_time = time.time()
    
    fold_count = 1
    for train_index, test_index in kf.split(X):
        # Split the data into training and test sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train the SVM model
        model = train_svm(X_train, y_train)
        
        # Make predictions
        y_pred, y_prob = make_predictions(model, X_test)
        
        # Calculate metrics
        cm, accuracy, fpr, tpr, roc_auc = evaluate_metrics(y_test, y_pred, y_prob)
        
        # Store values for reporting
        accuracies.append(accuracy)
        tpr_list.append(np.interp(np.linspace(0, 1, 100), fpr, tpr))  # Interpolate TPR values
        fpr_list.append(np.linspace(0, 1, 100))  # Interpolate FPR values to be consistent
        roc_auc_list.append(roc_auc)
        
        # Print performance for the current fold
        print_fold_performance(fold_count, cm, accuracy, roc_auc)
        
        fold_count += 1
    
    end_time = time.time()
    avg_accuracy = np.mean(accuracies)
    avg_roc_auc = np.mean(roc_auc_list)
    runtime = end_time - start_time
    
    # Calculate average TPR and FPR for plotting the ROC curve
    mean_fpr, mean_tpr = calculate_mean_roc(tpr_list, fpr_list)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC curve (area = {avg_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    return avg_accuracy, avg_roc_auc, runtime

# Example usage
# Assuming 'audit_data' is your dataset
# Evaluate the SVM and report results for a 6-fold cross-validation
if __name__ == "__main__":
    data = load_data("Homework1/dataset/wdbc.data")  # Change to your dataset location
    acc, roc_auc, time_taken = evaluate_svm(data, folds=6)

    # Print overall performance after all folds
    print(f"\nOverall Performance (after 6 folds):")
    print(f"Average Accuracy: {acc:.4f}")
    print(f"Average ROC AUC: {roc_auc:.4f}")
    print(f"Total Runtime: {time_taken:.4f} seconds")