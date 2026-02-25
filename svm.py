#!/usr/bin/env python3
"""
CMPSC 165 - Machine Learning
Homework 2, Problem 2: Support Vector Machine (SVM)
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt


def load_data(X_path: str, y_path: str = None):
    """Load features and labels from CSV files."""
    # first, read the features file into an array
    X_df = pd.read_csv(X_path, skipinitialspace=True)
    X = X_df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

    if y_path is None:
        return X

    y_df = pd.read_csv(y_path, skipinitialspace=True)  # header auto-detected
    y = y_df.iloc[:, 0].apply(pd.to_numeric, errors="raise").to_numpy(dtype=int)

    # searched this up but it basically lalows us to flatten it without copying memory
    y = y.ravel()

    return X, y


def preprocess_data(X_train, X_test):
    """Preprocess training and test data.""" # standardize data so that they are all roughly the same size
    # 1) compute per-column mean on X_train
    mean = X_train.mean(axis=0)

    # 2) compute per-column std on X_train
    std = X_train.std(axis=0)

    # 3) avoid division by zero for std
    std[std == 0] = 1.0

    # 4) standardize X_train and X_test using those stats
    X_train_new = (X_train - mean)/std
    X_test_new = (X_test - mean)/std

    X_train_new = np.hstack([X_train_new, np.ones((X_train_new.shape[0], 1), dtype=X_train_new.dtype)])
    X_test_new = np.hstack([X_test_new, np.ones((X_test_new.shape[0], 1), dtype=X_test_new.dtype)])

    if np.isnan(X_train_new).any() or np.isnan(X_test_new).any():
        raise ValueError("NaNs found after preprocessing — data loading/conversion is wrong.")
  
    return X_train_new, X_test_new


class SVMClassifier:
    """Support Vector Machine Classifier."""
    # initialize stuff
    def __init__(self, epochs=1, learning_rate = 0.003, lambda_ = 0.007):
        self.epochs = epochs
        # note: learning rate is how big of a step we take when updating weights
        # lambda is how much we punish large weights
        self.learning_rate = learning_rate
        self.lambda_ = lambda_

    def train(self, X, y):
        """Fit the classifier to training data."""
        # initialize weights as zeros
        w = np.zeros(X.shape[1])

        # loop epochs, loop samples
        for epoch in range(self.epochs):
            for i in range(len(X)):
                margin = y[i]*np.dot(w, X[i])

                if margin < 1:
                    # move w toward classifying x_i correctly and apply regularization shrink
                    w = w + self.learning_rate*(y[i]*X[i] - self.lambda_*w)
                else: 
                    # regilarization shrink
                    w = w - self.learning_rate*self.lambda_*w
                
        self.w = w

    def predict(self, X):
        """Predict labels for input samples."""
        predictions = []

        for i in range(len(X)):
            x = X[i]

            # predicted label y = sign(w_k dot x)
            pred = np.sign(np.dot(self.w, x)) 
            
            if pred == 0:
                pred = 1

            pred = int(pred)
            predictions.append(pred)

        # Return a 1D numpy array of ints (-1 or +1)
        return np.array(predictions)


def evaluate(y_true, y_pred):
    """Compute classification accuracy."""
    # accuracy is fraction of predictions that equal the true label
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # np.mean treats true as 1 and false as 0
    return np.mean(y_true == y_pred)


def run(Xtrain_file: str, Ytrain_file: str, test_data_file: str, pred_file: str):
    """Main function called by autograder."""
    # Load train
    X_train, y_train = load_data(Xtrain_file, Ytrain_file)

    # Load test
    X_test = load_data(test_data_file)

    # Preprocess
    X_train_p, X_test_p = preprocess_data(X_train, X_test)

    # Train
    model = SVMClassifier(epochs=3)  #tune epochs
    model.train(X_train_p, y_train)

    # Predict
    y_pred = model.predict(X_test_p).astype(int)

    # Save one integer per line, no header according to assignment instructions
    np.savetxt(pred_file, y_pred, fmt="%d")



# # this block is for testing and for generating my plots
# if __name__ == "__main__":
#     X, y = load_data("spam_X.csv", "spam_y.csv")

#     n = len(X)
#     split_90 = int(0.9*n)

#     X_train_full = X[:split_90]
#     y_train_full = y[:split_90]

#     X_test = X[split_90:]
#     y_test = y[split_90:]

#     # Preprocess AFTER split
#     X_train_p, X_test_p = preprocess_data(X_train_full, X_test)

#     # run experiment on different lambdas
#     # lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
#     # epochs = [1, 2, 3, 5]
#     learning_rate = [0.0001, 0.001, 0.01, 0.1, 1, 10]
#     results = []

#     for l in learning_rate:
#         model = SVMClassifier(epochs=2, learning_rate=l, lambda_=0.005)
#         model.train(X_train_p, y_train_full)

#         preds = model.predict(X_test_p)
#         acc = evaluate(y_test, preds)

#         results.append(acc)
#         print(l, acc)

# #     plt.plot(lambdas, results, marker='o')
# #     plt.xscale("log") # we use log scale according to report instructions


# #     plt.xlabel("Lambda Used (log scale)")
# #     plt.ylabel("Accuracy")
# #     plt.title("SVM Accuracy vs Lambda")
# #     plt.grid(True)
# #     plt.savefig("svm_plot.png")

