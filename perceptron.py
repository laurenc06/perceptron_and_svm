#!/usr/bin/env python3
"""
CMPSC 165 - Machine Learning
Homework 2, Problem 1: Voted Perceptron
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
    """Preprocess training and test data."""
    # standardize data so that they are all roughly the same size
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

class VotedPerceptron:
    """Voted Perceptron Classifier."""
    # initialize our stuff
    def __init__(self, epochs=1):
        self.epochs = epochs
        self.weights = []  # list of weight vectors w_k
        self.counts = []   # list of survival counts c_k

    def train(self, X, y):
        """Fit the classifier to training data."""
        # initialize weights as zeros, intialize count
        w = np.zeros(X.shape[1])
        c = 0

        # loop epochs, loop samples
        for epoch in range(self.epochs):
            for i in range(len(X)):
                # mistake defined as when y_i(w*x_i) <= 0
                score = np.dot(w, X[i]) 
                if y[i]*score <= 0:
                    if c > 0:
                        # save old weights before fixing
                        self.weights.append(w.copy()) # must be copy or else it will later be changed
                        self.counts.append(c)

                    # update
                    w = w + y[i]*X[i]
                    c = 1

                else: # if w survived
                    c += 1
            
        # store final (w, c) if c>0
        if c > 0:
            self.weights.append(w.copy())
            self.counts.append(c)

    def predict(self, X):
        """Predict labels for input samples."""
        predictions = []

        for i in range(len(X)):
            x = X[i]
            total = 0

            # total = sum over k of c_k * sign(w_k dot x)
            for k in range(len(self.weights)):
                vote = np.sign(np.dot(self.weights[k], x)) 
                weighted_vote = self.counts[k]*vote
                total += weighted_vote
        
            # pred = sign(total); if pred==0 choose +1
            pred = np.sign(total)
            if pred == 0:
                pred = 1

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

    # temp prints for testing
    # print("X_train", X_train.shape)
    # print("y_train", y_train.shape)
    # print("X_test", X_test.shape)

    # Train
    model = VotedPerceptron(epochs=3)  # tune epochs
    model.train(X_train_p, y_train)

    # Predict
    y_pred = model.predict(X_test_p).astype(int)

    # Save one integer per line, no header according to assignment instructions
    np.savetxt(pred_file, y_pred, fmt="%d")

# # This is a block of code for my testing and for generating my grpahs for the report
# if __name__ == "__main__":
#     X, y = load_data("spam_X.csv", "spam_y.csv")

#     n = len(X)
#     split_90 = int(0.9 * n)

#     X_train_full = X[:split_90]
#     y_train_full = y[:split_90]

#     X_test = X[split_90:]
#     y_test = y[split_90:]

#     # Preprocess AFTER split
#     X_train_p, X_test_p = preprocess_data(X_train_full, X_test)

#     # run experiment on different fractions of training data
#     # fractions = [0.01, 0.02, 0.05, 0.10, 0.20, 1.0]
#     epochs = [1, 2, 3, 5]
#     results = []

#     for e in epochs:
#         # size = int(frac*len(X_train_p))

#         # X_subset = X_train_p[:size]
#         # y_subset = y_train_full[:size]

#         model = VotedPerceptron(epochs=e)
#         # model.train(X_subset, y_subset)
#         model.train(X_train_p, y_train_full)

#         preds = model.predict(X_test_p)
#         acc = evaluate(y_test, preds)

#         results.append(acc)
#         print(e, acc)

#     # plt.plot([f*100 for f in fractions], results, marker='o')

#     # plt.xlabel("Percent of Remaining Training Data Used")
#     # plt.ylabel("Accuracy")
#     # plt.title("Voted Perceptron Accuracy vs Training Size")
#     # plt.grid(True)
#     # plt.savefig("perceptron_plot.png")
