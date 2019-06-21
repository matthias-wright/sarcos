# -*- coding: utf-8 -*-

"""
This module implements the k-nearest neighbors algorithm.
"""

import numpy as np


class NearestNeighbors:

    def __init__(self, X_train, Y_train):
        """
        :param X_train: training set
        :param Y_train: outputs of the training set
        """
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, x, k):
        """
        Predicts the output value of the provided example.
        :param x: Example input vector
        :param k: The number of neighbors used for computing the prediction
        :return: Prediction made based on the k neighbors
        """
        dist = np.linalg.norm(self.X_train - x, axis=1)
        idx = np.argpartition(dist, k)[:k]
        weights = 1 / dist[idx]**2
        return np.sum(np.squeeze(self.Y_train[idx]) * np.squeeze(weights)) / np.sum(weights)
        #return np.sum(np.squeeze(self.Y_train[idx])) / k

    def test(self, X_test, Y_test, k):
        """
        Tests the performance of the algorithm by means of the provided test set.
        :param X_test: Test set
        :param Y_test: Output values for the test set
        :param k: The number of neighbors used for computing the prediction
        :return: RMSE of the test set, predictions made on the test set
        """
        X_pred = np.zeros(Y_test.shape)
        for i in range(X_pred.shape[0]):
            X_pred[i][0] = self.predict(X_test[i], k)
        rmse = np.sqrt(np.mean(np.square(X_pred - Y_test)))
        return rmse, X_pred

    def get_best_k(self, X, Y, limit):
        """
        Tries out values for k in the range from 1 to 'limit', returns the k that yields the lowest RMSE.
        :param X: Cross validation set
        :param Y: Output values for the cross validation set
        :param limit: Range limit for k
        :return: Optimal k for the given range
        """
        best_k = -1
        best_rmse = np.inf
        for k in range(1, limit + 1):
            rmse, _ = self.test(X, Y, k)
            if rmse < best_rmse:
                best_rmse = rmse
                best_k = k
        return best_k
