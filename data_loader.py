# -*- coding: utf-8 -*-

"""
This module handles the data input.
"""

import numpy as np


class DataLoader:

    def __init__(self, path):
        """
        :param path: File path of the data
        """
        data = np.loadtxt(path, delimiter=',')
        np.random.seed(1)
        np.random.shuffle(data)
        self.X = data[:, :21]
        self.Y = data[:, 21:]

    def get_all_data(self, normalize=False):
        """
        Returns the entire data set.
        :param normalize: Whether or not the data will be normalized
        :return: Data set, output values
        """
        if normalize:
            return DataLoader._normalize(self.X), self.Y
        return self.X, self.Y

    def get_data_split(self, split, normalize=False):
        """
        Returns the data according to the specified splits.
        The 'split' parameter is a vector with either two or three components that in both cases have to sum to 1.
        If the 'split' vector contains two components, the first component specifies the portion of the data set
        that will be used for the training set and the second component specifies the portion of the data set that
        will be used for the test set.
        If the 'split' vector contains three components, the first and last components will be interpreted as described
        above and the middle component will be used to specify the portion of the data set that will be used for the
        cross validation set.
        :param split: A vector containing either 2 or 3 values.
        :param normalize: Whether or not the data will be normalized
        :return: Training set, output values for training set, cross validation set, output values for cross validation
                    set, test set, output values for cross validation set
        """
        assert np.sum(split) == 1
        X = self.X
        if normalize:
            X = DataLoader._normalize(self.X)
        if len(split) == 2:
            idx_train = int(np.round(self.X.shape[0] * split[0]))
            X_train = X[:idx_train, :]
            Y_train = self.Y[:idx_train, :]
            X_test = X[idx_train:, :]
            Y_test = self.Y[idx_train:, :]
            return X_train, Y_train, X_test, Y_test
        if len(split) == 3:
            idx_train = int(np.round(self.X.shape[0] * split[0]))
            idx_cv = idx_train + int(np.round(self.X.shape[0] * split[1]))
            idx_test = idx_cv + int(np.round(self.X.shape[0] * split[2]))
            X_train = X[:idx_train, :]
            Y_train = self.Y[:idx_train, :]
            X_cv = X[idx_train:idx_cv, :]
            Y_cv = self.Y[idx_train:idx_cv, :]
            X_test = X[idx_cv:idx_test, :]
            Y_test = self.Y[idx_cv:idx_test, :]
            return X_train, Y_train, X_cv, Y_cv, X_test, Y_test

    @staticmethod
    def _normalize(X):
        """
        Normalizes the data.
        :param X: Data
        :return: Normalized data
        """
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

