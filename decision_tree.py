# -*- coding: utf-8 -*-

"""
This module implements a decision tree.
"""

import numpy as np


class Node:

    def __init__(self, is_leaf, mean, std, feature, split, left, right):
        """
        :param is_leaf: Whether or not it is a leaf node
        :param mean: The current mean of all the examples contained in the node
        :param std: The current standard deviation of all the examples contained in the node
        :param feature: The feature by which the split is performed
        :param split: The actual threshold value for the split
        :param left: Left child node
        :param right: Right child node
        """
        self.is_leaf = is_leaf
        self.mean = mean
        self.std = std
        self.feature = feature
        self.split = split
        self.left = left
        self.right = right


class DecisionTree:

    def __init__(self, X_train, Y_train, min_nodes):
        """
        :param X_train: Training set
        :param Y_train: Output values for the training set
        :param min_nodes: If a node contains equal or less than 'min_nodes',
                            then the recursion is terminated and a leaf node is created
        """
        self.root = self._create_tree(X_train, Y_train, min_nodes)

    def predict(self, x):
        """
        Predicts the output value of the input vector.
        :param x: Input vector
        :return: Mean of the leaf node, standard deviation of the leaf node
        """
        return self._predict(self.root, x)

    def predict_all(self, X):
        """
        Predicts the output values of an entire set of input vectors.
        :param X: Set of input vectors
        :return: Mean of the leaf nodes, standard deviation of the leaf nodes
        """
        means = np.zeros((X.shape[0], 1))
        stds = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            mean, std = self._predict(self.root, X[i])
            means[i] = mean
            stds[i] = std
        return means, stds

    def _predict(self, node, x):
        """
        Helper function for the predict function.
        :param node: Current node in the tree
        :param x: Input vector for which we predict the output value
        :return: Mean of the leaf node, standard deviation of the leaf node
        """
        if node.is_leaf:
            return node.mean, node.std
        if x[node.feature] <= node.split:
            return self._predict(node.left, x)
        else:
            return self._predict(node.right, x)

    def _create_tree(self, X, Y, min_nodes):
        """
        Recursively creates a decision tree.
        :param X: Current set of training examples
        :param Y: Output values for the training examples
        :param min_nodes: If a node contains equal or less than 'min_nodes', then
                            the recursion is terminated and a leaf node is created
        :return: Eventually returns the root of the decision tree
        """
        if len(X) <= min_nodes:
            return Node(is_leaf=True, mean=np.mean(Y), std=np.std(Y), feature=None, split=None, left=None, right=None)
        feature, best_split, left_indices, right_indices = DecisionTree._best_split_random(X, Y)
        left = self._create_tree(X[left_indices], Y[left_indices], min_nodes)
        right = self._create_tree(X[right_indices], Y[right_indices], min_nodes)
        return Node(is_leaf=False, mean=np.mean(Y), std=np.std(Y), feature=feature, split=best_split, left=left, right=right)

    @staticmethod
    def _best_split_random(X, Y):
        """
        Computes the best split for a given set of training examples.
        Tries out only a random subset (of size num_features / 2 + 1) of the features.
        Tries out only a random subset (of size sqrt(num_values)*2) of all of the existing values for a specific feature.
        :param X: Set of training examples
        :param Y: Output values of training examples
        :return: Feature for split, threshold value for split, indices of left child, indices of right child
        """
        min_sse = np.inf
        feature = None
        best_split = None
        left = None
        right = None
        random_feature_indices_ = np.random.choice(np.arange(X.shape[1]), size=int(X.shape[1] // 2 + 1), replace=False)
        for j in range(len(random_feature_indices_)):
            i = random_feature_indices_[j]
            num_values = X[:, i].shape[0]
            random_values_indices = np.random.choice(np.arange(num_values), size=int(2 * np.sqrt(num_values)), replace=False)
            values = X[random_values_indices, i]
            for split in values:
                left_indices = X[:, i:(i+1)] <= split
                left_indices, _ = np.nonzero(left_indices)
                right_indices = X[:, i:(i+1)] > split
                right_indices, _ = np.nonzero(right_indices)
                sse = DecisionTree._sum_of_squared_errors(Y[left_indices], Y[right_indices])
                if sse < min_sse:
                    feature = i
                    best_split = split
                    left = left_indices
                    right = right_indices
                    min_sse = sse
        return feature, best_split, left, right

    @staticmethod
    def _best_split(X, Y):
        """
        Computes the best split for a given set of training examples.
        Tries out all of the existing values for a specific feature.
        :param X: Set of training examples
        :param Y: Output values of training examples
        :return: Feature for split, threshold value for split, indices of left child, indices of right child
        """
        min_sse = np.inf
        feature = None
        best_split = None
        left = None
        right = None
        for i in range(X.shape[1]):
            values = np.unique(X[:, i])
            for split in values:
                left_indices = X[:, i:(i+1)] <= split
                left_indices, _ = np.nonzero(left_indices)
                right_indices = X[:, i:(i+1)] > split
                right_indices, _ = np.nonzero(right_indices)
                sse = DecisionTree._sum_of_squared_errors(Y[left_indices], Y[right_indices])
                if sse < min_sse:
                    feature = i
                    best_split = split
                    left = left_indices
                    right = right_indices
                    min_sse = sse
        return feature, best_split, left, right

    @staticmethod
    def _sum_of_squared_errors(Y1, Y2):
        """
        Computes the sum of squared errors between two vectors.
        :param Y1: First vector
        :param Y2: Second vector
        :return: Sum of squared errors
        """
        if len(Y1) == 0:
            sse1 = 0
        else:
            sse1 = np.sum((Y1 - np.mean(Y1))**2)
        if len(Y2) == 0:
            sse2 = 0
        else:
            sse2 = np.sum((Y2 - np.mean(Y2)) ** 2)
        return sse1 + sse2




