# -*- coding: utf-8 -*-

"""
This module implements the random forests algorithm.
"""

import numpy as np
import decision_tree


class RandomForest:

    def __init__(self, X_train, Y_train, num_trees, training_set_size, min_nodes):
        """
        :param X_train: Training set
        :param Y_train: Output values for the training set
        :param num_trees: Number of decision trees
        :param training_set_size: Size of the training set that each decision tree is build with
        :param min_nodes: If a node contains equal or less than 'min_nodes',
                            then the recursion is terminated and a leaf node is created
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.tree_list = []
        self._create_forest(num_trees, training_set_size, min_nodes)

    def predict(self, x):
        """
        Predicts the output value of the input vector using
        the average over all of the prediction from the decision trees.
        :param x: Input vector
        :return: Mean of the leaf nodes, standard deviation of the leaf nodes
        """
        mean_all = 0
        std_all = 0
        for tree in self.tree_list:
            mean, std = tree.predict(x)
            mean_all = mean_all + mean
            std_all = std_all + std
        mean_all = mean_all / len(self.tree_list)
        std_all = std_all / len(self.tree_list)
        return mean_all, std_all

    def predict_all(self, X):
        """
        Predicts the output values of an entire set of input vectors
        using the average over all of the prediction from the decision trees.
        :param X: Set of input vectors
        :return: Mean of the leaf nodes, standard deviation of the leaf nodes
        """
        means = np.zeros((X.shape[0], 1))
        stds = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            mean, std = self.predict(X[i])
            means[i] = mean
            stds[i] = std
        return means, stds

    def _create_forest(self, num_trees, training_set_size, min_nodes):
        """
        Creates a random forest.
        :param num_trees: Number of decision trees
        :param training_set_size: Size of the training set that each decision tree is build with
        :param min_nodes: If a node contains equal or less than 'min_nodes',
                            then the recursion is terminated and a leaf node is created
        """
        for i in range(num_trees):
            tree = self._create_tree(training_set_size, min_nodes)
            self.tree_list.append(tree)

    def _create_tree(self, training_set_size, min_nodes):
        """
        Creates a decision tree from a random subset of the training examples.
        :param training_set_size: Size of the training set that each decision tree is build with
        :param min_nodes: If a node contains equal or less than 'min_nodes',
                            then the recursion is terminated and a leaf node is created
        :return: Decision tree
        """
        random_subset_indices = np.random.choice(np.arange(self.X_train.shape[0]), size=training_set_size, replace=False)
        X = self.X_train[random_subset_indices]
        Y = self.Y_train[random_subset_indices]
        tree = decision_tree.DecisionTree(X, Y, min_nodes=min_nodes)
        return tree
