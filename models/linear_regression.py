# -*- coding: utf-8 -*-

"""
This module implements linear regression.
"""

import numpy as np


class LinearRegression:

    def __init__(self, X_train, Y_train):
        """
        :param X_train: training set
        :param Y_train: outputs of the training set
        """
        self.X_train = X_train
        self.Y_train = Y_train
        n = self.X_train.shape[1]
        self.w = np.random.normal(0, 0.5, size=(n, 1))
        self.b = np.random.normal(0, 0.5)

    def train(self, iterations, learning_rate, print_iterations=False):
        """
        Optimizes the objective function with the gradient descent algorithm
        in order to find well suited parameters w and b.
        :param iterations: The number of gradient descent iterations
        :param learning_rate: Step size of the gradient descent algorithm
        :param print_iterations: Whether or not the current iteration and its training loss shall be printed to console
        """
        cost_prev = np.inf
        current_learning_rate = learning_rate
        for _ in range(iterations):
            prediction = np.dot(self.X_train, self.w) + self.b
            cost = self._cost(prediction)
            if print_iterations:
                print('Iteration ' + str(_) + ', cost: ' + str(cost))
            self._update_weights(learning_rate, prediction)

            if cost >= cost_prev:
                current_learning_rate = current_learning_rate * 0.5
                continue

            if np.fabs(cost - cost_prev) < 1e-12:
                print('Stopping early')
                break

            cost_prev = cost

    def test(self, X_test, Y_test):
        """
        Test the current parameters on the provided test set.
        :param X_test: Test set
        :param Y_test: Output values for the test set
        :return: RMSE for the test set, predictions on the test set
        """
        prediction = np.dot(X_test, self.w) + self.b
        rmse = np.sqrt(np.mean(np.square(prediction - Y_test)))
        return rmse, prediction

    def _update_weights(self, learning_rate, prediction):
        """
        Implements the parameter update in the direction of the negative
        gradient of the objective function.
        :param learning_rate: The step size
        :param prediction: The predictions from the current iteration
        :return:
        """
        grad_w, grad_b = self._compute_gradient(prediction)
        self.w = self.w - learning_rate * grad_w
        self.b = self.b - learning_rate * grad_b

    def _compute_gradient(self, prediction):
        """
        Computes the gradient for the parameters w and b.
        :param prediction: The predictions from the current iteration
        :return: gradient of w, gradient of b
        """
        m = prediction.shape[0]
        grad_w = np.dot(self.X_train.T, (prediction - self.Y_train)) / m
        grad_b = np.sum((prediction - self.Y_train)) / m
        return grad_w, grad_b

    def _cost(self, prediction):
        """
        Computes the cost for the current set of parameters.
        :param prediction: The predictions from the current iteration
        :return: Cost value
        """
        return np.sum((prediction - self.Y_train)**2) / (2 * self.Y_train.shape[0])

    def reset_parameters(self):
        """
        Resets the parameters of the model.
        """
        n = self.X_train.shape[1]
        self.w = np.random.normal(0, 0.5, size=(n, 1))
        self.b = np.random.normal(0, 0.5)

    def normal_equation(self, X_test, Y_test):
        """
        Computes the exact solution for the linear regression problem via the normal equation.
        :param X_test: Test set
        :param Y_test: Output values of the test set
        :return: RMSE of the test set, predictions for the test set
        """
        X = np.copy(self.X_train)
        X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        w = np.dot(np.dot(np.dot(X.T, X), X.T), self.Y_train)
        X_ = np.copy(X_test)
        X_ = np.append(np.ones((X_.shape[0],1)), X_, axis=1)
        print(w.shape)
        print(X_.shape)
        prediction = np.dot(X_, w)
        rmse = np.sqrt(np.mean(np.square(prediction - Y_test)))
        return rmse, prediction

