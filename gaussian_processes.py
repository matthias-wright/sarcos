# -*- coding: utf-8 -*-

"""
This module implements Gaussian process regression.
"""

import numpy as np


class GaussianProcess:

    def __init__(self, X_train, Y_train):
        """
        :param X_train: training set
        :param Y_train: outputs of the training set
        """
        self.X_train = X_train
        self.Y_train = Y_train

    def draw_from_prior(self, X_test, num_samples, length_scale):
        """
        Returns samples drawn from the prior distribution at the test inputs.
        :param X_test: test set
        :param num_samples: number of samples to be drawn
        :param length_scale: determines the smoothness of the function
        :return: samples
        """
        K__star_star = GaussianProcess.squared_exponential_kernel(X_test, X_test, length_scale)
        return np.dot(K__star_star, np.random.normal(size=(len(X_test), num_samples)))

    def draw_from_posterior(self, X_test, num_samples, length_scale):
        """
        Returns samples drawn from the posterior at the test inputs.
        :param X_test: test set
        :param num_samples: number of samples to be drawn
        :param length_scale: determines the smoothness of the function
        :return: samples
        """
        K = GaussianProcess.squared_exponential_kernel(self.X_train, self.X_train, length_scale)
        K_inverse = np.linalg.inv(K)
        K__star_star = GaussianProcess.squared_exponential_kernel(X_test, X_test, length_scale)
        K_star_ = GaussianProcess.squared_exponential_kernel(X_test, self.X_train, length_scale)
        K__star = GaussianProcess.squared_exponential_kernel(self.X_train, X_test, length_scale)
        K_conditioned = K__star_star - np.dot(np.dot(K_star_, K_inverse), K__star)
        mu = np.dot(np.dot(K_star_, K_inverse), self.Y_train)
        return mu + np.dot(K_conditioned, np.random.normal(size=(len(X_test), num_samples)))

    def predict(self, X_test, length_scale):
        """
        Predicts the output values for the provided inputs.
        :param X_test: Test set
        :param length_scale: determines the smoothness of the function
        :return: predictions, variance for the predictions
        """
        K = GaussianProcess.squared_exponential_kernel(self.X_train, self.X_train, length_scale)
        K_inverse = np.linalg.inv(K)
        K__star_star = GaussianProcess.squared_exponential_kernel(X_test, X_test, length_scale)
        K_star_ = GaussianProcess.squared_exponential_kernel(X_test, self.X_train, length_scale)
        K__star = GaussianProcess.squared_exponential_kernel(self.X_train, X_test, length_scale)
        K_conditioned = K__star_star - np.dot(np.dot(K_star_, K_inverse), K__star)
        sigma_squared = np.diag(K_conditioned)
        mu = np.dot(np.dot(K_star_, K_inverse), self.Y_train)
        return mu, sigma_squared

    def predict_cholesky(self, X_test, length_scale):
        """
        Predicts the output values for the provided inputs. Uses the Cholesky decomposition (more on that in the report)
        in order to reduce computation and improve numerical stability.
        :param X_test: Test set
        :param length_scale: determines the smoothness of the function
        :return: predictions, variance for the predictions
        """
        K = GaussianProcess.squared_exponential_kernel(self.X_train, self.X_train, length_scale)
        L = np.linalg.cholesky(K)
        K_star_ = GaussianProcess.squared_exponential_kernel(X_test, self.X_train, length_scale)
        v = np.linalg.solve(L, self.Y_train)
        w = np.linalg.solve(L.T, v)
        mu = np.dot(K_star_, w)
        q = np.linalg.solve(L, K_star_.T)
        z = np.linalg.solve(L.T, q)
        K__star_star = GaussianProcess.squared_exponential_kernel(X_test, X_test, length_scale)
        K_conditioned = K__star_star - np.dot(K_star_, z)
        sigma_squared = np.diag(K_conditioned)
        return mu, sigma_squared

    def test(self, X_test, Y_test, length_scale, use_cholesky):
        """
        Tests the algorithm on the provided test set.
        :param X_test: test set
        :param Y_test: output values of the test set
        :param length_scale: determines the smoothness of the function
        :param use_cholesky: True: use Cholesky decomposition for computation, False: do not
        :return: RMSE for the test set, predictions on the test set, variance for the predictions
        """
        if use_cholesky:
            mu, sigma = self.predict_cholesky(X_test, length_scale)
        else:
            mu, sigma = self.predict(X_test, length_scale)
        rmse = np.sqrt(np.mean(np.square(mu - Y_test)))
        return rmse, mu, sigma

    @staticmethod
    def squared_exponential_kernel(x1, x2, length_scale):
        """
        Returns the covariance matrix according to the squared exponential kernel between x1 and x2.
        :param x1: First set of input vectors
        :param x2: Second set of input vectors
        :param length_scale: Determines the smoothness of the function
        :return: Covariance matrix
        """
        dist = np.sum(x1**2, axis=1, keepdims=True) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
        return np.exp(-(1 / (2 * length_scale**2)) * dist)

