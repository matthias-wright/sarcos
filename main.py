# -*- coding: utf-8 -*-
import numpy as np
import data_loader
import nearest_neighbors
import linear_regression
import random_forests
import gaussian_processes


data = data_loader.DataLoader('sarcos_inv.csv')


def sarcos_nearest_neighbors():
    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = data.get_data_split([0.6, 0.2, 0.2], normalize=True)
    print('Nearest neighbors, sarcos:')
    print('\tThis will take about 30 seconds, please wait...')
    model = nearest_neighbors.NearestNeighbors(X_train, Y_train)
    #k = model.get_best_k(X_cv, Y_cv, 10)
    rmse, _ = model.test(X_test, Y_test, 4)
    print('\tRMSE: ' + str(rmse))


def sarcos_linear_regression():
    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = data.get_data_split([0.6, 0.2, 0.2], normalize=True)
    print('Linear regression, sarcos:')
    print('\tThis will take about 5 seconds, please wait...')
    model = linear_regression.LinearRegression(X_train, Y_train)
    model.train(10000, 0.1, print_iterations=False)
    rmse, prediction = model.test(X_test, Y_test)
    rmse_train, _ = model.test(X_train, Y_train)
    print('\tRMSE: ' + str(rmse))


def sarcos_regression_forest():
    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = data.get_data_split([0.6, 0.2, 0.2], normalize=False)
    print('Regression forests, sarcos:')
    print('\tThis will take about 6 minutes, please wait...')
    model = random_forests.RandomForest(X_train, Y_train, num_trees=30, training_set_size=4000, min_nodes=5)
    mean, std = model.predict_all(X_test)
    rmse = np.sqrt(np.mean(np.square(mean - Y_test)))
    print('\tRMSE: ' + str(rmse))


def sarcos_gaussian_processes():
    X_train, Y_train, X_cv, Y_cv, X_test, Y_test = data.get_data_split([0.6, 0.2, 0.2], normalize=True)
    print('Gaussian processes, sarcos:')
    print('\tThis will take about 15 minutes and use a lot of memory, please wait...')
    model = gaussian_processes.GaussianProcess(X_train, Y_train)
    rmse, mean, var = model.test(X_test, Y_test, length_scale=0.9, use_cholesky=True)
    print('\tRMSE: ' + str(rmse))


sarcos_nearest_neighbors()
sarcos_linear_regression()
sarcos_regression_forest()
sarcos_gaussian_processes()


