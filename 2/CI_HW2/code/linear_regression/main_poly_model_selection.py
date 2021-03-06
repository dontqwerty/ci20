#!/usr/bin/env python

import numpy as np
import json
import matplotlib.pyplot as plt
from plot_poly import plot_poly, plot_errors
import poly

"""
Assignment: Linear and Logistic Regression
Section: Linear Regression with polynomial features

This file:
1) loads the data from 'data_linreg.json'
2) trains and tests a linear regression model for K degrees
3) TODO: select the degree that minimizes validation error
4) plots the optimal results

TODO boxes are here and in 'poly.py'
"""


def main():
    # Number of possible degrees to be tested
    K = 30
    data_path = 'data_linreg.json'

    # Load the data
    f = open(data_path, 'r')
    data = json.load(f)
    for k, v in data.items():
        data[k] = np.array(v).reshape((len(v), 1))

    # Init vectors storing MSE (Mean squared error) values of each set at each degrees
    mse_train = np.zeros(K)
    mse_val = np.zeros(K)
    mse_test = np.zeros(K)
    theta_list = np.zeros(K, dtype=object)
    degrees = np.arange(K) + 1

    # Compute the MSE values
    for i in range(K):
        theta_list[i], mse_train[i], mse_val[i], mse_test[i] = poly.train_and_test(data, degrees[i])

    ######################
    #
    # TODO
    #
    # Find the best degree that minimizes the validation error.
    # Store it in the variable i_best for plotting the results
    #
    # TIPs:
    # - use the argmin function of numpy
    # - the code above is already giving the vectors of errors
    i_best_val = np.argmin(mse_val) # TODO: Change this
    best_degree_val = degrees[i_best_val]
    best_theta_val = theta_list[i_best_val]

    i_best_tr = np.argmin(mse_train) # TODO: Change this
    best_degree_tr = degrees[i_best_tr]
    best_theta_tr = theta_list[i_best_tr]

    #
    # END TODO
    ######################

    # Plot the training error as a function of the degrees
    plot_errors(i_best_val, degrees, mse_train, mse_val, mse_test)
    plot_poly(data, best_degree_val, best_theta_val)
    plt.show()

    plot_errors(i_best_tr, degrees, mse_train, mse_val, mse_test)
    plot_poly(data, best_degree_tr, best_theta_tr)
    plt.show()


if __name__ == '__main__':
    main()
