#!/usr/bin/env python
import numpy as np
from numpy.linalg import pinv

"""
Assignment: Linear and Logistic Regression
Section: Linear Regression with polynomial features

This file contains the main work to be done.
The functions are:
- TODO design_matrix: Create the design matrix including the polynomial expansions and the constant feature
- TODO train: finds the analytical solution of linear regression
- TODO compute_error: return the cost function of linear regression Mean Square Error
- train_and_test: call the compute error function and all sets and return the corresponding errors

"""


def design_matrix(x, degree):
    """
    Creates the design matrix given the data x.
    The design matrix is built of all polynomials of x from degree 0 to 'degree'.

    EX: for the data x = [0,1,2] and degree 2
    the function should return: [[1, 0, 0],
								 [1, 1, 1],
								 [1, 2, 4]] 

    :param x: numpy array of shape (N,1)
    :param degree: Higher degree of the polynomial
    :return: Expanded data in a numpy array of shape (N,degree+1)
    """

    ######################
    #
    # TODO
    #
    # Return the numpy array of shape (N,degree+1)
    # Storing the data of the form x_i^j at row i and column j
    # Look at the function description for more info
    #
    # TIP: use the power function from numpy
    X = np.zeros([np.size(x), degree+1])
    for i in range(degree + 1):
        for j in range(np.size(x)) :
            if (j == 0) :
                X[j][i] = 1
            else :
                X[j][i] = np.power(x[j], i)


    # X = x  # TODO: change me

    #
    # END TODO
    ######################

    return X


def train(x, y, degree):
    """
    Returns the optimal coefficients theta that minimizes the error
    ||  X * theta - y ||**2
    when X is the polynomial expansion of x_train of degree 'degree'.

    :param x: numpy array on the input
    :param y: numpy array containing the output
    :param degree: maximum polynomial degree in the polynomial expansion
    :return: a numpy array containing the coefficient of each polynomial degree in the regression
    """

    ######################
    #
    # TODO
    #
    # Returns the analytical solution of the linear regression
    #
    # TIPs:
    #  - Don't forget to first expand the data
    #  - WARNING:   With numpy array * is a term-term matrix multiplication
    #               The function np.dot performs a classic matrix multiplication (recent Python version accept @)
    #
    #  - To compute the pseudo inverse (A*A.T)^-1 * A.T with a more stable algorithm numpy provides the function pinv
    #   pinv is accessible in the sub-library numpy.linalg
    #
    X = np.zeros([np.size(x), degree+1])
    for i in range(degree + 1):
        for j in range(np.size(x)) :
            if (j == 0) :
                X[j][i] = 1
            else :
                X[j][i] = np.power(x[j], i)
    
    

    theta_opt = np.zeros(degree + 1)  # TODO: Change me
    # for r in range(np.size(theta_opt)) :
    
    X_ident = np.zeros([np.size(x), degree+1])
    for i in range(np.size(x)) :
        for j in range(degree + 1) :
            if (i == j) & i < (degree + 1) :
                X_ident[i][j] = 0    # used Lambda -> 0.5 (too small) / 0.55 (also too small but better) / 0.8 fine / 0 just to show the result without regularization
    
    theta_opt = np.dot(np.linalg.pinv(X + X_ident), y)


    # END TODO
    ######################

    return theta_opt


def compute_error(theta, degree, x, y):
    """
    Predicts the value of y given by the model given by theta and degree.
    Then compares the predicted value to y and provides the Mean squared error.

    :param theta: Coefficients of the linear regression
    :param degree: Degree in the polynomial expansion
    :param x: Input data
    :param y: Output data to be compared to prediction
    :return: err: Mean squared error
    """

    ######################
    #
    # TODO
    #
    # Returns the error (i.e. the cost function)
    #
    # TIPs:
    #  - WARNING:   With numpy array * is a term-term matrix multiplication
    #               The function np.dot performs a matrix multiplication
    #               A longer alternative is to first change your array to the matrix class using np.matrix,
    #               Then * becomes a matrix multiplication
    #
    #  - One can use the numpy function mean
    X = np.zeros([np.size(x), degree+1])
    for i in range(degree + 1):
        for j in range(np.size(x)) :
            if (j == 0) :
                X[j][i] = 1
            else :
                X[j][i] = np.power(x[j], i)

    x_matrix = np.matrix(X)     
    y_pred = x_matrix.dot(theta)
    # err = (y_pred - y)
    err = []
    
    for i in range(np.size(y_pred)) :
        err.append((y_pred[i] - y[i]) ** 2)
    #
    # END TODO
    ######################

    return np.mean(err)

def train_and_test(data, degree):
    """
    Trains the model with degree 'degree' and provides the MSE for the training, validation and testing sets

    :param data:
    :param degree:
    :return:
    """

    theta = train(data['x_train'], data['y_train'], degree)

    err_train = compute_error(theta, degree, data['x_train'], data['y_train'])
    err_val = compute_error(theta, degree, data['x_val'], data['y_val'])
    err_test = compute_error(theta, degree, data['x_test'], data['y_test'])

    return theta, err_train, err_val, err_test
