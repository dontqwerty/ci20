#!/usr/bin/env python
import numpy as np

from logreg_toolbox import sig

"""
Assignment: Linear and Logistic Regression
Section: Gradient descent (GD) and Logistic Regression
TODO Fill the cost function and the gradient
"""


def cost(theta, x, y):
    """
    Computes the cost of the logistic regression function.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: cost
    """
    N, n = x.shape

    ##############
    #
    # TODO
    #
    # Write the cost of logistic regression as defined in the lecture
    # Hint:
    #   - use the logistic function sig imported from the file toolbox
    #   - prefer numpy vectorized operations over for loops
    # 
    # WARNING: If you run into instabilities during the exercise this
    #   could be due to the usage log(x) with x very close to 0. Some
    #   implementations are more or less sensible to this issue, you
    #   may try another one. A (dirty) trick is to replace log(x) with
    #   log(x + epsilon) with epsilon a very small number like 1e-20
    #   or 1e-10 but the gradients might not be exact anymore. 

  
    c = 0
    c = -1/N * (y*np.log(sig(np.dot(x,theta)))+(1-y)*np.log(1-sig(np.dot(x,theta))))

    
    c = c.sum()

    # END TODO
    ###########

    return c


def grad(theta, x, y):
    """

    Computes the gradient of the cost of logistic regression

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: gradient
    """
    N, n = x.shape

    ##############
    #
    # TODO
    #
    #   - prefer numpy vectorized operations over for loops

 #   print("1 ",np.dot(x,theta))
   # print("2 ",np.dot(x[0],theta))
    g = np.zeros(theta.shape)
   
    for j in range (0,n):
        for i in range (0,N):
            g[j] += 1/N * ((sig(np.dot(x[i],theta))-y[i])*x[i][j])
    
    # END TODO
    ###########

    return g
