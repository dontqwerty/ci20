import numpy as np
import matplotlib.pyplot as plt
from svm_plot import plot_decision_function 

"""
Assignment 4: Support Vector Machine, Kernels & Multiclass classification
TODOs. Fill the cost function, the gradient function and gradient descent solver.
"""

def ex_4_a(x, y):
    
    # TODO: Split x, y (take 80% of x, and corresponding y). You can simply use indexing, since the dataset is already shuffled.
    
    X_train=x[0:80]
    y_train=y[0:80]
    X_test=x[80:101]
    y_test=y[80:101]
    C=1
    eta=0.01
    max_iter=100
    # Define the functions of the parameter we want to optimize
    f = lambda th: cost(th, X_train, y_train, C)
    df = lambda th: grad(th, X_train, y_train, C)
    
    # TODO: Initialize w and b to zeros. What is the dimensionality of w?
    w=np.zeros(2)
    b=np.zeros(1)
    theta_opt, E_list = gradient_descent(f, df, (w, b), eta, max_iter)
    w, b = theta_opt
    print("w optimal ",w)
    print("b optimal ",b)
    y_pred = np.zeros(20)
    # TODO: Calculate the predictions using the test set
    for i in range(X_test.shape[0]):
        if(np.dot(np.transpose(w),X_test[i])+b >= 0):
            y_pred[i] = 1
        else:
            y_pred[i] = -1
    print("ypred ", y_pred)
    print("y_test ",y_test)
    print("acc ",y_pred==y_test)
    # TODO: Calculate the accuracy
    acc =0
    for i in range(y_test.size):
        if(y_pred[i] == y_test[i]):
            acc +=1
    acc = acc/(y_test.size)
    acc=np.mean(y_test == y_pred)
    
    print("acc ",acc)
    # Plot the list of errors
    if len(E_list) > 0:
        fig, ax = plt.subplots(1)
        ax.plot(E_list, linewidth=2)
        ax.set_xlabel('Iteration number')
        ax.set_ylabel('Error')
        ax.set_title('Error monitoring')
        
    # TODO: Call the function for plotting (plot_decision_function).
        
    plot_decision_function(theta_opt, X_train, X_test, y_train, y_test)

def gradient_descent(f, df, theta0, learning_rate, max_iter):
    """
    Finds the optimal solution of the function f(x) using gradient descent:
    Until the number of iteration is reached, decreases the parameter x by the gradient times the learning_rate.
    The function should return the minimal argument x and the list of errors at each iteration in a numpy array.

    :param f: function to minimize
    :param df: gradient of f
    :param theta0: initial point
    :param learning_rate:
    :param max_iter: maximal number of iterations
    :return: x (solution), E_list (array of errors over iterations)
    """
    ##############
    #
    # TODO
    #
    # Implement a gradient descent algorithm

    E_list = np.zeros(max_iter)
    w, b = theta0
    theta = theta0
    for i in range (0,max_iter):
        theta_old = theta
        
    
        w_df, b_df = df(theta)
        
        w=w-learning_rate*w_df
        b=b-learning_rate*b_df
        theta=(w,b)
        E_list[i] = f(theta) 

    # END TODO
    ###########

    return (w, b), E_list


def cost(theta, x, y, C):
    """
    Computes the cost of the SVM objective.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :param C: penalty term
    :return: cost
    """
    cost = 0 # TODO 
    w,b = theta
  
    cost = 1/2*np.sum(w**2)
    cost_tmp = 0
    for i in range(0,80):
        cost_tmp += np.max([0,1 -y[i]*(np.dot(np.transpose(w),x[i])+b)])
    
    cost +=  C/80 *cost_tmp
    print("cost ",cost )
    return cost


def grad(theta, x, y, C):
    """

    Computes the gradient of the SVM objective.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :param C: penalty term
    :return: grad_w, grad_b
    """
    w, b = theta
    
    
    grad_w = 0  # TODO 
    grad_b = 0  # TODO 
    
    w_tmp = 0
    b_tmp = 0
    
    for i in range(x.shape[0]):
        if(1-y[i]*(np.dot(np.transpose(w),x[i]+b)) <= 0):
            Ii = 0
        else:
            Ii = 1
        w_tmp += Ii*y[i]*x[i]
        b_tmp = Ii*y[i]
    
    grad_w = w - C/x.shape[0] *w_tmp
    grad_b = -C/x.shape[0] *b_tmp
   
    return grad_w, grad_b
