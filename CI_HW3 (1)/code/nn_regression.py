import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt

from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, \
    plot_learned_function, plot_mse_vs_alpha

"""
Assignment 3: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODOs.
"""


def calculate_mse(nn, x, y):
    """
    Calculates the mean squared error on the training and test data given the NN model used.
    :param nn: An instance of MLPRegressor or MLPClassifier that has already been trained using fit
    :param x: The data
    :param y: The targets
    :return: Training MSE, Testing MSE
    """
    ## TODO
    mse = 0
    
    mse = mean_squared_error(y, nn.predict(x))
    return mse


def ex_1_1_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 a)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO

    nh = 50
    nn = MLPRegressor(activation='logistic',solver='lbfgs',max_iter=5000,alpha= 0,hidden_layer_sizes=(nh,))
    nn.fit(x_train,y_train)
    y_pred_train = nn.predict(x_train)
   
    y_pred_test = nn.predict(x_test)
    plot_learned_function(nh, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)
    pass


def ex_1_1_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    nh = 5
    max_it = 10
    min_i_train = 11
    min_i_test = 11
    max_train = 0
    min_train = 1
    min_test = 11
    mean_train = 1
    std_train = 0
    mse = np.zeros(shape=(10))
    for i in range(0,max_it):
         nn = MLPRegressor(activation='logistic',solver='lbfgs',max_iter=5000,alpha= 0,hidden_layer_sizes=(nh,),random_state=i)
         nn.fit(x_train,y_train)
         mse_train = calculate_mse(nn, x_train, y_train)
         mse_test  = calculate_mse(nn, x_test, y_test)
         mse[i] = mse_train
         print(i,'.)MSE train ',mse_train)
        # print(i,'.)MSE test  ',mse_test)
         if(mse_train < min_train):
             min_train = mse_train
             min_i_train = i
         if(mse_test < min_test):
             min_test = mse_test
             min_i_test = i
             
         if(mse_train > max_train):
            max_train = mse_train
    
    print("Minimum Train: ",min_train)
    print("Maximum Train: ",max_train)
    mean_train = mse.mean()
    print("Mean Train: ",mean_train)
    std_train = np.std(mse)
    print("Standard Derivation Train: ",std_train)
    print("Min Index train: ",min_i_train," & test: ",min_i_test)
    pass


def ex_1_1_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 c)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    nh = [1,2,4,6,8,12,20,40]
    mse_all_train = np.zeros(shape=(8,10))
    mse_all_test  = np.zeros(shape=(8,10))

    for i in range(0,10):
        for j in range(0,8):   
            seed = np.random.randint(1,100)
            nn = MLPRegressor(activation='logistic',solver='lbfgs',max_iter=5000,alpha= 0,hidden_layer_sizes=(nh[j],),random_state=seed)
            nn.fit(x_train,y_train)
            mse_train = calculate_mse(nn, x_train, y_train)
            mse_test  = calculate_mse(nn, x_test, y_test)
            mse_all_train[j][i] = mse_train
            mse_all_test[j][i] = mse_test
    plot_mse_vs_neurons(mse_all_train, mse_all_test, nh)
    
    nn = MLPRegressor(activation='logistic',solver='lbfgs',max_iter=5000,alpha= 0,hidden_layer_sizes=(nh[2],))
    nn.fit(x_train,y_train)
    y_pred_train = nn.predict(x_train)
   
    y_pred_test = nn.predict(x_test)
 
    plot_learned_function(nh[2], x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)

    pass


def ex_1_1_d(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 d)
    Remember to set alpha to 0 when initializing the model.
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    nh = [2,5,50]
    seed = 0
    iterations = 5000
    mse_all_train = np.zeros(shape=(3,iterations))
    mse_all_test  = np.zeros(shape=(3,iterations))
    for j in range(0,3): 
         nn = MLPRegressor(activation='logistic',solver='lbfgs',warm_start = True,max_iter=1,alpha= 0,hidden_layer_sizes=(nh[j],),random_state=seed)
         for i in range(0,iterations):
       
           
            nn.fit(x_train,y_train)
            mse_train = calculate_mse(nn, x_train, y_train)
            mse_test  = calculate_mse(nn, x_test, y_test)
            mse_all_train[j][i] = mse_train
            mse_all_test[j][i] = mse_test
            
    plot_mse_vs_iterations(mse_all_train, mse_all_test, iterations, nh)
    ## TODO
    pass


def ex_1_2(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 a)
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO#
    nh = 50
    alp = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    seed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    iterations = 5000
    mse_all_train = np.zeros(shape=(9,10))
    mse_all_test  = np.zeros(shape=(9,10))
    for j in range(0,9):
        for i in range(0, 10) :
            nn = MLPRegressor(activation='logistic',solver='lbfgs',max_iter=5000,alpha= alp[j], hidden_layer_sizes=(50,), random_state=seed[i])         #for i in range(0,iterations):
            nn.fit(x_train,y_train)
            mse_train = calculate_mse(nn, x_train, y_train)
            mse_test  = calculate_mse(nn, x_test, y_test)
            mse_all_train[j][i] = mse_train
            mse_all_test[j][i] = mse_test
            
    plot_mse_vs_alpha(mse_all_train, mse_all_test, alp)
   
    pass
