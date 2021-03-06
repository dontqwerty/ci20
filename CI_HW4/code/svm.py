import numpy as np
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix

from svm_plot import plot_svm_decision_boundary, plot_score_vs_degree, plot_score_vs_gamma, plot_mnist, \
    plot_confusion_matrix

"""
Assignment 4: Support Vector Machine, Kernels & Multiclass classification
TODOS are contained here.
"""


def ex_1_a(x, y):
    """
    Solution for exercise 1 a)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########
    clf = svm.SVC(kernel='linear')
    clf.fit(x,y)
    plot_svm_decision_boundary(clf, x, y)
    pass


def ex_1_b(x, y):
    """
    Solution for exercise 1 b)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########
    point_x = [4, 0]
    point_y = [1]
    x_extended = np.vstack((x, point_x))
    y_extended = np.hstack((y, point_y))

    clf = svm.SVC(kernel='linear')
    clf.fit(x_extended, y_extended)
    plot_svm_decision_boundary(clf, x_extended, y_extended)

    pass


def ex_1_c(x, y):
    """
    Solution for exercise 1 c)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel with different values of C
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########
    Cs = [1e6, 1, 0.1, 0.001]

    point_x = [4, 0]
    point_y = [1]
    x_extended = np.vstack((x, point_x))
    y_extended = np.hstack((y, point_y))
    for i in Cs:
        clf = svm.SVC(kernel='linear', C=i)
        clf.fit(x_extended, y_extended)
        plot_svm_decision_boundary(clf, x_extended, y_extended)

    pass


def ex_2_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel for the given dataset
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    plot_svm_decision_boundary(clf, x_train, y_train, x_test, y_test)
    score = clf.score(x_test, y_test)
    print(score)

    pass


def ex_2_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with polynomial kernels for different values of the degree
    ## (Remember to set the 'coef0' parameter to 1)
    ## and plot the variation of the training and test scores with polynomial degree using 'plot_score_vs_degree' func.
    ## Plot the decision boundary and support vectors for the best value of degree
    ## using 'plot_svm_decision_boundary' function
    ###########
    degrees = range(1, 21)
    train_scores = list()
    test_scores = list()
    for i in degrees:
        clf = svm.SVC(kernel='poly', degree=i, coef0=1)
        clf.fit(x_train, y_train)
        test_scores.append(clf.score(x_test, y_test))
        train_scores.append(clf.score(x_train, y_train))
    plot_score_vs_degree(train_scores, test_scores, degrees)
    j = max(test_scores)
    best_degree = 0
    for i in range(0, 21):
        if test_scores[i] == j :
            best_degree = i
            clf = svm.SVC(kernel='poly', degree=i, coef0=1)
            clf.fit(x_train, y_train)
            plot_svm_decision_boundary(clf, x_train, y_train, x_test, y_test)
            print(clf.score(x_test, y_test))
            print(best_degree)
            break
    


def ex_2_c(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 c)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with RBF kernels for different values of the gamma
    ## and plot the variation of the test and training scores with gamma using 'plot_score_vs_gamma' function.
    ## Plot the decision boundary and support vectors for the best value of gamma
    ## using 'plot_svm_decision_boundary' function
    ###########
    gammas = np.arange(0.01, 2, 0.02)

    train_scores = list()
    test_scores = list()
    for i in gammas:
        clf = svm.SVC(kernel='rbf', gamma=i)
        clf.fit(x_train, y_train)
        test_scores.append(clf.score(x_test, y_test))
        train_scores.append(clf.score(x_train, y_train))
    plot_score_vs_gamma(train_scores, test_scores, gammas)

    j = max(test_scores)
    best_gamma = test_scores.index(j)
    
    clf = svm.SVC(kernel='rbf', gamma=gammas[best_gamma])
    clf.fit(x_train, y_train)
    plot_svm_decision_boundary(clf, x_train, y_train, x_test, y_test)
    print(clf.score(x_test, y_test))
    print(gammas[best_gamma])


def ex_3_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with one-versus-rest strategy with
    ## - linear kernel
    ## - rbf kernel with gamma going from 10**-5 to 10**5
    ## - plot the scores with varying gamma using the function plot_score_versus_gamma
    ## - Note that the chance level is not .5 anymore and add the score obtained with the linear kernel as optional argument of this function (parameter baseline)
    ###########
   
    gammas= [1e-5,1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3,1e4,1e5]
    decision_function_shape='ovr'
    train_scores = list()
    test_scores = list()
  
    clf_lin= svm.SVC(kernel='linear',C=10)
    clf_lin.fit(x_train,y_train)
    test_scores_lin = (clf_lin.score(x_test, y_test))
    train_scores_lin = (clf_lin.score(x_train, y_train))
    for j in range(0,11):
        
        clf = svm.SVC(decision_function_shape='ovr',kernel='rbf',gamma=gammas[j],C=10)
        clf.fit(x_train,y_train)
        test_scores.append(clf.score(x_test, y_test))
        train_scores.append(clf.score(x_train, y_train))
        
    plot_score_vs_gamma(train_scores,test_scores,gammas,lin_score_train=train_scores_lin,lin_score_test=test_scores_lin,baseline=.2)
  
def ex_3_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with a LINEAR kernel
    ## Use the sklearn.metrics.confusion_matrix to plot the confusion matrix.
    ## Find the index for which you get the highest error rate.
    ## Plot the confusion matrix with plot_confusion_matrix.
    ## Plot the first 10 images classified as the most misclassified digit using plot_mnist.
    ###########
    
    clf = svm.SVC(kernel="linear",decision_function_shape='ovr',C=10)
    clf.fit(x_train,y_train)
    y_pred =clf.predict(x_test)

    labels = range(1, 6)

    plot_confusion_matrix(confusion_matrix(y_test, y_pred), labels)
    print("conf: ",confusion_matrix(y_test, y_pred))
   
    sel_err = np.array([9,25,643,654,668,685,696,727,738,739])  # CHANGE ME! Numpy indices to select all images that are misclassified.
    i = 0  # CHANGE ME! Should be the label number corresponding the largest classification error.
    i=2
    j= 0
    print("sel_err ",sel_err)
  
  
    # Plot with mnist plot
    plot_mnist(x_test[sel_err], y_pred[sel_err], labels=labels[i], k_plots=10, prefix='Predicted class')
