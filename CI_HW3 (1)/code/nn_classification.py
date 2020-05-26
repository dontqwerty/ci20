from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from nn_classification_plot import plot_hidden_layer_weights, plot_boxplot, plot_image
import numpy as np


"""
Assignment 3: Neural networks
Part 2: Classification with Neural Networks: Fashion MNIST

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODOs.
"""


def ex_2_1(X_train, y_train, X_test, y_test):
    """
    Solution for exercise 2.1
    :param X_train: Train set
    :param y_train: Targets for the train set
    :param X_test: Test set
    :param y_test: Targets for the test set
    :return:
    """

    # >>> from sklearn.neural_network import MLPClassifier
    # >>> from sklearn.datasets import make_classification
    # >>> from sklearn.model_selection import train_test_split
    # >>> X, y = make_classification(n_samples=100, random_state=1)
    # >>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
    # ...                                                     random_state=1)
    # >>> clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    # >>> clf.predict_proba(X_test[:1])
    # array([[0.038..., 0.961...]])
    # >>> clf.predict(X_test[:5, :])
    # array([1, 0, 1, 0, 1])
    # >>> clf.score(X_test, y_test)
    # 0.8...
    
    ## TODO
    train_accuracy = [0,0,0,0,0]
    test_accuracy = [0,0,0,0,0]
    clf = [0,0,0,0,0]
    for seed in range(1, 6):
      clf[seed -1] = MLPClassifier(hidden_layer_sizes=(10,), activation='tanh',
                          max_iter=50, random_state=(seed * 3))
      clf[seed -1].fit(X_train, y_train)
      train_accuracy[seed-1] = clf[seed -1].score(X_train, y_train)
      test_accuracy[seed-1] = clf[seed -1].score(X_test, y_test)

      print(train_accuracy[seed-1])
      print(test_accuracy[seed-1])
    plot_boxplot(train_accuracy, test_accuracy)

    y_test_pred = clf[4].predict(X_test)
    confusion = confusion_matrix(y_test, y_test_pred)

    # plot misclass
    misclass_coun = 0
    for ix, y in enumerate(y_test):
      if (y != y_test_pred[ix]):
        plot_image(X_test[ix])
        misclass_coun += 1
      if misclass_coun == 3:
        break

    # TODO: plot missclassified images based on confusion matrix
    print(confusion)

    # TODO: plot weights between input and hidden
    plot_hidden_layer_weights(clf[4].coefs_[0])

    pass
