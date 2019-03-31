# -*- coding:utf-8 -*-

__author__ = 'Andy Yang'

import copy
import data, data_visual, perceptron_algor,choose_model,svm_algor

if __name__ == "__main__":
    # Acquire data
    N = 250
    train_data, test_data = data.makeLinearSeparableData(N)
    # Data visualisation
    visualisation = data_visual.data_visual(
        copy.deepcopy(train_data), copy.deepcopy(test_data))
    # Show train_data
    visualisation.showTrainData()
    # Set K-fold cross-validation arg
    K = 4
    # Get the model using percetron algorithm
    model = perceptron_algor.perceptron(copy.deepcopy(train_data), K=4)
    # Get the optimum model
    opt_model = choose_model.chooseModel(model)
    # Model Visualisation and Test
    visualisation.showModel(opt_model[0], opt_model[1])
    # Use SVM algorithm
    svm_algor.svmAlgor()