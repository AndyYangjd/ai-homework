# -*- coding:utf-8 -*-
'''
考虑到SVM中SMO算法比较复杂，而且有成熟的库模块可以使用。
因此基于避免重复造轮子的考虑，直接使用scikit-learn库中的SVM算法模型，重点在于库API的使用。
另外本模块不再使用之前的方法通过函数获得数据，而是换个思路使用pands来处理data.py中生成的txt数据。
注：scikit-learn是一个广泛应用的用于实现机器学习算法的库。
参考:https://github.com/lawlite19/MachineLearning_Python/blob/master/SVM/SVM_scikit-learn.py
'''
__author__ = 'Andy Yang'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

def svmAlgor():
    # Acquire data from txt using pandas
    tra_file_name = r"./train_data.csv"
    tra_file = pd.read_csv(
        tra_file_name, header=None, names=['x0', 'x1', 'class'])

    test_file_name = r"./test_data.csv"
    test_file = pd.read_csv(
        test_file_name, header=None, names=['x0', 'x1', 'class'])
    # Translate type to matrix
    tra_data = tra_file.to_numpy()
    num_cols = tra_data.shape[1]
    tra_X = tra_data[:, 0:num_cols - 1]
    tra_label = tra_data[:, -1]

    test_data = test_file.to_numpy()
    num_cols = test_data.shape[1]
    test_X = test_data[:, 0:num_cols - 1]
    test_label = test_data[:, -1]
    # Create model
    # gamma为核函数的系数，值越大，拟合的越好
    model = svm.SVC(kernel='linear', C=10, gamma=1).fit(tra_X, tra_label)
    plotDecision(tra_X, tra_label, test_X, test_label, model)

def plotData(tra_X, tra_label, test_X, test_label):
    LIM = 100
    plt.figure(figsize=(100, 100))
    tra_positive = np.where(tra_label == 1)
    tra_negative = np.where(tra_label == -1)
    plt.plot(
        np.ravel(tra_X[tra_positive, 0]),
        np.ravel(tra_X[tra_positive, 1]),
        'r+',
        label='tra_positive-points')
    plt.plot(
        np.ravel(tra_X[tra_negative, 0]),
        np.ravel(tra_X[tra_negative, 1]),
        'b+',
        label='tra_negative-points')

    test_positive = np.where(test_label == 1)
    test_negative = np.where(test_label == -1)
    plt.plot(
        np.ravel(test_X[test_positive, 0]),
        np.ravel(test_X[test_positive, 1]),
        'ro',
        label='test_positive-points')
    plt.plot(
        np.ravel(test_X[test_negative, 0]),
        np.ravel(test_X[test_negative, 1]),
        'bo',
        label='test_negative-points')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-LIM, LIM)
    plt.ylim(-LIM, LIM)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.12), borderaxespad=0)
    return plt

def plotDecision(tra_X, tra_label, test_X, test_label, model):
    plt = plotData(tra_X, tra_label, test_X, test_label)
    LIM = 100
    w = model.coef_
    b = model.intercept_
    print(w)
    print(b)
    xp = range(-LIM, LIM, 1)
    yp = -(w[0, 0] * xp + b) / w[0, 1]
    plt.plot(xp, yp, 'b-')
    plt.draw()
    plt.pause(3)
    plt.savefig('svm_model.png', dpi=100)
    plt.close()

if __name__ == "__main__":
    svmAlgor()