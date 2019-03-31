# -*- coding:utf-8 -*-
"""
采用随机梯度法设计感知机算法，使误分类的点数最小。
采用K折交叉验证法计算模型。
"""

__author__ = 'Andy Yang'

import numpy as np

def perceptron(data, K=4):
    # 取整除，取商的整数部分
    num_validation = len(data) // K

    model = []
    for i in range(K):
        val_data = data[i * num_validation:(i + 1) * num_validation]
        tra_data = data[:i * num_validation] + data[(i + 1) * num_validation:]
        # Acquire tra_X and tra_y
        tra_X = [i[:2] for i in tra_data]
        tra_y = [i[-1] for i in tra_data]
        # convert list to array
        tra_X = np.array(tra_X)
        tra_y = np.array(tra_y)
        # Initialize weights
        w = np.zeros(2)
        b = 0
        # Define learning rate
        lr = 1
        # Iterate tra_X
        j = 0
        SEPARATED = False
        while not SEPARATED and j < len(tra_X):
            tmp = tra_y[j] * (w.dot(tra_X[j]) + b)
            if tmp > 0:
                j += 1
            else:
                SEPARATED = False
                w += lr * tra_y[j] * tra_X[j]
                b += lr * tra_y[j]
                j = 0
        # Validate the model(w,b) in val_data
        # Acquire tra_X and tra_y
        val_X = [i[:2] for i in val_data]
        val_y = [i[-1] for i in val_data]
        # convert list to array
        val_X = np.array(val_X)
        val_y = np.array(val_y)
        # Calculate the score
        score = 0
        for m in range(len(val_X)):
            score += -val_y[m] * (w.dot(val_X[m]) + b)
        # Save the w, b, score
        model.append([w, b, score])
    return model
