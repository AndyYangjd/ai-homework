# -*- coding:utf-8 -*-
"""
实现功能：
1、可视化训练集，测试集
"""

__author__ = 'Andy Yang'

import sys
import numpy as np
from matplotlib import pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D

class data_visual():
    def __init__(self, train_data, test_data):
        # Acquire point coordinate and label
        self.train_point = [i[:2] for i in train_data]
        self.train_label = [i[-1] for i in train_data]
        self.test_point = [i[:2] for i in test_data]
        self.test_label = [i[-1] for i in test_data]

        # Get the train_data index
        self.tra_postive = []
        self.tra_negative = []
        for index, value in enumerate(self.train_label):
            if value == 1:
                self.tra_postive.append(index)
            elif value == -1:
                self.tra_negative.append(index)
            else:
                sys.exit(1)
        # Get the classified point
        self.tra_postive_point = [
            self.train_point[i] for i in self.tra_postive
        ]
        self.tra_negative_point = [
            self.train_point[i] for i in self.tra_negative
        ]
        # Get the coordinate of point
        self.postive_x = [point[0] for point in self.tra_postive_point]
        self.postive_y = [point[1] for point in self.tra_postive_point]
        self.negative_x = [point[0] for point in self.tra_negative_point]
        self.negative_y = [point[1] for point in self.tra_negative_point]
        # Define limit
        self.LIM = 100

    def showTrainData(self):
        plt.figure('Train_Data')
        plt.plot(self.postive_x, self.postive_y, 'r+')
        plt.plot(self.negative_x, self.negative_y, 'b+')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Train_Data')
        plt.xlim(-self.LIM, self.LIM)
        plt.ylim(-self.LIM, self.LIM)
        plt.draw()
        plt.pause(3)
        plt.savefig('train_data.png')
        plt.close()

    def showModel(self, w, b):
        plt.figure('Model')
        plt.plot(self.postive_x, self.postive_y, 'r+')
        plt.plot(self.negative_x, self.negative_y, 'b+')
        if w[-1] == 0:
            if w[0] == 0:
                print('Error in model function')
                sys.exit(1)
            x = -b / w[0]
            ax = np.ones(self.LIM * 2) * x
            ay = np.arange(-self.LIM, self.LIM, 1)
        else:
            ax = np.arange(-self.LIM, self.LIM, 1)
            ay = -(w[0] * ax + b) / w[-1]
        plt.plot(ax, ay, 'y')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Model')
        plt.xlim(-self.LIM, self.LIM)
        plt.ylim(-self.LIM, self.LIM)
        plt.draw()
        plt.pause(3)
        plt.savefig('Model.png')
        plt.close()