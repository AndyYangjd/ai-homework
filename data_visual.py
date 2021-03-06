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
        self.tra_postive_x = [point[0] for point in self.tra_postive_point]
        self.tra_postive_y = [point[1] for point in self.tra_postive_point]
        self.tra_negative_x = [point[0] for point in self.tra_negative_point]
        self.tra_negative_y = [point[1] for point in self.tra_negative_point]

        # Get the test_data index
        self.test_postive = []
        self.test_negative = []
        for index, value in enumerate(self.test_label):
            if value == 1:
                self.test_postive.append(index)
            elif value == -1:
                self.test_negative.append(index)
            else:
                sys.exit(1)
        # Get the classified point
        self.test_postive_point = [
            self.test_point[i] for i in self.test_postive
        ]
        self.test_negative_point = [
            self.test_point[i] for i in self.test_negative
        ]
        # Get the coordinate of point
        self.test_postive_x = [point[0] for point in self.test_postive_point]
        self.test_postive_y = [point[1] for point in self.test_postive_point]
        self.test_negative_x = [point[0] for point in self.test_negative_point]
        self.test_negative_y = [point[1] for point in self.test_negative_point]

        # Define limit
        self.LIM = 100

    def showTrainData(self):
        plt.figure('Train_Data',figsize=(100,100))
        plt.plot(
            self.tra_postive_x,
            self.tra_postive_y,
            'r+',
            label='Postive-Train-Points')
        plt.plot(
            self.tra_negative_x,
            self.tra_negative_y,
            'b+',
            label='Negative-Train-Points')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Train_Data')
        plt.xlim(-self.LIM, self.LIM)
        plt.ylim(-self.LIM, self.LIM)
        plt.legend(
            loc='upper right', bbox_to_anchor=(1, 1.08), borderaxespad=0)
        plt.draw()
        plt.pause(3)
        plt.savefig('train_data.png',dpi=100)
        plt.close()

    def showModel(self, w, b):
        plt.figure('Model',figsize=(100,100))
        plt.plot(
            self.tra_postive_x,
            self.tra_postive_y,
            'r+',
            label='Postive-Train-Points')
        plt.plot(
            self.tra_negative_x,
            self.tra_negative_y,
            'b+',
            label='Negative-Train-Points')
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
        plt.plot(ax, ay, 'y', label='Classification-Line ')
        plt.plot(
            self.test_postive_x,
            self.test_postive_y,
            'ro',
            label='Postive-Test-Points')
        plt.plot(
            self.test_negative_x,
            self.test_negative_y,
            'bo',
            label='Negative-Test-Points')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Model')
        plt.xlim(-self.LIM, self.LIM)
        plt.ylim(-self.LIM, self.LIM)
        plt.legend(loc='upper right',bbox_to_anchor=(1,1.15),borderaxespad=0)
        plt.draw()
        plt.pause(3)
        plt.savefig('Model.png',dpi=100)
        plt.close()