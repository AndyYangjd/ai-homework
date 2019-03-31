# -*- coding:utf-8 -*-
"""
输入：数据集个数
输出：标记过的可二分类的训练集和测试集(数量比例4:1)
设计数据集数据结构：dataSets=[ [],[],...,[] ]，整体是一个列表，其中每个数据元素点也是一个列表。
数据元素点列表结构为1*3,其中最后一个元素代表标记值。
设计算法：
1、随机产生N组2维数据点（代表平面上的一个点）,限定所有数据点的x和y轴范围均在[-100,100]。
   即生成一个N*2的矩阵，其中每个元素的值范围都在[-100,100]。
2、设定模型：人为设定分类线为 x-y=0，法向量为[1,-1]。
3、标记数据点：将第1步产生的数据点根据第2步的模型进行标记正负，正为1,负为-1。
4、将标记过的数据集按照4:1的比例分割为训练集和测试集进行返回。
"""

__author__ = 'Andy Yang'

import numpy as np
import sys

def makeLinearSeparableData(N=250):
    """
    N(int):The total amount of dataSets.
    """
    if not isinstance(N, int):
        return sys.exit(1)

    if N <= 0:
        print("Please input a positive value of N.")
        return sys.exit(1)

    # Define limit
    LIM = 100

    # Generate random data
    randData = np.random.rand(N, 2) * 2 * LIM - LIM

    # Define the normal vector of line
    normal_vector = np.array([1, -1])

    # Define data list
    data = []
    for i in range(N):
        x = randData[i]
        inner_product = np.inner(x, normal_vector)
        x = x.tolist()
        if inner_product <= 0:
            x.append(-1)
        else:
            x.append(1)
        data.append(x)

    # Segment data
    n_train = int(np.trunc(N * 4 / 5))

    # Save train_data, test_data
    train_data = data[:n_train]
    fileObject = open('train_data.txt', 'w')
    for line in train_data:
        line = str(line)
        fileObject.write(line[1:-1])
        fileObject.write('\n')
    fileObject.close()

    test_data = data[n_train:]
    fileObject = open('test_data.txt', 'w')
    for line in test_data:
        line = str()
        fileObject.write(line[1:-1])
        fileObject.write('\n')
    fileObject.close()

    # Return train_data, test_data
    return train_data, test_data
