# -*- coding:utf-8 -*-
'''
选择分数最低的模型即为最佳模型
'''
__author__ = 'Andy Yang'
    
def chooseModel(model):
    # choose the min score and index
    score = [i[-1] for i in model]
    index = score.index(min(score))
    # choose the best model
    optimum_model = model[index]
    return optimum_model