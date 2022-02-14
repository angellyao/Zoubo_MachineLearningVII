#!/usr/bin/python
# -*- coding:utf-8 -*-

# import numpy as np
import pandas as pd
# from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression # 逻辑回归
from sklearn.preprocessing import StandardScaler, PolynomialFeatures # 标准化与特征选择
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
# import matplotlib as mpl
import os
from sklearn.externals import joblib # Joblib是一组在Python中提供轻量级管道的工具

if __name__ == "__main__":
    data = pd.read_csv('iris.data', header=None)
    x = data[[0, 1]]
    y = pd.Categorical(data[4]).codes

    if os.path.exists('iris.model'):
        print('Load Model...')
        lr = joblib.load('iris.model')  # 加载构造的模型并赋值给lr
    else:
        print('Train Model...')       
        # 构造pipeline
        lr = Pipeline([('sc',   StandardScaler()), 
                       ('poly', PolynomialFeatures(degree=3)),
                       ('clf',  LogisticRegression())])
        lr.fit(x, y.ravel())
    y_hat = lr.predict(x)
    joblib.dump(lr, 'iris.model')   # 保存
    print('y_hat = \n', y_hat)
    print('accuracy = %.3f%%' % (100*accuracy_score(y, y_hat)))