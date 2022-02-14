#!/usr/bin/python
# -*- coding:utf-8 -*-
# 波士顿房价数据集

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
# import sklearn.datasets
from pprint import pprint
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error     # 计算mse
from sklearn.ensemble import RandomForestRegressor # 随机森林
import warnings

mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings(action='ignore')
np.set_printoptions(suppress=True)

# 判断非空
def not_empty(s):
    return s != ''

if __name__ == "__main__":
    
    # 1.读取数据
    # 只是简单的读取，没有把每一行的多个数据做分离
    file_data = pd.read_csv('housing.data', header=None)
    # a = np.array([float(s) for s in str if s != ''])

    
    # 这里不知道做了什么
    data = np.empty((len(file_data), 14)) # 生成file_data行和14列的空矩阵
    for i, d in enumerate(file_data.values):
        d = list(map(float, list(filter(not_empty, d[0].split(' ')))))  
        data[i] = d
    x, y = np.split(data, (13, ), axis=1) # 以第13列作为分隔，前13列分给x，后面的分给y
    # 如下是从sklearn.datasets中读取数据并对属性和标签做分离
    # data = sklearn.datasets.load_boston()
    # x = np.array(data.data)
    # y = np.array(data.target)
  

    print('样本个数：%d, 特征个数：%d' % x.shape)
    print(y.shape)
    y = y.ravel()
    
    
    
    # 2.分离数据集做模型
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
    model = Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures(degree=3, include_bias=True)),
        ('linear', ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.99, 1], alphas=np.logspace(-3, 2, 5),
                                fit_intercept=False, max_iter=1e3, cv=3))
    ])
    # model = RandomForestRegressor(n_estimators=50, criterion='mse')
    print('开始建模...')
    model.fit(x_train, y_train)
    # linear = model.get_params('linear')['linear']
    # print(u'超参数：', linear.alpha_)
    # print(u'L1 ratio：', linear.l1_ratio_)
    # print(u'系数：', linear.coef_.ravel())
    
    # 模型验证
    order = y_test.argsort(axis=0)
    y_test = y_test[order]
    x_test = x_test[order, :]
    y_pred = model.predict(x_test)
    r2 = model.score(x_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print('R2:', r2)
    print('均方误差：', mse)
    
    # 模型预测
    t = np.arange(len(y_pred))
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', lw=2, label='真实值')
    plt.plot(t, y_pred, 'g-', lw=2, label='估计值')
    plt.legend(loc='best')
    plt.title('波士顿房价预测', fontsize=18)
    plt.xlabel('样本编号', fontsize=15)
    plt.ylabel('房屋价格', fontsize=15)
    plt.grid()
    plt.show()