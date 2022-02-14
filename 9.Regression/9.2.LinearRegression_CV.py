#!/usr/bin/python
# -*- coding:utf-8 -*-
# =============================================================================
# 不知道怎么看数据的回归系数项和回归截距项
# 线性回归，超参数调整
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split # 分离数据集
from sklearn.linear_model import Ridge # L2正则化Ridge
from sklearn.linear_model import Lasso # L1正则化Lasso
from sklearn.model_selection import GridSearchCV # 网格搜索优化参数


if __name__ == "__main__":
    
    # 1.分离属性和变量
    # pandas读入
    data = pd.read_csv('Advertising.csv')    # TV、Radio、Newspaper、Sales
    # print(data)
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']
    # print(x)
    # print(y)
    
    # 2.训练模型获取参数；网格搜索获取最优超参数
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)
    # model = Lasso()
    model = Ridge()
    alpha_can = np.logspace(-3, 2, 10)
    np.set_printoptions(suppress=True)
    print('alpha_can = ', alpha_can)
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    lasso_model.fit(x_train, y_train)
    print('超参数：\n', lasso_model.best_params_)
    
    # 3.测试模型，返回均方误差mse和均方误差的平方根rmse
    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    y_hat = lasso_model.predict(x_test)
    # print(lasso_model.score(x_test, y_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    # print(mse, rmse)
    
    # 4.绘图
    t = np.arange(len(x_test))
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', linewidth=2, label='真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='预测数据')
    plt.title('线性回归预测销量', fontsize=18)
    plt.legend(loc='upper left')
    plt.grid(b=True, ls=':')
    plt.show()