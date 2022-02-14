#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from datetime import datetime
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore')
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 这部分是对数据做了些调整，但是没用上
# =============================================================================
# def extend(a, b):
#     return 1.05*a-0.05*b, 1.05*b-0.05*a
# =============================================================================

# 调整时间格式
def date_parser(date):
    # return pd.datetime.strptime(date, '%Y-%m')
    return datetime.strptime(date, '%Y-%m')

if __name__ == '__main__':
    # 1.读取数据    
    pd.set_option('display.width', 100)
    np.set_printoptions(linewidth=100, suppress=True)
    data = pd.read_csv('AirPassengers.csv', header=0, parse_dates=['Month'], date_parser=date_parser, index_col=['Month'])
    data.rename(columns={'#Passengers': 'Passengers'}, inplace=True) # 重命名
    x = data['Passengers'].astype(np.float)
    x = np.log(x)
    
    # 2.构造模型
    show = 'prime'   # 'prime'
    # show = 'ma'   # 'ma'
    # show = 'diff'   # 'diff'
    d = 1
    diff = x - x.shift(periods=d)    # 做差分，阶数为1
    ma = x.rolling(window=12).mean() # 计算12期的平均数
    xma = x - ma                     # 实际值减去12期平均值    
    p = 2
    q = 2
    model = ARIMA(endog=x, order=(p, d, q))     # 自回归函数p,差分d,移动平均数q
    arima = model.fit(disp=-1)                  # disp<0:不输出过程
    prediction = arima.fittedvalues
    # print(type(prediction))
    y = prediction.cumsum() + x[0]   # 预测的结果
    mse = ((x - y)**2).mean()        # 对结果的评价
    rmse = np.sqrt(mse)
    
    # 3.作数据对比图
    plt.figure(facecolor='w')
    plt.figure(figsize=(8, 6))   # 设置图片大小   
    if show == 'diff':
        plt.plot(x, 'r-', lw=2, label='原始数据')
        plt.plot(diff, 'g-', lw=2, label='%d阶差分' % d)
        #plt.plot(prediction, 'r-', lw=2, label=u'预测数据')
        title = '乘客人数变化曲线 - 取对数'
    elif show == 'ma':
        #plt.plot(x, 'r-', lw=2, label=u'原始数据')
        #plt.plot(ma, 'g-', lw=2, label=u'滑动平均数据')
        plt.plot(xma, 'g-', lw=2, label='ln原始数据 - ln滑动平均数据')
        plt.plot(prediction, 'r-', lw=2, label='预测数据')
        title = '滑动平均值与MA预测值'
    else:
        plt.plot(x, 'r-', lw=2, label='原始数据')
        plt.plot(y, 'g-', lw=2, label='预测数据')
        title = '对数乘客人数与预测值(AR=%d, d=%d, MA=%d)：RMSE=%.4f' % (p, d, q, rmse)
    plt.legend(loc='upper left')
    plt.grid(b=True, ls=':')
    plt.title(title, fontsize=16)
    plt.tight_layout(2)
    # plt.savefig('%s.png' % title)
    plt.show()