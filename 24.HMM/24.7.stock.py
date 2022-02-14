# =============================================================================
# 
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    pd.set_option('display.width', 500)
    data = pd.read_csv('SZ000725.txt', skiprows=1, header=0, sep='\t', encoding='GBK')
    data = data.iloc[:-1, :]
    columns = u'日期,开盘,最高,最低,收盘,成交量,成交额'.split(',')
    data.rename(columns=dict(zip(data.columns, columns)), inplace=True)
    index = data[u'收盘'].diff() / data[u'收盘']
    print(index)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.hist(index, bins=np.arange(-0.11, 0.11, 0.01), range=(index.min(), index.max()), color='g', alpha=0.8, edgecolor='k')
    plt.grid(True)
    plt.xlabel(u'涨跌幅', fontsize=14)
    plt.ylabel(u'频率', fontsize=14)
    plt.title(u'股票涨跌幅与发生频率关系直方图', fontsize=18)
    plt.show()
    condition = np.logical_and(index > 0.02, index < 0.05)
    # condition = index > 0.07
    data_ = data[condition]
    d = data_.sort_values(by=u'成交量')
    d = d[-100:]
    d.sort_index(inplace=True)
    index = list(d.index)
    print(d)
    n = len(data[u'收盘'])
    x = np.zeros(n+1)
    x[d.index] = 1
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(data[u'收盘'], 'r-', lw=2, label=u'股票价格')
    plt.plot(data_[u'收盘'], 'go', label=u'建议购买点', markeredgecolor='k')
    plt.xlabel(u'时间', fontsize=14)
    plt.ylabel(u'股票收盘价', fontsize=14)
    plt.grid(True)
    plt.title(u'股票购买点探索', fontsize=18)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
