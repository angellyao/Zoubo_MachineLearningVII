# =============================================================================
# 
# =============================================================================

import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics.pairwise import pairwise_distances_argmin
import warnings


def expand(a, b):
    d = (b - a) * 0.05
    return a-d, b+d


if __name__ == "__main__":
    warnings.filterwarnings("ignore")   # hmmlearn(0.2.0) < sklearn(0.18)

    # 0日期  1开盘  2最高  3最低  4收盘  5成交量  6成交额
    x = np.loadtxt('SH600000.txt', delimiter='\t', skiprows=2, usecols=(4, 5, 6, 2, 3))
    close_price = x[:, 0]
    volumn = x[:, 1]
    amount = x[:, 2]
    amplitude_price = x[:, 3] - x[:, 4] # 每天的最高价与最低价的差
    diff_price = np.diff(close_price)   # 涨跌值
    volumn = volumn[1:]                 # 成交量
    amount = amount[1:]                 # 成交额
    amplitude_price = amplitude_price[1:]   # 每日振幅
    sample = np.column_stack((diff_price, volumn, amount, amplitude_price))    # 观测值
    n = 5
    model = hmm.GaussianHMM(n_components=n, covariance_type='full')
    model.fit(sample)
    y = model.predict_proba(sample)
    np.set_printoptions(suppress=True, edgeitems=100)
    print('proba:\n', y)

    t = np.arange(len(diff_price))
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10,6.7), facecolor='w')
    plt.subplot(421)
    plt.plot(t, diff_price, 'r-')
    plt.grid(True)
    plt.title('涨跌幅')
    plt.subplot(422)
    plt.plot(t, volumn, 'g-')
    plt.grid(True)
    plt.title('交易量')

    clrs = plt.cm.terrain(np.linspace(0, 0.8, n))
    plt.subplot(423)
    for i, clr in enumerate(clrs):
        plt.plot(t, y[:, i], '-', lw=0.3, color=clr, alpha=0.7)
    plt.title('所有组分')
    plt.grid(True)
    for i, clr in enumerate(clrs):
        axes = plt.subplot(4, 2, i+4)
        plt.plot(t, y[:, i], '-', color=clr)
        print('组分概率：', y[:, i])
        plt.title('组分%d' % (i+1))
        plt.grid(True)
    plt.suptitle('SH600000股票：GaussianHMM分解隐变量', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
