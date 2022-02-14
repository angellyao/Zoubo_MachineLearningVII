# =============================================================================
# 决策树多目标问题？
# =============================================================================
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    N = 400
    x = np.random.rand(N) * 8 - 4     # rand(N)：N个[0,1]之间均匀分布的随机样本。[-4,4)
    x = np.random.rand(N) * 4 * np.pi     # [-4,4)
    x.sort()

    """
    y1 = np.sin(x) + 3 + np.random.randn(N) * 0.1
    y2 = np.cos(0.3*x) + np.random.randn(N) * 0.01
    y1 = np.sin(x) + np.random.randn(N) * 0.05
    y2 = np.cos(x) + np.random.randn(N) * 0.1
    """    

    y1 = 16 * np.sin(x) ** 3 + np.random.randn(N)*0.5
    y2 = 13 * np.cos(x) - 5 * np.cos(2*x) - 2 * np.cos(3*x) - np.cos(4*x) + np.random.randn(N)*0.5

    np.set_printoptions(suppress=True)
    y = np.vstack((y1, y2)).T
    print('Data = \n', np.vstack((x, y1, y2)).T)
    x = x.reshape(-1, 1)  # 转置后，得到N个样本，每个样本都是1维的
    
    # 决策树，最大深度为10
    deep = 10
    reg = DecisionTreeRegressor(criterion='mse', max_depth=deep)
    dt = reg.fit(x, y)
    x_test = np.linspace(x.min(), x.max(), num=1000).reshape(-1, 1)
    y_hat = dt.predict(x_test)    
    
    plt.figure(facecolor='w')
    plt.scatter(y[:, 0], y[:, 1], c='r', marker='s', edgecolor='k', s=60, label='真实值', alpha=0.8)
    plt.scatter(y_hat[:, 0], y_hat[:, 1], c='g', marker='o', edgecolor='k', edgecolors='g', s=30, label='预测值', alpha=0.8)
    plt.legend(loc='lower left', fancybox=True, fontsize=12)
    plt.xlabel('$Y_1$', fontsize=12)
    plt.ylabel('$Y_2$', fontsize=12)
    plt.grid(b=True, ls=':', color='#606060')
    plt.title('决策树多输出回归', fontsize=15)
    plt.tight_layout(1)
    plt.show()