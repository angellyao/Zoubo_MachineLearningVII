# =============================================================================
# 决策树回归?
# 使用模拟的数据，做值预测。
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    # 1.模拟数据
    N = 100
    x = np.random.rand(N) * 6 - 3     # [-3,3)
    x.sort()
    y = np.sin(x) + np.random.randn(N) * 0.05
    # print(y)
    x = x.reshape(-1, 1)
    # print(x)
    
    # 2.做决策树回归模型
    # 决策树回归，最大深度为9
    dt = DecisionTreeRegressor(criterion='mse', max_depth=9)
    dt.fit(x, y)
    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    y_hat = dt.predict(x_test)    
    plt.figure(facecolor='w')
    plt.plot(x, y, 'r*', markersize=10, markeredgecolor='k', label='实际值')
    plt.plot(x_test, y_hat, 'g-', linewidth=2, label = '预测值')
    plt.legend(loc = 'upper left', fontsize=12)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(b=True, ls=':', color='#606060')
    plt.title('决策树回归', fontsize=15)
    plt.tight_layout(2)
    plt.show()

    # 3.比较决策树的深度影响：不同深度的树的影响
    depth = [2, 4, 6, 8, 10]
    clr = 'rgbmy'
    dtr = DecisionTreeRegressor(criterion='mse')
    plt.figure(facecolor = 'w')
    plt.plot(x, y, 'ro', ms = 5, mec = 'k', label = '实际值')
    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    
    for d, c in zip(depth, clr):
        dtr.set_params(max_depth=d)
        dtr.fit(x, y)
        y_hat = dtr.predict(x_test)
        plt.plot(x_test, y_hat, '-', color=c, linewidth=2, markeredgecolor='k', label='Depth=%d' % d)
        
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(b=True, ls=':', color='#606060')
    plt.title('决策树回归结果对比', fontsize=15)
    plt.tight_layout(2)
    plt.show()