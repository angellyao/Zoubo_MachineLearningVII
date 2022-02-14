# =============================================================================
# 使用svm，拟合数据，再使用网格搜索做预测
# =============================================================================

import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV    # 0.17 grid_search
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    # 1.模拟数据
    N = 50
    np.random.seed(0)
    x = np.sort(np.random.uniform(0, 6, N), axis=0)
    y = 2*np.sin(x) + 0.1*np.random.randn(N)
    x = x.reshape(-1, 1)
    print('x =\n', x)
    print('y =\n', y)
    
    # 2.构造模型
    model = svm.SVR(kernel='rbf')
    c_can = np.logspace(-2, 2, 10)
    gamma_can = np.logspace(-2, 2, 10)
    svr = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=5)
    svr.fit(x, y)
    print('验证参数：\n', svr.best_params_)

    x_test = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_hat = svr.predict(x_test)
    
    # 3.查看结果
    sp = svr.best_estimator_.support_
    plt.figure(facecolor='w')
    plt.scatter(x[sp], y[sp], s=120, c='r', marker='*', edgecolors='k', label='Support Vectors', zorder=3)
    plt.plot(x_test, y_hat, 'r-', linewidth=2, label='RBF Kernel')
    plt.plot(x, y, 'go', markeredgecolor='k', markersize=5)
    plt.legend(loc='upper right')
    plt.title('SVR', fontsize=16)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, ls=':', color='#A0A0A0')
    plt.show()