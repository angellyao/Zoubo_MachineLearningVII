# =============================================================================
# EM算法简单引入
# 算法介绍：文档：各种算法的简单解释.note
# http://note.youdao.com/noteshare?id=9e04e2bf851974a475a6be3c7d954d8b&sub=19734443A3F440AFBDB4E81490605EDF
# 最大化期望模型
# =============================================================================

import numpy as np
from scipy.stats import multivariate_normal # 多元正态分布
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':    
    # 1.按照一定的分布，生成随机数
    # style = 'myself'
    style = 'sklearn'
    np.random.seed(0)
    # 正例属性
    mu1_fact = (0, 0, 0)
    cov1_fact = np.diag((1, 2, 3)) # 主对角小元素为(1,2,3)
    data1 = np.random.multivariate_normal(mu1_fact, cov1_fact, 400) # 400行3列
    # 反例属性
    mu2_fact = (2, 2, 1)
    cov2_fact = np.array(((1, 1, 3), (1, 2, 1), (0, 0, 1)))
    data2 = np.random.multivariate_normal(mu2_fact, cov2_fact, 100)
    # 数据汇总
    data = np.vstack((data1, data2))
    y = np.array([True] * 400 + [False] * 100) # 400个正例，100个反例
    
    # 2.构造模型
    if style == 'sklearn':
        # n_components混合高斯模型个数。默认1
        # covariance_type协方差类型。默认full
        # tol为EM迭代停止阈值。默认1e-3
        # max_iter最大迭代次数。默认100
        g = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000)
        g.fit(data)
        print('类别概率:\t', g.weights_[0])
        print('均值:\n', g.means_, '\n')
        print('方差:\n', g.covariances_, '\n')
        mu1, mu2 = g.means_
        sigma1, sigma2 = g.covariances_
    else:
        num_iter = 100
        n, d = data.shape
        # 随机指定
        # mu1 = np.random.standard_normal(d)
        # print mu1
        # mu2 = np.random.standard_normal(d)
        # print mu2
        mu1 = data.min(axis=0)
        mu2 = data.max(axis=0)
        sigma1 = np.identity(d)
        sigma2 = np.identity(d)
        pi = 0.5
        # EM
        for i in range(num_iter):
            # E Step
            norm1 = multivariate_normal(mu1, sigma1)
            norm2 = multivariate_normal(mu2, sigma2)
            tau1 = pi * norm1.pdf(data)
            tau2 = (1 - pi) * norm2.pdf(data)
            gamma = tau1 / (tau1 + tau2)

            # M Step
            mu1 = np.dot(gamma, data) / np.sum(gamma)
            mu2 = np.dot((1 - gamma), data) / np.sum((1 - gamma))
            sigma1 = np.dot(gamma * (data - mu1).T, data - mu1) / np.sum(gamma)
            sigma2 = np.dot((1 - gamma) * (data - mu2).T, data - mu2) / np.sum(1 - gamma)
            pi = np.sum(gamma) / n
            print(i, ":\t", mu1, mu2)
        print('类别概率:\t', pi)
        print('均值:\t', mu1, mu2)
        print('方差:\n', sigma1, '\n\n', sigma2, '\n')

    # 3.预测分类
    norm1 = multivariate_normal(mu1, sigma1)
    norm2 = multivariate_normal(mu2, sigma2)
    tau1 = norm1.pdf(data)
    tau2 = norm2.pdf(data)
    
    # 4.绘制结果图
    fig = plt.figure(figsize=(10, 5), facecolor='w')
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', s=30, marker='o', edgecolors='k', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('原始数据', fontsize=15)
    ax = fig.add_subplot(122, projection='3d')
    order = pairwise_distances_argmin([mu1_fact, mu2_fact], [mu1, mu2], metric='euclidean')
    print(order)
    if order[0] == 0:
        c1 = tau1 > tau2
    else:
        c1 = tau1 < tau2
    c2 = ~c1
    acc = np.mean(y == c1)
    print('准确率：%.2f%%' % (100*acc))
    ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c='r', s=30, marker='o', edgecolors='k', depthshade=True)
    ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c='g', s=30, marker='^', edgecolors='k', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('EM算法分类', fontsize=15)
    plt.suptitle('EM算法的实现', fontsize=18)
    plt.subplots_adjust(top=0.90)
    plt.tight_layout()
    plt.show()