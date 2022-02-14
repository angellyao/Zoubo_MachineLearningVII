# 决策树与随机森林解决分类问题
# 决策树作为一种基础模型，随机森林作为一个集成模型。
# xgboost
# 对基础的分类器做加权，得到更好的分类器
# AdaBoost，对错误分类的样本权重加大，对正确分类的样本权重减小

# =============================================================================
# agaricus_train、agaricus_test有1个类别标签，剩下的都是one-hot编码的数据
# https://blog.csdn.net/qq_37960402/article/details/88638082
# =============================================================================

import xgboost as xgb  # 分布式梯度增强树
import numpy as np
# from sklearn.tree import DecisionTreeClassifier

# =============================================================================
# # 1、xgBoost的基本使用
# # 2、自定义损失函数的梯度和二阶导
# # 3、binary:logistic/logitraw
# =============================================================================

# 定义f: theta * x
def g_h(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0-p)
    return g, h

# 计算错误率
def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)

if __name__ == "__main__":
    # 1.读取数据
    data_train = xgb.DMatrix('agaricus_train.txt')
    data_test = xgb.DMatrix('agaricus_test.txt')
    # print(data_train) # 没有显式地打印数据

    # 2.设置参数
    # ‘max_depth’：树的深度，默认值是6
    # ‘eta’：步长用于防止过拟合，范围是0～1，默认值0.3
    # ‘objective’：指定你想要的类型的学习者,包括线性回归、逻辑回归、泊松回归等，默认值设置为reg:linear
    param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic'} # logitraw
    # param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'} # logitraw
    # param = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'reg:logistic'}
    watchlist = [(data_test, 'test'), (data_train, 'train')] # 定义参数，用于参看模型的状态
    n_round = 7
    # bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    # 使用g_h(y_hat, y)函数；error_rate(y_hat, y)函数
    
    # 3.使用训练数据做模型
    # obj定制的目标函数。
    # 下面的命令会默认打印一些训练结果
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=g_h, feval=error_rate)

    # 4.使用训练的模型做预测，并计算正确率
    y_hat = bst.predict(data_test) # 预测值
    y = data_test.get_label()
    
    error = sum(y != (y_hat > 0.5))
    error_rate = float(error) / len(y_hat)
    print('样本总数：\t', len(y_hat))
    print('错误数目：\t%4d' % error)
    print('错误率：\t%.5f%%' % (100*error_rate))