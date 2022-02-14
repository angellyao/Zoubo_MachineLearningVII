# =============================================================================
# wine数据集，第1列为类别标签，后面13列数值类属性
# Ridge分类、随机森林分类、梯度提升决策树
# 对比一下，看看参数在设定时有和不同
# =============================================================================

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # 1.读取并分离数据集
    data = pd.read_csv('wine.data', header=None)
    x, y = data.iloc[:, 1:], data[0]
    x = MinMaxScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.7)    
    
    # 2.使用Logistic回归算法
    lr = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=3)
    lr.fit(x_train, y_train.ravel()) # 这一步有FutureWarning
    print('1.Logistic回归：')
    print('参数alpha=%.2f' % lr.alpha_)
    y_train_pred = lr.predict(x_train)
    y_test_pred = lr.predict(x_test)
    print('Logistic回归训练集准确率：', accuracy_score(y_train, y_train_pred))
    print('Logistic回归测试集准确率：', accuracy_score(y_test, y_test_pred))
    
    # 3.使用随机森林做算法
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, oob_score=True)
    rf.fit(x_train, y_train.ravel())
    print('2.随机森林：')
    print('OOB Score=%.5f' % rf.oob_score_)
    y_train_pred = rf.predict(x_train)
    y_test_pred = rf.predict(x_test)
    print('随机森林训练集准确率：', accuracy_score(y_train, y_train_pred))
    print('随机森林测试集准确率：', accuracy_score(y_test, y_test_pred))
    
    # 4.使用GBDT算法
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2)
    gb.fit(x_train, y_train.ravel())
    print('3.GBDT：')
    y_train_pred = gb.predict(x_train)
    y_test_pred = gb.predict(x_test)
    print('GBDT训练集准确率：', accuracy_score(y_train, y_train_pred))
    print('GBDT测试集准确率：', accuracy_score(y_test, y_test_pred))

    y_train[y_train == 3] = 0
    y_test[y_test == 3] = 0
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    params = {'max_depth': 1, 'eta': 0.9, 'objective': 'multi:softmax', 'num_class': 3}
    # params = {'max_depth': 1, 'eta': 0.9, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}
    bst = xgb.train(params, data_train, num_boost_round=5, evals=watch_list)
    y_train_pred = bst.predict(data_train)
    y_test_pred = bst.predict(data_test)
    print('4.XGBoost：')
    print('XGBoost训练集准确率：', accuracy_score(y_train, y_train_pred))
    print('XGBoost测试集准确率：', accuracy_score(y_test, y_test_pred))