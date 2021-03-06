# =============================================================================
# 泰坦尼克数据分析
# 整体过程是
# 1.数据预处理
# 2.写出处理后的数据
# 3.做预测
# =============================================================================

import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归
from sklearn.ensemble import RandomForestClassifier # 随机森林分类
from sklearn.metrics import accuracy_score          # 正确率分析
import pandas as pd
import csv


"""
# 显示正确率
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    print('%s正确率：%.3f%%' % (tip, acc_rate))
    return acc_rate
"""

# 数据处理过程
# 读取数据并对训练数据和测试数据做不同的处理
def load_data(file_name, is_train): # 输入的参数为文件、是否训练数据。文件以指定文件夹形式给出
    data = pd.read_csv(file_name)  # 数据文件路径
    pd.set_option('display.width', 200)
    # print('data.describe() = \n', data.describe())

    # 1.性别字段编码
    # data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
    data['Sex'] = pd.Categorical(data['Sex']).codes

    # 2.补齐船票价格缺失值
    if len(data.Fare[data.Fare == 0]) > 0:
        fare = np.zeros(3)
        for f in range(0, 3):
            fare[f] = data[data['Pclass'] == f + 1]['Fare'].dropna().median()
        # print(fare)
        for f in range(0, 3):  # loop 0 to 2
            data.loc[(data.Fare == 0) & (data.Pclass == f + 1), 'Fare'] = fare[f]
    # print('data.describe() = \n', data.describe())
    
    # 3.年龄：使用均值代替缺失值
    # mean_age = data['Age'].dropna().mean()
    # data.loc[(data.Age.isnull()), 'Age'] = mean_age
    if is_train:
        # 年龄：使用随机森林预测年龄缺失值
        print('随机森林预测缺失年龄：--start--')
        data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]   # 年龄不缺失的数据
        age_null = data_for_age.loc[(data.Age.isnull())]
        # print(age_exist)
        x = age_exist.values[:, 1:]
        y = age_exist.values[:, 0]
        rfr = RandomForestRegressor(n_estimators=20)
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.values[:, 1:])
        # print age_hat
        data.loc[(data.Age.isnull()), 'Age'] = age_hat
        # print('随机森林预测缺失年龄：--over--')
    else:
        print('随机森林预测缺失年龄2：--start--')
        data_for_age = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]  # 年龄不缺失的数据
        age_null = data_for_age.loc[(data.Age.isnull())]
        # print(age_exist)
        x = age_exist.values[:, 1:]
        y = age_exist.values[:, 0]
        rfr = RandomForestRegressor(n_estimators=1000)
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.values[:, 1:])
        # print age_hat
        data.loc[(data.Age.isnull()), 'Age'] = age_hat
        print('随机森林预测缺失年龄2：--over--')
    data['Age'] = pd.cut(data['Age'], bins=6, labels=np.arange(6))

    # 4.起始城市
    data.loc[(data.Embarked.isnull()), 'Embarked'] = 'S'  # 保留缺失出发城市
    embarked_data = pd.get_dummies(data.Embarked)
    print('embarked_data = ', embarked_data)
    # # embarked_data = embarked_data.rename(columns={'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown', 'U': 'UnknownCity'})
    embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))
    data = pd.concat([data, embarked_data], axis=1)
    # print(data.describe())
    
    data.to_csv('New_Data.csv')
    x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    # x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = None
    if 'Survived' in data:
        y = data['Survived']

    x = np.array(x)
    y = np.array(y)

    # 思考：这样做，其实发生了什么？
    x = np.tile(x, (5, 1))
    y = np.tile(y, (5, ))
    if is_train:
        return x, y
    return x, data['PassengerId']


# 写出结果
def write_result(c, c_type):
    file_name = 'Titanic.test.csv'
    x, passenger_id = load_data(file_name, False)

    if type == 3:
        x = xgb.DMatrix(x)
    y = c.predict(x)
    y[y > 0.5] = 1
    y[~(y > 0.5)] = 0

    predictions_file = open("Prediction_%d.csv" % c_type, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId", "Survived"])
    open_file_object.writerows(list(zip(passenger_id, y)))
    predictions_file.close()


# 模型过程
if __name__ == "__main__":
    x, y = load_data('Titanic.train.csv', True)
    # print('x = ', x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    
    # 1.Logistic回归
    lr = LogisticRegression(penalty='l2', max_iter=500)  #  max_iter=500是后来加上的，不然会报错。
    lr.fit(x_train, y_train)
    y_hat = lr.predict(x_test)
    lr_acc = accuracy_score(y_test, y_hat)
    # write_result(lr, 1)
    
    # 2.随机森林
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(x_train, y_train)
    y_hat = rfc.predict(x_test)
    rfc_acc = accuracy_score(y_test, y_hat)
    # write_result(rfc, 2)

    # 3.XGBoost
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    # param = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}
    param = {'max_depth': 6, 'eta': 0.8, 'objective': 'binary:logistic'}
             # 'subsample': 1, 'alpha': 0, 'lambda': 0, 'min_child_weight': 1}
    bst = xgb.train(param, data_train, num_boost_round=20, evals=watch_list)
    y_hat = bst.predict(data_test)
    # write_result(bst, 3)
    y_hat[y_hat > 0.5] = 1
    y_hat[~(y_hat > 0.5)] = 0
    xgb_acc = accuracy_score(y_test, y_hat)

    print('1.Logistic回归：%.3f%%' % (100*lr_acc))
    print('2.随机森林：%.3f%%' % (100*rfc_acc))
    print('3.XGBoost：%.3f%%' % (100*xgb_acc))