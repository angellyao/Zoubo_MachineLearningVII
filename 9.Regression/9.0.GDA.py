#!/usr/bin/python
# -*- coding:utf-8 -*-

# 梯度下降的简单示例
# 设置梯度变化，梯度是cur**2 - a，学习率是learning_rate
import math

if __name__ == "__main__":
    learning_rate = 0.01  # 学习率0.01
    for a in range(1,100):  # 计算1-99的所有数据的平方根
        cur = 0
        for i in range(1000): # 999次迭代
            cur -= learning_rate*(cur**2 - a)
            # cur  = cur -learning_rate*(cur**2 - a)
        print(' %d的平方根(近似)为：%.8f，真实值是：%.8f' % (a, cur, math.sqrt(a))) # 取8位小数
