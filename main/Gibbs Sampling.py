"""
M-H采样已经可以很好的解决蒙特卡罗方法需要的任意概率分布的样本集的问题。
但是M-H采样有两个缺点：
1）：需要计算接受率，在高维时计算量大。并且由于接受率的原因导致算法收敛时间变长
2）：有些高维数据，特征的条件概率分布好求，但是特征的联合分布不好求
Gibbs采样是M-H算法的一种特殊形式，它能够保证接受率为1，加速了MCMC算法的收敛
但是，Gibbs采样必需要知道条件分布
参考：https://www.cnblogs.com/pinard/p/6645766.html
"""
# !/usr/bin/python
# -*- coding: utf-8 -*-
from mpl_toolkits.mplot3d import Axes3D     # 看似没用，其实是画三维图必备的
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random
import math
import numpy as np
import seaborn as sns
sns.set(context='paper', style='ticks')


sample_source = multivariate_normal(mean=[5, -1], cov=[[1, 0.5], [0.5, 4]])
rho = 0.5       # 皮尔斯相关系数 \rho * \sigma_x * \sigma_y = cov(x, y)


def py_given_x(x, m1, m2, s1, s2):
    """二维正态分布的条件概率，对于数据(x,y)，计算P(y|x)"""
    return random.normalvariate(m2 + rho * s2 / s1 * (x - m1), math.sqrt(1 - rho ** 2) * s2)


def px_given_y(y, m1, m2, s1, s2):
    """二维正态分布的条件概率，对于数据(x,y)，计算P(x|y)"""
    return random.normalvariate(m1 + rho * s1 / s2 * (y - m2), math.sqrt(1 - rho ** 2) * s1)


if __name__ == '__main__':
    mu = [5, -1]
    cov = [[1, 0.5], [0.5, 2]]
    sample_source = multivariate_normal(mean=mu, cov=cov)
    record = list()
    record.append(mu)
    prob = list()
    prob.append(sample_source.pdf(record[0]))

    for t in range(1000):
        x = px_given_y(record[-1][1], *mu, *np.diag(cov))
        y = py_given_x(x, *mu, *np.diag(cov))
        z = sample_source.pdf([x, y])
        record.append([x, y])
        prob.append(z)

    record = np.array(record)
    # prob = np.array(prob).reshape(1, -1)
    num_bins = 50
    plt.figure()
    plt.subplot(121)
    plt.hist(record[:, 0], num_bins, alpha=0.5, density=1, facecolor='red', label='Feature x')
    plt.hist(record[:, 1], num_bins, alpha=0.5, density=1, facecolor='green', label='Feature y')
    plt.legend(loc='best')

    # 绘制三维散点图
    ax = plt.subplot(122, projection='3d')
    for i in range(len(prob)):
        # print(record[i, 0], record[i, 1], prob[i])
        ax.scatter(record[i, 0], record[i, 1], prob[i], color='blue')
    plt.show()