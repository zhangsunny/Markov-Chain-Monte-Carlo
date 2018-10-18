"""
Metropolis-Hastings 采样算法解决了Metropolis要求变量分布对称性的问题
也可以将Metropolis看作是Metropolis-Hastings的特殊情况，即q_{ij} = q_{ji}
测试Metropolis-Hastings 算法对多变量分布采样
对多变量分布采样有两种方法：BlockWise和ComponentWise
BlockWise: 需要与样本属性数量相同的多变量分布，每次生成一条数据
ComponentWise：每次生成一条数据的一个属性，相较于BlockWise没有前提要求
参考：https://blog.csdn.net/google19890102/article/details/51785156

这个代码有点问题，在于计算alpha时并没有用到q分布，也就是忽略了q_{ij} 和 q_{ji}
"""
# !/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set(context='paper', style='ticks')
zh_font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')


def biv_exp(theta1, theta2):
    lam1 = 0.5
    lam2 = 0.1
    lam = 0.01
    max_val = 8
    y = math.exp(-(lam1 + lam) * theta1 - (lam2 + lam) * theta2 - lam * max_val)
    return y


def block_wise_sampling():
    T = 10000
    theta = np.zeros((T, 2))
    theta_min = 0
    theta_max = 8
    theta[0] = np.random.uniform(theta_min, theta_max, size=(1, 2))

    for t in range(1, T):
        # 直接生成一条数据的所有变量
        theta_hat = np.random.uniform(theta_min, theta_max, size=2)
        alpha = min(1.0, biv_exp(*theta_hat) / biv_exp(*theta[t - 1]))
        u = np.random.uniform(0, 1)
        theta[t] = np.array(theta_hat) if u <= alpha else np.array(theta[t-1])

    return theta


def component_wise_sampling():
    T = 10000
    theta = np.zeros((T, 2))
    theta_min = 0
    theta_max = 8
    theta[0] = np.random.uniform(theta_min, theta_max, size=(1, 2))

    for t in range(1, T):
        for i in range(theta.shape[-1]):
            # 每次只产生一个属性的值
            theta_hat = np.random.uniform(theta_min, theta_max, size=1)
            theta_tmp = np.array(theta[t-1])
            theta_tmp[i] = theta_hat
            # 注意此时计算alpha，分子的参数只改变当前属性的值，其余值不变
            alpha = min(1.0, biv_exp(*theta_tmp) / biv_exp(*theta[t-1]))
            u = np.random.uniform(0, 1)
            theta[t][i] = np.array(theta_hat) if u <= alpha else np.array(theta[t - 1][i])

    return theta


def draw(theta, method_name):
    num_bins = 50
    plt.figure()
    plt.subplot(221)
    plt.plot(theta[:, 0], color='green')
    plt.title('%s采样状态转移图-1' % method_name, fontproperties=zh_font)

    plt.subplot(222)
    plt.hist(theta[:, 0], num_bins, density=1, alpha=0.5, facecolor='red')
    plt.title('%s采样状态分布直方图-1' % method_name, fontproperties=zh_font)

    plt.subplot(223)
    plt.plot(theta[:, 1], color='green')
    plt.title('%s采样状态转移图-2' % method_name, fontproperties=zh_font)

    plt.subplot(224)
    plt.hist(theta[:, 1], num_bins, density=1, alpha=0.5, facecolor='red')
    plt.title('%s采样状态分布直方图-2' % method_name, fontproperties=zh_font)

    plt.tight_layout(True)


if __name__ == '__main__':
    theta_1 = block_wise_sampling()
    draw(theta_1, 'BlockWise')
    theta_2 = component_wise_sampling()
    draw(theta_2, 'ComponentWise')
    plt.show()
