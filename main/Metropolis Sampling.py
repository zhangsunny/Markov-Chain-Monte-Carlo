"""
测试Metropolis算法，它是MCMC算法的一种
该算法要求已知的分布是对称的，即Q_{ij} = Q_{ji}，例如：正态分布、柯西分布、均匀分布
参考：https://blog.csdn.net/google19890102/article/details/51755242
假设目标概率密度函数为：f(\theta) = \frac{1}{\pi(1+\theta^2)}
"""
# !/usr/bin/python
# -*- coding: utf-8 -*-
import random
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import seaborn as sns

sns.set(context='paper', style='ticks')
# 解决matplotlib画图中文乱码问题
zh_font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')


def cauchy(theta):
    """采样的目标概率密度函数，柯西分布"""
    return 1 / (1 + theta**2)


if __name__ == '__main__':
    T = 10000
    sigma = 1
    theta = np.zeros(T)
    theta_min = -30
    theta_max = 30
    theta[0] = random.uniform(theta_min, theta_max)

    for t in range(1, T):
        # 计算下一个可能的状态\theta^*，选择均值为\theta_{t-1},方差为1的正态分布作为Q分布，计算Q_{ij}=Q_{ji}
        theta_hat = norm.rvs(loc=theta[t-1], scale=sigma, size=1, random_state=None)
        # 计算状态转移概率\alpha
        alpha = min(1, cauchy(theta_hat[0])/cauchy(theta[t-1]))

        u = random.uniform(0, 1)
        theta[t] = theta_hat if u <= alpha else theta[t-1]

    plt.figure('Metropolis采样')
    plt.subplot(411)
    plt.plot(theta, color='green')
    plt.title('生成状态转移图', fontproperties=zh_font)
    plt.subplot(412)
    num_bins = 50
    plt.hist(theta, num_bins, density=1, facecolor='red', alpha=0.5)
    plt.title('生成状态分布图', fontproperties=zh_font)
    plt.subplot(413)
    sns.kdeplot(theta, shade=True)
    plt.title('生成状态核密度估计图', fontproperties=zh_font)
    plt.subplot(414)
    plt.title('目标概率密度函数图', fontproperties=zh_font)
    x = np.arange(min(theta), max(theta), 0.01)
    y = cauchy(x)
    plt.plot(x, y)
    plt.tight_layout(True)
    plt.show()
