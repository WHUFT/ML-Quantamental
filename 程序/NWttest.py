# -*- coding:utf-8 -*-
'''
@description:
NW-t检验所用包
'''

import numpy as np
from collections import namedtuple
from scipy.stats import distributions


def _ttest_finish(df, t):
    #from scipy.stats
    '''

    :param df:自由度
    :param t: t值
    :return: 输出t和对应p值
    '''
    prob = distributions.t.sf(np.abs(t), df) * 2  # use np.abs to get upper tail
    if t.ndim == 0:
        t = t[()]
    return t, prob

NWt_1sampleResult = namedtuple('NWT_1sampResult', ('statistic', 'pvalue'))
def nwttest_1samp(a, popmean, axis=0,L=1):
    '''
    主函数
    :param a: 数据列表
    :param popmean: 原假设值u0
    :param axis: 行还是列，默认行
    :param L: lag， 滞后多少，默认1
    :return: 输出nw-t和对应p值
    '''
    a = np.array(a)
    N = len(a)
    df = N-1
    e = a - np.mean(a)
    residuals = np.sum(e**2)
    Q = 0
    for i in range(L):
        w_l = 1 - (i+1)/(1+L)
        for j in range(1,N):
            Q += w_l*e[j]*e[j-(i+1)]
    S = residuals + 2*Q
    nw_var = S/N
    d = np.mean(a,axis) - popmean
    nw_sd = np.sqrt(nw_var / float(df))
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.divide(d, nw_sd)
    t,prob = _ttest_finish(df,t)

    return NWt_1sampleResult(t,prob)
