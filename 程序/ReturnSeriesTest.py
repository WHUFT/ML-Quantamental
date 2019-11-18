#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@description:
    在获取各个算法构建多空组合月度收益序列后，对各个收益序列与OLS回归（benchmark）
    和DFN（表现最好的深度算法）序列是否存在显著差异进行NW-T检验
"""

import glob,os
import pandas as pd
from NWttest import nwttest_1samp
import warnings
warnings.filterwarnings('ignore')

#
def returnseriestest(length):
    path = r'..\DataBase\returnseries'+'\\'+str(length)
    file = glob.glob(os.path.join(path, "*.csv"))
    ols = pd.read_csv(path+'\\OLS'+str(length)+'.csv')
    dfn=pd.read_csv(path+'\\DFN'+str(length)+'.csv')
    k = []  # 每个算法一个df
    for i in range(len(file)):
        k.append(pd.read_csv(file[i]))
    #OLS与其他算法区别
    for i in range(len(k)):
        t = []
        t1 = nwttest_1samp(k[i].iloc[:, 1] - ols['long-short'], 0)
        t.append(t1.statistic)
        t2 = nwttest_1samp(k[i].iloc[:, 2] - ols['long'], 0)
        t.append(t2.statistic)
        t3 = nwttest_1samp(k[i].iloc[:, 3] - ols['short'], 0)
        t.append(t3.statistic)
        print('ols-'+file[i][27:-4], t)
    #DFN与其他算法区别
    for i in range(len(k)):
        t = []
        t1 = nwttest_1samp(-k[i].iloc[:, 1] + dfn['long-short'], 0)
        t.append(t1.statistic)
        t2 = nwttest_1samp(-k[i].iloc[:, 2] + dfn['long'], 0)
        t.append(t2.statistic)
        t3 = nwttest_1samp(-k[i].iloc[:, 3] + dfn['short'], 0)
        t.append(t3.statistic)
        print('dfn-'+file[i][27:-4], t)
    return




