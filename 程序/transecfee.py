# -*- coding:utf-8 -*-
'''
@description
  用于输出交易费用之后的结果
'''
import numpy as np
import pandas as pd
import os
import gc
from NWttest import nwttest_1samp
import statsmodels.formula.api as smf

factorlist = os.listdir('../Database/returnseries/12')
rf=pd.read_csv('../Database/RF.csv')#无风险rf
rf12=rf.iloc[12:-1,:]
def trasecfee(feerate):
    '''
    :param feerate: 交易费率
    :return:不输出只打印结果
    '''
    for i in range(len(factorlist)):
        temp = pd.read_csv('../Database/returnseries/12/' + factorlist[i])
        long_short = np.array(temp['long-short'])
        afterLS = list(long_short - feerate*2)
        length = 12
        name = factorlist[i]
        T_value = []
        Mean = []
        p_value = []
        sharpratio = []
        Std = []
        TO = [afterLS]
        for l in TO:
            t_test = nwttest_1samp(l, 0, L=1)
            mean = np.average(l) * 12- rf12.mean().tolist()[0] * 12 / 100
            STD = np.std(l) * np.sqrt(12)
            sharp = (mean) / STD
            T_value.append(t_test.statistic)
            p_value.append(t_test.pvalue)
            Mean.append(mean)
            Std.append(STD)
            sharpratio.append(sharp)
        print(name, 'long-short')
        print('mean', Mean[0] / 12)
        print('t-statistic', '('+str(round(T_value[0],4))+')')
        ff3 = pd.read_csv('../Database/ff3.csv')
        ff5 = pd.read_csv('../Database/ff5.csv')
        A = pd.DataFrame(afterLS, columns=['long-short'])
        M = pd.concat([A], axis=1)
        alpha3 = []
        t3 = []
        t5 = []
        alpha5 = []
        for i in range(1):
            X1 = ff3.iloc[length:, 1:]
            X2 = ff5.iloc[length:, 1:]
            Y = M.iloc[:-2, i]
            Y.index = X1.index
            Y = Y - rf12.RF[:-1] / 100
            used1 = {'X': X1, 'Y': Y}
            reg = smf.ols(formula='Y~1+X', data=used1).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
            t3.append(reg.tvalues[0])
            alpha3.append(reg.params[0] * 12)
            used2 = {'X': X2, 'Y': Y}
            reg = smf.ols(formula='Y~1+X', data=used2).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
            t5.append(reg.tvalues[0])
            alpha5.append(reg.params[0] * 12)
        print('alpha-FF3', alpha3[0]/12)
        print('t-statistic', '('+str(round(t3[0],4))+')')
        print('alpha-FF5', alpha5[0]/12,)
        print('t-statistic','('+str(round(t5[0],4))+')')
        print('sharpe', sharpratio[0])
        gc.collect()
        print('*'*30)#分隔开不同收益序列


def showtrasecfee(transectionfee):
    '''
    :param transectionfee: 交易费用比例
    :return: 无返回直接打印交易费用调整后的结果
    '''
    trasecfee(transectionfee)# 方便调用