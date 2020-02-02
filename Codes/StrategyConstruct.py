#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@description:
    1.投资组合构建通用函数（output),函数最终输出多空组合月度收益，FF3/5-alpha,sharpe ratio
    2.FC和ensemble无内置算法包，此处单独构建FC和ensemblenn

"""
import glob,os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso,ElasticNet,Ridge
from NWttest import nwttest_1samp
from sklearn.svm import SVR
import warnings
from sklearn.neural_network import MLPRegressor
import gc


#************************3.EN-ANN、FC算法函数******************************#
#python里没有EN-ANN和FC的对应算法包 此处先定义算法计算方式
#EN-ANN
class ensemblenn(object):
    def __init__(self,ensemblenumbers,modeluse = MLPRegressor(solver = 'lbfgs'),pickpercent = 0.5):
        self.ensemblenumbers = ensemblenumbers
        self.modellist = []
        self.score = [0]*ensemblenumbers
        for i in range(self.ensemblenumbers):
            self.modellist.append(modeluse)
        self.pickpercent = pickpercent
    def fit(self,X,Y):
        for i in range(self.ensemblenumbers):
            self.modellist[i].fit(X,Y)
            self.score[i] = self.modellist[i].loss_
    def predict(self,xtest):
        usemodel = np.array(self.modellist)[np.argsort(np.array(self.score))]
        usemodel = usemodel[0:self.ensemblenumbers//2]
        predict = []
        for i in range(self.ensemblenumbers//2):
            predict.append(usemodel[i].predict(xtest))
        return list(np.mean(predict,axis=0))
#FC
def FC(length,rf, timeseries, lenn=96, na='FC'):
    #length为滑动窗口长度：取值{3,12,24,36}
    #na为输出文件名称
    #rf为无风险利率，取值与length对应{rf3,rf12,rf24,rf36}
    Long_Short = []
    Long = []
    Short = []
    for i in range(len(timeseries) - length):
        print(i)
        FINALm = pd.concat(timeseries[i:i + (length +1)], axis=0)
        FINALm = FINALm.fillna(0)
        FINAL_X = FINALm.iloc[:, :-2]
        FINAL_x = scale(FINAL_X)
        final = pd.concat(timeseries[i:i + length], axis=0)
        x_train = FINAL_x[:len(final)]
        x_test = FINAL_x[len(final):]
        y_train = final.iloc[:, -1]
        test = timeseries[i + length]
        clf = LinearRegression()
        k = []
        for i in range(lenn):
            x = x_train[:, i].reshape(-1, 1)
            clf.fit(x, y_train)
            k.append(clf.coef_[0])
        PREDICTION = []
        for i in range(len(x_test)):
            test0 = np.array(x_test[i])
            y = 0
            for j in range(lenn):
                y = y + test0[j] * k[j]
            PREDICTION.append(y)
        y_test = test.iloc[:, -1]
        # 构建投资组合
        r_predict = pd.DataFrame(PREDICTION, columns=['predict'])
        r_ture = pd.DataFrame(y_test)
        r_ture.columns = ['ture']
        r_ture.index = r_predict.index
        FINAL = pd.concat([r_predict, r_ture], axis=1)
        FINAL_sort = FINAL.sort_values(by='predict', axis=0)
        r_final = np.array(FINAL_sort['ture'])
        m = int(len(r_final) * 0.1) + 1
        r_final = r_final.tolist()
        long = r_final[-m:]
        short = r_final[:m]
        r_end = (np.sum(long) - np.sum(short)) / m
        Long_Short.append(r_end)
        Long.append(np.average(long))
        Short.append(np.average(short))
    T_value = []
    Mean = []
    p_value = []
    sharpratio = []
    Std = []
    TO = [Long_Short, Long, Short]
    for l in TO:
        t_test = nwttest_1samp(l, 0)
        mean = np.average(l) * 12
        STD = np.std(l) * np.sqrt(12)
        sharp = (mean - rf.mean().tolist()[0] * 12 / 100) / STD
        T_value.append(t_test.statistic)
        p_value.append(t_test.pvalue)
        Mean.append(mean)
        Std.append(STD)
        sharpratio.append(sharp)
    name = na
    length = length
    print(name, 'long-short', 'long', 'short')
    print('mean', Mean[0] / 12, Mean[1] / 12, Mean[2] / 12)
    print('t-statistic', '(' + str(round(T_value[0], 4)) + ')', '(' + str(round(T_value[1], 4)) + ')',
          '(' + str(round(T_value[2], 4)) + ')')
    A = pd.DataFrame(Long_Short, columns=['long-short'])
    B = pd.DataFrame(Long, columns=['long'])
    C = pd.DataFrame(Short, columns=['short'])
    M = pd.concat([A, B, C], axis=1)
    M.to_csv('..\output\\'+name + '.csv')
    ff3 = pd.read_csv('..\DataBase\\ff3.csv')
    ff5 = pd.read_csv('..\DataBase\\ff5.csv')
    alpha3 = []
    t3 = []
    t5 = []
    alpha5 = []
    for i in range(3):
        X1 = ff3.iloc[length:, 1:]
        X2 = ff5.iloc[length:, 1:]
        Y = M.iloc[:-2, i]
        Y.index = X1.index
        Y = Y - rf.RF[:-1] / 100
        x1 = sm.add_constant(X1)
        reg = sm.OLS(Y, x1).fit()
        t3.append(reg.tvalues[0])
        alpha3.append(reg.params[0] * 12)
        x2 = sm.add_constant(X2)
        reg = sm.OLS(Y, x2).fit()
        t5.append(reg.tvalues[0])
        alpha5.append(reg.params[0] * 12)
    print('alpha-FF3', alpha3[0] / 12, alpha3[1] / 12, alpha3[2] / 12)
    print('t-statistic', '(' + str(round(t3[0], 4)) + ')', '(' + str(round(t3[1], 4)) + ')',
          '(' + str(round(t3[2], 4)) + ')')
    print('alpha-FF5', alpha5[0] / 12, alpha5[1] / 12, alpha5[2] / 12)
    print('t-statistic', '(' + str(round(t5[0], 4)) + ')', '(' + str(round(t5[1], 4)) + ')',
          '(' + str(round(t5[2], 4)) + ')')
    print('sharpe', sharpratio[0], sharpratio[1], sharpratio[2])


#***********************4.投资组合构建主函数****************************************#
def output(length,CLF,name,rf,timeseries):
    #length为滑动窗口长度：取值{3,12,24,36}
    #CLF为预测选取的机器学习模型
    #name为输出文件名称（type:string)
    #rf为无风险利率，取值与length对应{rf3,rf12,rf24,rf36}
    Long_Short = []
    Long = []
    Short = []
    for i in range(len(timeseries) - (length)):
        print(i)
        FINALm = pd.concat(timeseries[i:i + (length+1)], axis=0)
        FINALm = FINALm.fillna(0)#因子缺失值以0填充
        FINAL_X = FINALm.iloc[:, :-2]
        FINAL_x = scale(FINAL_X)
        final = pd.concat(timeseries[i:i + length], axis=0)
        x_train = FINAL_x[:len(final)]
        x_test = FINAL_x[len(final):]
        y_train = final.iloc[:, -1]
        test = timeseries[i + length]
        y_test = test.iloc[:, -1]
        # 基准-linear
        clf = CLF
        clf.fit(x_train, y_train)
        PREDICTION = clf.predict(x_test)
        # 构建投资组合
        prediction = pd.DataFrame(PREDICTION)
        r_predict = pd.DataFrame(PREDICTION, columns=['predict'])
        r_ture = pd.DataFrame(y_test)
        r_ture.columns = ['ture']
        r_ture.index = r_predict.index
        FINAL = pd.concat([r_predict, r_ture], axis=1)
        FINAL_sort = FINAL.sort_values(by='predict', axis=0)
        r_final = np.array(FINAL_sort['ture'])
        m = int(len(r_final) * 0.1) + 1
        r_final = r_final.tolist()
        long = r_final[-m:]
        short = r_final[:m]
        r_end = (np.sum(long) - np.sum(short)) / m
        Long_Short.append(r_end)
        Long.append(np.average(long))
        Short.append(np.average(short))
    T_value = []
    Mean = []
    p_value = []
    sharpratio = []
    Std = []
    TO = [Long_Short, Long, Short]
    for l in TO:
        t_test = nwttest_1samp(l, 0)
        mean = np.average(l) * 12
        STD = np.std(l) * np.sqrt(12)
        sharp = (mean- rf.mean().tolist()[0]*12/100 ) / STD
        T_value.append(t_test.statistic)
        p_value.append(t_test.pvalue)
        Mean.append(mean)
        Std.append(STD)
        sharpratio.append(sharp)
    print(name, 'long-short', 'long', 'short')
    print('mean', Mean[0]/12, Mean[1]/12, Mean[2]/12)
    print('t-statistic', '('+str(round(T_value[0],4))+')','('+str(round(T_value[1],4))+')', '('+str(round(T_value[2],4))+')')
    A = pd.DataFrame(Long_Short, columns=['long-short'])
    B = pd.DataFrame(Long, columns=['long'])
    C = pd.DataFrame(Short, columns=['short'])
    M = pd.concat([A, B, C], axis=1)
    M.to_csv('..\output\\'+name+'.csv')
    ff3 = pd.read_csv('..\DataBase\\ff3.csv')
    ff5 = pd.read_csv('..\DataBase\\ff5.csv')
    alpha3 = []
    t3 = []
    t5 = []
    alpha5 = []
    for i in range(3):
        X1 = ff3.iloc[length:, 1:]
        X2 = ff5.iloc[length:, 1:]
        Y = M.iloc[:-2, i]
        Y.index = X1.index
        Y = Y - rf.RF[:-1] / 100
        x1 = sm.add_constant(X1)
        reg = sm.OLS(Y, x1).fit()
        t3.append(reg.tvalues[0])
        alpha3.append(reg.params[0] * 12)
        x2 = sm.add_constant(X2)
        reg = sm.OLS(Y, x2).fit()
        t5.append(reg.tvalues[0])
        alpha5.append(reg.params[0] * 12)
    print('alpha-FF3', alpha3[0]/12, alpha3[1]/12, alpha3[2]/12)
    print('t-statistic', '('+str(round(t3[0],4))+')','('+str(round(t3[1],4))+')', '('+str(round(t3[2],4))+')')
    print('alpha-FF5', alpha5[0]/12, alpha5[1]/12, alpha5[2]/12)
    print('t-statistic', '('+str(round(t5[0],4))+')','('+str(round(t5[1],4))+')', '('+str(round(t5[2],4))+')')
    print('sharpe', sharpratio[0], sharpratio[1], sharpratio[2])

# 因为LSTM与RNN一个步长内所用数据形状必须一致，设置专用的主函数供使用
def output2(length,CLF,name,rf,timeseries2):
    # length为滑动窗口长度：取值{3,12,24,36}
    # CLF为预测选取的机器学习模型
    # name为输出文件名称（type:string)
    # rf为无风险利率，取值与length对应{rf3,rf12,rf24,rf36}
    Long_Short = []
    Long = []
    Short = []
    for i in range(len(timeseries2) - length):
        FINALm = pd.concat(timeseries2[i:(i + length + 1)], axis=0)
        FINALm[~FINALm['ret'].isin(['null'])] = FINALm[~FINALm['ret'].isin(['null'])].fillna(0)
        FINAL_X = FINALm.iloc[:, :-2]
        FINAL_x = FINAL_X
        FINAL_x[~FINALm['ret'].isin(['null'])] = scale(FINAL_X[~FINALm['ret'].isin(['null'])])
        FINAL_x[FINALm['ret'].isin(['null'])] = 0
        FINALm[FINALm['ret'].isin(['null'])] = 0
        x_train = [FINAL_x.iloc[j * 3571:(j + 1) * 3571, :].values for j in range(length)]
        y_train = [FINALm.iloc[j * 3571:(j + 1) * 3571, -1].values for j in range(length)]
        x_test = np.array([FINAL_x.iloc[j * 3571:(j + 1) * 3571, :].values for j in range(1, length + 1)])
        y_test = list(FINALm.iloc[(length) * 3571:(length + 1) * 3571, -1].values)
        # 基准-linear
        clf = CLF
        clf.fit(x_train, y_train)
        PREDICTION = clf.predict(x_test)
        PREDICTION = [PREDICTION[m][0] for m in range(3571 * (length - 1), 3571 * length)]
        # 构建投资组合
        r_predict = pd.DataFrame(PREDICTION, columns=['predict'])
        r_ture = pd.DataFrame(y_test)
        r_ture.columns = ['ture']
        r_ture.index = r_predict.index
        FINAL = pd.concat([r_predict, r_ture], axis=1)
        FINAL = FINAL[~timeseries2[i + length]['ret'].isin(['null'])]
        FINAL_sort = FINAL.sort_values(by='predict', axis=0)
        r_final = np.array(FINAL_sort['ture'])
        m = int(len(r_final) * 0.1) + 1
        r_final = r_final.tolist()
        long = r_final[-m:]
        short = r_final[:m]
        r_end = (np.sum(long) - np.sum(short)) / m
        Long_Short.append(r_end)
        Long.append(np.average(long))
        Short.append(np.average(short))
        gc.collect()
    T_value = []
    Mean = []
    p_value = []
    sharpratio = []
    Std = []
    TO = [Long_Short, Long, Short]
    for l in TO:
        t_test = nwttest_1samp(l, 0, L=1)
        mean = np.average(l) * 12- rf.mean().tolist()[0] * 12 / 100
        STD = np.std(l) * np.sqrt(12)
        sharp = (mean ) / STD
        T_value.append(t_test.statistic)
        p_value.append(t_test.pvalue)
        Mean.append(mean)
        Std.append(STD)
        sharpratio.append(sharp)
    print(name, 'long-short', 'long', 'short')
    print('mean', Mean[0] / 12 + rf.mean().tolist()[0] / 100, Mean[1] / 12 + rf.mean().tolist()[0] / 100, Mean[2] / 12 + rf.mean().tolist()[0] / 100)
    print('t-statistic', '('+str(round(T_value[0],4))+')', '('+str(round(T_value[1],4))+')', '('+str(round(T_value[2],4))+')')
    A = pd.DataFrame(Long_Short, columns=['long-short'])
    B = pd.DataFrame(Long, columns=['long'])
    C = pd.DataFrame(Short, columns=['short'])
    M = pd.concat([A, B, C], axis=1)
    M.to_csv('..\output\\' + name + '.csv')
    ff3 = pd.read_csv('..\DataBase\\ff3.csv')
    ff5 = pd.read_csv('..\DataBase\\ff5.csv')
    alpha3 = []
    t3 = []
    t5 = []
    alpha5 = []
    for i in range(3):
        X1 = ff3.iloc[length:, 1:]
        X2 = ff5.iloc[length:, 1:]
        Y = M.iloc[:-2, i]
        Y.index = X1.index
        Y = Y - rf.RF[:-1] / 100
        x1 = sm.add_constant(X1)
        reg = sm.OLS(Y, x1).fit()
        t3.append(reg.tvalues[0])
        alpha3.append(reg.params[0] * 12)
        x2 = sm.add_constant(X2)
        reg = sm.OLS(Y, x2).fit()
        t5.append(reg.tvalues[0])
        alpha5.append(reg.params[0] * 12)
    print('alpha-FF3', alpha3[0] / 12, alpha3[1] / 12, alpha3[2] / 12)
    print('t-statistic', '(' + str(round(t3[0], 4)) + ')', '(' + str(round(t3[1], 4)) + ')',
          '(' + str(round(t3[2], 4)) + ')')
    print('alpha-FF5', alpha5[0] / 12, alpha5[1] / 12, alpha5[2] / 12)
    print('t-statistic', '(' + str(round(t5[0], 4)) + ')', '(' + str(round(t5[1], 4)) + ')',
          '(' + str(round(t5[2], 4)) + ')')
    print('sharpe', sharpratio[0], sharpratio[1], sharpratio[2])

# 针对所有机器学习模型集成所设置的主函数
def comboutput(length, clf, name, rf,timeseries2, index):
    Long_Short = []
    Long = []
    Short = []
    for i in range(len(timeseries2) - length):
        print(i)
        # LSTM数据
        FINALm = pd.concat(timeseries2[i:(i + length + 1)], axis=0)
        FINALm[~FINALm['ret'].isin(['null'])] = FINALm[~FINALm['ret'].isin(['null'])].fillna(0)
        FINAL_X = FINALm.iloc[:, :-2]
        FINAL_x = FINAL_X
        FINAL_x[~FINALm['ret'].isin(['null'])] = scale(FINAL_X[~FINALm['ret'].isin(['null'])])
        FINAL_x[FINALm['ret'].isin(['null'])] = 0
        FINALm[FINALm['ret'].isin(['null'])] = 0
        Nx_train = [FINAL_x.iloc[j * 3571:(j + 1) * 3571, :].values for j in range(length)]
        Ny_train = [FINALm.iloc[j * 3571:(j + 1) * 3571, -1].values for j in range(length)]
        Nx_test = [FINAL_x.iloc[j * 3571:(j + 1) * 3571, :].values for j in range(1, length + 1)]
        ## 传统数据
        dl = timeseries2[i:i + (length + 1)]
        p = [dl[j][~index[i+j]] for j in range(len(dl))]
        TTX = pd.concat(p, axis=0)
        TTX = TTX.fillna(0)
        TXX = TTX.iloc[:, :-2]
        TXx = scale(TXX)
        final = pd.concat(p[:length], axis=0)
        Tx_train = TXx[:len(final)]
        Tx_test = TXx[len(final):]
        Ty_train = final.iloc[:, -1]
        test = p[-1]
        Ty_test = test.iloc[:, -1]
        # 基准-linear
        clf = clf
        clf.fit(Tx_train, Ty_train, Nx_train, Ny_train)
        PREDICTION = clf.predict(Tx_test, Nx_test, index[i+length], length)
        r_predict = pd.DataFrame(PREDICTION, columns=['predict'])
        r_ture = pd.DataFrame(Ty_test)
        r_ture.columns = ['ture']
        r_ture.index = r_predict.index
        FINAL = pd.concat([r_predict, r_ture], axis=1)
        FINAL_sort = FINAL.sort_values(by='predict', axis=0)
        r_final = np.array(FINAL_sort['ture'])
        m = int(len(r_final) * 0.1) + 1
        r_final = r_final.tolist()
        long = r_final[-m:]
        short = r_final[:m]
        r_end = (np.sum(long) - np.sum(short)) / m
        Long_Short.append(r_end)
        Long.append(np.average(long))
        Short.append(np.average(short))
        gc.collect()
    T_value = []
    Mean = []
    p_value = []
    sharpratio = []
    Std = []
    TO = [Long_Short, Long, Short]
    for l in TO:
        t_test = nwttest_1samp(l, 0, L=1)
        mean = np.average(l) * 12 - rf.mean().tolist()[0] * 12 / 100
        STD = np.std(l) * np.sqrt(12)
        sharp = (mean) / STD
        T_value.append(t_test.statistic)
        p_value.append(t_test.pvalue)
        Mean.append(mean)
        Std.append(STD)
        sharpratio.append(sharp)
    print(name, 'long-short', 'long', 'short')
    print('mean', Mean[0] / 12, Mean[1] / 12, Mean[2] / 12)
    print('t-statistic', '(' + str(round(T_value[0], 4)) + ')', '(' + str(round(T_value[1], 4)) + ')',
          '(' + str(round(T_value[2], 4)) + ')')
    A = pd.DataFrame(Long_Short, columns=['long-short'])
    B = pd.DataFrame(Long, columns=['long'])
    C = pd.DataFrame(Short, columns=['short'])
    M = pd.concat([A, B, C], axis=1)
    M.to_csv('..\output\\' + name + '.csv')
    ff3 = pd.read_csv('..\DataBase\\ff3.csv')
    ff5 = pd.read_csv('..\DataBase\\ff5.csv')
    alpha3 = []
    t3 = []
    t5 = []
    alpha5 = []
    for i in range(3):
        X1 = ff3.iloc[length:, 1:]
        X2 = ff5.iloc[length:, 1:]
        Y = M.iloc[:-2, i]
        Y.index = X1.index
        Y = Y - rf.RF[:-1] / 100
        x1 = sm.add_constant(X1)
        reg = sm.OLS(Y, x1).fit()
        t3.append(reg.tvalues[0])
        alpha3.append(reg.params[0] * 12)
        x2 = sm.add_constant(X2)
        reg = sm.OLS(Y, x2).fit()
        t5.append(reg.tvalues[0])
        alpha5.append(reg.params[0] * 12)
    print('alpha-FF3', alpha3[0] / 12, alpha3[1] / 12, alpha3[2] / 12)
    print('t-statistic', '(' + str(round(t3[0], 4)) + ')', '(' + str(round(t3[1], 4)) + ')',
          '(' + str(round(t3[2], 4)) + ')')
    print('alpha-FF5', alpha5[0] / 12, alpha5[1] / 12, alpha5[2] / 12)
    print('t-statistic', '(' + str(round(t5[0], 4)) + ')', '(' + str(round(t5[1], 4)) + ')',
          '(' + str(round(t5[2], 4)) + ')')
    print('sharpe', sharpratio[0], sharpratio[1], sharpratio[2])
