# -*- coding:utf-8 -*-
'''
@description
用于筛选特征
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
import gc


def dropimportant(length,CLF,name, factor, timeseries, base =  0.028813732000000005,inpath = 'output'):
    '''
    主函数1，用于非循环神经网络特征筛选
    :param length: 滑动窗口长度
    :param CLF: 模型
    :param name: 筛选用模型名称
    :param timeseries: 数据
    :param factor: 因子名称
    :param base: 多空对应月度收益
    :param inpath:输出路径
    :return:月度收益差
    '''
    meanlist = []
    for j in range(96):
        print(j)
        print(factor[j])
        Long_Short = []
        Long = []
        Short = []
        for i in range(len(timeseries) - (length+1)):

            FINALm = pd.concat(timeseries[i:i + (length+1)], axis=0)
            FINALm = FINALm.fillna(0)
            FINAL_X= FINALm.iloc[:, :-2].copy()
            FINAL_X.drop(columns=factor[j],inplace=True)
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
            gc.collect()
        A = pd.DataFrame(Long_Short, columns=['long-short'])
        B = pd.DataFrame(Long, columns=['long'])
        C = pd.DataFrame(Short, columns=['short'])
        M = pd.concat([A, B, C], axis=1)
        meanlist.append(np.average(Long_Short))
        if (i%10) == 0:
            print(i)
    gc.collect()
    u = base - np.array(meanlist)
    u = pd.DataFrame(u)
    u.to_csv('../' + inpath + '/'+name+'returngap.csv')
    return u

def dropimportant2(length,CLF,name, factor, timeseries,base =  0.028813732000000005,inpath = 'output',a=0,b=96):
    '''
    主函数2 用于循环神经网络特征筛选
    :param length: 滑动窗口大小
    :param CLF: 筛选用模型
    :param name: 模型名称
    :param factor: 因子名称
    :param timeseries: 数据
    :param base: 月度收益
    :param inpath: 输出位置
    :param a: 起始因子
    :param b: 终止因子
    :return: 返回收益差
    '''
    meanlist = []
    for j in range(a,b):
        print(j)
        print(factor[j])
        Long_Short = []
        Long = []
        Short = []
        for i in range(len(timeseries) - length):
            FINALm = pd.concat(timeseries[i:(i + length + 1)], axis=0)
            FINALm[~FINALm['ret'].isin(['null'])] = FINALm[~FINALm['ret'].isin(['null'])].fillna(0)
            FINAL_X = FINALm.iloc[:, :-2]
            FINAL_X.drop(columns=factor[j], inplace=True)
            FINAL_x = FINAL_X
            FINAL_x[~FINALm['ret'].isin(['null'])] = scale(FINAL_X[~FINALm['ret'].isin(['null'])])
            FINAL_x[FINALm['ret'].isin(['null'])] = 0
            FINALm[FINALm['ret'].isin(['null'])] = 0
            x_train = [FINAL_x.iloc[j * 3571:(j + 1) * 3571, :].values for j in range(length)]
            y_train = [FINALm.iloc[j * 3571:(j + 1) * 3571, -1].values for j in range(length)]
            x_test = [FINAL_x.iloc[j * 3571:(j + 1) * 3571, :].values for j in range(1, length + 1)]
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
            FINAL = FINAL[~timeseries[i + length]['ret'].isin(['null'])]
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
        A = pd.DataFrame(Long_Short, columns=['long-short'])
        B = pd.DataFrame(Long, columns=['long'])
        C = pd.DataFrame(Short, columns=['short'])
        M = pd.concat([A, B, C], axis=1)
        meanlist.append(np.average(Long_Short))
        if (i%10) == 0:
            print(i)
    gc.collect()
    u = base - np.array(meanlist)
    u = pd.DataFrame(u)
    u.to_csv('../' + inpath + '/' + name + 'returngap.csv')
    return u

def FCselect(factor, timeseries):
    '''
    用于FC筛选因子
    :param factor:因子名称
    :param timeseries:数据
    :return:
    '''
    length = 12
    base = 0.0228
    name = 'FC'
    inpath = 'output'
    meanlist = []
    for j in range(96):
        print(j)
        print(factor[j])
        Long_Short = []
        Long = []
        Short = []
        for i in range(len(timeseries) - (length + 1)):
            FINALm = pd.concat(timeseries[i:i + (length + 1)], axis=0)
            FINALm = FINALm.fillna(0)
            FINAL_X = FINALm.iloc[:, :-2].copy()
            FINAL_X.drop(columns=factor[j], inplace=True)
            FINAL_x = scale(FINAL_X)
            final = pd.concat(timeseries[i:i + length], axis=0)
            x_train = FINAL_x[:len(final)]
            x_test = FINAL_x[len(final):]
            y_train = final.iloc[:, -1]
            test = timeseries[i + length]
            y_test = test.iloc[:, -1]
            clf = LinearRegression()
            k = []
            for ax in range(95):
                x = x_train[:, ax].reshape(-1, 1)
                clf.fit(x, y_train)
                k.append(clf.coef_[0])
            PREDICTION = []
            for ap in range(len(x_test)):
                test0 = np.array(x_test[ap])
                y = 0
                for ass in range(95):
                    y = y + test0[ass] * k[ass]
                PREDICTION.append(y)
            y_test = test.iloc[:, -1]
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
            gc.collect()
        meanlist.append(np.average(Long_Short))
        if (i % 10) == 0:
            print(i)
    gc.collect()

    u = base - np.array(meanlist)
    u = pd.DataFrame(u)
    u.to_csv('../' + inpath + '/' + name +'returngap.csv')