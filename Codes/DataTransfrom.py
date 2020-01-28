#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@description:
    导入基础数据并进行一些预处理，最后变成每个截面一个Dataframe,列名为[各因子名称+’stock'+'ret']，index代表单只股票
"""
import glob,os
import pandas as pd
import warnings

#*************************1.导入因子数据，无风险利率，股票月度收益数据*****************************#
warnings.filterwarnings('ignore')
def datatransfrom(datapath):
    path=datapath
    file = glob.glob(os.path.join(path, "*.csv"))
    k=[]
    for i in range(len(file)):
        k.append(pd.read_csv(file[i]))
    #股票月度收益
    ret=pd.read_csv('..\DataBase\\final_return.csv')
    #无风险利率
    rf=pd.read_csv('..\DataBase\\RF.csv')
    rf3=rf.iloc[3:-1,:]
    rf12=rf.iloc[12:-1,:]
    rf24=rf.iloc[24:-1,:]
    rf36=rf.iloc[36:-1,:]
    riskfree = [rf3, rf12, rf24, rf36]
    #因子名称
    factor=[]
    for i in range(len(file)):
        factor.append(file[i][20:-4])
    factor.append('stock')


    #*****对原始数据进行预处理，每个截面一个Dataframe,列名为{96因子名称+’stock'+'ret'}，index代表单只股票******#
    timeseries=[]
    for i in range(len(ret.columns)-1):
        kl=pd.concat([k[j].iloc[:,i+1] for j in range(len(file))],axis=1)
        kl['stock'] = ret.iloc[:,0]
        kl.columns = factor
        kl=kl.iloc[:-2,:]
        timeseries.append(kl)
    #删除月度收益不存在的数据条
    for i in range(len(timeseries)):
        timeseries[i]['ret']=ret.iloc[:,i+1]
        timeseries[i]['ret']=timeseries[i]['ret'].fillna('null')
        timeseries[i]=timeseries[i][~timeseries[i]['ret'].isin(['null'])]
    return riskfree,timeseries,factor

## 为LSTM\RNN设计的数据读取函数
def datatransfrom2(datapath, after=False):
    path=datapath
    file = glob.glob(os.path.join(path, "*.csv"))
    k=[]
    for i in range(len(file)):
        k.append(pd.read_csv(file[i]))
    #股票月度收益
    ret=pd.read_csv('..\DataBase\\final_return.csv')
    #因子名称
    factor=[]
    for i in range(len(file)):
        factor.append(file[i][20:-4])
    factor.append('stock')

    #*****对原始数据进行预处理，每个截面一个Dataframe,列名为{96因子名称+’stock'+'ret'}，index代表单只股票******#
    timeseries2=[]
    index = []
    for i in range(len(ret.columns)-1):
        kl=pd.concat([k[j].iloc[:,i+1] for j in range(len(file))],axis=1)
        kl['stock'] = ret.iloc[:,0]
        kl.columns = factor
        if after:# 保证筛选后因子个数为3571个
            kl = kl.iloc[:, :]
        else:
            kl = kl.iloc[:-2,:]
        timeseries2.append(kl)
    # 加入月度收益，令月度收益不存在为null，方便下一步函数处理
    for i in range(len(timeseries2)):
        timeseries2[i]['ret'] = ret.iloc[:, i + 1]
        timeseries2[i]['ret'] = timeseries2[i]['ret'].fillna('null')
        index.append(timeseries2[i]['ret'].isin(['null']))
    return timeseries2, index

