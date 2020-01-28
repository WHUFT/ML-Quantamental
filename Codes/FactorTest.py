#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@description:
1.在去掉市值最小的30%股票后分别根据市值（size)和收益价格比（EP）分组后构建MKT/SMB/VMG 3因子
2.单因子10分组10-1/1-10多空组合因子调整收益
3.单因子10分组检验各组因子调整收益
4.各因子与size因子独立双变量分组检验结果
5.各因子与BM因子独立双变量分组检验结果
6.96项因子fama macbeth回归检验结果

"""
#*****1.在去掉市值最小的30%股票后分别根据市值（size)和收益价格比（EP）分组后构建MKT/SMB/VMG 3因子*****#

import pandas as pd
import numpy as np
ret=pd.read_csv('..\DataBase\\final_return.csv')
size=pd.read_csv('..\DataBase\\factor\\01size.csv')
EP=pd.read_csv('..\DataBase\\factor\\32EP.csv')

faceq=pd.DataFrame(columns=['MKT','SMB','VMG'],index=ret.columns[1:])
facvw=pd.DataFrame(columns=['MKT','SMB','VMG'],index=ret.columns[1:])
rf=pd.read_csv('RF.csv')
for i in range(len(ret.columns)-1):
    total=pd.concat([ret.iloc[:,i+1],size.iloc[:,i+1],EP.iloc[:,i+1]],axis=1)
    total.columns=['ret','size','EP']
    total=total.dropna()
    total = total.sort_values(by='size')
    final = total.iloc[int(len(total) * 0.3):, :]#去掉30%小市值
    #MKT
    faceq.iloc[i,0] = final.ret.mean()-rf.RF[i]/100
    final['VW'] = final.apply(lambda x: x['size'] * x['ret'], axis=1)
    facvw.iloc[i,0] = final.VW.sum()/final.iloc[:,1].sum()-rf.RF[i]/100
    #SMB/VMG
    SIZE1=final.iloc[:int(len(final)/2),:]#小
    SIZE1=SIZE1.sort_values(by='EP')
    SV=SIZE1.iloc[int(2*len(SIZE1)/3):,:]
    sveq=SV['ret'].mean()
    svvw=SV.VW.sum()/SV.iloc[:,1].sum()
    SM=SIZE1.iloc[int(1*len(SIZE1)/3):int(2*len(SIZE1)/3),:]
    smeq=SM['ret'].mean()
    smvw=SM.VW.sum()/SM.iloc[:,1].sum()
    SG=SIZE1.iloc[:int(1*len(SIZE1)/3),:]
    sgeq=SG['ret'].mean()
    sgvw=SG.VW.sum()/SG.iloc[:,1].sum()

    SIZE2 = final.iloc[int(len(final) / 2):,:]#大
    SIZE2 = SIZE2.sort_values(by='EP')
    BV=SIZE2.iloc[int(2*len(SIZE2)/3):,:]
    bveq=BV['ret'].mean()
    bvvw=BV.VW.sum()/BV.iloc[:,1].sum()
    BM=SIZE2.iloc[int(1*len(SIZE2)/3):int(2*len(SIZE2)/3),:]
    bmeq=BM['ret'].mean()
    bmvw=BM.VW.sum()/BM.iloc[:,1].sum()
    BG=SIZE2.iloc[:int(1*len(SIZE2)/3),:]
    bgeq=BG['ret'].mean()
    bgvw=BG.VW.sum()/BG.iloc[:,1].sum()
    faceq.iloc[i, 1] = (sveq +smeq +sgeq )/3-(bveq +bmeq +bgeq )/3
    facvw.iloc[i, 1] = (svvw +smvw +sgvw )/3-(bvvw +bmvw +bgvw )/3
    faceq.iloc[i, 2] = (sveq +bveq)/2-(sgeq+bgeq )/2
    facvw.iloc[i, 2] = (svvw +bvvw)/2-(sgvw+bgvw )/2
    print(i)

#*******************2.单因子10分组10-1多空组合因子调整收益**********************8#
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from NWttest import nwttest_1samp
factoreq=pd.read_csv('..\DataBase\\factorEW.csv')
factorvw=pd.read_csv('..\DataBase\\factorVW.csv')
resultew=pd.DataFrame(columns=factoreq.columns[1:],index=['ret','t','exret','t','capmret','t','ff3ret','t','ff30ret','t'] )
resultvw=pd.DataFrame(columns=factoreq.columns[1:],index=['ret','t','exret','t','capmret','t','ff3ret','t','ff30ret','t'] )
rf=pd.read_csv('RF.csv')
for i in range(len(factoreq.columns)-1):
    if factoreq.iloc[:,i+1].mean()>0:
        faceq=factoreq.iloc[:,i+1]
    else:
        faceq = -factoreq.iloc[:, i + 1]
    if factorvw.iloc[:,i+1].mean()>0:
        facvw=factorvw.iloc[:,i+1]
    else:
        facvw = -factorvw.iloc[:, i + 1]
    ####return
    reteq0 = faceq.dropna()
    resultew.iloc[0, i] = reteq0.mean()
    ttest = nwttest_1samp(reteq0, 0)
    resultew.iloc[1, i] = ttest.statistic
    retvw0 = facvw.dropna()
    resultvw.iloc[0, i] = retvw0.mean()
    ttest = nwttest_1samp(retvw0, 0)
    resultvw.iloc[1, i] = ttest.statistic

    ##excess return
    exeq=faceq-rf.RF/100
    exvw=facvw-rf.RF/100
    exeq0=exeq.dropna()
    resultew.iloc[2, i] = exeq0.mean()
    #ttest=stats.ttest_1samp(exeq0,0)
    ttest=nwttest_1samp(exeq0, 0)
    resultew.iloc[3, i]=ttest.statistic
    exvw0 = exvw.dropna()
    resultvw.iloc[2, i] = exvw0.mean()
    # ttest=stats.ttest_1samp(exvw0,0)
    ttest = nwttest_1samp(exvw0, 0)
    resultvw.iloc[3, i] = ttest.statistic
    ##CAPM
    ff3 = pd.read_csv('ff3.csv')
    capm=ff3.iloc[:,1]
    x1 = sm.add_constant(capm)
    X1=pd.concat([exeq[:-2],x1],axis=1)
    X1 = X1.dropna()
    X2 = pd.concat([exvw[:-2], x1], axis=1)
    X2 = X2.dropna()
    regeq = sm.OLS(X1.iloc[:,0],X1.iloc[:,1:]).fit()
    regvw = sm.OLS(X2.iloc[:,0],X2.iloc[:,1:]).fit()
    resultew.iloc[4, i]=regeq.params[0]
    resultew.iloc[5, i]=regeq.tvalues[0]
    resultvw.iloc[4, i] = regvw.params[0]
    resultvw.iloc[5, i] = regvw.tvalues[0]

    x1 = sm.add_constant(ff3.iloc[:,1:])
    X1 = pd.concat([exeq[:-2], x1], axis=1)
    X1 = X1.dropna()
    X2 = pd.concat([exvw[:-2], x1], axis=1)
    X2 = X2.dropna()
    regeq = sm.OLS(X1.iloc[:, 0], X1.iloc[:, 1:]).fit()
    regvw = sm.OLS(X2.iloc[:, 0], X2.iloc[:, 1:]).fit()
    resultew.iloc[6, i] = regeq.params[0]
    resultew.iloc[7, i] = regeq.tvalues[0]
    resultvw.iloc[6, i] = regvw.params[0]
    resultvw.iloc[7, i] = regvw.tvalues[0]

    ff30=pd.read_csv('..\DataBase\\ff30.csv')
    x1 = sm.add_constant(ff30.iloc[:, 1:])
    X1 = pd.concat([exeq, x1], axis=1)
    X1 = X1.dropna()
    X2 = pd.concat([exvw, x1], axis=1)
    X2 = X2.dropna()
    regeq = sm.OLS(np.asarray(X1.iloc[:, 0]), np.asarray(X1.iloc[:, 1:])).fit()
    regvw = sm.OLS(np.asarray(X2.iloc[:, 0]), np.asarray(X2.iloc[:, 1:])).fit()
    resultew.iloc[8, i] = regeq.params[0]
    resultew.iloc[9, i] = regeq.tvalues[0]
    resultvw.iloc[8, i] = regvw.params[0]
    resultvw.iloc[9, i] = regvw.tvalues[0]
    print(i)

#*****************3.单因子10分组检验各组因子调整收益***********************#
import pandas as pd
import numpy as np
import glob,os
import statsmodels.api as sm
from scipy import stats
from NWttest import nwttest_1samp
ret=pd.read_csv('final_return.csv')
size=pd.read_csv('..\DataBase\\factor\\01size.csv')
path = r'..\DataBase\\factor'
file = glob.glob(os.path.join(path, "*.csv"))
rf=pd.read_csv('RF.csv')
k = []  # 每个因子一个表
for i in range(96):
    k.append(pd.read_csv(file[i]))
factor = []#因子名称
for i in range(len(file)):
    factor.append(file[i][29:-4])
for i in range(10):
    resultew=pd.DataFrame (columns=factor,index=['ret','t','exret','t','capmret','t','ff3ret','t','ff30ret','t'])
    resultvw = pd.DataFrame(columns=factor,index=['ret', 't', 'exret', 't', 'capmret', 't', 'ff3ret', 't', 'ff30ret', 't'])
    for j in range(96):
        totalew =[]
        totalvw =[]
        for m in range(len(ret.columns)-1):
            final=pd.concat([size.iloc[:,m+1],k[j].iloc[:,m+1],ret.iloc[:,m+1]],axis=1)
            final.columns=['size','factor','ret']
            final=final.dropna()
            if len(final)==0:
                totalew.append(np.nan)
                totalvw.append(np.nan)
            else:
                final = final.sort_values(by='factor')
                final['VW'] = final.apply(lambda x: x['size'] * x['ret'], axis=1)
                total = final.iloc[int(len(final) / 10) * i:int(len(final) / 10) * (i + 1), :]
                totalew.append(total['ret'].mean())
                totalvw.append(total['VW'].sum() / total['size'].sum())
        faceq = pd.Series(totalew)
        facvw = pd.Series(totalvw)

        ####return
        reteq0 = faceq.dropna()
        resultew.iloc[0, j] = reteq0.mean()
        ttest = nwttest_1samp(reteq0, 0)
        resultew.iloc[1, j] = ttest.statistic
        retvw0 = facvw.dropna()
        resultvw.iloc[0, j] = retvw0.mean()
        ttest = nwttest_1samp(retvw0, 0)
        resultvw.iloc[1, j] = ttest.statistic

        ##excess return
        exeq = faceq - rf.RF / 100
        exvw = facvw - rf.RF / 100
        exeq0 = exeq.dropna()
        resultew.iloc[2, j] = exeq0.mean()
        # ttest=stats.ttest_1samp(exeq0,0)
        ttest = nwttest_1samp(exeq0, 0)
        resultew.iloc[3, j] = ttest.statistic
        exvw0 = exvw.dropna()
        resultvw.iloc[2, j] = exvw0.mean()
        # ttest=stats.ttest_1samp(exvw0,0)
        ttest = nwttest_1samp(exvw0, 0)
        resultvw.iloc[3, j] = ttest.statistic
        ##CAPM-alpha/FF3-alpha/去掉市值最小30%股票后adj-FF3-alpha
        ff3 = pd.read_csv('ff3.csv')
        capm = ff3.iloc[:, 1]
        x1 = sm.add_constant(capm)
        X1 = pd.concat([exeq[:-2], x1], axis=1)
        X1 = X1.dropna()
        X2 = pd.concat([exvw[:-2], x1], axis=1)
        X2 = X2.dropna()
        regeq = sm.OLS(X1.iloc[:, 0], X1.iloc[:, 1:]).fit()
        regvw = sm.OLS(X2.iloc[:, 0], X2.iloc[:, 1:]).fit()
        resultew.iloc[4, j] = regeq.params[0]
        resultew.iloc[5, j] = regeq.tvalues[0]
        resultvw.iloc[4, j] = regvw.params[0]
        resultvw.iloc[5, j] = regvw.tvalues[0]

        x1 = sm.add_constant(ff3.iloc[:, 1:])
        X1 = pd.concat([exeq[:-2], x1], axis=1)
        X1 = X1.dropna()
        X2 = pd.concat([exvw[:-2], x1], axis=1)
        X2 = X2.dropna()
        regeq = sm.OLS(X1.iloc[:, 0], X1.iloc[:, 1:]).fit()
        regvw = sm.OLS(X2.iloc[:, 0], X2.iloc[:, 1:]).fit()
        resultew.iloc[6, j] = regeq.params[0]
        resultew.iloc[7, j] = regeq.tvalues[0]
        resultvw.iloc[6, j] = regvw.params[0]
        resultvw.iloc[7, j] = regvw.tvalues[0]

        ff30 = pd.read_csv('..\DataBase\\ff30.csv')
        x1 = sm.add_constant(ff30.iloc[:, 1:])
        X1 = pd.concat([exeq, x1], axis=1)
        X1 = X1.dropna()
        X2 = pd.concat([exvw, x1], axis=1)
        X2 = X2.dropna()
        regeq = sm.OLS(X1.iloc[:, 0], X1.iloc[:, 1:]).fit()
        regvw = sm.OLS(X2.iloc[:, 0], X2.iloc[:, 1:]).fit()
        resultew.iloc[8, j] = regeq.params[0]
        resultew.iloc[9, j] = regeq.tvalues[0]
        resultvw.iloc[8, j] = regvw.params[0]
        resultvw.iloc[9, j] = regvw.tvalues[0]
    resultew.to_csv('RESLew'+str(i)+'.csv')
    resultvw.to_csv('RESLvw' + str(i) + '.csv')

#**********************4.各因子与size因子独立双变量分组检验结果**********************************#
import pandas as pd
import numpy as np
import glob,os
import statsmodels.api as sm
from NWttest import nwttest_1samp
ret=pd.read_csv('..\DataBase\\final_return.csv')
size=pd.read_csv('..\DataBase\\factor\\01size.csv')
size=size.iloc[:,:-1]
size.columns=ret.columns
path = r'..\DataBase\\factor'
file = glob.glob(os.path.join(path, "*.csv"))
rf=pd.read_csv('..\DataBase\\RF.csv')
k = []  # 每个因子一个表
for i in range(96):
    k.append(pd.read_csv(file[i]))
factor = []#因子名称
for i in range(len(file)):
    factor.append(file[i][29:-4])

def gendata(x1,x2,ret,size,m=5,n=5):
    x1 = x1[~np.isnan(ret)]
    x1=x1[~np.isnan(size)]
    x2 = x2[~np.isnan(ret)]
    x2 = x2[~np.isnan(size)]
    x1sort = x1.apply(lambda x: np.argsort(x), axis=0)
    x2sort = x2.apply(lambda x: np.argsort(x), axis=0)
    datacolumn = []
    for a in range(m):
        for b in range(n):
            datacolumn.append(str(a + 1) + 'X' + str(b + 1))
    dfeq=pd.DataFrame(columns=datacolumn,index=[i for i in range(len(ret.columns)-1)])
    dfvw = pd.DataFrame(columns=datacolumn, index=[i for i in range(len(ret.columns) - 1)])
    for i in range(len(ret.columns)-1):
        truelistx1 = x1sort.iloc[:,i+1 ]
        truelistx2 = x2sort.iloc[:, i+1]
        truelistx1 = truelistx1[~(truelistx1 == -1)].sort_values()
        truelistx1=truelistx1.index
        truelistx2 = truelistx2[~(truelistx2 == -1)].sort_values()
        truelistx2 = truelistx2.index
        if len(truelistx1)<5 or len(truelistx2)<5:
            dfeq.iloc[i,:] =np.nan
            dfvw.iloc[i,:] =np.nan
        else:
            x1lines = np.array_split(np.array(truelistx1), m)
            x2lines = np.array_split(np.array(truelistx2), n)
            for a in range(m):
                x1use = x1lines[a]
                for b in range(n):
                    x2use = x2lines[b]
                    tempindex = np.intersect1d(x1use, x2use)
                    if len(tempindex)==0:
                        dfeq.iloc[i, a * m + b]=np.nan
                        dfvw.iloc[i, a * m + b] = np.nan
                    else:
                        dfeq.iloc[i, a * m + b] = ret.iloc[tempindex, i + 1].mean()
                        total = pd.concat([ret.iloc[tempindex, i + 1], size.iloc[tempindex, i + 1]], axis=1)
                        total.columns = ['ret', 'size']
                        total['vw'] = total.apply(lambda x: x['size'] * x['ret'], axis=1)
                        dfvw.iloc[i, a * m + b] = total['vw'].sum() / total['size'].sum()
    return dfeq,dfvw


RETEQ,RETVQ=[],[]
for i in range(len(k)):
    k[i] = k[i].iloc[:, :264]
    k[i].columns = ret.columns
    reteq, retvw = gendata(size, k[i], ret, size)
    RETEQ.append(reteq)
    RETVQ.append(retvw)

#删除SIZE因子本身（不与自身分组）
del RETEQ[0]
del RETVQ[0]
del factor[0]

for l in range(5):
    longew=pd.concat([i.iloc[:, l*5+4] for i in RETEQ], axis=1)
    shortew = pd.concat([i.iloc[:, l * 5 ] for i in RETEQ], axis=1)
    longvw = pd.concat([i.iloc[:, l * 5 + 4] for i in RETVQ], axis=1)
    shortvw = pd.concat([i.iloc[:, l * 5] for i in RETVQ], axis=1)
    longew.columns = factor
    longvw.columns = factor
    shortew.columns = factor
    shortvw.columns = factor
    lsew=longew-shortew
    lsvw=longvw-shortvw
    resultew = pd.DataFrame(columns=factor,
                            index=['ret', 't', 'exret', 't', 'capmret', 't', 'ff3ret', 't', 'ff30ret', 't'])
    resultvw = pd.DataFrame(columns=factor,
                            index=['ret', 't', 'exret', 't', 'capmret', 't', 'ff3ret', 't', 'ff30ret', 't'])
    for j in range(len(factor)):
        faceq, facvw = lsew.iloc[:, j], lsvw.iloc[:, j]
        reteq0 = faceq.dropna()
        resultew.iloc[0, j] = reteq0.mean()
        ttest = nwttest_1samp(reteq0, 0)
        resultew.iloc[1, j] = ttest.statistic
        retvw0 = facvw.dropna()
        resultvw.iloc[0, j] = retvw0.mean()
        ttest = nwttest_1samp(retvw0, 0)
        resultvw.iloc[1, j] = ttest.statistic
        ##excess return
        exeq = faceq - rf.RF / 100
        exvw = facvw - rf.RF / 100
        exeq0 = exeq.dropna()
        resultew.iloc[2, j] = exeq0.mean()
        # ttest=stats.ttest_1samp(exeq0,0)
        ttest = nwttest_1samp(exeq0, 0)
        resultew.iloc[3, j] = ttest.statistic
        exvw0 = exvw.dropna()
        resultvw.iloc[2, j] = exvw0.mean()
        # ttest=stats.ttest_1samp(exvw0,0)
        ttest = nwttest_1samp(exvw0, 0)
        resultvw.iloc[3, j] = ttest.statistic
        ##CAPM
        ff3 = pd.read_csv('..\DataBase\\ff3.csv')
        capm = ff3.iloc[:, 1]
        x1 = sm.add_constant(capm)
        X1 = pd.concat([exeq[:-2], x1], axis=1)
        X1 = X1.dropna()
        X2 = pd.concat([exvw[:-2], x1], axis=1)
        X2 = X2.dropna()
        regeq = sm.OLS(np.asarray(X1.iloc[:, 0]), np.asarray(X1.iloc[:, 1:])).fit()
        regvw = sm.OLS(np.asarray(X2.iloc[:, 0]), np.asarray(X2.iloc[:, 1:])).fit()
        resultew.iloc[4, j] = regeq.params[0]
        resultew.iloc[5, j] = regeq.tvalues[0]
        resultvw.iloc[4, j] = regvw.params[0]
        resultvw.iloc[5, j] = regvw.tvalues[0]
        x1 = sm.add_constant(ff3.iloc[:, 1:])
        X1 = pd.concat([exeq[:-2], x1], axis=1)
        X1 = X1.dropna()
        X2 = pd.concat([exvw[:-2], x1], axis=1)
        X2 = X2.dropna()
        regeq = sm.OLS(np.asarray(X1.iloc[:, 0]), np.asarray(X1.iloc[:, 1:])).fit()
        regvw = sm.OLS(np.asarray(X2.iloc[:, 0]), np.asarray(X2.iloc[:, 1:])).fit()
        resultew.iloc[6, j] = regeq.params[0]
        resultew.iloc[7, j] = regeq.tvalues[0]
        resultvw.iloc[6, j] = regvw.params[0]
        resultvw.iloc[7, j] = regvw.tvalues[0]
        ff30 = pd.read_csv('..\DataBase\\ff30.csv')
        x1 = sm.add_constant(ff30.iloc[:, 1:])
        X1 = pd.concat([exeq, x1], axis=1)
        X1 = X1.dropna()
        X2 = pd.concat([exvw, x1], axis=1)
        X2 = X2.dropna()
        regeq = sm.OLS(np.asarray(X1.iloc[:, 0]), np.asarray(X1.iloc[:, 1:])).fit()
        regvw = sm.OLS(np.asarray(X2.iloc[:, 0]), np.asarray(X2.iloc[:, 1:])).fit()
        resultew.iloc[8, j] = regeq.params[0]
        resultew.iloc[9, j] = regeq.tvalues[0]
        resultvw.iloc[8, j] = regvw.params[0]
        resultvw.iloc[9, j] = regvw.tvalues[0]
    resultew.to_csv('longshort55SIZEEW' + str(l) + '.csv')
    resultvw.to_csv('longshort55SIZEVW' + str(l) + '.csv')

for u in range(25):
    DATAEQ=pd.concat([i.iloc[:,u] for i in RETEQ],axis=1)
    DATAVQ = pd.concat([i.iloc[:, u] for i in RETVQ],axis=1)
    DATAEQ.columns=factor
    DATAVQ.columns = factor
    resultew = pd.DataFrame(columns=factor,
                                index=['ret', 't', 'exret', 't', 'capmret', 't', 'ff3ret', 't', 'ff30ret', 't'])
    resultvw = pd.DataFrame(columns=factor,
                                index=['ret', 't', 'exret', 't', 'capmret', 't', 'ff3ret', 't', 'ff30ret', 't'])
    for j in range(len(factor)):
        faceq, facvw=DATAEQ.iloc[:,j],DATAVQ.iloc[:,j]
        reteq0 = faceq.dropna()
        resultew.iloc[0, j] = reteq0.mean()
        ttest = nwttest_1samp(reteq0, 0)
        resultew.iloc[1, j] = ttest.statistic
        retvw0 = facvw.dropna()
        resultvw.iloc[0, j] = retvw0.mean()
        ttest = nwttest_1samp(retvw0, 0)
        resultvw.iloc[1, j] = ttest.statistic

        ##excess return
        exeq = faceq - rf.RF / 100
        exvw = facvw - rf.RF / 100
        exeq0 = exeq.dropna()
        resultew.iloc[2, j] = exeq0.mean()
        # ttest=stats.ttest_1samp(exeq0,0)
        ttest = nwttest_1samp(exeq0, 0)
        resultew.iloc[3, j] = ttest.statistic
        exvw0 = exvw.dropna()
        resultvw.iloc[2, j] = exvw0.mean()
        # ttest=stats.ttest_1samp(exvw0,0)
        ttest = nwttest_1samp(exvw0, 0)
        resultvw.iloc[3, j] = ttest.statistic
        ##CAPM
        ff3 = pd.read_csv('..\DataBase\\ff3.csv')
        capm = ff3.iloc[:, 1]
        x1 = sm.add_constant(capm)
        X1 = pd.concat([exeq[:-2], x1], axis=1)
        X1 = X1.dropna()
        X2 = pd.concat([exvw[:-2], x1], axis=1)
        X2 = X2.dropna()
        regeq = sm.OLS(np.asarray(X1.iloc[:, 0]), np.asarray(X1.iloc[:, 1:])).fit()
        regvw = sm.OLS(np.asarray(X2.iloc[:, 0]), np.asarray(X2.iloc[:, 1:])).fit()
        resultew.iloc[4, j] = regeq.params[0]
        resultew.iloc[5, j] = regeq.tvalues[0]
        resultvw.iloc[4, j] = regvw.params[0]
        resultvw.iloc[5, j] = regvw.tvalues[0]

        x1 = sm.add_constant(ff3.iloc[:, 1:])
        X1 = pd.concat([exeq[:-2], x1], axis=1)
        X1 = X1.dropna()
        X2 = pd.concat([exvw[:-2], x1], axis=1)
        X2 = X2.dropna()
        regeq = sm.OLS(np.asarray(X1.iloc[:, 0]), np.asarray(X1.iloc[:, 1:])).fit()
        regvw = sm.OLS(np.asarray(X2.iloc[:, 0]), np.asarray(X2.iloc[:, 1:])).fit()
        resultew.iloc[6, j] = regeq.params[0]
        resultew.iloc[7, j] = regeq.tvalues[0]
        resultvw.iloc[6, j] = regvw.params[0]
        resultvw.iloc[7, j] = regvw.tvalues[0]

        ff30 = pd.read_csv('C:\\Users\\15083\Desktop\\ff30.csv')
        x1 = sm.add_constant(ff30.iloc[:, 1:])
        X1 = pd.concat([exeq, x1], axis=1)
        X1 = X1.dropna()
        X2 = pd.concat([exvw, x1], axis=1)
        X2 = X2.dropna()
        regeq = sm.OLS(np.asarray(X1.iloc[:, 0]), np.asarray(X1.iloc[:, 1:])).fit()
        regvw = sm.OLS(np.asarray(X2.iloc[:, 0]), np.asarray(X2.iloc[:, 1:])).fit()
        resultew.iloc[8, j] = regeq.params[0]
        resultew.iloc[9, j] = regeq.tvalues[0]
        resultvw.iloc[8, j] = regvw.params[0]
        resultvw.iloc[9, j] = regvw.tvalues[0]
    resultew.to_csv('EW55size'+reteq.columns[u]+'.csv')
    resultvw.to_csv('VW55size' + reteq.columns[u] + '.csv')


#**********************5.各因子与BM因子独立双变量分组检验结果**********************************#
import pandas as pd
import numpy as np
import glob, os
import statsmodels.api as sm
from NWttest import nwttest_1samp

ret = pd.read_csv('..\DataBase\\final_return.csv')
size = pd.read_csv('..\DataBase\\factor\\01size.csv')
BM = pd.read_csv('..\DataBase\\factor\\28BM.csv')
size = size.iloc[:, :-1]
size.columns = ret.columns
path = r'..\DataBase\\factor'
file = glob.glob(os.path.join(path, "*.csv"))
rf = pd.read_csv('..\DataBase\\RF.csv')
k = []  # 每个因子一个表
for i in range(96):
    k.append(pd.read_csv(file[i]))
factor = []  # 因子名称
for i in range(len(file)):
    factor.append(file[i][29:-4])

BM = BM.iloc[:, :-1]
BM.columns = ret.columns


def gendata(x1, x2, ret, size, m=5, n=5):
    x1 = x1[~np.isnan(ret)]
    x1 = x1[~np.isnan(size)]
    x2 = x2[~np.isnan(ret)]
    x2 = x2[~np.isnan(size)]
    x1sort = x1.apply(lambda x: np.argsort(x), axis=0)
    x2sort = x2.apply(lambda x: np.argsort(x), axis=0)
    datacolumn = []

    for a in range(m):
        for b in range(n):
            datacolumn.append(str(a + 1) + 'X' + str(b + 1))
    dfeq = pd.DataFrame(columns=datacolumn, index=[i for i in range(len(ret.columns) - 1)])
    dfvw = pd.DataFrame(columns=datacolumn, index=[i for i in range(len(ret.columns) - 1)])
    for i in range(len(ret.columns) - 1):
        truelistx1 = x1sort.iloc[:, i + 1]
        truelistx2 = x2sort.iloc[:, i + 1]
        truelistx1 = truelistx1[~(truelistx1 == -1)].sort_values()
        truelistx1 = truelistx1.index
        truelistx2 = truelistx2[~(truelistx2 == -1)].sort_values()
        truelistx2 = truelistx2.index
        if len(truelistx1) < 5 or len(truelistx2) < 5:
            dfeq.iloc[i, :] = np.nan
            dfvw.iloc[i, :] = np.nan
        else:
            x1lines = np.array_split(np.array(truelistx1), m)
            x2lines = np.array_split(np.array(truelistx2), n)
            for a in range(m):
                x1use = x1lines[a]
                lsew=[]
                lsvw = []
                for b in range(n):
                    x2use = x2lines[b]
                    tempindex = np.intersect1d(x1use, x2use)
                    if len(tempindex) == 0:
                        dfeq.iloc[i, a * m + b] = np.nan
                        dfvw.iloc[i, a * m + b] = np.nan
                        lsew.append(0)
                        lsvw.append(0)
                    else:
                        dfeq.iloc[i, a * m + b] = ret.iloc[tempindex, i + 1].mean()
                        total = pd.concat([ret.iloc[tempindex, i + 1], size.iloc[tempindex, i + 1]], axis=1)
                        total.columns = ['ret', 'size']
                        total['vw'] = total.apply(lambda x: x['size'] * x['ret'], axis=1)
                        dfvw.iloc[i, a * m + b] = total['vw'].sum() / total['size'].sum()
                        lsew.append(ret.iloc[tempindex, i + 1].mean())
                        lsvw.append(total['vw'].sum() / total['size'].sum())
    return dfeq, dfvw


RETEQ, RETVQ = [], []
for i in range(len(k)):
    k[i] = k[i].iloc[:, :264]
    k[i].columns = ret.columns
    reteq, retvw = gendata(BM, k[i], ret, size)
    RETEQ.append(reteq)
    RETVQ.append(retvw)

#删除BM因子本身（不与自身分组）
del RETEQ[27]
del RETVQ[27]
del factor[27]

for l in range(5):
    longew=pd.concat([i.iloc[:, l*5+4] for i in RETEQ], axis=1)
    shortew = pd.concat([i.iloc[:, l * 5 ] for i in RETEQ], axis=1)
    longvw = pd.concat([i.iloc[:, l * 5 + 4] for i in RETVQ], axis=1)
    shortvw = pd.concat([i.iloc[:, l * 5] for i in RETVQ], axis=1)
    longew.columns = factor
    longvw.columns = factor
    shortew.columns = factor
    shortvw.columns = factor
    lsew=longew-shortew
    lsvw=longvw-shortvw
    resultew = pd.DataFrame(columns=factor,
                            index=['ret', 't', 'exret', 't', 'capmret', 't', 'ff3ret', 't', 'ff30ret', 't'])
    resultvw = pd.DataFrame(columns=factor,
                            index=['ret', 't', 'exret', 't', 'capmret', 't', 'ff3ret', 't', 'ff30ret', 't'])
    for j in range(len(factor)):
        faceq, facvw = lsew.iloc[:, j], lsvw.iloc[:, j]
        reteq0 = faceq.dropna()
        resultew.iloc[0, j] = reteq0.mean()
        ttest = nwttest_1samp(reteq0, 0)
        resultew.iloc[1, j] = ttest.statistic
        retvw0 = facvw.dropna()
        resultvw.iloc[0, j] = retvw0.mean()
        ttest = nwttest_1samp(retvw0, 0)
        resultvw.iloc[1, j] = ttest.statistic
        ##excess return
        exeq = faceq - rf.RF / 100
        exvw = facvw - rf.RF / 100
        exeq0 = exeq.dropna()
        resultew.iloc[2, j] = exeq0.mean()
        # ttest=stats.ttest_1samp(exeq0,0)
        ttest = nwttest_1samp(exeq0, 0)
        resultew.iloc[3, j] = ttest.statistic
        exvw0 = exvw.dropna()
        resultvw.iloc[2, j] = exvw0.mean()
        # ttest=stats.ttest_1samp(exvw0,0)
        ttest = nwttest_1samp(exvw0, 0)
        resultvw.iloc[3, j] = ttest.statistic
        ##CAPM
        ff3 = pd.read_csv('..\DataBase\\ff3.csv')
        capm = ff3.iloc[:, 1]
        x1 = sm.add_constant(capm)
        X1 = pd.concat([exeq[:-2], x1], axis=1)
        X1 = X1.dropna()
        X2 = pd.concat([exvw[:-2], x1], axis=1)
        X2 = X2.dropna()
        regeq = sm.OLS(np.asarray(X1.iloc[:, 0]), np.asarray(X1.iloc[:, 1:])).fit()
        regvw = sm.OLS(np.asarray(X2.iloc[:, 0]), np.asarray(X2.iloc[:, 1:])).fit()
        resultew.iloc[4, j] = regeq.params[0]
        resultew.iloc[5, j] = regeq.tvalues[0]
        resultvw.iloc[4, j] = regvw.params[0]
        resultvw.iloc[5, j] = regvw.tvalues[0]
        x1 = sm.add_constant(ff3.iloc[:, 1:])
        X1 = pd.concat([exeq[:-2], x1], axis=1)
        X1 = X1.dropna()
        X2 = pd.concat([exvw[:-2], x1], axis=1)
        X2 = X2.dropna()
        regeq = sm.OLS(np.asarray(X1.iloc[:, 0]), np.asarray(X1.iloc[:, 1:])).fit()
        regvw = sm.OLS(np.asarray(X2.iloc[:, 0]), np.asarray(X2.iloc[:, 1:])).fit()
        resultew.iloc[6, j] = regeq.params[0]
        resultew.iloc[7, j] = regeq.tvalues[0]
        resultvw.iloc[6, j] = regvw.params[0]
        resultvw.iloc[7, j] = regvw.tvalues[0]
        ff30 = pd.read_csv('..\DataBase\\ff30.csv')
        x1 = sm.add_constant(ff30.iloc[:, 1:])
        X1 = pd.concat([exeq, x1], axis=1)
        X1 = X1.dropna()
        X2 = pd.concat([exvw, x1], axis=1)
        X2 = X2.dropna()
        regeq = sm.OLS(np.asarray(X1.iloc[:, 0]), np.asarray(X1.iloc[:, 1:])).fit()
        regvw = sm.OLS(np.asarray(X2.iloc[:, 0]), np.asarray(X2.iloc[:, 1:])).fit()
        resultew.iloc[8, j] = regeq.params[0]
        resultew.iloc[9, j] = regeq.tvalues[0]
        resultvw.iloc[8, j] = regvw.params[0]
        resultvw.iloc[9, j] = regvw.tvalues[0]
    resultew.to_csv('longshort55BMEW' + str(l) + '.csv')
    resultvw.to_csv('longshort55BMVW' + str(l) + '.csv')

for u in range(25):
    DATAEQ = pd.concat([i.iloc[:, u] for i in RETEQ], axis=1)
    DATAVQ = pd.concat([i.iloc[:, u] for i in RETVQ], axis=1)
    DATAEQ.columns = factor
    DATAVQ.columns = factor
    resultew = pd.DataFrame(columns=factor,
                            index=['ret', 't', 'exret', 't', 'capmret', 't', 'ff3ret', 't', 'ff30ret', 't'])
    resultvw = pd.DataFrame(columns=factor,
                            index=['ret', 't', 'exret', 't', 'capmret', 't', 'ff3ret', 't', 'ff30ret', 't'])
    for j in range(len(factor)):
        faceq, facvw = DATAEQ.iloc[:, j], DATAVQ.iloc[:, j]
        reteq0 = faceq.dropna()
        resultew.iloc[0, j] = reteq0.mean()
        ttest = nwttest_1samp(reteq0, 0)
        resultew.iloc[1, j] = ttest.statistic
        retvw0 = facvw.dropna()
        resultvw.iloc[0, j] = retvw0.mean()
        ttest = nwttest_1samp(retvw0, 0)
        resultvw.iloc[1, j] = ttest.statistic
        ##excess return
        exeq = faceq - rf.RF / 100
        exvw = facvw - rf.RF / 100
        exeq0 = exeq.dropna()
        resultew.iloc[2, j] = exeq0.mean()
        # ttest=stats.ttest_1samp(exeq0,0)
        ttest = nwttest_1samp(exeq0, 0)
        resultew.iloc[3, j] = ttest.statistic
        exvw0 = exvw.dropna()
        resultvw.iloc[2, j] = exvw0.mean()
        # ttest=stats.ttest_1samp(exvw0,0)
        ttest = nwttest_1samp(exvw0, 0)
        resultvw.iloc[3, j] = ttest.statistic
        ##CAPM
        ff3 = pd.read_csv('..\DataBase\\ff3.csv')
        capm = ff3.iloc[:, 1]
        x1 = sm.add_constant(capm)
        X1 = pd.concat([exeq[:-2], x1], axis=1)
        X1 = X1.dropna()
        X2 = pd.concat([exvw[:-2], x1], axis=1)
        X2 = X2.dropna()
        regeq = sm.OLS(np.asarray(X1.iloc[:, 0]), np.asarray(X1.iloc[:, 1:])).fit()
        regvw = sm.OLS(np.asarray(X2.iloc[:, 0]), np.asarray(X2.iloc[:, 1:])).fit()
        resultew.iloc[4, j] = regeq.params[0]
        resultew.iloc[5, j] = regeq.tvalues[0]
        resultvw.iloc[4, j] = regvw.params[0]
        resultvw.iloc[5, j] = regvw.tvalues[0]
        x1 = sm.add_constant(ff3.iloc[:, 1:])
        X1 = pd.concat([exeq[:-2], x1], axis=1)
        X1 = X1.dropna()
        X2 = pd.concat([exvw[:-2], x1], axis=1)
        X2 = X2.dropna()
        regeq = sm.OLS(np.asarray(X1.iloc[:, 0]), np.asarray(X1.iloc[:, 1:])).fit()
        regvw = sm.OLS(np.asarray(X2.iloc[:, 0]), np.asarray(X2.iloc[:, 1:])).fit()
        resultew.iloc[6, j] = regeq.params[0]
        resultew.iloc[7, j] = regeq.tvalues[0]
        resultvw.iloc[6, j] = regvw.params[0]
        resultvw.iloc[7, j] = regvw.tvalues[0]
        ff30 = pd.read_csv('..\DataBase\\ff30.csv')
        x1 = sm.add_constant(ff30.iloc[:, 1:])
        X1 = pd.concat([exeq, x1], axis=1)
        X1 = X1.dropna()
        X2 = pd.concat([exvw, x1], axis=1)
        X2 = X2.dropna()
        regeqq = sm.OLS(np.asarray(X1.iloc[:, 0]), np.asarray(X1.iloc[:, 1:])).fit()
        regvwq = sm.OLS(np.asarray(X2.iloc[:, 0]), np.asarray(X2.iloc[:, 1:])).fit()
        resultew.iloc[8, j] = regeqq.params[0]
        resultew.iloc[9, j] = regeqq.tvalues[0]
        resultvw.iloc[8, j] = regvwq.params[0]
        resultvw.iloc[9, j] = regvwq.tvalues[0]
    resultew.to_csv('EW55BM' + reteq.columns[u] + '.csv')
    resultvw.to_csv('VW55BM' + reteq.columns[u] + '.csv')


#******************************6.96项因子fama macbeth回归检验结果*****************************************#
import pandas as pd
import numpy as np
import glob,os
import statsmodels.api as sm
from sklearn.preprocessing import scale
from factorNWttest import nwttest_1samp
ret=pd.read_csv('final_return.csv')
size=pd.read_csv('..\DataBase\\factor\\01size.csv')
path = r'..\DataBase\\factor'
file = glob.glob(os.path.join(path, "*.csv"))
rf=pd.read_csv('..\DataBase\\RF.csv')
k = []  # 每个因子一个表
for i in range(96):
    k.append(pd.read_csv(file[i]))
factor = []#因子名称
for i in range(len(file)):
    factor.append(file[i][29:-4])
factor.insert(0, 'constant')
xishu=pd.DataFrame(index=ret.columns[1:],columns=factor)
for i in range(ret.shape[1]-1):
    final=pd.concat([j.iloc[:,i+1] for j in k],axis=1)
    final.columns=factor[1:]
    final = sm.add_constant(final)
    final['ret']=ret.iloc[:,i+1]
    X1=final.dropna()
    if len(X1)==0:
        xishu.iloc[i,:]=np.nan
    else:
        X1.iloc[:, 1:-1] = scale(X1.iloc[:, 1:-1], axis=0)
        reg = sm.OLS(np.asarray(X1.iloc[:, 0]), np.asarray(X1.iloc[:, 1:])).fit()
        xishu.iloc[i,:]=reg.params
FM=pd.DataFrame(columns=xishu.columns,index=['mean','t-stats'])
gen=xishu.dropna()
for i in range(xishu.shape[1]):
    fa = gen.iloc[:, i]
    FM.iloc[0, i] = fa.mean()#截面系数时序均值
    ttest = nwttest_1samp(fa, 0)#t检验-因子系数是否异于0
    FM.iloc[1, i] = ttest.statistic

FM.to_csv('FM.csv')