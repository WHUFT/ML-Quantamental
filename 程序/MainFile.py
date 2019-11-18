#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@description:
    构建机器学习驱动多因子投资策略主函数
    1.首先输入各算法参数（参数根据第一个滑动窗口网格调参确定，此处直接输入）
    2.全样本3/12/24/36个月滑动窗口函数运行,最终直接输出多空组合月度收益，FF3/5-alpha,sharpe ratio，并将收益序列保存在’output'文件夹内
    3.全样本3/12/24/36个月滑动窗口各个算法多空组合月度收益序列是否存在显著差异NW-T检验
    4.不同交易费率下的多空组合绩效结果
    5.全样本12个月华东窗口下各个算法的特征筛选
    6.特征筛选后16项因子12个月滑动窗口函数运行,最终输出多空组合月度收益，FF3/5-alpha,sharpe ratio，并将收益序列保存在’output'文件夹内
    注：深度学习算法要在使有GPU的环境下进行训练
"""
from StrategyConstruct import FC, output, output2, comboutput, ensemblenn
from selectFactor import dropimportant, dropimportant2, FCselect
from DataTransfrom import datatransfrom, datatransfrom2
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import DFN
import RNNMODEL as rm
import Ensembleall as ea
import warnings
from mxnet import gpu
import os
from transecfee import showtrasecfee
from ReturnSeriesTest import returnseriestest
warnings.filterwarnings('ignore')


#各个算法参数（根据第一个窗口网格调参确定）
window=[3,12,24,36]
PLS_params=[2,2,1,1]
lasso_params=[1e-3,5e-4,0.01,0.01]
ridge_params=[0.1,0.005,0.01,0.005]
elasticnet_params={'alpha':[0.01,1e-3,0.01,0.1],'l1_ratio':[0.3,0.3,0.7,0.3]}
SVR_params={'kernel':['linear','linear','rbf','rbf'],'gamma':[1e-3,1e-3,1e-3,1e-4],'C':[0.01,0.001,0.01,1e-4]}
GBDT_params={'learning_rate':[0.1,0.1,0.1,0.1],'maxdepth':[2,3,2,2],'n_estimators':[100,100,100,100]}#XGBOOST与GBDT相同 此处共用
ENANN_params = {'max_iter': [100, 100, 200, 300], 'p': [0.3, 0.5, 0.7, 0.5]}
DFN_params = {'learning_rate':[0.1, 0.1, 0.1, 0.001], 'batch': [300, 400, 300, 400]}
LSTM_params = {'learning_rate':[1e-4, 1e-5, 1e-4, 1e-6], 'depth': [2, 2, 1, 2], 'hidden_number': [256]*4}
RNN_params = {'learning_rate':[0.1, 0.1, 0.1, 0.001], 'depth': [1, 1, 2, 1], 'hidden_number': [256]*4}


#**********************2.全样本3/12/24/36个月滑动窗口函数运行**********************************#
path = r'..\DataBase\factor'#96项因子所在路径
factorname = [x[1:-4] for x in os.listdir(path)]
riskfree, timeseries, factor, timeseries2, index = datatransfrom(path)[0], datatransfrom(path)[1], datatransfrom(path)[2], datatransfrom2(path)[0], datatransfrom2(path)[1]
for i in range(4):
    i= 0
    output(window[i],LinearRegression(),'OLS'+str(window[i]),riskfree[i], timeseries)
    FC(window[i], riskfree[i], timeseries, 96,'FC')
    output(window[i], PLSRegression(PLS_params[i]), 'PLS' + str(window[i]), riskfree[i], timeseries)
    output(window[i],Lasso(alpha=lasso_params[i]),'Lasso'+ str(window[i]), riskfree[i], timeseries)
    output(window[i],Ridge(alpha=ridge_params[i]),'Ridge'+str(window[i]),riskfree[i], timeseries)
    output(window[i],ElasticNet(alpha= elasticnet_params['alpha'] [i],l1_ratio= elasticnet_params['l1_ratio'][i]),'ElasticNet'+str(window[i]),riskfree[i], timeseries)
    output(window[i],SVR(kernel=SVR_params['kernel'][i],gamma= SVR_params ['gamma'][i],C= SVR_params ['C'][i] ),'SVR'+str(window[i]),riskfree[i], timeseries)
    output(window[i], GradientBoostingRegressor(n_estimators=GBDT_params['n_estimators'][i],max_depth=GBDT_params['maxdepth'][i],learning_rate=GBDT_params['learning_rate'][i]), 'GBDT' + str(window[i]),riskfree[i], timeseries)
    output(window[i], XGBRegressor(n_estimators=GBDT_params['n_estimators'][i],max_depth=GBDT_params['maxdepth'][i], learning_rate=GBDT_params['learning_rate'][i]), 'XGBOOST' + str(window[i]), riskfree[i], timeseries)
    output(window[i], ensemblenn(5,modeluse = MLPRegressor(solver = 'lbfgs', max_iter=ENANN_params['max_iter'][i]), pickpercent=ENANN_params['p'][i]), 'ENANN' + str(window[i]), riskfree[i], timeseries)
    output(window[i], DFN.DFN(outputdim=1, neuralset=[96, 50, 25, 10, 5, 2], ctx=gpu(0), epoch=10, batch_size=DFN_params['batch'][i], lr=DFN_params['learning_rate'][i]), 'DFN' + str(window[i]), riskfree[i], timeseries)
    output2(window[i], rm.lstmmodule(96, LSTM_params['hidden_number'][i], LSTM_params['depth'][i], 100, 3571, lr=LSTM_params['learning_rate'][i]), 'LSTM'+ str(window[i]) ,riskfree[i], timeseries2)
    output2(window[i], rm.lstmmodule(96,  RNN_params['hidden_number'][i], RNN_params['depth'][i], 100, 3571, lr=RNN_params['learning_rate'][i], ntype='RNN'), 'RNN'+ str(window[i]), riskfree[i], timeseries2)
    modellist = [DFN.DFN(outputdim=1, neuralset=[96, 50, 25, 10, 5, 2], ctx=gpu(0), epoch=10, batch_size=DFN_params['batch'][i], lr=DFN_params['learning_rate'][i]),
                 ensemblenn(5,modeluse = MLPRegressor(solver = 'lbfgs', max_iter=ENANN_params['max_iter'][i]), pickpercent=ENANN_params['p'][i]),
                 XGBRegressor(n_estimators=GBDT_params['n_estimators'][i],max_depth=GBDT_params['maxdepth'][i], learning_rate=GBDT_params['learning_rate'][i]),
                 GradientBoostingRegressor(n_estimators=GBDT_params['n_estimators'][i],max_depth=GBDT_params['maxdepth'][i],learning_rate=GBDT_params['learning_rate'][i]),
                 PLSRegression(PLS_params[i]),
                 Ridge(alpha=ridge_params[i]),
                 SVR(kernel=SVR_params['kernel'][i],gamma= SVR_params ['gamma'][i],C= SVR_params ['C'][i])]# PLS一定要放在倒数第三个
    nmolist = [rm.lstmmodule(96, LSTM_params['hidden_number'][i], LSTM_params['depth'][i], 100, 3571, lr=LSTM_params['learning_rate'][i]),
               rm.lstmmodule(96,  RNN_params['hidden_number'][i], RNN_params['depth'][i], 100, 3571, lr=RNN_params['learning_rate'][i], ntype='RNN')]# 循环神经网络模型
    modelname = ['DFN', 'En-ann', 'xgboost', 'GBDT', 'lasso', 'Elasticnet', 'pls', 'Ridge', 'svm', 'LSTM', 'RNN']
    ensemblemodel = ea.Ensemblelr(modellist, nmolist, modelname)
    comboutput(window[i],ensemblemodel, 'Ensemble'+str(window[i]),riskfree[i], timeseries2, index)
#*********************************3..各算法收益序列差异NW-t检验****************************************#

for i in window:
    returnseriestest(i)

#*********************************4.不同交易费率情形****************************************#

showtrasecfee(0.005)
showtrasecfee(0.0075)
showtrasecfee(0.01)

#****************************5.全样本12个月特征筛选过程************************************#

i = 1#选取12个月滑动窗口筛选因子
dropimportant(window[i] ,LinearRegression(), 'OLS'+str(window[i]), factorname, timeseries,0.0201)
FCselect(factorname, timeseries)
dropimportant(window[i], PLSRegression(PLS_params[i]), 'PLS', factorname, timeseries, 0.0230)
dropimportant(window[i], Lasso(alpha=lasso_params[i]), 'Lasso', factorname, timeseries, 0.0208)
dropimportant(window[i], Ridge(alpha=ridge_params[i]), 'Ridge', factorname, timeseries, 0.0208)
dropimportant(window[i], ElasticNet(alpha= elasticnet_params['alpha'] [i],l1_ratio= elasticnet_params['l1_ratio'][i]), 'ElasticNet', factorname, timeseries, 0.0212)
dropimportant(window[i], SVR(kernel=SVR_params['kernel'][i],gamma= SVR_params ['gamma'][i],C= SVR_params ['C'][i] ), 'SVR', factorname, timeseries, 0.0225)
dropimportant(window[i], GradientBoostingRegressor(n_estimators=GBDT_params['n_estimators'][i],max_depth=GBDT_params['maxdepth'][i],learning_rate=GBDT_params['learning_rate'][i]), 'GBDT', factorname, timeseries, 0.0268)
dropimportant(window[i], XGBRegressor(n_estimators=GBDT_params['n_estimators'][i],max_depth=GBDT_params['maxdepth'][i], learning_rate=GBDT_params['learning_rate'][i]), 'XGBOOST',  factorname, timeseries, 0.0273)
dropimportant(window[i], ensemblenn(5,modeluse = MLPRegressor(solver = 'lbfgs', max_iter=ENANN_params['max_iter'][i]), pickpercent=ENANN_params['p'][i]), 'ENANN', factorname, timeseries, 0.0234)
dropimportant(window[i], DFN.DFN(outputdim=1, neuralset=[96, 50, 25, 10, 5, 2], ctx=gpu(0), epoch=10, batch_size=DFN_params['batch'][i], lr=DFN_params['learning_rate'][i]), 'DFN', factorname, timeseries, 0.0278)
dropimportant2(window[i], rm.lstmmodule(95, LSTM_params['hidden_number'][i], LSTM_params['depth'][i], 100, 3571, lr=LSTM_params['learning_rate'][i]), 'LSTM', factorname, timeseries2, 0.0257)
dropimportant2(window[i], rm.lstmmodule(95,  RNN_params['hidden_number'][i], RNN_params['depth'][i], 100, 3571, lr=RNN_params['learning_rate'][i], ntype='RNN'), 'RNN', factorname, timeseries2, 0.0210)


#****************************6.特征筛选后16项因子12个月滑动窗口函数运行************************************#
path = r'..\DataBase\factorselect'#经过筛选后因子集合所在路径
riskfree,timeseries,factor,timeseries2=datatransfrom(path)[0],datatransfrom(path)[1],datatransfrom(path)[2],datatransfrom2(path,after=True)[0]
i=1 #选取12个月滑动窗口测试筛选后因子集合绩效表现
output(window[i],LinearRegression(),'OLS'+str(window[i]),riskfree[i], timeseries)
FC(window[i], riskfree[i], timeseries, 11,'FC')
output(window[i], PLSRegression(PLS_params[i]), 'PLS' + str(window[i]), riskfree[i], timeseries)
output(window[i],Lasso(alpha=lasso_params[i]),'Lasso'+ str(window[i]), riskfree[i], timeseries)
output(window[i],Ridge(alpha=ridge_params[i]),'Ridge'+str(window[i]),riskfree[i], timeseries)
output(window[i],ElasticNet(alpha= elasticnet_params['alpha'] [i],l1_ratio= elasticnet_params['l1_ratio'][i]),'ElasticNet'+str(window[i]),riskfree[i], timeseries)
output(window[i],SVR(kernel=SVR_params['kernel'][i],gamma= SVR_params ['gamma'][i],C= SVR_params ['C'][i] ),'SVR'+str(window[i]),riskfree[i], timeseries)
output(window[i], GradientBoostingRegressor(n_estimators=GBDT_params['n_estimators'][i],max_depth=GBDT_params['maxdepth'][i],learning_rate=GBDT_params['learning_rate'][i]), 'GBDT' + str(window[i]),riskfree[i], timeseries)
output(window[i], XGBRegressor(n_estimators=GBDT_params['n_estimators'][i],max_depth=GBDT_params['maxdepth'][i], learning_rate=GBDT_params['learning_rate'][i]), 'XGBOOST' + str(window[i]), riskfree[i], timeseries)
output(window[i], ensemblenn(5,modeluse = MLPRegressor(solver = 'lbfgs', max_iter=ENANN_params['max_iter'][i]), pickpercent=ENANN_params['p'][i]), 'ENANN' + str(window[i]), riskfree[i], timeseries)
output(window[i], DFN.DFN(outputdim=1, neuralset=[16, 50, 25, 10, 5, 2], ctx=gpu(0), epoch=10, batch_size=DFN_params['batch'][i], lr=DFN_params['learning_rate'][i]), 'DFN' + str(window[i]), riskfree[i], timeseries)
output2(window[i], rm.lstmmodule(11, LSTM_params['hidden_number'][i], LSTM_params['depth'][i], 100, 3571, lr=LSTM_params['learning_rate'][i]), 'LSTM'+ str(window[i]) ,riskfree[i], timeseries2)
output2(window[i], rm.lstmmodule(11,  RNN_params['hidden_number'][i], RNN_params['depth'][i], 100, 3571, lr=RNN_params['learning_rate'][i], ntype='RNN'), 'RNN'+ str(window[i]), riskfree[i], timeseries2)


