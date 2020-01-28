# -*- coding:utf-8 -*-
"""
@description:
    集成所有函数模型模块
"""
import numpy as np
class Ensemblelr(object):
    def __init__(self,classifymodellist,newmodellist,modelname):
        '''

        :param classifymodellist: 非RNN/LSTM模型
        :param newmodellist: RNN/LSTM模型
        :param modelname: 所有模型名称，
        注：PLS放到传统模型倒数第三位
        '''
        self.cmodelist = classifymodellist
        self.nml = newmodellist
        self.modelname = modelname
        self.cmodelnum = len(classifymodellist)
        self.nmodelnum = len(newmodellist)
        print('Ensembleall model initializing')

    def fit(self,TX,TY,NX,NY):
        for i in range(self.cmodelnum):
            self.cmodelist[i].fit(TX, TY)
            print(self.modelname[i]+' finished')
        for i in range(self.nmodelnum):
            self.nml[i].fit(NX, NY)
            print(self.modelname[self.cmodelnum + i])
        print('fit finished')

    def predict(self, TXtest, NXtest,indlist,length):
        predictlist = []
        for i in range(self.cmodelnum):
            predictlist.append(self.cmodelist[i].predict(TXtest))
        predictlist[-3] = list(map(lambda x: x[0], predictlist[-3]))  # 为了PLS的输出结果变成数值
        for i in range(self.nmodelnum):
            KP = self.nml[i].predict(NXtest)
            predictlist.append(np.array([KP[m][0] for m in range(3571 * (length - 1), 3571 * length)])[~indlist])
        return list(np.mean(predictlist, axis=0))


