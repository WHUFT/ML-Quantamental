# -*- coding:utf-8 -*-
'''
@description:
深度前馈网络库函数，依赖mxnet和gpu, 在有GPU的环境下运行。
'''
from mxnet import gluon,nd,gpu,autograd
from mxnet.gluon import loss as gloss,nn,data as gdata
import mxnet as mx
import numpy as np

class DFN(nn.Block):
    def __init__(self, outputdim, batch_size=10, epoch=10, depth=1, neuralset=[],
                 activiatemethod='relu', ctx=gpu(), trainmethod='sgd', lr=0.5, **kwargs):
        '''
        主函数，用于深度前馈网络训练
        :param outputdim: 输出维度
        :param batch_size: 小批量大小
        :param epoch: 训练论述
        :param depth: 神经网络深度
        :param neuralset: 各层神经元数，list形式 eg. [96,50,10]
        :param activiatemethod: 激活函数
        :param ctx: 训练用设备 默认gpu
        :param trainmethod: 训练器
        :param lr: 学习率
        :param kwargs: 其他参数
        '''
        super(DFN,self).__init__(**kwargs)
        mx.random.seed(521,ctx=gpu())
        mx.random.seed(521)
        self.ctx = ctx
        self.outputdim = outputdim
        self.net = nn.Sequential()
        self.neuralset = neuralset
        self.activiatemethod = activiatemethod
        self.depth = depth
        self.trainmethod = trainmethod
        self.lr = lr
        if neuralset:
            pass
        else:
            neuralset = [256]*depth
        for i in range(depth):
            self.net.add(nn.Dense(neuralset[i], activation=activiatemethod))
        self.net.add(nn.Dense(outputdim))
        self.net.initialize(ctx=self.ctx)
        self.Trainer = gluon.Trainer(self.net.collect_params(), trainmethod, {'learning_rate': lr})
        self.loss = gloss.L2Loss()
        self.batch_size = batch_size
        self.epoch = epoch
        print('network initializing finished')

    def retrain(self,Xnd,Ynd,data_iter,k):
        knew = 0
        self.net.initialize(ctx=self.ctx, force_reinit=True)
        self.Trainer = gluon.Trainer(self.net.collect_params(), self.trainmethod, {'learning_rate': self.lr/2})
        print('reinitialize finished and retrain begin')
        for i in range(k - 1):
            # with mx.autograd.record():
            #     l = self.loss(self.net(Xnd),Ynd).mean()
            # l.backward()
            # self.Trainer.step(4400)
            for xtrain, ytrain in data_iter:
                ytrain = nd.array(ytrain, ctx=self.ctx)
                with mx.autograd.record():
                    l = self.loss(self.net(xtrain), ytrain).mean()
                l.backward()
                self.Trainer.step(1)
                print('#')
            knew += 1
            lossout = self.loss(self.net(Xnd), Ynd)
            print('epoch ' + str(i) + ': loss is %f' % lossout.mean().asnumpy())
            if lossout.mean().asnumpy()>1:
                self.retrain(Xnd, Ynd, data_iter, knew)
                break
            elif np.isnan(lossout.mean().asnumpy()):
                self.retrain(Xnd, Ynd, data_iter, knew)
                break


        # print(lossout)


    def fit(self,X,Y):
        Xnd = nd.array(X,ctx = self.ctx)
        Ynd = nd.array(Y,ctx = self.ctx)
        dataset = gdata.ArrayDataset(Xnd, Ynd)
        # 随机读取小批量
        data_iter = gdata.DataLoader(dataset, self.batch_size, shuffle=False)
        k = 1
        for i in range(self.epoch):
            # with mx.autograd.record():
            #     l = self.loss(self.net(Xnd),Ynd).mean()
            # l.backward()
            # self.Trainer.step(4400)
            for xtrain, ytrain in data_iter:
                ytrain = nd.array(ytrain,ctx=self.ctx)
                with mx.autograd.record():
                    l = self.loss(self.net(xtrain), ytrain).mean()
                l.backward()
                self.Trainer.step(1)
                print('#', )
            k += 1
            lossout = self.loss(self.net(Xnd), Ynd)
            print('epoch ' + str(i) + ': loss is %f' % lossout.mean().asnumpy())
            if lossout.mean().asnumpy()>1:
                retrainNN = True
                break
            elif np.isnan(lossout.mean().asnumpy()):
                retrainNN = True
                break
            else:
                retrainNN = False
        if retrainNN:
            self.retrain(Xnd,Ynd,data_iter,k)
        print('all finished')

    def predict(self,Xtest):
        Xtestnd = nd.array(Xtest,ctx = self.ctx)
        return list(map(lambda x:x[0],nd.array(self.net(Xtestnd),ctx=mx.cpu()).asnumpy()))
