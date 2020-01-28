# -*- coding:utf-8 -*-
'''
@description:
  用于构建循环神经网络
'''

import mxnet as mx
from mxnet import nd,gluon,gpu
from mxnet.gluon import rnn,nn,loss as gloss,data as gdata
import numpy as np

class BaseRnn(nn.Block):
    def __init__(self,num_feature,num_hidden,num_layers,ntype='RNN',bidirectional = False,dropout=0,**kwargs):
        '''
        基本循环神经网络单元
        :param num_feature: 特征个数
        :param num_hidden: 隐层神经元个数
        :param num_layers: 隐层数目
        :param ntype: 神经网络类型，默认RNN
        :param bidirectional: 是否双向，默认否
        :param dropout: 丢弃率，即神经元多大概率不激活
        :param kwargs: 其他参数
        '''
        super(BaseRnn,self).__init__(**kwargs)
        self.num_hidden = num_hidden
        if ntype == 'RNN':
            with self.name_scope():
                self.rnn = rnn.RNN(num_hidden,num_layers,input_size=num_feature,bidirectional=bidirectional, layout='TNC',dropout=dropout)
                self.decoder = nn.Dense(1, in_units=num_hidden)
        elif ntype == 'LSTM':
            with self.name_scope():
                self.rnn = rnn.LSTM(num_hidden,num_layers,input_size=num_feature,bidirectional=bidirectional, layout='TNC',dropout=dropout)
                self.decoder = nn.Dense(1, in_units=num_hidden)
        elif ntype == 'GRU':
            with self.name_scope():
                self.rnn = rnn.GRU(num_hidden,num_layers,input_size=num_feature,bidirectional=bidirectional, layout='TNC',dropout=dropout)
                self.decoder = nn.Dense(1, in_units=num_hidden)

    def forward(self, inputs, hidden):
        output, hidden = self.rnn(inputs, hidden)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


class lstmmodule(object):
    def __init__(self,num_feature,num_hidden,num_layers,epoch,batch_size,ntype='LSTM',bidirectional = False,dropout=0,trainmethod = 'adam',\
                 lr=0.01,loss = gloss.L1Loss(),ctx = gpu(0),datashuffle=False,initialfunc=mx.init.Xavier(),prin=False,**kwargs):
        '''
        用于循环神经网络的训练和预测， 主函数
        :param num_feature: 特征个数
        :param num_hidden: 隐层神经元数目
        :param num_layers: 隐层数
        :param epoch: 训练轮数
        :param batch_size: 小批量大小，在这里是股票实体数目多少
        :param ntype: 神经网络类型，默认LSTM
        :param bidirectional: 是否双向神经网络，默认否
        :param dropout: 丢弃率，默认不丢弃
        :param trainmethod: 训练器，默认adam
        :param lr: 学习率
        :param loss: 损失函数
        :param ctx: 训练设备，默认gpu
        :param datashuffle: 是否随机打乱数据，默认否
        :param initialfunc: 初始化方法，默认Xaiver
        :param prin: 是否输出训练指示，默认否
        :param kwargs: 其他参数
        '''
        super(lstmmodule,self).__init__(**kwargs)
        mx.random.seed(521,ctx=gpu())
        mx.random.seed(521)
        self.ctx = ctx
        self.net = BaseRnn(num_feature,num_hidden,num_layers,ntype,bidirectional,dropout)
        self.trainmethod = trainmethod
        self.lr = lr
        self.intialfunc = initialfunc
        self.net.collect_params().initialize(self.intialfunc, ctx=self.ctx)
        self.Trainer = gluon.Trainer(self.net.collect_params(), trainmethod, {'learning_rate': lr})
        self.loss = loss
        self.datashuffle = datashuffle
        self.epoch = epoch
        self.batch_size = batch_size
        self.prin = prin

    def fit(self,X,Y):
        Xnd = nd.array(X,ctx = self.ctx)
        Ynd = nd.array(Y,ctx = self.ctx)
        for i in range(self.epoch):
            state = self.net.begin_state(batch_size=self.batch_size, ctx=self.ctx)
            for s in state:
                s.detach()
            with mx.autograd.record():
                (output, state) = self.net(Xnd, state)
                l = self.loss(output.reshape(len(output)//self.batch_size,self.batch_size), Ynd).mean()
            l.backward()
            self.Trainer.step(1)
            if self.prin:
                print('#')
                if i == self.epoch -1:
                    print('the last epoch loss is ', l.asnumpy())
        if self.prin:
            print('all finished')


    def predict(self,Xtest):
        state = self.net.begin_state(batch_size=self.batch_size, ctx=self.ctx)
        Xtestnd = nd.array(Xtest,ctx = self.ctx)
        (Y,statenew) = self.net(Xtestnd,state)
        return nd.array(Y,ctx=mx.cpu()).asnumpy()

