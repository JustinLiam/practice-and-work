import numpy as np


class BPNet(object):

    def __init__(self,num_inputs,num_hiddens,num_outputs,lr=0.1,epochs=100000):
        """
        初始化
        :param num_inputs: 输入的个数
        :param num_hiddens: 隐藏层神经元的个数
        :param num_outputs: 输出层神经元的个数
        :param lr: 学习速率
        :param epochs: 训练迭代次数
        """
        self.w1 = np.random.random((num_inputs,num_hiddens))*2 - 1     # 隐藏层的权重，取之范围为 [ -1,1 ]
        self.b1 = np.zeros(num_hiddens)     # 隐藏层的偏置
        self.w2 = np.random.random((num_hiddens,num_outputs))*2 - 1     # 输出层的权重，取之范围为 [ -1,1 ]
        self.b2 = np.zeros(num_outputs)     # 输出层偏置
        self.lr = lr
        self.epochs = epochs

    # sigmoid 激活函数
    def sigmoid(self,X):
        return 1/(1+np.exp(-X))

    # sigmoid 函数的导数
    def dsigmoid(self,X):
        return X*(1-X)


    def fit(self,X,y):
        """
        训练网络
        :param X: 训练样本输入
        :param y: 真实值
        """
        for epoch in range(self.epochs):
            self.update(X,y)


    def forward(self,X):
        """
        逐层进行进行向前传播
        :param X: 输入
        :return: 隐藏层与输出层的向前传播的结果
        """
        ########## Begin ##########
        # 输入层到隐藏层的向前传播
        hidden = self.sigmoid(np.dot(X,self.w1) + self.b1)
        output = self.sigmoid(np.dot(hidden,self.w2) + self.b2)
        print(np.shape(self.w2))
        # 隐藏层到输出层的向前传播
        ########## End ##########
        return hidden,output

    def update(self,X,y):
        """
        根据误差，更新权重与偏置
        :param X: 输入
        :param y: 真实值
        """
        hidden,output = self.forward(X)
        ########## Begin ##########
        # 输出层误差改变量
        L2_delta = (y.T - output)*self.dsigmoid(output)
        # 隐藏层误差改变量
        L1_delta = L2_delta.dot(self.w2.T)*self.dsigmoid(hidden)
        # 输出层对隐藏层的权重改变量
        w2_c = self.lr * hidden.dot(L2_delta)
        # 隐藏层对输入层的权重改变量
        w1_c = self.lr * X.T.dot(L1_delta)
        # 更新权重以及偏置
        self.w2 = self.w2 + w2_c
        self.w1 = self.w1 + w1_c
        self.b2 = self.b2 + self.lr * L2_delta
        self.b1 = self.b1 + self.lr * L1_delta
        ########## End ##########




