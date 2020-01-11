# -*- coding: utf-8 -*-
import numpy as np


# 双曲正切函数,该函数为奇函数
def tanh(x):
    return np.tanh(x)


# tanh()的导函数为 f'(t) = 1 - f(x)^2
def tanh_Der(x):
    return 1.0 - tanh(x)**2


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        :参数layers: 神经网络的结构(输入层-隐含层-输出层包含的结点数列表)
        :参数activation: 激活函数类型
        """

        # 通过参数判断模型使用的激活函数
        # 这里使用的是双曲正切函数tanh()
        # 也可以用其它的激活函数
        if activation == 'tanh':
            self.activation = tanh
            self.activation_Der = tanh_Der
        else:
            # 可以在这里添加其他的激活函数
            # 我们没有其他激活函数，这里就直接跳过
            pass

        # 初始化存储权值的矩阵
        # 用于之后存储和输出权重矩阵，从而保存计算好的模型
        self.weights = []

        for i in range(1, len(layers) - 1):
            # 初始化输入层和隐藏层连接权重
            # 取随机数的范围的上下限需要各+1
            # 留给输入层和隐藏层的偏置节点
            tmp = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) - 1
            self.weights.append(tmp)

        # 初始化输出层与隐藏层之间的连接权重
        tmp = 2*np.random.random((layers[i] + 1, layers[i+1])) - 1
        self.weights.append(tmp)

    def Train(self, X, Y, learning_rate=0.05, epochs=600000):
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # 训练固定次数，通过函数参数赋值，默认是100000
        for k in range(epochs):
            if k % 1000 == 0:
                print('epochs:', k)

            # 获得区间[0，low）内离散均匀分布的随机整数
            # 从而实现从m= X.shape[0]个输入样本中随机选一个样本
            i = np.random.randint(X.shape[0], high=None)
            a = [X[i]]

            for l in range(len(self.weights)):
                # 权值矩阵中每一列代表该层中的一个结点与上一层所有结点之间的权值
                dot_value = np.dot(a[l], self.weights[l])
                activation = self.activation(dot_value)
                a.append(activation)

            # 反向递推计算delta:从输出层开始,先算出该层的delta,再向前计算
            # 计算输出层delta
            error = Y[i] - a[-1]
            deltas = [error * self.activation_Der(a[-1])]

            # 从倒数第2层开始反向计算delta
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)
                              * self.activation_Der(a[l]))

            # 逆转列表中的元素
            deltas.reverse()    
            # 反向传播
            # 1.将其输出增量与输入激活量相乘，得到权重的梯度。
            # 2.从权重中减去梯度的一个比率（百分比）。
            # 逐层调整权值
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                # 每输入一次样本,就更新一次权值
                self.weights[i] += learning_rate * \
                    np.dot(layer.T, delta)  

    def predict(self, x):
        a = np.concatenate((np.ones(1), np.array(x)))       
        for l in range(0, len(self.weights)):               
            a = self.activation(np.dot(a, self.weights[l]))
        return a


if __name__ == '__main__':
    # 网络结构: 2输入1输出,1个隐含层(包含2个结点)
    nn = NeuralNetwork([2, 10, 1])     
    # 输入矩阵(每行代表一个样本,每列代表一个特征)
    X = np.array([[0, 0],           
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    # 期望输出
    Y = np.array([0, 1, 1, 0])      
    # 训练网络
    nn.Train(X, Y)                    
    # 调整后的权值列表
    print('w:', nn.weights)          

    # 真实值
    y_ = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [1], [0], [3], [2], [5], [4], [7], [6], [2], [3], [0], [1], [6], [7], [4], [5], [3], [2], [1], [0], [7], [6], [5], [
                  4], [4], [5], [6], [7], [0], [1], [2], [3], [5], [4], [7], [6], [1], [0], [3], [2], [6], [7], [4], [5], [2], [3], [0], [1], [7], [6], [5], [4], [3], [2], [1], [0]])
    # 误差允许范围
    re = 0.0005

    pre = []
    for i in range(8):
        for j in range(8):
            pre.append(['{:03b}'.format(i), '{:03b}'.format(j)])
    print(pre)

    res = []
    for s in pre:
        tmp = 0
        for i in range(3):
            tmp += nn.predict([int(s[0][i]), int(s[1][i])])*2**(2-i)
        res.append(tmp)
    for i in res:
        print(i)

    print("loss:", np.square(np.subtract(res, y_)).mean())

    count = 0
    for i in range(64):
        sub = res[i][0].item()-y_[i][0].item()
        print(y_[i][0], res[i][0], sub)
        if np.abs(sub) <= re:
            count += 1
    print("accuracy:", count, '/', len(y_), '=', count/len(y_))
