import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        # 初始化权值矩阵，存储学习速率
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha


    # 定义阶跃函数
    def step(self, x):
        # 应用阶跃函数
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # 训练感知机
        X = np.c_[X, np.ones((X.shape[0]))]
        # 循环每一个epoch
        for epoch in np.arange(0, epochs):
            # 循环遍历每个单独的数据点
            for (x, target) in zip(X, y):
                # 得到每一个单独数据点对应的阶跃函数值
                p = self.step(np.dot(x, self.W))

                # 如果预测不正确的话，更新权重
                if p != target:
                    error = p - target
                    #更新权重
                    self.W += -self.alpha * error * x
                    print("[INFO] current weight is {}".format(self.W))


    # 定义的预测函数
    def predict(self, X, addBias=True):
        # 确定输入是一个矩阵
        X = np.atleast_2d(X)
        # 检查是否应该增加bias列
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        return self.step(np.dot(X, self.W))

