from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activation(x):
    # 对制定的一个input计算sigmoid激活函数的指定值
    return 1.0 / (1 + np.exp(-x))

def predict(X, W):
    # 取特征和权值矩阵之间的点积
    preds = sigmoid_activation(X.dot(W))
    # 应用一个阶跃函数来阈值输出到二进制分类标签
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1

    return preds


def next_batch(X, y, batchSize):
    # 以小批量循环我们的数据集' X '，生成一个当前batch数据集和标签的元组
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i: i + batchSize], y[i: i + batchSize])


#  construct  the  argument  parse  and  parse  the  arguments
ap  =  argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float,  default=100, help="#  of  epochs")
ap.add_argument("-a", "--alpha", type=float,  default=0.01,help="learning  rate")
ap.add_argument("-b", "--batch-size", type=int,  default=32,help="size  of  SGD  mini-batches")
args = vars(ap.parse_args())


# 生成一个包含1000个数据点的2类分类问题，每个数据点是一个2D特征向量
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# 插入一列1作为特性矩阵的最后一项，这个小技巧可以让我们处理偏差，作为权重矩阵中的可训练参数
X = np.c_[X, np.ones((X.shape[0]))]

# 将数据分割成训练和测试分割，使用50%数据用于训练，其余50%用于测试
(trainX,testX,trainY,testY) = train_test_split(X, y,test_size=0.5,random_state=42)

# 初始化我们的权值矩阵和损失列表
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []


# 循环每一个epoch
for epoch in np.arange(0, args["epochs"]):
    # 初始化最终的losses数组对每一个epoch
    epochLoss = []

    # 循环得到当前的Batch对应的数据集
    for (batchX, batchY) in next_batch(X, y,args["batch_size"]):
        # 使用当前的小批量数据集x点乘上权重矩阵
        preds = sigmoid_activation(batchX.dot(W))
        # 得到当前的偏差
        error = preds - batchY
        epochLoss.append(np.sum(error ** 2))

        gradient = batchX.T.dot(error)
        W += -args["alpha"] * gradient


    # 得到每一个epoch中的loss的平均值，然后将loss的平均值放入到losses中
    loss = np.average(epochLoss)
    losses.append(loss)

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}， loss={:.7f}".format(int(epoch + 1), loss))


# 评估分类器
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))


# 绘制(测试)分类数据
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:,0], testX[:, 1], marker="o", c=testY, s=30)


# 绘制loss
plt.style.use("ggplot")

plt.figure()
plt.plot(np.arange(0,  args["epochs"]),  losses)
plt.title("Training  Loss")
plt.xlabel("Epoch  #")
plt.ylabel("Loss")
plt.show()





