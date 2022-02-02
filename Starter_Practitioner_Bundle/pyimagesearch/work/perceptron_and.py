import sys
import os
sys.path.append(os.path.abspath(".."))


from pyimagesearch.nn import Perceptron
import numpy as np

# 构造and数据集
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])


# 定义感知机并进行训练
print("[INFO] training perception...")

p = Perceptron.Perceptron(X.shape[1], alpha=0.1)

p.fit(X, y, epochs=20)


# 测试感知机
print("[INFO] testing perception...")


for (x, target) in zip(X, y):
    pred = p.predict(x)
    print("[INFO] data={}, ground-truth={}, pred={}".format(x, target[0], pred))


