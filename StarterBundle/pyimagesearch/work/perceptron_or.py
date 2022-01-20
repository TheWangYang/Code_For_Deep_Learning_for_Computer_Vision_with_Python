from pyimagesearch.nn import Perceptron
import numpy as np

# 构造OR数据集
X = np.array([[0, 0],  [0, 1],  [1, 0],  [1, 1]])
y = np.array([[0],  [1],  [1],  [1]])


# 定义感知机，并进行训练
print("[INFO] training perceptron...")
p = Perceptron.Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)



# 评价感知机
print("[INFO] testing perceptron...")

# 遍历数据点
for (x, target) in zip(X, y):
    # 做一个预测，并将结果显示在其中
    pred = p.predict(x)
    # ground-truth是真实值的意思
    print("[INFO] data={}, ground-truth={}, pred={}".format(x, target[0], pred))






