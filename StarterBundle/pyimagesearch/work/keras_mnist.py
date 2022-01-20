from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse

# 上述代码解释：
# 导入所需的Python包。LabelBinarizer将被用来将我们的整数标签热编码为矢量标签。
# one - hot 编码将分类标签从单个整数转换为向量。许多机器学习算法(包括神经网络)受益于这种类型的标签表示。在本节的后面，我将更详细地讨论单热编码，并提供多个示例(包括使用LabelBinarizer)。


# 接着导入必要的包，用Keras创建一个简单的前馈神经网络。顺序类表示我们的网络将被前馈，层次将顺序地添加到类中，一个在另一个之上。第6行上的Dense类是我们完全连接的层的实现。为了让我们的网络真正学习，我们需要应用SGD(第7行)来优化网络的参数。最后，为了访问完整的MNIST数据集，我们需要在第8行从scikit-learn中导入数据集帮助器。



# 这里我们只需要一个开关——output，它是我们绘制的随时间变化的损失和精度图的路径，它将被保存到磁盘上
ap = argparse.ArgumentParser()
ap.add_argument("-o", "-output", required=True,help="path to the output loss/accuracy plot")

args = vars(ap.parse_args)


# 加载MINIST数据集
print("[INFO] loading MNIST(full) dataset...")

dataset = datasets.fetch_openml("mnist_784", data_home="./dataset")  # 加载数据集，如果之前没有用过，那么会自动下载

data = dataset.data.astype("float") / 255.0  # 将数据进行归一化，将像素强度缩放到0,1范围

# 划分训练集和测试集的比例
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25)


# 编码我们的标签，将数据集中的标签转换为向量标签
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# 使用keras定义我们的网络结构
model = Sequential()
model.add(Dense(256, input_shape=(784, ), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

# 我们的网络是一个前馈体系结构，由第37行上的Sequential类实例化——这个体系结构意味着各个层将彼此堆叠在一起，前一层的输出将反馈到下一层。



# 接下来进行训练
print("[INFO] training network...")

# 初始化SGD迭代优化器
sgd = SGD(0.01)

# 使用类别交叉熵损失函数作为损失度量
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# 实际训练过程，并得到测试集的输出
# 实际上这里用到了交叉验证集，这里比较宽容地使用了测试集进行对参数的调优，但是实际模型的训练中，必须使用测试集之外的数据
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)


# 对模型进行评估
print("[INFO] evaluating network...")

predictions = model.predict(testX, batch_size=128)

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))


# 将结果进行可视化展示
#  plot  the  training  loss  and  accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100),  H.history["loss"],  label="train_loss")
plt.plot(np.arange(0, 100),  H.history["val_loss"],  label="val_loss")
plt.plot(np.arange(0, 100),  H.history["acc"],  label="train_acc")
plt.plot(np.arange(0, 100),  H.history["val_acc"],  label="val_acc")
plt.title("Training  Loss  and  Accuracy")
plt.xlabel("Epoch  #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])











