from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse


# 用于解析输入的参数
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output",  required=True, help="path  to  the  output  loss/accuracy  plot")
args = vars(ap.parse_args())

# 加载数据集
print("[INFO]  loading  CIFAR-10  data...")
((trainX,  trainY),  (testX,  testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))

# 为了使训练集和测试集中的每个图像变平，我们只需使用NumPy的.重塑函数。执行这个函数之后，trainX现在拥有形状(50000,3072)，而testX拥有形状(10000,3072)。

# 将类标签转换为向量矩阵
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# 初始化标签
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]



# 定义网络结构
model = Sequential()
model.add(Dense(1024,  input_shape=(3072,),  activation="relu"))
model.add(Dense(512,  activation="relu"))
model.add(Dense(10,  activation="softmax"))

# 第37行实例化了Sequential类。然后，我们添加第一个稠密层，其input_shape为3072，这是设计矩阵中3072个扁平像素值的每个节点——这一层负责学习1024个权值。我们还将用ReLU激活来替换过时的sigmoid，以期提高网络性能。

# 下一个全连接层(第39行)学习512个权值，而最后一层(第40行)学习对应于10种可能的输出分类的权值，以及一个softmax分类器，以获得每个类的最终输出概率。


# 训练网络
print("[INFO] training network...")

sgd = SGD(0.01)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)

# 评估网络

print("[INFO] evaluating network...")

predictions = model.predict(testX, batch_size=32)

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))



# 最后将loss和精确度展示出来
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






