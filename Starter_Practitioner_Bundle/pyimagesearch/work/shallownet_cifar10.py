# 输入需要的库
import sys
import os
sys.path.append(os.path.abspath(".."))


from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import ShallowNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np


# 加载训练数据集和测试数据集，然后缩放到0,1范围内
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# 将标签从整数转换为向量形式
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# 对数据集初始化标签名字
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
             "dog", "frog", "horse", "ship", "truck"]

# 初始化优化器和模型
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = ShallowNet.ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 训练NN
print("[INFO] training work...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=10, verbose=1)

# 评价NN
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

# 绘制训练中的loss损失和精确度
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()















