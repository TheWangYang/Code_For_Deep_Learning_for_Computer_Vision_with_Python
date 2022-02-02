# 导入需要包
import sys
import os
sys.path.append(os.path.abspath(".."))

from pyimagesearch.nn.conv import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np



# 加载MNIST数据集
print("[INFO] accessing MNIST...")
dataset = datasets.fetch_openml("mnist_784", data_home="./dataset")
data = dataset.data

if K.image_data_format() == "channel_first":
    data = data.reshape(data.shape[0], 1, 28,28)
else:
    data = data.reshape(data.shape[0], 28, 28, 1)


# 将上述的各个图片像素缩放到0,1之间
(trainX, testX, trainY, testY) = \
    train_test_split(data / 255.0, dataset.target.astype("int"), test_size=0.25, random_state=42)


# 将标签从整数转换为向量
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.fit_transform(testY)


# 初始化优化器和模型
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 训练网络
print("[INFO] training work...")
H = model.fit(trainX, trainY, validation_data=(testX,testY), batch_size=128, epochs=20, verbose=1)


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1),
target_names=[str(x) for x in le.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()












