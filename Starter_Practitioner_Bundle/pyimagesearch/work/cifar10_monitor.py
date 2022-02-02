import sys
import os
sys.path.append(os.path.abspath(".."))

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# 导入需要的包
from pyimagesearch.callbacks import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse

# 获得命令行输入参数
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output directory")
args = vars(ap.parse_args())

# 根据进程id展示信息
print("[INFO process ID: {}".format(os.getpid()))

# 加载训练和测试数据集并缩放数据集到0,1范围内
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0


# 将标签从整数转换为向量形式
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# 初始化标签列表
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

# 初始化优化器，但是不设置任何降低学习率的模块
print("[INFO] compiling model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 构建对应的监控模块
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor.TrainingMonitor(figPath, jsonPath=jsonPath)]

# 训练网络，并显示实时的训练和测试loss和accuracy
print("[INFO] training network...")
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=100, callbacks=callbacks, verbose=1)

















