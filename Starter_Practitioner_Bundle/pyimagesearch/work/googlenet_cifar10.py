import sys
import os
sys.path.append(os.path.abspath('../..'))

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import MiniGoogLeNet
from pyimagesearch.callbacks import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import os



# 初始化epochs和学习率
NUM_EPOCHS = 70
INIT_LR = 5e-3

def poly_decay(epoch):
    # 初始化最大epochs, 基础学习率,和幂p
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # 基于多项式衰减计算新的学习率
    alpha = baseLR * (1 - (epoch / maxEpochs)) ** power

    return alpha


# 构建命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-o", "--output", required=True, help="path to output directory (logs, plots, etc.)")
args = vars(ap.parse_args())


# 记载数据集并将int类型数转换为float类型的
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# 应用均值像素减法
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# 将标签从int类型转换为向量形式
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# 为了防止过拟合，使用数据增强
aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
                         fill_mode="nearest")

# 构建回调函数列表，监视训练过程
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor.TrainingMonitor(figPath, jsonPath=jsonPath), LearningRateScheduler(poly_decay)]


# 训练网络
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = MiniGoogLeNet.MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 训练网络
print("[INFO] training network...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
validation_data=(testX, testY), steps_per_epoch=len(trainX) // 64,
epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)


# 像磁盘中保存训练好的Model
print("[INFO] serializing network...")
model.save(args["model"])




























