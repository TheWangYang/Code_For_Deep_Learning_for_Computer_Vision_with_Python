import sys
import os
sys.path.append(os.path.abspath("../.."))

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import ResNet
from pyimagesearch.callbacks import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model
import keras.backend as K
import numpy as np
import argparse


# set a high recursion limit so Theano doesn’t complain
sys.setrecursionlimit(5000)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--checkpoints", help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
# ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
ap.add_argument("-om", "--output_model", required=True, help="序列化（保存）模型到磁盘")
args = vars(ap.parse_args())


# 下一步是从磁盘加载CIFAR-10数据集(预拆分为训练和测试)，进行均值减法，对整数标签进行一次热编码为向量:

# load the training and testing data, converting the images from
# integers to floats
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")
# apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")


# 在这种情况下，我们从第一个时代就开始训练ResNet，我们需要实例化网络架构:
# if there is no specific model checkpoint supplied, then initialize
# the network (ResNet-56) and compile the model
if args["model"] is None:
    print("[INFO] compiling model...")
    opt = SGD(lr=1e-1)
    model = ResNet.ResNet.build(32, 32, 3, 10, (9, 9, 9), (64, 64, 128, 256), reg=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# otherwise, load the checkpoint from disk
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    # update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-5)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))


# construct the set of callbacks
callbacks = [TrainingMonitor.TrainingMonitor("output/resnet56_cifar10.png", jsonPath="output/resnet56_cifar10.json")]


# train the network
print("[INFO] training network...")
model.fit_generator(
aug.flow(trainX, trainY, batch_size=128), validation_data=(testX, testY),
steps_per_epoch=len(trainX) // 128, epochs=10,
callbacks=callbacks, verbose=1)


# 序列化模型到磁盘
print("[INFO] save model...")
model.save(args["output_model"])































