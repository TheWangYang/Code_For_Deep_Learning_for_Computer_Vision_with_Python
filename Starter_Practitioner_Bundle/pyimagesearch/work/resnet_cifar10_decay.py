import os
import sys
sys.path.append(os.path.abspath("../.."))

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import ResNet
from pyimagesearch.callbacks import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import sys
import os


# set a high recursion limit so Theano doesn’t complain
sys.setrecursionlimit(5000)


# 定义学习率衰减时间表
NUM_EPOCHS = 100
INIT_LR = 1e-1


def poly_decay(epoch):
    # 初始化最大epochs数，基本学习率和幂
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # 计算新的学习率根据epoch
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    return alpha



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-o", "--output", required=True, help="path to output directory (logs, plots, etc.)")
args = vars(ap.parse_args())


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
height_shift_range=0.1, horizontal_flip=True,
fill_mode="nearest")



# construct the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor.TrainingMonitor(figPath, jsonPath=jsonPath),
LearningRateScheduler(poly_decay)]


# initialize the optimizer and model (ResNet-56)
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = ResNet.ResNet.build(32, 32, 3, 10, (9, 9, 9),
(64, 64, 128, 256), reg=0.0005)
model.compile(loss="categorical_crossentropy", optimizer=opt,
metrics=["accuracy"])




# train the network
print("[INFO] training network...")
model.fit_generator(
aug.flow(trainX, trainY, batch_size=128),
validation_data=(testX, testY),
steps_per_epoch=len(trainX) // 128, epochs=10,
callbacks=callbacks, verbose=1)

# save the network to disk
print("[INFO] serializing network...")
model.save(args["model"])























