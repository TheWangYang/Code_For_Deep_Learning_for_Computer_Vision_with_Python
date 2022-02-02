import os
import sys
sys.path.append(os.path.abspath("../.."))

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# 导入需要的库
from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import ResNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import load_model
import keras.backend as K
import argparse
import json
import sys


# set a high recursion limit so Theano doesn’t complain
sys.setrecursionlimit(5000)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
# ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
# ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
ap.add_argument("-om", "--output_model", type=str, help="磁盘上保存模型的路径")
args = vars(ap.parse_args())


# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
horizontal_flip=True, fill_mode="nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())


 # initialize the image preprocessors
sp = SimplePreprocessor.SimplePreprocessor(64, 64)
mp = MeanPreprocessor.MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor.ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator.HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug=aug,
preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator.HDF5DatasetGenerator(config.VAL_HDF5, 64,
preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)


# 编译模型
print("[INFO] compiling model...")
model = ResNet.ResNet.build(64, 64, 3, config.NUM_CLASSES, (3, 4, 6),
(64, 128, 256, 512), reg=0.0005, dataset="tiny_imagenet")
opt = SGD(lr=1e-1, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
metrics=["accuracy"])

# 构建回调列表
callbacks = [TrainingMonitor.TrainingMonitor(config.FIG_PATH, jsonPath=config.JSON_PATH)]


# 训练网络
# train the network
model.fit_generator(
trainGen.generator(),
steps_per_epoch=trainGen.numImages // 64,
validation_data=valGen.generator(),
validation_steps=valGen.numImages // 64,
epochs=50,
max_queue_size=64 * 2,
callbacks=callbacks, verbose=1)


# 序列化网络到磁盘中
print("[INFO] save model...")
model.save(args["model"])


# 关闭数据库
trainGen.close()
valGen.close()





























