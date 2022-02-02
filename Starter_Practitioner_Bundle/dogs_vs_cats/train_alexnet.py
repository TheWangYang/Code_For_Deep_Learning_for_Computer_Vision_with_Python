import sys
import os
sys.path.append(os.path.abspath('..'))

# 导入需要的库
import matplotlib
matplotlib.use("Agg")

from dogs_vs_cats.config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import PatchPreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import AlexNet
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import json

# 构建命令行参数
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
horizontal_flip=True, fill_mode="nearest")


# 为数据集加载RGB均值
means = json.loads(open(config.DATASET_MEAN).read())

# 初始化图片预处理器
sp = SimplePreprocessor.SimplePreprocessor(227, 227)
pp = PatchPreprocessor.PatchPreprocessor(227, 227)
mp = MeanPreprocessor.MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor.ImageToArrayPreprocessor()

# 初始化训练和验证集生成器
trainGen = HDF5DatasetGenerator.HDF5DatasetGenerator(config.TRAIN_HDF5, 128, aug=aug,
preprocessors=[pp, mp, iap], classes=2)

valGen = HDF5DatasetGenerator.HDF5DatasetGenerator(config.VAL_HDF5, 128,
preprocessors=[sp, mp, iap], classes=2)


# 初始化网络架构
print("[INFO]compiling model...")
model = AlexNet.AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", metrics=["accuracy"])

# construct the set of callbacks
# path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
# callbacks = [TrainingMonitor.TrainingMonitor(path)]


# 训练网络
model.fit(
trainGen.generator(),
steps_per_epoch=trainGen.numImages // 128,
validation_data=valGen.generator(),
validation_steps=valGen.numImages // 128,
epochs=75,
max_queue_size=128 * 2, verbose=1)


# 保存模型
print("[INFO] save model...")
model.save(config.MODEL_PATH, overwrite=True)


# 关闭HDF5数据集
trainGen.close()
valGen.close()













