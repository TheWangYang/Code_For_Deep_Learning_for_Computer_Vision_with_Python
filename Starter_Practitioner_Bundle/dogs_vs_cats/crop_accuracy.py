import sys
import os
sys.path.append(os.path.abspath(".."))

from dogs_vs_cats.config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.preprocessing import CropPreprocessor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.utils.ranked import rank5_accuracy
from tensorflow.python.keras.models import load_model
import numpy as np
import progressbar
import json


# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor.SimplePreprocessor(227, 227)
mp = MeanPreprocessor.MeanPreprocessor(means["R"], means["G"], means["B"])
cp = CropPreprocessor.CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor.ImageToArrayPreprocessor()

# load the pretrained network
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# 初始化测试集生成器并在测试集合上进行预测
print("[INFO] predicting on the test data (no corps)...")
testGen = HDF5DatasetGenerator.HDF5DatasetGenerator(config.TEST_HDF5, 64,
preprocessors=[sp, mp, iap], classes=2)
predictions = model.predict_generator(testGen.generator(),
steps=testGen.numImages // 64, max_queue_size=64 * 2)


# 只计算rank1的准确率
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()


# 重新初始化测试集生成器，本次去除sp
testGen = HDF5DatasetGenerator.HDF5DatasetGenerator(config.TEST_HDF5, 64,
preprocessors=[mp], classes=2)
predictions = []

# 初始化进度条
widgets = ["Evaluating: ", progressbar.Percentage(), " ",
progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=testGen.numImages // 64,
widgets=widgets).start()

# 对测试数据单个遍历进行循环
for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):
    # 循环遍历单个图片
    for image in images:
        # 应用crop技术在每个图片上，然后对每个采样得到的局部图片从图片转换为数组
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops], dtype="float32")

        # 在这些crops上进行预测，然后求平均值作为最后的结果
        pred = model.predict(crops)
        predictions.append(pred.mean(axis=0))

    # update the progress bar
    pbar.update(i)


# compute the rank-1 accuracy
pbar.finish()
print("[INFO] predicting on test data (with crops)...")
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()
































