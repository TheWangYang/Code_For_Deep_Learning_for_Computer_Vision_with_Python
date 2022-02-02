import sys
sys.path.append(r"E:\pythonNeedSoftware\PyCharmWorkPlace\Starter_Practitioner_Bundle")

# import the necessary packages
from dogs_vs_cats.config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os


# 得到图像路径
trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[1].split(".")[0] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# 从训练集中进行分层抽样，从训练数据中构建22 #测试分割
split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES, stratify=trainLabels, random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

# 构建验证集
split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_VAL_IMAGES, stratify=trainLabels, random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split


# 构建一个列表，将训练、验证和测试36 #的图像路径与其对应的标签和输出HDF5 37 #文件配对
datasets = [
("train", trainPaths, trainLabels, config.TRAIN_HDF5),
("val", valPaths, valLabels, config.VAL_HDF5),
("test", testPaths, testLabels, config.TEST_HDF5)]

# 初始化图像预处理器和RGB通道
aap = AspectAwarePreprocessor.AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

# 使用for循环构建数据集
for (dType, paths, labels, outputPath) in datasets:
    # 创建HDF5数据集
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter.HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)

    # 初始化进度条
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # 加载图像
        image = cv2.imread(path)
        image = aap.preprocess(image)

        # 如果我们正在构建训练数据集，那么计算图像中每个通道的值，然后更新各自的列表
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # 增加图像和标签
        writer.add([image], [label])
        pbar.update(i)

    # 关闭writer
    pbar.finish()
    writer.close()


# construct a dictionary of averages, then serialize the means to a
# JSON file
print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()









