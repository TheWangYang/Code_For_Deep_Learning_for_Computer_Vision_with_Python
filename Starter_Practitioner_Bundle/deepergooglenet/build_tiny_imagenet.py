import sys
import os
sys.path.append(os.path.abspath('..'))

# import the necessary packages
from deepergooglenet.config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2

# grab the paths to the training images, then extract the training
# class labels and encode them
trainPaths = list(paths.list_images(config.TRAIN_IMAGES))
trainLabels = [p.split(os.path.sep)[-3] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)


# perform stratified sampling from the training set to construct a
# a testing set
split = train_test_split(trainPaths, trainLabels,
test_size=config.NUM_TEST_IMAGES, stratify=trainLabels, random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split


# load the validation filename => class from file and then use these
# mappings to build the validation paths and label lists
M = open(config.VAL_MAPPINGS).read().strip().split("\n")
M = [r.split("\t")[:2] for r in M]
valPaths = [os.path.sep.join([config.VAL_IMAGES, m[0]]) for m in M]
valLabels = le.transform([m[1] for m in M])


# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5
# files
datasets = [
("train", trainPaths, trainLabels, config.TRAIN_HDF5),
("val", valPaths, valLabels, config.VAL_HDF5),
("test", testPaths, testLabels, config.TEST_HDF5)]


# 我们也将初始化RGB平均值
# initialize the lists of RGB channel averages
(R, G, B) = ([], [], [])

# 最后，我们准备为Tiny ImageNet构建HDF5数据集:
for (dType, paths, labels, outputPath) in datasets:
    # create HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter.HDF5DatasetWriter((len(paths), 64, 64, 3), outputPath)
    # initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    # 循环每个数据集路径进行保存
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # 加载图片
        image = cv2.imread(path)

        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        writer.add([image], [label])
        pbar.update(i)

    pbar.finish()
    writer.close()

# construct a dictionary of averages, then serialize the means to a
# JSON file
print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()



































