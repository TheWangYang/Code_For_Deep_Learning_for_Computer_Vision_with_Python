import sys
import os
sys.path.append(os.path.abspath('..'))

# 导入需要的库
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

# 创建命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-o", "--output", required=True, help="path to output HDF5 file")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="batch size of images to be passed through network")
ap.add_argument("-s", "--buffer-size", type=int, default=1000, help="size of feature extraction buffer")
args = vars(ap.parse_args())


# 存储batch_size到变量中
bs = args["batch_size"]

# 抓取我们将描述的图像列表，然后随机打乱他们的排序，以方便训练和测试分割通过数组切片
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)


# 从图像路径中提取类标签，然后对类标签进行编码
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)


# 加载ResNet网络
print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=False)  # 去掉网络最后的FC层

# 初始化HDF5数据集写入器，然后存储类标签在数据集中
dataset = HDF5DatasetWriter.HDF5DatasetWriter((len(imagePaths), 2048), args["output"],
                            dataKey="features", bufSize=args["buffer_size"])

dataset.storeClassLabels(le.classes_)

# 初始化进度条
widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()


# 循环图像在构造的数组中，按照每次bs个数据进行循环
for i in np.arange(0, len(imagePaths), bs):
    # 得到当前批次的路径，和类标签，并创建当前的保存图片数组
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []

    # 循环当前批量中的图片对象
    for (j, imagePath) in enumerate(batchPaths):
        # 使用Keras库保证提取的图片大小为特定的224 * 224
        image = load_img(imagePath, target_size=(224, 224))
        # 将图片转换为像素数组
        image = img_to_array(image)

        # 预处理图像：扩展维度；
        image = np.expand_dims(image, axis=0)

        # 从ImageNet数据集减去平均RGB像素强度
        image = imagenet_utils.preprocess_input(image)

        # 增加图片到batchImages数组中
        batchImages.append(image)

    # 将图片传入到cnn中，然后将输出作为实际的特征
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)

    # 重塑特征，使每个图像由'maxpooling2d '输出的一个扁平的特征向量表示
    # 这里的第一维为batch_size的大小
    features = features.reshape((features.shape[0], 2048))

    # 增加特征和标签到hdf5格式文件中
    dataset.add(features, batchLabels)

    # 在progressBar中更新i
    pbar.update(i)


# 关闭数据集
dataset.close()
pbar.finish()














