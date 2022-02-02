# 导入需要的库
from keras.utils import np_utils
import numpy as np
import h5py


class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None,
                 aug=None, binarize=True, classes=2):
        # 存储以上参数
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes


        # 打开HDF5数据库，确定数据总数
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]


    # 定义生成器函数
    def generator(self, passes=np.inf):
        # 初始化epoch变量
        epochs = 0

        # 持续循环
        while epochs < passes:
            for i in np.arange(0, self.numImages, self.batchSize):
                # 循环HDF5数据集
                # 从HDF5数据集中提取图片和标签
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]

                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)
                # 查看是否需要使用图片预处理进行处理
                if self.preprocessors is not None:
                    # 初始化预处理程序的图片列表
                    procImages = []

                    # 循环图片
                    for image in images:
                        # 应用每一个预处理程序到每一个图像中
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        # 更新预处理列表
                        procImages.append(image)
                    # 更新图片数组为预处理之后的数组
                    images = np.array(procImages)

                # 判断是否需要对图片数据进行增强
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels, batch_size
                                                          =self.batchSize))
                # 将处理之后的图片和标签生成一个二元组
                yield(images, labels)

            # 增加epochs
            epochs += 1

    # 设置关闭函数
    def close(self):
        self.db.close()



