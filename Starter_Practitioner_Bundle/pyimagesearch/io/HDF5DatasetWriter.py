# 导入需要的库
import h5py
import os


# 创建的将图片保存为hdf5格式的helper
class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):

        # 检查output路径是否存在，如果存在那么报错如下
        if os.path.exists(outputPath):
            raise ValueError("The supplied‘outputPath‘ already exists and cannot be overwritten. Manually delete "
        "the file before continuing.", outputPath)

        # 打开hdf5文件创建两个数据集:
        # 一个用来存储图像或特征， 另外一个用来存储类别标签
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")

        # 存储到缓冲区, 然后根据数据集中的index索引初始化缓冲区
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        # 增加行和标签到缓冲区中
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)
        # 检查是否缓冲区的存储需要冲洗到磁盘中
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # 将缓冲区中的数据写到磁盘，然后清理缓冲区
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):
        # 创造一个数据集存储真实的标签名称，然后存储类标签
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names", (len(classLabels), ), dtype=dt)
        labelSet[:] = classLabels


    def close(self):
        # 将缓冲区中的所有数据都冲洗到磁盘中
        if len(self.buffer["data"]) > 0:
            self.flush()

        self.db.close()














