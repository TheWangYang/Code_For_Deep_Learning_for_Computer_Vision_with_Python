from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse


# 设置的解析命令行语句的方法
# ——dataset:输入图像数据集在磁盘上的路径。
# ——neighbors:可选，使用k- nn算法时需要应用的邻居数k。
# ——jobs:可选，计算输入数据点与训练集之间的距离时要运行的并发作业数量。-1的值将使用处理器上所有可用的内核。

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",  required=True, help="path  to  input  dataset")
ap.add_argument("-k", "--neighbors", type=int,  default=1, help="#  of  nearest  neighbors  for  classification")
ap.add_argument("-j", "--jobs", type=int,  default=-1, help="#  of  jobs  for  k-NN  distance  (-1  uses  all  available  cores)")
args = vars(ap.parse_args())


# 现在进行图像的加载和预处理
print("[INFO] loading images...")

imagePaths = list(paths.list_images(args["dataset"]))

# 初始化图像处理器和从硬盘中加载数据集
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))


# 显示一些关于图像内存消耗的信息
print("[INFO]  features  matrix:  {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))


# 将标签编码为整数
le = LabelEncoder()
labels = le.fit_transform(labels)

# 将数据集分为训练和测试部分，使用75%作为训练集，使用25%作为测试集
(trainX,  testX,  trainY,  testY) = train_test_split(data,  labels, test_size=0.25,random_state=42)


# 然后评估我们的kNN分类器
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY,  model.predict(testX), target_names=le.classes_))





