import sys
sys.path.append(r"E:\pythonNeedSoftware\PyCharmWorkPlace\Deep_learning_For_CV_With_Python")

# 导入需要的库
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from pyimagesearch.nn.conv import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

# 构建命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

# 初始化图片数据集合标签数组
data = []
labels = []

# 循环输入的图片
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    # 加载图片，预处理图片，并将图片数据加入到data数组中
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)  # 将图片转换为对应的向量
    data.append(image)

    # 从图片的路径中提取图片的分类信息，然后加入到数组中
    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)


# 缩放图片的像素到0,1范围内
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# 将标签从整数转换为向量格式
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# 统计笑 和 不笑数据的占比，然后初始化对应的权重
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals


# 使用20用来测试，80%用来训练
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)


# 初始化模型
print("[INFO] compiling model...")
model = LeNet.LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练网络
print("[INFO] training network...")
# 添加了权重
H = model.fit(trainX, trainY, validation_data=(testX, testY), class_weight=classWeight, batch_size=64, epochs=30, verbose=1)


# 评价网络
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# 向磁盘保存模型
print("[INFO] serializing network...")
model.save(args["model"])


# 绘制loss和accuracy
plt.style.use("ggplot")
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["acc"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()











































