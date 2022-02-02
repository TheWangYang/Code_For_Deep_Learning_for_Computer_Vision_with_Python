# 导入需要的库
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from pyimagesearch.nn.conv import LeNet
from pyimagesearch.utils.captchahelper import preprocess
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())


# 初始化数据和标签
data = []
labels = []


# 循环输入的图像
for imagePath in paths.list_images(args["dataset"]):
    # 加载图像，预处理图片，并存放到data list中
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, 28, 28)
    image = img_to_array(image)
    data.append(image)

    # 提取类标签从路径中，更新标签list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)


# 缩放像素到0,1之间
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# 分割数据集
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# 将标签从整数转换为向量
lb = LabelBinarizer().fit(trainY)
trainY = lb.transform(trainY)
testY = lb.transform(testY)



# 初始化优化器和LeNet网络
print("[INFO] compiling model ...")

model = LeNet.LeNet.build(width=28, height=28, depth=1,classes=9)
opt = SGD(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# 训练网络
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=15, verbose=1)


# 评价网络
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.arfmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# 保存模型到磁盘中
print("[INFO] save model into disk")

model.save(args["model"])


# 绘制训练和test过程中的loss 和 acc
plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["acc"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()












