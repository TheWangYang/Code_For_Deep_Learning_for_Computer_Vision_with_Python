import sys
import os
sys.path.append(os.path.abspath(".."))

from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import ShallowNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse


# 解析命令行参数
ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())


# # 获取我们将要描述的图像列表
# print("[INFO] loading images...")
# imagePaths = list(paths.list_images(args["dataset"]))


# 加载cifar_10数据集
# 加载训练数据集和测试数据集，然后缩放到0,1范围内
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0


# # 初始化图片进程
# sp = SimplePreprocessor.SimplePreprocessor(32, 32)
# iap = ImageToArrayPreprocessor.ImageToArrayPreprocessor()
#
# # 从磁盘加载数据集然后将原始像素缩放到0，1范围内
# sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp, iap])
# (data, labels) = sdl.load(imagePaths, verbose=500)
# data = data.astype("float") / 255.0
#
# # 下一步是将我们的数据划分为训练和测试分块，同时将我们的标签编码为向量:
# # 使用数据的75%进行训练，25%用于测试
# (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)


# 将标签从整数转换为向量
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)


# 对数据集初始化标签名字
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
             "dog", "frog", "horse", "ship", "truck"]

# 初始化优化器和模型
print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 训练网络
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=10, verbose=1)

# 保存模型
print("[INFO] serializing network...")
model.save(args["model"])

# # 然后是进行可视化展示loss和accuracy的变化曲线
# # 评价NN
# print("[INFO] evaluating network...")
# predictions = model.predict(testX, batch_size=32)
# print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
#                             target_names=labelNames))
#
# # 绘制训练中的loss损失和精确度
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
# plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
# plt.show()







