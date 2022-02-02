import sys
import os
sys.path.append(os.path.abspath(".."))

# 导入需要的库
import matplotlib
matplotlib.use("Agg")


# 导入需要的库
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os



# 解析命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory")
ap.add_argument("-m", "--models", required=True, help="path to output models directory")
ap.add_argument("-n", "--num-models", type=int, default=5, help="# of models to train")
args = vars(ap.parse_args())


# 加载数据集，并将Y预测缩放到0,1范围内
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# 将整形转换为向量形式
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# 初始化了数据集中的类别标签
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck"]


# 构建数据增强器
aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
height_shift_range=0.1, horizontal_flip=True,
fill_mode="nearest")


 # 循环number个数量的MiniVGG模型进行训练
for i in np.arange(0, args["num_models"]):
    # 初始化优化器和模型
    print("[INFO] training model {}/{}".format(i + 1,
    args["num_models"]))
    opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9,
    nesterov=True)
    model = MiniVGGNet.MiniVGGNet.build(width=32, height=32, depth=3,
    classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

    # 训练网络
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
    validation_data = (testX, testY), epochs=40,
    steps_per_epoch = len(trainX) // 64, verbose=1)
    # 向磁盘中保存模型
    p = [args["models"], "model_{}.model".format(i)]
    model.save(os.path.sep.join(p))

    # 评价每一个网络
    predictions = model.predict(testX, batch_size=64)
    report = classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names = labelNames)

    # 保存训练报告
    p = [args["output"], "model_{}.txt".format(i)]
    f = open(os.path.sep.join(p), "w")
    f.write(report)
    f.close()

    # 绘制训练和验证中的loss和accuracy曲线
    p = [args["output"], "model_{}.png".format(i)]
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 40), H.history["loss"], label = "train_loss")
    plt.plot(np.arange(0, 40), H.history["val_loss"], label = "val_loss")
    plt.plot(np.arange(0, 40), H.history["acc"], label = "train_acc")
    plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy for model {}".format(i))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(os.path.sep.join(p))
    plt.close()























