# 导入需要的库
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import argparse
import glob
import os


# 构建命令行参数列表
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True, help="path to models directory")
args = vars(ap.parse_args())


# 加载数据集，然后缩放到0,1范围内
(testX, testY) = cifar10.load_data()[1]
testX = testX.astype("float") / 255.0
# 初始化数据集标签，并将Y标签从整数转为向量形式
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck"]
lb = LabelBinarizer()
testY = lb.fit_transform(testY)


# 收集模型路径，并初始化模型列表
modelPaths = os.path.sep.join([args["models"], "*.model"])
modelPaths = list(glob.glob(modelPaths))
models = []


# 从磁盘加载每一个模型,添加到model list列表中
for (i, modelPath) in enumerate(modelPaths):
    print("[INFO] loading model {}/{}".format(i + 1, len(modelPaths)))
    models.append(load_model(modelPath))


# 初始化预测列表
print("[INFO] evaluating ensemble...")
predictions = []


# 循环所有的模型
for model in models:
    # 使用列表中的每一个模型对testX数据集进行预测
    predictions.append(model.predict(testX, batch_size=64))

# 平均所有的模型预测结果，并做出来报告分析
predictions = np.average(predictions, axis=0)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=labelNames))











