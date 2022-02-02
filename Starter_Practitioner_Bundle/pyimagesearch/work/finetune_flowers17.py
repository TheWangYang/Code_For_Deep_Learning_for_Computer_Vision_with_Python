import sys
import os
sys.path.append(os.path.abspath(".."))

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor, SimpleDatasetLoader
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.nn.conv import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import argparse


# 构建命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True,  help="path to output model")
args = vars(ap.parse_args())

# 构建数据增强器
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
horizontal_flip=True, fill_mode="nearest")

# 抓取图片列表，从路径中得到图片的标签
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]


# 初始化图片预处理器
aap = AspectAwarePreprocessor.AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor.ImageToArrayPreprocessor()


# 从磁盘加载数据集然后缩放原始像素到0,1之间
sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0


# 分割数据集，75%作为训练集，25%作为验证集
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)


# 将y预测结果从整形转换为向量形式
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# 加载VGG16网络，确保VGG16的FC层不包含进去
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))


# 初始化新的FC层，然后跟随着一个softmax分类器
headModel = FCHeadNet.FCHeadNet.build(baseModel, len(classNames), 256)


# 将头部FC模型放在基础模型的顶部——这将成为我们将要训练的实际模型
model = Model(inputs=baseModel.input, outputs=headModel)

# 使用for循环给baseModel的每一层设置权重不改变
for layer in baseModel.layers:
    layer.trainable = False


# 给新组成的model进行预热
print("[INFO] computing model...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 使用一些epochs训练新设置的FC层
print("[INFO] training head...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), epochs=25,
                    steps_per_epoch=len(trainX) // 32, verbose=1)


# 初始化之后评价网络
print("[INFO] evaluating after initialization..")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# 结合CONV的其他层继续训练
for layer in baseModel.layers[15:]:
    layer.trainable = True



# 重新编译模型，使得模型修改生效
print("[INFO] re-compiling model...")
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# 重新训练模型，对CONV层和FC层进行微调
print("[INFO] fine-tuning model...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                    validation_data=(testX, testY), epochs=100,
                    steps_per_epoch=len(trainX) // 32, verbose=1)

# 评价微调之后的模型
print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=classNames))

# 向磁盘中保存模型
print("[INFO] serializing model...")
model.save(args["model"])


