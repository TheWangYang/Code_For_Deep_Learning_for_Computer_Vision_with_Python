# import the necessary packages
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to the input image")
ap.add_argument("-model", "--model", type=str, default="vgg16",help="name of pre-trained network to use")
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
"vgg16": VGG16,
"vgg19": VGG19,
"inception": InceptionV3,
"xception": Xception, # TensorFlow ONLY
"resnet": ResNet50
}

# ensure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should "
    "be a key in the‘MODELS‘ dictionary")


# 初始化输入图像的形状及预处理函数（可能会根据选择的模型进行改变）
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# 如果我们是用Inception V3或者Xception，那么改变输入的图片大小
if args["model"] in ("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input


# 从磁盘加载预训练模型权重，如果是第一次，那么需要下载
print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")


# 加载图像，并使用Keras库的helper助手，保证输入图像和inputShape保持一致
print("[INFO] loading and pre-processing image...")
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)

# 调整输入图像的形式
image = np.expand_dims(image, axis=0)

# 使用恰当的预处理函数处理图片
image = preprocess(image)

# 分类图像
print("[INFO] classifying image with {}...".format(args["model"]))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

# 循环所有的预测，并显示排名前5的预测
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:2f}%".format(i + 1, label, prob * 100))


# 通过OpenCV加载图像，在图像上绘制顶部的预测和显示图像到我们的屏幕
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}", format(label), (10, 30),cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2)

cv2.imshow("Classification", orig)
cv2.waitKey(0)


















