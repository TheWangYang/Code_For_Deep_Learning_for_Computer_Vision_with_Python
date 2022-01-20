# 导入需要的库
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from keras.models import load_model
from keras.datasets import cifar10
from imutils import paths
import numpy as np
import argparse
import cv2


# 构造命令行参数解析器
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
args = vars(ap.parse_args())

# 对数据集初始化标签名字
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
             "dog", "frog", "horse", "ship", "truck"]


# 抓取图像的列表，然后随机采样10个索引到图像路径列表
print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]


# 初始化图像处理器
sp = SimplePreprocessor.SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor.ImageToArrayPreprocessor()

# 从磁盘加载数据集然后缩放原始像素到0,1之间
sdl = SimpleDatasetLoader.SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0


# 加载预训练网络
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])


# 预测10张图片
print("[INFO] predicting...")
preds = model.predict(data, batch_size=32)

# 循环10张图片的预测
for (i, imagePath) in enumerate(imagePaths):
    # 加载图片，得到预测结果并展示它
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label:{}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)









