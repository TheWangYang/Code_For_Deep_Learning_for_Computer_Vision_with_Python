# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from pyimagesearch.utils.captchahelper import preprocess
from imutils import contours
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images")
ap.add_argument("-m", "--model", required=True, help="path to input model")
args = vars(ap.parse_args())


# 加载预训练网络
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])


# 随机得到几张图片
imagePaths = list(paths.list_images(args["input"]))
imagePaths = np.random.choice(imagePaths, size=(10, ), replace=False)

# 循环得到所有的图片
for imagePath in imagePaths:
    # 加载图片然后转换为灰度图像，然后填充图片，保证输入的像素和CNN匹配。
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)


    # 然后使用阈值图像显示数字
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)



# 找到轮廓在图像中，保存四个最高的，然后从左到右排序
cnts = cv2.findContours(thresh.coty(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reversed=True)[:4]
cnts = contours.sort_contours(cnts)[0]


# 初始化输出图像作为灰度图像，有三通道，并输出预测结果
output = cv2.merge([gray] * 3)
predictions = []


# 循环轮廓
for c in cnts:
    # 计算box对于轮廓，然后得到数字
    (x, y, w, h) = cv2.boundingRect(c)
    roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]


    # 预处理ROI并分类
    roi = preprocess(roi, 28, 28)
    roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
    pred = model.predict(roi).argmax(axis=1)[0] + 1
    predictions.append(str(pred))

    # 绘制预测结果在输出图像上
    cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
    cv2.putText(output,str(pred), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)



# 展示输出图片
print("[INFO] captcha: {}".format("".join(predictions)))
cv2.imshow("Output", output)
cv2.waitKey()














