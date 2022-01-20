# 导入需要的库
import imutils
import cv2

def preprocess(image, width, height):
    # 获得图像尺寸，然后初始化填充值
    (h, w) = image.shape[:2]

    # 如果宽度大于高度，那么按照宽度调整大小
    if w > h:
        image = imutils.resize(image, width=height)

    # 否则，如果高度大于宽度那么按照高度进行大小调整
    else:
        image = imutils.resize(image, height=height)

    # 确定高度和宽度需要的填充
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)


    # 填充图像，然后应用更多的resize来解决四舍五入问题
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # 返回一个预处理图片
    return image











