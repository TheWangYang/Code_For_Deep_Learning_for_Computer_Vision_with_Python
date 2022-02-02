# 导入需要的库
import cv2


class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        # 在一个训练集上存储红、绿、蓝通道的平均值
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean

    def preprocess(self, image):
        # 将图片分为各自的红 绿 蓝通通道
        (B, G, R) = cv2.split(image.astype("float32"))

        # 减去每个通道的平均值
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        # 返回图像
        return cv2.merge([B, G, R])


    
