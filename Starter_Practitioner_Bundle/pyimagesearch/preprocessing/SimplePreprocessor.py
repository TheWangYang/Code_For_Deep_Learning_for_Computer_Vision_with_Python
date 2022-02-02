import cv2

# 定义的简单的图像处理类
class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self,image):
        return cv2.resize(image,(self.width,self.height),interpolation=self.inter)


