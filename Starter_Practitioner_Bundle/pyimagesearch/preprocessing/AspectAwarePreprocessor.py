# 导入需要的库
import imutils
import cv2

class AspectAwarePreprocessor:
    def __init__(self,width,height,inter=cv2.INTER_AREA):
        # 存储目标图片宽度，高度和重置大小的插入方法
        self.width = width
        self.height = height
        self.inter = inter

    # 定义的预处理函数
    def preprocess(self,image):
        # 获得图像尺寸，然后初始化增量以用于裁剪
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        # 如果宽度小于高度，那么先按照宽度裁剪
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)

        # 否则先按照高度裁剪
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        # 重新得到图片的长宽
        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]

        # 最后使用cv2库重新设置宽高，为了保证输入时的尺寸固定
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)





