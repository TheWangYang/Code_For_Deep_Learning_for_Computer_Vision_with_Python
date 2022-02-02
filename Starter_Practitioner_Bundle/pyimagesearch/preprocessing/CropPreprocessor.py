# import the necessary packages
import numpy as np
import cv2


class CropPreprocessor:
    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
        # store the target image width, height, whether or not
        # horizontal flips should be included, along with the
        # interpolation method used when resizing
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    def preprocess(self, image):
        # 初始化作物列表
        crops = []

        # 获取图像的宽度和高度，然后使用这些尺寸来定义图像的角
        (h, w) = image.shape[:2]

        coords = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h]]

        # 计算图像的中心裁剪
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))

        coords.append([dW, dH, w - dW, h - dH])

        # 使用循环得到图片的作物裁剪
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX: endX]
            crop = cv2.resize(crop, (self.width, self.height), interpolation=self.inter)
            crops.append(crop)


        if self.horiz:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)


        return np.array(crops)


    



