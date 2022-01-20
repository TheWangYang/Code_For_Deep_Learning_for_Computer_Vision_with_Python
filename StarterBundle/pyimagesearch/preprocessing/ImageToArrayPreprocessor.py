#  导入需要的库
from keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # 存储图像数据格式
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # 应用keras库函数重新排列图片的维度
        return img_to_array(image, data_format=self.dataFormat)


