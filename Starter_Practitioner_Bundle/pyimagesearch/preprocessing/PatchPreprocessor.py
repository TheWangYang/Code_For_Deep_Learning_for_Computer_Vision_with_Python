# 导入需要的库
from sklearn.feature_extraction.image import extract_patches_2d



class PatchPreprocessor:
    def __init__(self, width, height):
        # 存储图片的目标宽度和高度
        self.width = width
        self.height = height

    def preprocess(self, image):
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]


