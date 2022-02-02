# 导入相关库
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class ShallowNet:
    @ staticmethod
    def build(width, height,depth, classes):
        # 初始化模型，使用输入形状并包含通道顺序
        model = Sequential()
        inputShape = (height, width, depth)

        # 如果使用：通道在第一个位置那么使用下面的代码
        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)


        # 定义网络的第一个conv（卷积层）-> ReLu层
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))

        # 设置softmax分类器
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # 返回构建好的网络
        return model

