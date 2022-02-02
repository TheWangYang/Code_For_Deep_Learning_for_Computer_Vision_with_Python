# 导入库
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K



# 建立网络架构
class MiniVGGNet:
    @ staticmethod
    def build(width, height, depth, classes):
        # 初始化模型设置输入格式
        model = Sequential()
        inputShape = (height, width, depth)
        # 设置通道尺寸
        chanDim = -1

        # 对通道有限的像素输入实现泛化
        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)
            chanDim = 1 # 当通道首先出现的时候，设置BN归一化的位置在index = 1处


        # 定义第一层
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 定义第二层
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 第一个全连接层
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # 使用softmax分类器
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

    





