# 导入需要的库
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras import backend as K



class MiniGoogLeNet:
    @ staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
        # 定义一个卷积层到批量归一化到激活层
        x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)

        return x

    @ staticmethod
    def inception_module(x, numK1x1, numK3x3, chanDim):
        # 定义两个conv模块，然后在相同维度进行连接
        conv_1x1 = MiniGoogLeNet.conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
        conv_3x3 = MiniGoogLeNet.conv_module(x, numK3x3, 3, 3, (1, 1), chanDim)
        x = concatenate([conv_1x1, conv_3x3], axis=chanDim)
        return x


    @ staticmethod
    def downsample_module(x, K, chanDim):
        # 定义卷积层和池化层，然后在通道维度上连接起来
        conv_3x3 = MiniGoogLeNet.conv_module(x, K, 3, 3, (2, 2), chanDim, padding="valid")
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=chanDim)
        return x


    @staticmethod
    def build(width, height, depth, classes):
        # 初始化
        inputShape = (height, width, depth)
        chanDim = -1


        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)
            chanDim = 1



        # 定义模型输入和第一个conv层
        inputs = Input(shape=inputShape)
        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chanDim)


        # 堆叠两个inception模块和一个下采样模块
        x = MiniGoogLeNet.inception_module(x, 32, 32, chanDim)
        x = MiniGoogLeNet.inception_module(x, 32, 48, chanDim)
        x = MiniGoogLeNet.downsample_module(x, 80, chanDim)

        # 四个inception模块之后跟着一个下采样模块
        x = MiniGoogLeNet.inception_module(x, 112, 48, chanDim)
        x = MiniGoogLeNet.inception_module(x, 96, 64, chanDim)
        x = MiniGoogLeNet.inception_module(x, 80, 80, chanDim)
        x = MiniGoogLeNet.inception_module(x, 48, 96, chanDim)
        x = MiniGoogLeNet.downsample_module(x, 96, chanDim)

        # 两个inception模块之后跟着一个平均池化层，和一个dropout层
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        # softmax 分类器
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        # 创建模型
        model = Model(inputs, x, name="googlenet")

        # 返回创建好的模型
        return model






















