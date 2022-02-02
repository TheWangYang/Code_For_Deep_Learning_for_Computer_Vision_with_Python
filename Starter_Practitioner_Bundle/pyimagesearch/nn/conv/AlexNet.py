# 导入需要库
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import backend as K


class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        # 初始化模型
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1


        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)
            chanDim = -1

        # 定义网路中的第一个层
        model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=inputShape, padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        # Block #2: second CONV => RELU => POOL layer set
        model.add(Conv2D(256, (5, 5), padding="same",
        kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #3: CONV => RELU => CONV => RELU => CONV => RELU
        model.add(Conv2D(384, (3, 3), padding="same",
        kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(384, (3, 3), padding="same",
        kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same",
        kernel_regularizer = l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #4: first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # Block #5: second set of FC => RELU layers
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))
        # return the constructed network architecture
        return model



























