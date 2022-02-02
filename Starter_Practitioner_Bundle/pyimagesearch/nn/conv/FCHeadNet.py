from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense


# 设置一个自定义的FC全连接层
class FCHeadNet:
    @staticmethod
    def build(baseModel, classes, D):
        # 初始化头部模型，它将被放置在基础之上，然后添加一个FC层
        headModel = baseModel.output  # 得到基础网络
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)

        # 增加一个softmax层
        headModel = Dense(classes, activation="softmax")(headModel)

        # 返回模型
        return headModel


