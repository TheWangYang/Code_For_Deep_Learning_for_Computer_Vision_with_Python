# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse


def step_decay(epoch):
    # 初始化基础的学习率lr，下降因子，和每一个epoch下降的数
    initAlpha = 0.01
    factor  = 0.25
    dropEvery = 5


    # 对当前的epoch计算学习速率
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))


    # 返回学习率
    return float(alpha)








