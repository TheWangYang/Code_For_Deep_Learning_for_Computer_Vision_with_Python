import sys
sys.path.append(r"E:\pythonNeedSoftware\PyCharmWorkPlace\Deep_learning_For_CV_With_Python")

# 导入需要的库
from pyimagesearch.nn.conv import LeNet
from keras.utils import plot_model

# 初始化LeNet并将网络结构写入到磁盘中
model = LeNet.LeNet.build(28,28,1,10)
plot_model(model, to_file="lenet.png",show_shapes=True)








