from pyimagesearch.nn import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets


# 开始训练MINIST数据集
print("[INFO]  loading  MNIST  (sample)  dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())  # 归一化处理
print("[INFO]  samples:  {},  dim:  {}".format(data.shape[0], data.shape[1]))

# 将数据集进行分分类，75%作为训练集，25%作为测试集
(trainX,  testX,  trainY,  testY)  =  train_test_split(data, digits.target,  test_size=0.25)

# 使用one-hot编码将对应的标签进行向量化
trainY  =  LabelBinarizer().fit_transform(trainY)
testY  =  LabelBinarizer().fit_transform(testY)


# 训练网络
print("[INFO]  training  network...")
nn  =  NeuralNetwork.NeuralNetwork([trainX.shape[1], 32, 16, 10])
print("[INFO]  {}".format(nn))
nn.fit(trainX,  trainY,  epochs=1000)


# 测试网络
print("[INFO]  evaluating  network...")
predictions  =  nn.predict(testX)
predictions  =  predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1),  predictions))







