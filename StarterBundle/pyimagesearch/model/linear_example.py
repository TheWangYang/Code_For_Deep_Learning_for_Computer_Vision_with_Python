import numpy as np
import cv2

# 初始化类标签并设置伪随机的种子数字生成器，这样我们就可以重现我们的结果
labels = ["dog", "cat", "panda"]
np.random.seed(1)


# 初始化权重矩阵和偏差向量
# 第14行初始化权矩阵W，其值来自均匀分布的随机值，采样范围为[0,1]。这个权重矩阵有3行(每个类标签对应一行)和3072列(32 × 32 × 3图像中的每个像素对应一行)。
# 然后我们在第15行初始化偏置向量——这个向量也被随机填充为分布[0,1]上均匀采样的值。偏差向量有3行(对应于类标签的数量)和1列。

W = np.random.randn(3, 3072)
b = np.random.randn(3)

# 加载图像
# 第19行通过cv2.imread从磁盘加载图像。然后我们在第20行将图像的大小调整为32×32像素(忽略长宽比)——我们的图像现在被表示为(32,32,3)NumPy数组，我们将其平化为3072 -dim向量。
#
orig = cv2.imread("beagle.png")
image = cv2.resize(orig, (32, 32)).flatten()

# 计算分类标签的分数
scores = W.dot(image) + b

# 将每一个评分值写入终端
for (label, score) in zip(labels, scores):
    print("[INFO] {} : {:.2f}".format(label, score))

cv2.putText(orig, "Label : {}".format(labels[np.argmax(scores)]), (10, 30),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.9,(0, 255, 0), 2)

cv2.imshow("Image",orig)

cv2.waitKey(0)

