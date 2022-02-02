# 导入需要的包
from skimage.exposure import rescale_intensity
import numpy as np
import  argparse
import cv2



# 定义卷积方法
def convolve(image, K):
    # 获取图像和核的空间维度
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    # 为输出分配内存
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    # 循环输入图像，“滑动”内核
    # 每个(x, y)从左到右和从上到下的坐标
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # 通过提取当前(x, y)坐标维度的中心区域，提取图像的ROI
            roi = image[y - pad : y + pad + 1 ,x - pad:x + pad + 1]

            # #执行实际的卷积，在ROI和核之间进行元素的乘法，然后对矩阵求和
            k = (roi * K).sum()

            # #将卷积值存储在输出图像的(x, y)坐标中
            output[y - pad, x - pad] = k


    # 然后，完成输出
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output


# 设置命令行参数，主要设置为输入的图片
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())


# 构造用于平滑图像的平均模糊核
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# 构造一个锐化过滤器
sharpen = np.array((
[0, -1, 0],
[-1, 5, -1],
[0, -1, 0]), dtype="int")


# 构造用于检测图像边缘区域的拉普拉斯核
laplacian = np.array((
[0, 1, 0],
[1, -4, 1],
[0, 1, 0]), dtype="int")


# 构造 Sobelx轴 核
sobelX = np.array((
[-1, 0, 1],
[-2, 0, 2],
[-1, 0, 1]), dtype="int")

# 构造 Sobely轴 核
sobelY = np.array((
[-1, -2, -1],
[0, 0, 0],
[1, 2, 1]), dtype="int")

 # 构造一个emboss核
emboss = np.array((
[-2, -1, 0],
[-1, 1, 1],
[0, 1, 2]), dtype="int")


# 构造内核库，一个我们将用的内核列表
# 同时使用我们自己构造的‘convole‘函数和OpenCV库中的‘filter2D‘函数
kernelBank = (
("small_blur", smallBlur),
("large_blur", largeBlur),
("sharpen", sharpen),
("laplacian", laplacian),
("sobel_x", sobelX),
("sobel_y", sobelY),
("emboss", emboss))


# 下面的code演示了使用上述内核库的方法
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 循环覆盖内核库
for (kernelName, K) in kernelBank:
    # 使用我们自定义的' convolve '函数和OpenCV的' filter2d '函数将内核应用到灰度图像
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)


    # 展示输出的图像
    cv2.imshow("Original", gray)
    cv2.imshow("{} - convole".format(kernelName), convolveOutput)
    cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()









