# 导入需要的库
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse



# 解析命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-o", "--output", required=True, help="path to output directory to store augmentation examples")
ap.add_argument("-p", "--prefix", type=str, default="image", help="output filename prefix")
args = vars(ap.parse_args())

# 加载图像，转换为一个Numpy数组，增加一个额外的维度
print("[INFO] loading example image...")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# 构造用于数据增强的图像生成器，然后初始化到目前为止生成的图像总数
aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
total = 0

# 构建一个真实的python生成器
print("[INFO] generating images...")
imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
                    save_prefix=args["prefix"],save_format=".jpg")

# 循环我们的图像数据增强生成器中的示例
for image in imageGen:
    # 增加计数器
    total += 1

    # 如果已经达到了10个样例，那么跳出循环
    if total == 10:
        break







