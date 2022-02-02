# 导入需要的库
from keras.applications import VGG16
import argparse


# 构建命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include-top", type=int, default=1, help="whether or not to include top of CNN")
args = vars(ap.parse_args())

# load the VGG16 network
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=args["include_top"] > 0)


print("[INFO] showing layers...")
# for循环显示网络中的每一层
for (i, layer) in enumerate(model.layers):
    print("[INFO] {}\t{}".format(i, layer.__class__.__name__))




