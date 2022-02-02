# 导入需要的库
import argparse
import requests
import time
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory of images")
ap.add_argument("-n", "--num-images", type=int, default=500, help="# of images to download")
args = vars(ap.parse_args())

# 初始化URL,网络无法访问（因为是国外网址）
url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 0

# 循环下载图片
for i in range(0, args["num_images"]):
    try:
        # 尝试下载一个图片
        r = requests.get(url, timeout=60, verify=False)
        # 保存图片到本地目录中
        p = os.path.sep.join([args["output"], "{}.jpg".format(str(total).zfill(5))])
        f = open(p, "wb")
        f.write(r.content)
        f.close()

        # 更新计数器
        print("[INFO] download:{}".format(p))
        total += 1

    except Exception as e:
        print(repr(e))

    # 插入一个sleep线程为了更好的服务
    time.sleep(0.1)






