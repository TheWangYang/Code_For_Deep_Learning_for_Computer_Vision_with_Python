# import the necessary packages
from imutils import paths
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
help="path to input directory of images")
ap.add_argument("-a", "--annot", required=True,
help="path to output directory of annotations")
args = vars(ap.parse_args())


# grab the image paths then initialize the dictionary of character
# counts
imagePaths = list(paths.list_images(args["input"]))
counts = {}


# 循环图片的输入路径
for (i, imagePath) in enumerate(imagePaths):
    # display an update to the user
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))

    try:
        # load the image and convert it to grayscale, then pad the
        # image to ensure digits caught on the border of the image
        # are retained
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # 在图像中找到轮廓并找到其中四个最大的
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

        # 对于四个轮廓，通过计算边界框来得到数字值
        for c in cnts:
            # 计算数字的边界框，然后提取数字
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y - 5:y + h + 5, x - 5 : x + w + 5]

            # 显示字符，使其足够大，以便我们看到，然后等待按键
            cv2.imshow("ROI", imutils.resize(roi, width=28))
            key = cv2.waitKey()

            # 如果'键被按下，那么忽略当前的数字
            if key == ord("'"):
                print("[INFO] ignoring character")
                continue

            # 获取按下的键并构造路，得到输出目录
            key = chr(key).upper()
            dirPath = os.path.sep.join([args["annot"], key])

            # 如果输出目录不存在，请创建该目录
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            # 将标签字符写入文件
            count = counts.get(key, 1)

            p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])

            cv2.imwrite(p, roi)

            # 增加当前键的计数
            counts[key] = count + 1

    except KeyboardInterrupt:
        print("[INFO] manually leaving script")
        break

    # 此特定图像出现未知错误
    except:
        print("[INFO] skipping image for some particular reasons...")












