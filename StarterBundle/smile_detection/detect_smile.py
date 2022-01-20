# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2


# 构建命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())


# 加载检测器和模型
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])


# 如果视频资源没有提供，那么就使用电脑摄像头
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# 否则，加载video
else:
    camera = cv2.VideoCapture(args["video"])


# 持续循环，得到视频中的帧
while True:
    # 得到当前帧
    (grabbed, frame) = camera.read()

    # 如果使用的是视频资源，那么直接读取一次帧即可，然后跳出循环
    if args.get("video") and not grabbed:
        break

    # 重新缩放帧到合适大小，然后复制帧，用于标记和放入分类信息
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    # 在帧中检测人脸, 然后克隆帧
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # 在bounding box中得到RoI区域
    for (fX, fY, fW, fH) in rects:
        # 得到RoI，并缩放RoI到合适尺寸，便于作为输入到CNN中
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # 确定笑或不笑的概率，从而得到分类标签
        (notSmiling, smiling) = model.predict(roi)[0]
        label = "Smiling" if smiling > notSmiling else "Not Smiling"

        # 在输出帧中标定bounding box，然后输出分类信息
        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    # 展示图片
    cv2.imshow("Face", frameClone)

    # 如果q键被按下，停止循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 清除camera并关闭任何打开的窗口
camera.release()
cv2.destroyAllWindows()



