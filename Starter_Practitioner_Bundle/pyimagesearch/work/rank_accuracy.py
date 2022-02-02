import sys
import os
sys.path.append(os.path.abspath(".."))


# 导入需要的库
from pyimagesearch.utils.ranked import rank5_accuracy
import argparse
import pickle
import h5py


# 构建命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="path HDF5 database")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
args = vars(ap.parse_args())



# 加载预训练模型
print("[INFO] loading pre-trained model...")
model = pickle.loads(open(args["model"], "rb").read())


# 打开HDF5数据集读取数据
db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)



# 最后计算rank1和rank5精度
print("[INFO] predicting...")
preds = model.predict_proba(db["features"][i:])
(rank1, rank5) = rank5_accuracy(preds, db["labels"][i:])

# 显示两种精确度
print("[INFO] rank-1:{:.2f}%".format(rank1 * 100))
print("[INFO] rank-5:{:.2f}%".format(rank5 * 100))


# 关闭数据库
db.close()






