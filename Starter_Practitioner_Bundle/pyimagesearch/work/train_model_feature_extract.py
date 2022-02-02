# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py


# 创建命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="path HDF5 database")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())


# 打开HDF5文件，得到训练和测试集的分开index索引，前提是需要在hdf5格式化的时候将数据打乱shuffle
db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)


# 训练logistic逻辑回归分类器
print("[INFO] tuning hyperparameters...")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=args["jobs"])
model.fit(db["features"][:i], db["labels"][:i])  # 训练分类器
print("[INFO] best hyperparameters :{}".format(model.best_params_))


# 评价模型
print("[INFO] evaluating...")
preds = model.predict(db["features"][i:])  # 预测最后的分类结果
print(classification_report(db["labels"][i:], preds, target_names=db["label_names"]))


# 最后将logistic模型保存到磁盘上，然后关闭数据
print("[INFO] saving model...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model.best_estimator_))  # 保存分类器模型的最佳估算
f.close()
db.close()
















