# 导入需要的库
# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import argparse
import pickle
import h5py

# 构建命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="path HDF5 database")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())


# 打开hdf5数据集
db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)

# 设置一些参数，然后得到最佳参数
print("[INFO] tuning hyperparameters...")
params = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=args["jobs"])
model.fit(db["features"][:i], db["labels"][:i])
print("[INFO] best hyperparameters: {}".format(model.best_params_))


# 评价模型
print("[INFO] evaluating...")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds, target_names=db["label_names"]))

# 使用额外的精度计算原始准确度
acc = accuracy_score(db["labels"][i:], preds)
print("[INFO] score: {}".format(acc))


# 保存模型
print("[INFO] saving model...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

# 关闭数据集
db.close()

















