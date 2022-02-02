
# 导入需要的包
import numpy as np


def rank5_accuracy(preds, labels):
    # 初始化rank1和rank5精确度
    rank1 = 0
    rank5 = 0

    # 循环预测和ground_truth
    for(p, gt) in zip(preds, labels):
        # 降序排序
        p = np.argsort(p)[::-1]

        # 检查是否和groud-truth框重合在top5之间
        if gt in p[:5]:
            rank5 += 1

        if gt == p[0]:
            rank1 += 1

        # 计算最终的rank-1和rank-5准确率
        rank1 /= float(len(labels))
        rank5 /= float(len(labels))

        # 返回结果
        return rank1, rank5





