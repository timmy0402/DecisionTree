import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

train_data = pd.read_csv("train.csv")


# node design
class Node:
    def __init__(
        self,
        feature,
        data,
        threshold,
        left,
        right,
        isRoot,
        isLeaf,
        sampleSize,
        purity,
        values,
        label,
    ):
        self.feature = feature
        self.data = data
        self.threshold = threshold
        self.left = left
        self.right = right
        self.isRoot = isRoot
        self.isLeaf = isLeaf
        self.sampleSize = sampleSize
        self.purity = purity
        self.values = values
        self.label = label


def get_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
