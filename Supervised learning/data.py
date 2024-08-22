import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
读取数据并合并
"""


f_s = 1.28 * 1000
sample_0 = pd.read_csv("./Supervised learning/data/train/0.csv")
sample_1 = pd.read_csv("./Supervised learning/data/train/1.csv")
sample_2 = pd.read_csv("./Supervised learning/data/train/2.csv")
sample_3 = pd.read_csv("./Supervised learning/data/train/3.csv")
sample_4 = pd.read_csv("./Supervised learning/data/train/4.csv")
# 这里pandas的合并更好因为它可以把列数不同的dataframe合并成一个dataframe,缺失值变成nan
sample = pd.concat([sample_0, sample_1, sample_2, sample_3, sample_4], axis=1)
# 数据
data_values = np.array(sample)
data_labels = np.concatenate(
    (
        np.array([0] * sample_0.shape[1]),
        np.array([1] * sample_1.shape[1]),
        np.array([2] * sample_2.shape[1]),
        np.array([3] * sample_3.shape[1]),
        np.array([4] * sample_4.shape[1]),
    )
)

if __name__ == "__main__":
    print(data_values.shape)
