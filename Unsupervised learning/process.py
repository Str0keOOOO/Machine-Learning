import numpy as np
import pandas as pd


"""
训练集
"""


train_data = pd.read_csv("./Unsupervised learning/data/etch_train.csv")
grouped = train_data.groupby("Run ID")
train_data_split = {key: group for key, group in grouped}
data_trained = np.ndarray((len(train_data_split.keys()), 39))
feature_count = 19
for i, (key, value) in enumerate(train_data_split.items()):
    ar1 = np.empty((0, feature_count))
    ar2 = np.empty((0, feature_count))
    data_trained[i, 0] = key
    for ii in range(value.shape[0]):
        if value.iloc[ii, 2] == 4:
            ar1 = np.concatenate((ar1, np.array(value.iloc[ii, 3:]).reshape(1, -1)))
        else:
            ar2 = np.concatenate((ar2, np.array(value.iloc[ii, 3:]).reshape(1, -1)))
    ar1_mean = np.mean(ar1, axis=0)
    ar2_mean = np.mean(ar2, axis=0)
    data_trained[i, 1:] = np.append(ar1_mean, ar2_mean)


"""
测试集
"""


test_data = pd.read_csv("./Unsupervised learning/data/etch_test.csv")
grouped = test_data.groupby("Run ID")
test_data_split = {key: group for key, group in grouped}
data_tested = np.ndarray((len(test_data_split.keys()), 39))
feature_count = 19
for i, (key, value) in enumerate(test_data_split.items()):
    ar1 = np.empty((0, feature_count))
    ar2 = np.empty((0, feature_count))
    data_tested[i, 0] = key
    for ii in range(value.shape[0]):
        if value.iloc[ii, 2] == 4:
            ar1 = np.concatenate((ar1, np.array(value.iloc[ii, 3:]).reshape(1, -1)))
        else:
            ar2 = np.concatenate((ar2, np.array(value.iloc[ii, 3:]).reshape(1, -1)))
    ar1_mean = np.mean(ar1, axis=0)
    ar2_mean = np.mean(ar2, axis=0)
    data_tested[i, 1:] = np.append(ar1_mean, ar2_mean)


if __name__ == "__main__":
    for key, sub_df in train_data_split.items():
        print(f"DataFrame for {key}:")
        print(sub_df)
        print("\n")
    print(data_trained.shape)
    for key, sub_df in test_data_split.items():
        print(f"DataFrame for {key}:")
        print(sub_df)
        print("\n")
    print(data_tested.shape)
