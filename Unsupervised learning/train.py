import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from process import data_trained, data_tested


"""
PCA降维,K-means
"""


scaler = StandardScaler()
data_trained_scaled = scaler.fit_transform(data_trained[:, 1:])

pca = PCA(n_components=2)
data_trained_pca = pca.fit_transform(data_trained_scaled)

print("贡献率:", pca.explained_variance_ratio_)

kmeans = KMeans(n_clusters=1, random_state=42)
kmeans.fit(data_trained_scaled)
labels = kmeans.labels_

scaler_tested = StandardScaler()
data_tested_scaled = scaler_tested.fit_transform(data_tested[:, 1:])

pca_tested = PCA(n_components=2)
data_tested_pca = pca_tested.fit_transform(data_tested_scaled)

print("测试集贡献率:", pca_tested.explained_variance_ratio_)

kmeans_tested = KMeans(n_clusters=2, random_state=42)
kmeans_tested.fit(data_tested_scaled)
labels_tested = kmeans_tested.labels_


"""
画图
"""


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
scatter = plt.scatter(
    data_trained_pca[:, 0],
    data_trained_pca[:, 1],
    c=labels,
    cmap="viridis",
    edgecolor="k",
    alpha=0.7,
)
scatter = plt.scatter(
    data_tested_pca[:, 0],
    data_tested_pca[:, 1],
    c=labels_tested,
    cmap="viridis",
    edgecolor="k",
    alpha=0.7,
)
plt.title("K-means Clustering of PCA-transformed data_trained and data_tested")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.grid(True)
plt.subplot(1, 2, 2)
scatter = plt.scatter(
    data_tested_pca[:, 0],
    data_tested_pca[:, 1],
    c=labels_tested,
    cmap="viridis",
    edgecolor="k",
    alpha=0.7,
)
plt.title("K-means Clustering of PCA-transformed data_trained")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.grid()
plt.show()


"""
预测
"""


result = np.concatenate(
    (data_tested[:, 0].reshape(-1, 1), labels_tested.reshape(-1, 1)), axis=1
)
result_csv = pd.DataFrame(result, columns=["Run ID", "Label"])
result_csv.to_csv("./Unsupervised learning/predict.csv", index=False)
