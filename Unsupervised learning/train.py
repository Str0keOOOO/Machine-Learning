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

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

# Subplot 1
scatter = axes[0].scatter(
    data_trained_pca[:, 0],
    data_trained_pca[:, 1],
    c=labels,
    cmap="viridis",
    edgecolor="k",
    alpha=0.7,
)
scatter = axes[0].scatter(
    data_tested_pca[:, 0],
    data_tested_pca[:, 1],
    c=labels_tested,
    cmap="viridis",
    edgecolor="k",
    alpha=0.7,
)
axes[0].set_title("K-means Clustering of PCA-transformed data_trained and data_tested")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
axes[0].legend(*scatter.legend_elements(), title="Clusters")
axes[0].grid()

# Subplot 2
scatter = axes[1].scatter(
    data_trained_pca[:, 0],
    data_trained_pca[:, 1],
    c=labels,
    cmap="viridis",
    edgecolor="k",
    alpha=0.7,
)
axes[1].set_title("K-means Clustering of PCA-transformed data_trained")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")
axes[1].legend(*scatter.legend_elements(), title="Clusters")
axes[1].grid()

# Subplot 3
scatter = axes[2].scatter(
    data_tested_pca[:, 0],
    data_tested_pca[:, 1],
    c=labels_tested,
    cmap="viridis",
    edgecolor="k",
    alpha=0.7,
)
axes[2].set_title("K-means Clustering of PCA-transformed data_tested")
axes[2].set_xlabel("Principal Component 1")
axes[2].set_ylabel("Principal Component 2")
axes[2].legend(*scatter.legend_elements(), title="Clusters")
axes[2].grid()

plt.show()


"""
预测
"""


result = np.concatenate(
    (data_tested[:, 0].reshape(-1, 1), labels_tested.reshape(-1, 1)), axis=1
)
result_csv = pd.DataFrame(result, columns=["Run ID", "Label"])
result_csv.to_csv("./Unsupervised learning/predict.csv", index=False)
