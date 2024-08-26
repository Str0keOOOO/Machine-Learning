import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data import data_values, data_labels, f_s
from Process import get_all_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data_features = get_all_features(data_values, f_s, 30, 3)

"""
分割数据集
"""


x_train, x_test, y_train, y_test = train_test_split(
    data_features,
    data_labels,
    test_size=0.4,
    shuffle=True,
    stratify=data_labels,
    random_state=1,
)
# x_train,x_test,y_train,y_test=train_test_split(data,label.ravel().astype(int))

"""
训练

逻辑回归,贝叶斯,KMeans,决策树,随机森林,支持向量机,xgboost,lightgbm,BP神经网络,线性判别
"""


LR = LogisticRegression(solver="liblinear")
LR.fit(x_train, y_train)
print(f"逻辑回归训练集得分:{LR.score(x_train,y_train)}")
print(f"逻辑回归测试集得分:{LR.score(x_test,y_test)}")

MNB = MultinomialNB()
MNB.fit(x_train, y_train)
print(f"朴素贝叶斯训练集得分:{MNB.score(x_train,y_train)}")
print(f"朴素贝叶斯测试集得分:{MNB.score(x_test,y_test)}")

BNB = BernoulliNB()
BNB.fit(x_train, y_train)
print(f"伯努利贝叶斯训练集得分:{BNB.score(x_train,y_train)}")
print(f"伯努利贝叶斯测试集得分:{BNB.score(x_test,y_test)}")

GNB = GaussianNB()
GNB.fit(x_train, y_train)
print(f"高斯贝叶斯训练集得分:{GNB.score(x_train,y_train)}")
print(f"高斯贝叶斯测试集得分:{GNB.score(x_test,y_test)}")

KNN = KNeighborsClassifier()
KNN.fit(x_train, y_train)
print(f"KNN训练集得分:{KNN.score(x_train,y_train)}")
print(f"KNN测试集得分:{KNN.score(x_test,y_test)}")

DTC = DecisionTreeClassifier()
DTC.fit(x_train, y_train)
print(f"决策树训练集得分:{DTC.score(x_train,y_train)}")
print(f"决策树测试集得分:{DTC.score(x_test,y_test)}")

RFC = RandomForestClassifier(n_estimators=1000)
RFC.fit(x_train, y_train)
print(f"随机森林训练集得分:{RFC.score(x_train,y_train)}")
print(f"随机森林测试集得分:{RFC.score(x_test,y_test)}")

SM = svm.SVC(kernel="rbf")
SM.fit(x_train, y_train)
print(f"支持向量机训练集得分:{SM.score(x_train,y_train)}")
print(f"支持向量机测试集得分:{SM.score(x_test,y_test)}")

XGB = XGBClassifier()
XGB.fit(x_train, y_train)
print(f"xgboost训练集得分:{XGB.score(x_train,y_train)}")
print(f"xgboost测试集得分:{XGB.score(x_test,y_test)}")

LGBM = LGBMClassifier(verbose=-1)
LGBM.fit(x_train, y_train)
print(f"lightgbm训练集得分:{LGBM.score(x_train,y_train)}")
print(f"lightgbm测试集得分:{LGBM.score(x_test,y_test)}")

BP = MLPClassifier(max_iter=10000, random_state=62)
BP.fit(x_train, y_train)
print(f"BP神经网络训练集得分:{BP.score(x_train,y_train)}")
print(f"BP神经网络测试集得分:{BP.score(x_test,y_test)}")

LDA = LinearDiscriminantAnalysis()
LDA.fit(x_train, y_train)
print(f"线性判别训练集得分:{LDA.score(x_train,y_train)}")
print(f"线性判别测试集得分:{LDA.score(x_test,y_test)}")

"""
预测

随机森林,lightgbm
"""

predict_data_values = pd.read_csv("./Supervised learning/data/test.csv")
predict_data_values = np.array(predict_data_values)
predict_data_features = get_all_features(predict_data_values, f_s, 30, 3)

predict_data_labels_RFC = RFC.predict(predict_data_features)
predict_data_labels_LGBM = LGBM.predict(predict_data_features)
print(f"随机森林预测为{predict_data_labels_RFC}")
print(f"Lightgbm预测为{predict_data_labels_LGBM}")


"""画图"""
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
# 特征贡献
x = np.linspace(1, len(RFC.feature_importances_), len(RFC.feature_importances_))
plt.figure(1, figsize=(4, 3))
plt.title("特征贡献")
plt.bar(x, RFC.feature_importances_, color="gray")
# # 决策树可视化
plt.figure(2, figsize=(12, 8))
plot_tree(DTC, fontsize=10)
plt.show()
# pairplot
plt.figure(3)
data = np.concatenate((data_features[:, :12], data_labels.reshape(-1, 1)), axis=1)
data = pd.DataFrame(data)  # 没有设置index
data_len = data.shape[1] - 1
kind_dict = {
    0: "健康",
    1: "内圈故障",
    2: "外圈故障",
    3: "滚动体故障",
    4: "保持架故障",
}
data[data_len] = data[data_len].map(kind_dict)
sns.pairplot(data, hue=data_len)
plt.savefig("./Supervised learning/pairplot.png", dpi=300)
