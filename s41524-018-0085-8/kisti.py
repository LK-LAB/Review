from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


from sbs import *

df = pd.read_csv('selected.csv', low_memory=False)

feat_labels = df.columns[1:-1]

forest = RandomForestClassifier(n_estimators=500, random_state=1)

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# T_sep 설정
# 논문 첫 부분에서 T_sep = 10(K)로 두고 설명
T_sep = 10

# T_sep보다 Tc가 높으면 1, 아니면 0
y = np.where(y > T_sep, 1, 0)

# Dataset에서 85%는 train에, 나머지 15%는 test에 사용하기 위해 분할 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0, stratify=y)

feature_set = [ft for ft in X.columns]

stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

sbs = SBS(forest, k_features=1)
sbs.fit(X_train_std, y_train, feature_set)

k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker="o")
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.savefig("graph.png", dpi=1000)
