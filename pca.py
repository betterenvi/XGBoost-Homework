#! /usr/bin/python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("train.csv", index_col="Id")
data = pd.get_dummies(df)
X, Y = data.iloc[:,17:].values, data["Score"].values
n = int(X.shape[0] * 0.8)

xx = data.iloc[:,1:17].values
pca = PCA(10)
xx = pca.fit_transform(xx)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
print(xx.shape, X.shape)
X = np.hstack((xx, X))
print(X.shape)

dtrain = xgb.DMatrix(X[:n], label=Y[:n])
dtest = xgb.DMatrix(X[n:], label=Y[n:])
param = {'objective': 'reg:linear',
        'max_depth': 2,
        'eta': 0.1,
        'gamma': 0.1,
        'silent': 1,
        'nthread': 4,
        'save_period': 0}
num_round = 1000
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, watchlist)
