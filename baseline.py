#! /usr/bin/python
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("train.csv")
df.drop("Id", axis=1, inplace=True)
data = pd.get_dummies(df)
X, Y = data.iloc[:,1:].values, data["Score"].values
n = int(X.shape[0] * 0.8)

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
