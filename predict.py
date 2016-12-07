#! /usr/bin/python
import pandas as pd
import xgboost as xgb

df = pd.read_csv("train.csv")
df.drop("Id", axis=1, inplace=True)
data = pd.get_dummies(df)
X, Y = data.iloc[:,1:].values, data["Score"].values
print('train shape', X.shape)

dtrain = xgb.DMatrix(X, label=Y)
param = {'objective': 'count:poisson',
        'max_depth': 3,
        'eta': 0.04375073666234408,
        'gamma': 0,
        'silent': 1,
        'nthread': 4,
        'save_period': 0,
        'eval_metric': 'rmse'}
num_round = 1191
bst = xgb.train(param, dtrain, num_round)

df = pd.read_csv('test.csv')
Id = df['Id']
df.drop("Id", axis=1, inplace=True)
X = pd.get_dummies(df).values
print('test shape', X.shape)
dtest = xgb.DMatrix(X)
Y = bst.predict(dtest)

res = pd.DataFrame({}, index=Id)
res['Score'] = Y
res.to_csv('1601214449_predict.csv', header=True, index=True, sep=',')
