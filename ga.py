#! /usr/bin/python
import random, copy, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from deap import base
from deap import creator
from deap import tools
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from MyGA import *

df = pd.read_csv("train.csv")
df.drop("Id", axis=1, inplace=True)
data = pd.get_dummies(df)
X, Y = data.iloc[:,1:].values, data["Score"].values
allY = {'Y':Y, 'log':np.log(Y + 1), 'sqrt':np.sqrt(Y), 'sq':Y**2}
RANGE_Y = range(0, 71)
n = int(X.shape[0] * 0.8)
ntrain, ntest = n, X.shape[0] - n

#CUT_BINS = [-1, 5, 10, 20, 75]
CUT_BINS = [-1, 5, 75]
CUT_BINS = [-1, 75]
Y_bins = pd.cut(Y, CUT_BINS)
Y_bins_set = sorted(set(Y_bins))
nbins = len(Y_bins_set)
Y_bins_set2number = Series(range(len(Y_bins_set)), index=Y_bins_set)
Y_bins_number = Y_bins_set2number[Y_bins].values
dtrain = xgb.DMatrix(X[:n], label=Y[:n])
dtest = xgb.DMatrix(X[n:], label=Y[n:])
cls_dtrain = xgb.DMatrix(X[:n], label=Y_bins_number[:n])
cls_dtest = xgb.DMatrix(X[n:], label=Y_bins_number[n:])

gen_max_depth = lambda : random.randint(1, 5)
gen_eta = lambda : random.random() * 0.5
gen_funcs = [gen_max_depth, gen_eta]
def cls_gen_param():
    param = list()
    for func in gen_funcs:
        param.append(func())
    return param

def cls_train(indiv, num_round=1500):
    param = {
    'objective': 'multi:softmax',
    'max_depth': indiv[0],
    'eta': indiv[1],
    'silent': 1,
    'nthread': 4,
    'num_class': nbins}
    watchlist = [(cls_dtrain, 'cls_dtrain'), (cls_dtest, 'cls_dtest')]
    bst = xgb.train(param, cls_dtrain, num_round, watchlist, early_stopping_rounds=150, verbose_eval=50)
    print(indiv, bst.best_iteration, bst.best_ntree_limit, bst.best_score)
    return bst

def cls_evaluate(indiv):
    bst = cls_train(indiv)
    return bst.best_score,

def cls_mut_indiv(indiv, indiv_pb):
    for i, ele in enumerate(indiv):
        if random.random() < indiv_pb:
            indiv[i] = cls_gen_funcs[i]()

cls_bst = cls_train([3, 0.05], 237)
cls_pred = cls_bst.predict(cls_dtest)

sub_dtrain = {}
sub_ntrain = {}
sub_dtest = {}
sub_ntest = {}
for yb_idx, yb in enumerate(Y_bins_set):
    train_sel = Y_bins[:n] == yb
    sub_dtrain[yb] = xgb.DMatrix(X[:n][train_sel], label=Y[:n][train_sel])
    sub_ntrain[yb] = sum(train_sel)
    test_sel = cls_pred == yb_idx
    sub_dtest[yb] = xgb.DMatrix(X[n:][test_sel], label=Y[n:][test_sel])
    sub_ntest[yb] = sum(test_sel)


# best ? ['count:poisson', 3, 0.04375073666234408, 0], 3.656079
gen_objective = lambda : random.choice(['count:poisson'])
gen_objective = lambda : random.choice(['reg:linear', 'count:poisson'])
gen_max_depth = lambda : random.randint(1, 10)
#gen_eta = lambda : random.choice([0.04375073666234408, 0.3464085982132564,0.40045324570183705, 0.645319692228763 ]) #random.random() * 0.3
gen_eta = lambda : random.random() * 1
gen_gamma = lambda : random.randint(0, 10)
gen_funcs = [gen_objective, gen_max_depth, gen_eta, gen_gamma]
num_gen_funcs = len(gen_funcs)
def gen_param():
    param = list()
    for b in range(len(Y_bins_set)):
        for func in gen_funcs:
            param.append(func())
    return param

def train(indiv):
    overall_score = 0.0
    sub_bst = dict()
    for yb_idx, yb in enumerate(Y_bins_set):
        sub_indiv = indiv[(yb_idx * num_gen_funcs):((yb_idx + 1) * num_gen_funcs)]
        param = {'objective': sub_indiv[0],
            'max_depth': sub_indiv[1],
            'eta': sub_indiv[2],
            'gamma': sub_indiv[3],
            'silent': 1,
            'nthread': 4,
            'save_period': 0,
            'eval_metric': 'rmse'}
        num_round = 1502
        watchlist = [(sub_dtrain[yb], 'train'), (sub_dtest[yb], 'eval')]
        bst = xgb.train(param, sub_dtrain[yb], num_round, watchlist, early_stopping_rounds=150, verbose_eval=False)
        # print(yb, bst.best_iteration, bst.best_ntree_limit, bst.best_score)
        overall_score += (bst.best_score ** 2) * sub_ntest[yb]
        print '\t', sub_indiv, yb, bst.best_score, sub_ntest[yb]
        sub_bst[yb] = bst
    overall_score = np.sqrt(overall_score / ntest)
    print overall_score
    return sub_bst, overall_score

def evaluate(indiv):
    sub_bst, overall_score = train(indiv)
    return overall_score,


def mut_indiv(indiv, indiv_pb):
    for i, ele in enumerate(indiv):
        if random.random() < indiv_pb:
            tp = type(indiv[i])
            func_idx = i % num_gen_funcs
            if func_idx == 0:
                indiv[i] = gen_funcs[func_idx]()
            elif func_idx == 1:
                indiv[i] = max(1, indiv[i] + random.choice([1, -1]))
            elif func_idx == 2:
                indiv[i] = min(0.9, indiv[i] * (random.random() + 0.5))
            elif func_idx == 3:
                indiv[i] = max(0, indiv[i] + random.choice([1, -1]))

ga = MyGA(gen_param, evaluate, mut_indiv, CXPB=0.5, MUTPB=0.2)
ga.init_pop(NPOP=30)
ga.iterate(NGEN=10)
