import pandas as pd
import random
import numpy as np
from copy import deepcopy

import os, pickle, sys

import warnings
warnings.filterwarnings("ignore")
from UnbiasedGBM import UnbiasedBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

data = pd.read_csv(f'./data/train_x.csv')
label = pd.read_csv(f'./data/train_y.csv')

data.rename(columns={c:f'feat{i}' for i,c in enumerate(data.columns)}, inplace=True)

cat_feats = [c for c in data.select_dtypes(exclude=np.number).columns]
categorical_indicator = [(c in cat_feats) for c in data.columns]

from utils import prepare_data
data = prepare_data(data, categorical_indicator, 'advance2.5')

X_tv, X_test, y_tv, y_test = train_test_split(data, label, test_size=0.33, random_state=42, stratify=label)
X_train, X_valid, y_train, y_valid = train_test_split(X_tv, y_tv, test_size=0.5, random_state=42, stratify=y_tv)

X_train = X_train.reset_index(drop=True)
X_valid = X_valid.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)[y_train.columns[0]]
y_valid = y_valid.reset_index(drop=True)[y_valid.columns[0]]
y_test = y_test.reset_index(drop=True)[y_test.columns[0]]


import optuna
study = optuna.load_study(study_name="tmp", storage=f"sqlite:///data/tune_ugb.db")
params = study.best_params

metric = lambda x, y : roc_auc_score(x,y)

model = UnbiasedBoost('logloss', params['n_est'], params['min_leaf'], params['thresh'], 3, params['lr'], 1)
score, pred = model.fit(X_train, y_train, testset=(metric, X_test, y_test.values), mono_h=1, large_leaf_reward=params['power'], score_type='advance')

pred = model.predict(X_test)
print('UnbiasedGBM auc score:', score)
print(metric(y_test.values, pred))
print(model.calc_self_imp())

# baseline --------------------------

if True:
    study = optuna.load_study(study_name="tmp", storage=f"sqlite:///data/tune_lgb.db")
    params = study.best_params
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=params['n_est'],
        min_child_weight=params['min_leaf'],
        min_split_gain=abs(params['thresh']),
        learning_rate=params['lr'],
        n_jobs=1,
        seed=1,
    )
    model.fit(X_train, y_train, verbose=False)
    pred = model.predict_proba(X_test)[:,1]
    score = metric(y_test.values, pred)
    print('baseline auc score:',score)