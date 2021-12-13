#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 10:09:30 2021

@author: april
"""

### 1.lightgbm

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, roc_auc_score

my_data = pd.read_csv('first.txt', sep='\t')
my_target = np.array(my_data['label'])
my_data = np.array(my_data.drop(['label'], 1))

exter_data = pd.read_csv('second.txt', sep='\t')
exter_target = np.array(exter_data['label'])
exter_data = np.array(exter_data.drop(['label'], 1))

# split dataset
seed = 268
X = my_data
y = my_target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.33)
lgb_train = lgb.Dataset(X_tr, y_tr)

# In LightGBM, the validation data should be aligned with training data.
lgb_eval = lgb.Dataset(X_te, y_te, reference=lgb_train)
ext_lgb_eval = lgb.Dataset(exter_data, exter_target, reference=lgb_train)

params = {}
params['objective'] = 'binary' # Two categories
params['metric'] = {'binary_logloss', 'auc'} 
params['num_threads'] = 4 
params['verbosity'] = 20
params['early_stopping_round'] = 5
clf = lgb.train(params, lgb_train,  valid_sets=[lgb_eval])

# prediction on the internal dataset
y_pred = clf.predict(X_te, num_iteration=clf.best_iteration)
#clf.save_model('model.txt')
threshold = 0.5
pred_result = []
for mypred in y_pred:
    if mypred > threshold:
        pred_result.append(1)
    else:
        pred_result.append(0)
pred_result = np.array(pred_result)
print(np.sum(pred_result == y_te)/(y_te.shape))
gbm_score = accuracy_score(y_te, pred_result) 
print('Accuracy:', gbm_score)
gbm_auc = roc_auc_score(y_te, pred_result) #gbm_auc值
print('gbm_auc:', gbm_auc)

# prediction on the external set
ext_pred = clf.predict(exter_data, num_iteration=clf.best_iteration)
threshold = 0.5
ext_result = []
for mypred in ext_pred:
    if mypred > threshold:
        ext_result.append(1)
    else:
        ext_result.append(0)
ext_result = np.array(ext_result)
print(np.sum(ext_result == exter_target)/(exter_target.shape))


### 2.XGboost
from xgboost import XGBClassifier

# split dataset
seed = 2635
test_size = 0.33
X = my_data
y = my_target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

##可视化测试集的loss
model = XGBClassifier()
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", 
          eval_set=eval_set, verbose=False)

# prediction on the internal dataset
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
xbm_auc=roc_auc_score(y_te, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0)) 

# prediction on the external set
exter_pred = model.predict(exter_data)
predictions = [round(value) for value in exter_pred]

accuracy = accuracy_score(exter_target, predictions)
ext_xbm_auc=roc_auc_score(exter_target, ext_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


### 3.GCForest
from GCForest import gcForest
import joblib

# split dataset
seed = 1234
X = my_data
y = my_target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.33)

gcf = gcForest(shape_1X=28, window=9, tolerance=0.0,min_samples_mgs=10, min_samples_cascade=7)
gcf.fit(X_tr, y_tr)

##### prediction on the test set
y_pred = gcf.predict(X_te)
gc_score = accuracy_score(y_te, y_pred)
gc_auc = roc_auc_score(y_te, y_pred) 
    
#prediction on the exter set
ext_pred = gcf.predict(exter_data)
ext_gc_score = accuracy_score(exter_target, ext_pred)
ext_gc_auc = roc_auc_score(exter_target, ext_pred) 
