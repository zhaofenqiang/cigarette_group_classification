#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:32:49 2018

@author: zfq
"""
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn import neighbors

#%%
demographic = pd.read_csv("/media/zfq/本地磁盘/cigarette/sub_info2.csv")
lh_all_area = pd.read_csv("/media/zfq/本地磁盘/cigarette/aparc_a2009s__stats_lh_all_area.csv",index_col='lh.aparc.a2009s.area')
rh_all_area = pd.read_csv("/media/zfq/本地磁盘/cigarette/aparc_a2009s__stats_rh_all_area.csv",index_col='rh.aparc.a2009s.area')

allSub = demographic['ID'].tolist()
allLabel = demographic['Group'].tolist()

trainIndex = []
testIndex = []
for i in range(113):
    if i % 4 == 0:
        testIndex = np.append(testIndex,i)
    else:
        trainIndex = np.append(trainIndex,i)
 
#%% Train
trainX = []
trainY = []
for i in range(len(trainIndex)):
    sub = allSub[trainIndex[i].astype(np.int32)]
    y = allLabel[trainIndex[i].astype(np.int32)]
#    if y == 2:
#        y = 1
#    if y == 3:
#        y = 0
    lh = lh_all_area.loc[sub,:].values
    rh = rh_all_area.loc[sub,:].values
#    x = np.append(lh,rh)
    x = lh
    trainX = np.concatenate((trainX, x), axis=0)  
    trainY = np.append(trainY, y)    
trainX = trainX.reshape(len(trainIndex),74)/6000.0

knn1 = neighbors.KNeighborsClassifier(5, 'uniform')
knn2 = neighbors.KNeighborsClassifier(5, 'distance')

C = 1.0
rbfsvm = svm.SVC(kernel='rbf', class_weight={0: 1.7})
linearsvm = svm.SVC(kernel='linear', C=C)
polysvm =  svm.SVC(kernel='poly', degree=3, C=C)

models = knn1
models.fit(trainX, trainY) 

#%% Test
testX = []
testY = []
for i in range(len(testIndex)):
    sub = allSub[testIndex[i].astype(np.int32)]
    y = allLabel[testIndex[i].astype(np.int32)]
#    if y == 2:
#        y = 1
#    if y == 3:
#        y = 0
    lh = lh_all_area.loc[sub,:].values
    rh = rh_all_area.loc[sub,:].values
#    x = np.append(lh,rh)
    x = lh
    testX = np.concatenate((testX, x), axis=0)  
    testY = np.append(testY, y)    
testX = testX.reshape(len(testIndex),74)/6000.0

Z = models.predict(testX)
print(Z)
acc = float(np.sum(Z == testY))/len(Z)
print(acc)
