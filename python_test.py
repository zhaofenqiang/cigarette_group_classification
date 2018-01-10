#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:15:54 2018

@author: zfq
"""

from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, y)
clf = SGDClassifier(loss="log").fit(X, y)
clf.predict_proba([[1., 1.]])  

trainX = np.array([[0,0],[0,1],[0,2],[1,1],[1,2],[1,-1],[2,0],[3,0],[2,-1],[3,1]])
trainY = np.array([0,0,0,0,0,1,1,1,1,1])

C = 1.0
models = svm.SVC(kernel='linear', gamma=0.7, C=C)
#(svm.SVC(kernel='linear', C=C),
#          svm.LinearSVC(C=C),
#          svm.SVC(kernel='poly', degree=3, C=C))
models = models.fit(trainX, trainY) 

testX = [[0.5,0.5],[2,2],[2,-0.5],[2.5,1]]
testY = [0,0,1,1]

Z = models.predict(testX)
Z = models.decision_function(testX) # get the signed distance to the hyperplane
print(Z)

models.support_vectors_
models.support_
models.n_support_

plt.scatter(trainX[:, 0], trainX[:, 1], c=trainY, s=50, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = models.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(models.support_vectors_[:, 0], models.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none')
plt.show()

#%%
rng = np.random.RandomState(0)
n_samples_1 = 1000
n_samples_2 = 100
X = np.r_[1.5 * rng.randn(n_samples_1, 2),
          0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
y = [0] * (n_samples_1) + [1] * (n_samples_2)


xx = np.linspace(-1, 5, 10)
