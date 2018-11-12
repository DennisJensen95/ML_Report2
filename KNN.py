#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import math
import numpy as np
from xlrd import open_workbook
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, contour, contourf, cm, colorbar
import sklearn.linear_model as lm
from toolbox_02450 import confmatplot, dbplotf
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# Import data_preproccessing
from data_preProcessing_interval_normalize import *

Y = zscore(X,0)
# Normalize and compute PCA (UNCOMMENT to experiment with PCA preprocessing)
U,S,V = np.linalg.svd(Y,full_matrices=False)
V = V.T
#Components to be included as features
k_pca = 2
X = X @ V[:,0:k_pca]
N, M = X.shape


# K-fold crossvalidation with stratified folds
K = 5
CV_1 = StratifiedKFold(K, shuffle=False)
# J-fold inner
J = 10
CV_2 = StratifiedKFold(J, shuffle=False)


options_knn = [2,3,4,5,6,7,8,9,10]
param_err_knn = np.zeros(len(options_knn))
k=0

y_est_gen = np.zeros(k)
best_error_inner = []
best_param_knn = list()
y_ESTKNN = list()
y_test_outer = list()
err_test_outer = list()
gen_inner_error = list()
best_knn_list = list()
X_test_outer = list()
for train_index, test_index in CV_1.split(X, y):

        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        for train_index_inner, test_index_inner in CV_2.split(X_train, y_train):
            # Fit classifier and classify the test points

            X_train_inner = X_train[train_index_inner]
            y_train_inner = y[train_index_inner]
            X_test_inner = X[test_index_inner]
            y_test_inner = y[test_index_inner]

            err_inner=0
            h = 0
            for o in options_knn:
                knclassifier = KNeighborsClassifier(n_neighbors=o)
                knclassifier.fit(X_train_inner, y_train_inner)
                y_est_inner = knclassifier.predict(X_test_inner)
                err_inner = sum(np.abs(y_est_inner != y_test_inner))
                # print('Computing CV fold: {0} for KNN equal to {1}'.format(k+1,o))
                param_err_knn[h] = err_inner
                h += 1

            best_index = np.argmin(param_err_knn)
            best_param_knn.append(options_knn[best_index])
            best_error_inner.append(param_err_knn[best_index])
            j += 1

        gen_inner_error.append(np.mean(param_err_knn).round(2))
        best_knn = KNeighborsClassifier(n_neighbors=best_param_knn[k])
        best_knn = best_knn.fit(X_train, y_train)
        y_est = best_knn.predict(X_test)
        X_test_outer.append(X_test)
        best_knn_list.append(best_knn)
        mis_classified = np.sum(y_est != y_test)
        relative_error = mis_classified / len(y_test)
        y_ESTKNN.append(y_est)

        y_test_outer.append(y_test)
        err_test_outer.append(relative_error)
        k += 1

print("The generalization error is {0} ".format(np.mean(err_test_outer)))
best_index = np.argmin(err_test_outer)

confmatplot(y_test_outer[best_index], y_ESTKNN[best_index])
show()

figure(1)
def neval(xval):
    return np.argmax(best_knn.predict_proba(xval),1)

if k_pca == 2:
    figure()
    dbplotf(X_test_outer[best_index],y_test_outer[best_index],neval,'auto')
    show()
