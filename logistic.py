#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from toolbox_02450 import confmatplot, dbplotf
from sklearn.model_selection import StratifiedKFold
from scipy.stats import zscore

from data_preProcessing_interval_normalize import *

Y = zscore(X,0)
# Normalize and compute PCA (UNCOMMENT to experiment with PCA preprocessing)
U,S,V = np.linalg.svd(Y,full_matrices=False)
V = V.T
#Components to be included as features
k_pca = 8
X = X @ V[:,0:k_pca]
N, M = X.shape

# K-fold crossvalidation with stratified folds
K = 5
CV_1 = StratifiedKFold(K, shuffle=False)
# J-fold crossvalidation with stratified folds
J = 3
CV_2 = StratifiedKFold(J, shuffle=False)

# train models
n_train = [0.001, 0.01, 0.1, 1, 10, 100]
# n_train = ["9", "10", "11", "12", "13", "14"]

Error_test = np.empty(K)
clf_list = list()
k=0

err_test = np.ones(J)*np.inf
clf_para = list()
clf_parameter_list = list()
clf_best_para = list()
err_inner_loop = []
err_para_loop = []
gen_err = np.zeros(K)*np.nan
best_est = []
y_test_outer = []
x_test_outer = []
for train_index, test_index in CV_1.split(X, y):
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]
    j = 0
    for train_index_2, test_index_2 in CV_2.split(X_train, y_train):
        X_train_2 = X_train[train_index_2, :]
        y_train_2 = y_train[train_index_2]
        X_test_2 = X_train[test_index_2, :]
        y_test_2 = y_train[test_index_2]

        for i in n_train:
            # extract training and test set for current CV fold
            logit_classifier = lm.LogisticRegression(C=i)
            logit_classifier.fit(X_train_2, y_train_2)
            Y_est = logit_classifier.predict(X_test_2)
            mis_classified = np.sum(logit_classifier.predict(X_train_2) != y_train_2)
            relative_error = mis_classified / len(y_train_2)
            clf_parameter_list.append(10+i)
            err_para_loop.append(relative_error)

        #Best performing model from innner loop
        best_index = np.argmin(err_para_loop)
        clf_best_para.append(clf_parameter_list[best_index])
        err_inner_loop.append(err_para_loop[best_index])
        j+=1

    #Refit the best performing model
    best_index = np.argmin(err_inner_loop)
    logit_classifier = lm.LogisticRegression(C=clf_best_para[best_index])
    logit_classifier.fit(X_train, y_train)
    y_est = logit_classifier.predict(X_test)
    y_test_outer.append(y_test)
    x_test_outer.append(X_test)
    clf_list.append(logit_classifier)
    best_est.append(y_est)
    mis_classified = np.sum(y_est != y_test)
    relative_error = mis_classified / len(y_test)
    gen_err[k] = relative_error
    k += 1
# show confusion matrix for model training using K-fold cross validation
best_index = gen_err.argmin()
figure()
confmatplot(best_est[best_index], y_test_outer[best_index])
print("The mean generalization error is {0}".format(np.mean(gen_err)))

#
#
#
# #show confusion matrix for model testing using Leave one out cross validation
# figure(1)
# Y_est2 = clf_list[best_index].predict(X_test)
# confmatplot(Y_test,Y_est2)
# show()
#
# Decision boundaries for the multinomial regression model

def nevallog(xval):
    return np.argmax(clf_list[best_index].predict_proba(xval),1)

if k_pca == 2:
    figure()
    dbplotf(x_test_outer[best_index], y_test_outer[best_index] ,nevallog,'auto')
    show()

