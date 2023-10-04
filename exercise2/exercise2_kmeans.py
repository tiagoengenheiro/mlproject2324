# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:37:29 2023

@authors: Tiago e João
"""

import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import cross_validate
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

X_train=np.load("X_train_regression2.npy") #shape -> (100, 4)
y_train=np.load("y_train_regression2.npy") #shape -> (100, 1)

kmeans = KMeans(n_clusters=2)

kmeans.fit(X_train)
labels = kmeans.labels_

X_train1 = X_train[labels == 0, :]
X_train2 = X_train[labels == 1, :]

y_train1 = y_train[labels == 0, :]
y_train2 = y_train[labels == 1, :]

print(X_train1.shape)
print(X_train2.shape)

print(y_train1.shape)
print(y_train2.shape)


best_mean_score = 1000
best_mean_score = 1000
#Alphas used for both Ridge and Lasso Regression models
alphas = [0.001,0.01, 0.1, 1, 10, 100, 1000]

X_train = X_train1
y_train = y_train1

regr = LinearRegression()
regr.fit(X_train,y_train)

#Running crossvalidation on the regular Linear Regression
scores=(-1)*cross_validate(regr, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_squared_error'))["test_neg_mean_squared_error"]
score_mean=np.mean(scores) #mean of the MSE for 5 folds

if score_mean<best_mean_score:
    best_mean_score=score_mean
    best_model="linear"
    best_alpha = 0

#Defining a cycle to get the best model among the models: Linear, Lasso and Ridge
for alpha in alphas:
    for model in ["ridge","lasso"]:
        if model=="ridge":
            model=Ridge(alpha)
        else:
            model=Lasso(alpha)
        model.fit(X_train,y_train)
        scores=(-1)*cross_validate(model, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_squared_error'))["test_neg_mean_squared_error"]
        score_mean=np.mean(scores)
        
        if score_mean<best_mean_score:
            best_mean_score=score_mean
            best_model=model
            best_alpha=alpha
    

#Best model - the one with the least mean MSE from crossvalidation with k=5
print("The best model is %s regression with an alpha of %s and a mean score of %s" % (
    best_model,
    best_alpha,
    best_mean_score))


X_train = X_train2
y_train = y_train2

regr = LinearRegression()
regr.fit(X_train,y_train)

#Running crossvalidation on the regular Linear Regression
scores=(-1)*cross_validate(regr, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_squared_error'))["test_neg_mean_squared_error"]
score_mean=np.mean(scores) #mean of the MSE for 5 folds

if score_mean<best_mean_score:
    best_mean_score=score_mean
    best_model="linear"
    best_alpha = 0

#Defining a cycle to get the best model among the models: Linear, Lasso and Ridge
for alpha in alphas:
    for model in ["ridge","lasso"]:
        if model=="ridge":
            model=Ridge(alpha)
        else:
            model=Lasso(alpha)
        model.fit(X_train,y_train)
        scores=(-1)*cross_validate(model, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_squared_error'))["test_neg_mean_squared_error"]
        score_mean=np.mean(scores)
    
        if score_mean<best_mean_score:
            best_mean_score=score_mean
            best_model=model
            best_alpha=alpha
    

#Best model - the one with the least mean MSE from crossvalidation with k=5
print("The best model is %s regression with an alpha of %s and a mean score of %s" % (
    best_model,
    best_alpha,
    best_mean_score))

