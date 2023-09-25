# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:37:29 2023

@author: Tiago
"""

import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate

x_train=np.load("X_train_regression1.npy") #shape -> (15,10)
y_train=np.load("y_train_regression1.npy") #shape -> (15,1)

#Centralized Data
#x_train=x_train - np.mean(x_train,axis=0) #axis=0 -> por coluna

#Normalized Data
#x_train= (x_train - np.min(x_train,axis=0)) / (np.max(x_train,axis=0) - np.min(x_train,axis=0))


n_examples,n_features=x_train.shape
column_of_ones=np.ones((n_examples,1)) #shape (15,1)

X_train=np.hstack((column_of_ones,x_train)) #shape -> (15,11)

X_train = x_train
# Closed form
# Beta=np.dot(np.dot(np.linalg.inv(np.dot(X_train.T,X_train)),X_train.T),y_train) # (11,1)
# y_pred=np.dot(X_train,Beta)


#Linear predictor
regr = LinearRegression()
regr.fit(X_train,y_train)

scores=(-1)*cross_validate(regr, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_squared_error'))["test_neg_mean_squared_error"]

alphas = [0.01, 0.1, 1, 10, 100, 1000]

best_mean_score=np.mean(scores)
best_method="linear"
best_alpha=0

print(best_alpha,best_method,best_mean_score)

#Lasso and Ridge
for alpha in alphas:
    for method in ["ridge","lasso"]:
        if method=="ridge":
            model=Ridge(alpha)
        else:
            model=Lasso(alpha)
        model.fit(X_train,y_train)
        scores=(-1)*cross_validate(model, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_squared_error'))["test_neg_mean_squared_error"]
        score_mean=np.mean(scores)
        
        # print(best_alpha,best_method,best_mean_score)
        print(alpha,method,score_mean)


        if score_mean<best_mean_score:
            best_mean_score=score_mean
            best_method=method
            best_alpha=alpha
    
#Best method
print("the best method is %s regression with an alpha of %s and a mean score of %s" % (
    best_method,
    best_alpha,
    best_mean_score))





#Computing the final predictions
model=Lasso(alpha=0.1) 

model.fit(X_train,y_train)
print(model.coef_.shape) 


x_test=np.load("X_test_regression1.npy") #shape -> (15,10)
n_examples,_=x_test.shape
column_of_ones=np.ones((n_examples,1)) #shape (15,1)
X_test=np.hstack((column_of_ones,x_test)) #shape -> (15,11)
y_test=model.predict(x_test).reshape(n_examples,1) #->(1000,1)

    
    