# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:37:29 2023

@author: Tiago
"""

import numpy as np
import random 
from sklearn.linear_model import LinearRegression,Lasso,Ridge,RidgeCV,LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,KFold

x_train=np.load("X_train_regression1.npy") #shape -> (15,10)
y_train=np.load("y_train_regression1.npy") #shape -> (15,1)


n_examples,n_features=x_train.shape
column_of_ones=np.ones((n_examples,1)) #shape (15,1)

X_train=np.hstack((column_of_ones,x_train)) #shape -> (15,11)


Beta=np.dot(np.dot(np.linalg.inv(np.dot(X_train.T,X_train)),X_train.T),y_train) # (11,1)

y_pred=np.dot(X_train,Beta)



regr = LinearRegression()

regr.fit(X_train,y_train)



scores=-1*cross_val_score(regr, X_train,y_train,cv=5,scoring="neg_mean_squared_error")
alphas = [0.01, 0.1, 1, 10, 100, 1000]

best_mean_score=np.mean(scores)
best_method="linear"
best_alpha=0

print(best_alpha,best_method,best_mean_score,np.std(scores))


for alpha in alphas:
    for method in ["ridge","lasso"]:
        if method=="ridge":
            model=Ridge(alpha)
        else:
            model=Lasso(alpha)
        model.fit(x_train,y_train)
        scores=-1*cross_val_score(model,x_train,y_train,cv=5,scoring="neg_mean_squared_error")
        score_mean=np.mean(scores)
        score_std=np.std(scores)
        print(alpha,method,score_mean,score_std)
        if score_mean<best_mean_score:
            best_mean_score=score_mean
            best_method=method
            best_alpha=alpha
    

print("the best method is %s regression with an alpha of %s and a mean score of %s" % (
    best_method,
    best_alpha,
    best_mean_score))

# def crosskvalidation(k,X_train,y_train):
#     guideArray=[i for i in range(0,X_train.shape[0],X_train.shape[0]//k)]+[X_train.shape[0]]
#     meanError=[]
#     for el1,el2 in zip(guideArray[:-1],guideArray[1:]):
#         print(el1,el2)
#         X_train_k=np.vstack((X_train[:el1],X_train[el2:]))
#         y_train_k=np.vstack((y_train[:el1],y_train[el2:]))
#         regr.fit(X_train_k,y_train_k)
#         y_pred=regr.predict(X_train[el1:el2])
#         print(mean_squared_error(y_train[el1:el2], y_pred))
#         meanError.append(mean_squared_error(y_train[el1:el2], y_pred))
#     return meanError

#print(np.mean(crosskvalidation(5, X_train, y_train)))

        
#TODO: why is Lasso better, indica mesmo que há variáveis irrelevantes para o outcome. Penalização suave, apenas com 0.1
    
    