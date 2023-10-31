# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:37:29 2023

@authors: Tiago e João
"""

import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge

from sklearn.model_selection import cross_validate

X_train=np.load("X_train_regression1.npy") #shape -> (15,10)
y_train=np.load("y_train_regression1.npy") #shape -> (15,1)

# Rescaling (min-max normalization)
# X_train= (X_train - np.min(X_train,axis=0)) / (np.max(X_train,axis=0) - np.min(X_train,axis=0))
# y_train= (y_train - np.min(y_train,axis=0)) / (np.max(y_train,axis=0) - np.min(y_train,axis=0))

#Since Sklearn models already center the data by default there is no need to add the bias column to X

# Closed form
# Beta=np.dot(np.dot(np.linalg.inv(np.dot(X_train.T,X_train)),X_train.T),y_train) # (11,1)
# y_pred=np.dot(X_train,Beta)

#Linear predictor
regr = LinearRegression()
regr.fit(X_train,y_train)

#Running crossvalidation on the regular Linear Regression
scores=(-1)*cross_validate(regr, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_squared_error'))["test_neg_mean_squared_error"]
best_mean_score=np.mean(scores) #mean of the MSE for 5 folds

#Defining a cycle to get the best model among the models: Linear, Lasso and Ridge

#Alphas used for both Ridge and Lasso Regression models
alphas = [0.01, 0.1, 1, 10, 100, 1000]

best_model="linear"
best_alpha=0

print(best_alpha,best_model,round(best_mean_score,3),round(np.std(scores),3))
models={
    "lasso":[],
    "ridge":[]
}
for alpha in alphas:
    for model_name in ["ridge","lasso"]:
        if model_name=="ridge":
            model=Ridge(alpha)
        else:
            model=Lasso(alpha)
        model.fit(X_train,y_train)
        scores=(-1)*cross_validate(model, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_squared_error'))["test_neg_mean_squared_error"]
        score_mean=np.mean(scores)
        models[model_name].append(round(score_mean,3))
        models[model_name].append(round(np.std(scores),3))
        # print(best_alpha,best_model,best_mean_score)
        print(alpha,model,score_mean,np.std(scores))

        if score_mean<best_mean_score:
            best_mean_score=score_mean
            best_model=model
            best_alpha=alpha
    
#Best model - the one with the least mean MSE from crossvalidation with k=5
print("the best model is %s regression with an alpha of %s and a mean score of %s" % (
    best_model,
    best_alpha,
    best_mean_score))

## Best model is Lasso with alpha=0.1
print(models)
#Fitting the data with the best model
model=Lasso(alpha=0.1) 
model.fit(X_train,y_train)

X_test=np.load("X_test_regression1.npy") #shape -> (15,10)
n_examples,_=X_test.shape
y_test=model.predict(X_test).reshape(n_examples,1) #->(1000,1)
np.save("y_test_regression1.npy",y_test)
y_test=np.load("y_test_regression1.npy")
print(y_test.shape)
    
    