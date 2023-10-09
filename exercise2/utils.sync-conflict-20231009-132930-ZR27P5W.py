import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import cross_validate
import pandas as pd
import matplotlib.pyplot as plt

def get_best_model(X_train,y_train):
    model = LinearRegression()
    model.fit(X_train,y_train)
    #Running crossvalidation on the regular Linear Regression
    scores=(-1)*cross_validate(model, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_squared_error'))["test_neg_mean_squared_error"]
    best_mean_score=np.mean(scores) #mean of the MSE for 5 folds


    #Defining a cycle to get the best model among the models: Linear, Lasso and Ridge

    #Alphas used for both Ridge and Lasso Regression models
    alphas = [0.001,0.01, 0.1, 1, 10, 100, 1000]

    best_model=model
    best_alpha=0

    #print(best_alpha,best_model,best_mean_score)

    for alpha in alphas:
        for model_names in ["ridge","lasso"]:
            if model_names=="ridge":
                model=Ridge(alpha)
            else:
                model=Lasso(alpha)
            model.fit(X_train,y_train)
            scores=(-1)*cross_validate(model, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_squared_error'))["test_neg_mean_squared_error"]
            score_mean=np.mean(scores)

            # print(best_alpha,best_model,best_mean_score)
            #print(alpha,model,score_mean)


            if score_mean<best_mean_score:
                best_mean_score=score_mean
                best_model=model
                best_alpha=alpha

    #Best model - the one with the least mean MSE from crossvalidation with k=5
    # print("the best model is %s regression with an alpha of %s and a mean score of %s" % (
    #     best_model,
    #     best_alpha,
    #     best_mean_score))
    return best_model,best_mean_score

def fit_linear_models(X_train,y_train,X_train_1,y_train_1,X_train_2,y_train_2,final=False):
    regr1 = LinearRegression()
    regr2 = LinearRegression()
    regr1.fit(X_train_1,y_train_1)
    regr2.fit(X_train_2,y_train_2)
    residual1=np.abs(regr1.predict(X_train)-y_train)
    residual2=np.abs(regr2.predict(X_train)-y_train)
    res_diff=np.squeeze(residual1<residual2) #Should this example be in model1?
    y_train_1=y_train[res_diff]
    X_train_1=X_train[res_diff]
    X_train_2=X_train[~res_diff]
    y_train_2=y_train[~res_diff]
    if final:
        print("List of Model1 indexes",[i for i,x in enumerate(res_diff) if x==True])
        print("List of Model2 indexes",[i for i,x in enumerate(res_diff) if x==False])
    return X_train_1,y_train_1,X_train_2,y_train_2



