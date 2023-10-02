import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import cross_validate
import pandas as pd
import matplotlib.pyplot as plt
X_train=np.load("X_train_regression2.npy") #shape -> (100, 4)
y_train=np.load("y_train_regression2.npy") #shape -> (100, 1)
from sklearn.preprocessing import RobustScaler

X_train=RobustScaler().fit_transform(X_train)
y_train=RobustScaler().fit_transform(y_train)

regr = LinearRegression()
regr.fit(X_train,y_train)

#Running crossvalidation on the regular Linear Regression
scores=(-1)*cross_validate(regr, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_squared_error'))["test_neg_mean_squared_error"]
best_mean_score=np.mean(scores) #mean of the MSE for 5 folds


#Defining a cycle to get the best model among the models: Linear, Lasso and Ridge

#Alphas used for both Ridge and Lasso Regression models
alphas = [0.001,0.01, 0.1, 1, 10, 100, 1000]

best_model="linear"
best_alpha=0

print(best_alpha,best_model,best_mean_score)

for alpha in alphas:
    for model in ["ridge","lasso"]:
        if model=="ridge":
            model=Ridge(alpha)
        else:
            model=Lasso(alpha)
        model.fit(X_train,y_train)
        scores=(-1)*cross_validate(model, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_squared_error'))["test_neg_mean_squared_error"]
        score_mean=np.mean(scores)

        # print(best_alpha,best_model,best_mean_score)
        print(alpha,model,score_mean)


        if score_mean<best_mean_score:
            best_mean_score=score_mean
            best_model=model
            best_alpha=alpha

#Best model - the one with the least mean MSE from crossvalidation with k=5
print("the best model is %s regression with an alpha of %s and a mean score of %s" % (
    best_model,
    best_alpha,
    best_mean_score))

# Print the correlation matrix
#Pequena contribuição da 1ºa e 2a feat independente
#Pouca contribuição da terceira feat


