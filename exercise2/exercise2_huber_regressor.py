import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge,HuberRegressor
from sklearn.model_selection import cross_validate
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_best_model,fit_linear_models
from sklearn.preprocessing import RobustScaler,StandardScaler
X_train=np.load("X_train_regression2.npy") #shape -> (100, 4)
y_train=np.load("y_train_regression2.npy") #shape -> (100, 1)



n_examples,n_features=X_train.shape
min_mse=(np.inf,np.inf)
model1,model2=None,None
best_epsilon=None
best_alpha=0
for epsilon in np.arange(1.05,1.2,0.01):
    print(epsilon)
    for alpha in [10.0**(exp) for exp in np.arange(-5,3,1)]:
        reg = HuberRegressor(epsilon=epsilon,alpha=alpha,tol=1e-5)
        reg.fit(X_train,y_train.ravel())
        model1,mse1=get_best_model(X_train[~reg.outliers_.ravel()],y_train[~reg.outliers_.ravel()])
        model2,mse2=get_best_model(X_train[reg.outliers_.ravel()],y_train[reg.outliers_.ravel()])
        results=np.hstack((model1.predict(X_train).reshape(n_examples,1),model2.predict(X_train).reshape(n_examples,1)))
        results = np.square(results-y_train)
        results = np.min(results,axis=1)
        results = np.sum(results,axis=0)
        print("SSE:",results, (mse1+mse2)/2)
        if (mse1+mse2) < sum(min_mse):
            min_mse=(mse1,mse2)
            best_epsilon=epsilon
            best_alpha=alpha

print(best_alpha)
print(best_epsilon)
print(model1,min_mse[0])
print(model2,min_mse[1])
print(np.mean([min_mse[0],min_mse[1]]))

reg = HuberRegressor(epsilon=best_epsilon,alpha=best_alpha,tol=1e-10)
reg.fit(X_train,y_train.ravel())
print("Number of inliers",len(y_train[~reg.outliers_.ravel()]))
model1,mse1=get_best_model(X_train[~reg.outliers_.ravel()],y_train[~reg.outliers_.ravel()])
model2,mse2=get_best_model(X_train[reg.outliers_.ravel()],y_train[reg.outliers_.ravel()])
results=np.hstack((model1.predict(X_train).reshape(n_examples,1),model2.predict(X_train).reshape(n_examples,1)))
results = np.square(results-y_train)
results = np.min(results,axis=1)
results = np.sum(results,axis=0)
print("SSE:",results)
