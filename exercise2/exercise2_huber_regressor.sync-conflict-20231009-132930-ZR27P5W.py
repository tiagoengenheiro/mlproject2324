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
best_n_samples=0
th=0
for n_samples in np.arange(1.35,1.8,0.05):
    reg = HuberRegressor(epsilon= n_samples,tol=1e-30)
    reg.fit(X_train,y_train.ravel())
    #print(reg.outliers_)
# print("Number of inliers",len(reg.inlier_mask_[reg.inlier_mask_==True]))
    model1,mse1=get_best_model(X_train[~reg.outliers_.ravel()],y_train[~reg.outliers_.ravel()])
    model2,mse2=get_best_model(X_train[reg.outliers_.ravel()],y_train[reg.outliers_.ravel()])
    print(n_samples,"MSE1",mse1,"MSE2",mse2)
    if (mse1+mse2) < sum(min_mse):
        min_mse=(mse1,mse2)
        best_n_samples=n_samples
print(best_n_samples)
print(model1,min_mse[0])
print(model2,min_mse[1])

reg = HuberRegressor(epsilon=n_samples)
reg.fit(X_train,y_train.ravel())
# print("Number of inliers",len(reg.inlier_mask_[reg.inlier_mask_==True]))
model1,mse1=get_best_model(X_train[~reg.outliers_.ravel()],y_train[~reg.outliers_.ravel()])
model2,mse2=get_best_model(X_train[reg.outliers_.ravel()],y_train[reg.outliers_.ravel()])
#results=np.hstack((RS.inverse_transform(model1.predict(X_train)),RS.inverse_transform(model2.predict(X_train))))
results=np.hstack((model1.predict(X_train),model2.predict(X_train).reshape(n_examples,1)))
results = np.square(results-y_train)
results = np.min(results,axis=1)
results = np.sum(results,axis=0)
print("SSE:",results)
