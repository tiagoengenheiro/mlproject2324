from sklearn.cluster import KMeans
import numpy as np
import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import cross_validate
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor,TheilSenRegressor,HuberRegressor
from utils import get_best_model

X_train=np.load("X_train_regression2.npy") #shape -> (100, 4)
y_train=np.load("y_train_regression2.npy") #shape -> (100, 1)
print(get_best_model(X_train,y_train))
n_examples,n_features=X_train.shape
reg = HuberRegressor().fit(X_train, y_train)

scores=(-1)*cross_validate(reg, X_train, y_train, cv=5, scoring=('r2', 'neg_mean_squared_error'))["test_neg_mean_squared_error"]
best_mean_score=np.mean(scores)
print(best_mean_score)
