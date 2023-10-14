import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import cross_validate
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_best_model,fit_linear_models
from sklearn.preprocessing import RobustScaler,StandardScaler
X_train=np.load("X_train_regression2.npy") #shape -> (100, 4)
y_train=np.load("y_train_regression2.npy") #shape -> (100, 1)

df=pd.DataFrame(np.hstack((X_train,y_train)))
df.columns=[f"Feature{i}" for i in range(1,5)]+["Output"]
print(df.corr())