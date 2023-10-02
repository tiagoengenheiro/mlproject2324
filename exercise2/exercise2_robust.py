import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import cross_validate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.preprocessing import RobustScaler
X_train_init=np.load("X_train_regression2.npy") #shape -> (100, 4)
y_train_init=np.load("y_train_regression2.npy") #shape -> (100, 1)

RS = RobustScaler()
X_train = RS.fit_transform(X_train_init)
y_train = RS.fit_transform(y_train_init)

# X_train = X_train_init
# y_train = y_train_init
data=np.hstack((X_train,y_train))
df=pd.DataFrame(data)
print(df.describe())
df.hist()
pyplot.show()