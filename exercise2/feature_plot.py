# -*- coding: utf-8 -*-
"""

@authors: Tiago e João

"""
import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
import os

X_train=np.load("X_train_regression2.npy") #shape -> (100, 4)
y_train=np.load("y_train_regression2.npy") #shape -> (100, 1)

save_folder = 'images'
os.makedirs(save_folder, exist_ok=True)

#x_values = [1, 2, 3, 4]
x_values = [1]
x_values = np.array(x_values)

print(X_train)

X_train = X_train.T
y_train = y_train.T


print(x_values.shape)
print(X_train.shape)

plt.figure(figsize=(10, 10))

# for col in range(X_train.shape[1]):
#     unique_x_values = np.unique(x_values)
#     y_values = X_train[:, col]
#     color = plt.cm.viridis(col / X_train.shape[1])
#     plt.scatter(unique_x_values, y_values) 


# plt.xlabel('Feature')
# plt.ylabel('Feature Value')
# plt.legend()

for col in range(y_train.shape[1]):
    unique_x_values = np.unique(x_values)
    y_values = y_train[:, col]
    color = plt.cm.viridis(col / y_train.shape[1])
    plt.scatter(unique_x_values, y_values) 


plt.xlabel('Feature')
plt.ylabel('Feature Value')
plt.legend()

plt.show()


    
    