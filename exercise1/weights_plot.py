# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate

x_train=np.load("X_train_regression1.npy") #shape -> (15,10)
y_train=np.load("y_train_regression1.npy") #shape -> (15,1)

#Centralized Data
x_train=x_train - np.mean(x_train,axis=0) #axis=0 -> por coluna

#Normalized Data
#x_train= (x_train - np.min(x_train,axis=0)) / (np.max(x_train,axis=0) - np.min(x_train,axis=0))

n_examples,n_features=x_train.shape
column_of_ones=np.ones((n_examples,1)) #shape (15,1)
X_train=np.hstack((column_of_ones,x_train)) #shape -> (15,11)

alphas = [0.01, 0.1, 1, 10, 100, 1000]
coefficients = []

#Lasso weights plot
for alpha in alphas:
    model=Lasso(alpha, fit_intercept = False)
    model.fit(X_train,y_train)
    print(model.coef_.shape)
    coefficients.append(model.coef_)
    
coefficients = np.array(coefficients) 

print(coefficients.shape)

plt.figure(figsize=(10, 6))  

for coef_index in range(coefficients.shape[1]):
    plt.plot(alphas, coefficients[:, coef_index], label=f'Coefficient {coef_index + 1}')

plt.xscale('log')  
plt.xlabel('Alpha (Log Scale)')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficient Paths')
plt.legend()
plt.grid(True)
# plt.show()


#Ridge weights plot
# for alpha in alphas:
#     model=Ridge(alpha)
#     model.fit(X_train,y_train)
#     print(model.coef_.shape)
#     coefficients.append(model.coef_)
    
# coefficients = np.array(coefficients) 

# plt.figure(figsize=(10, 6))  

# print(coefficients.shape)

# coefficients = coefficients.reshape(6, 11)

# print(coefficients.shape)

# for coef_index in range(coefficients.shape[1]):
#     plt.plot(alphas, coefficients[:, coef_index], label=f'Coefficient {coef_index + 1}')

# plt.xscale('log')
# plt.xlabel('Alpha (Log Scale)')
# plt.ylabel('Coefficient Value')
# plt.title('Ridge Coefficients')
# plt.legend()
# plt.grid(True)
# plt.show()



plt.savefig('lasso_coef3.png')
    
    