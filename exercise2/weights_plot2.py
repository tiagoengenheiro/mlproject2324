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

alphas = [0.01, 0.1, 1, 10, 100, 1000]

save_folder = 'images'
os.makedirs(save_folder, exist_ok=True)

for prepro in [""]:
    X_train=np.load("X_train_regression2.npy") #shape -> (15,10)
    y_train=np.load("y_train_regression2.npy") #shape -> (15,1)

    ridge_coefficients = []
    lasso_coefficients = []

    for model_id in ["ridge","lasso"]:
        for alpha in alphas:
            if model_id=="ridge":
                model=Ridge(alpha)
                model.fit(X_train,y_train)
                ridge_coefficients.append(np.squeeze(model.coef_))
            elif model_id == "lasso":
                model=Lasso(alpha)
                model.fit(X_train,y_train)
                lasso_coefficients.append(model.coef_)
                print(model.coef_.shape)


    ridge_coefficients = np.array(ridge_coefficients) 
    lasso_coefficients = np.array(lasso_coefficients)

    plt.figure(figsize=(10, 6))
    print(lasso_coefficients.shape)  
    for coef_index in range(lasso_coefficients.shape[1]):
        plt.plot(alphas, lasso_coefficients[:, coef_index], label=f'Coefficient {coef_index + 1}')

    plt.xscale('log')  
    plt.xlabel('Alpha (Log Scale)')
    plt.ylabel('Coefficient Value')
    plt.title('Lasso - Coefficient Variation')
    plt.legend()
    plt.grid(True)

    filename = f'lasso_coef_{prepro}.png'
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path)

    plt.figure(figsize=(10, 6))  
    
    for coef_index in range(ridge_coefficients.shape[1]):
        plt.plot(alphas, ridge_coefficients[:, coef_index], label=f'Coefficient {coef_index + 1}')

    plt.xscale('log')
    plt.xlabel('Alpha (Log Scale)')
    plt.ylabel('Coefficient Value')
    plt.title('Ridge Coefficients')
    plt.legend()
    plt.grid(True)

    filename = f'ridge_coef_{prepro}.png'
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path)


    
    