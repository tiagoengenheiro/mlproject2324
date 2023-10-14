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

    if prepro == "_zscore":
        X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
        # y_train = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
    elif prepro == "_norm":
        X_train= (X_train - np.min(X_train,axis=0)) / (np.max(X_train,axis=0) - np.min(X_train,axis=0))
        # y_train= (y_train - np.min(y_train,axis=0)) / (np.max(y_train,axis=0) - np.min(y_train,axis=0))

    ridge_coefficients = []
    lasso_coefficients = []

    for model_id in ["ridge","lasso"]:
        for alpha in alphas:
            if model_id=="ridge":
                model=Ridge(alpha)
                model.fit(X_train,y_train)
                ridge_coefficients.append(model.coef_)
            elif model_id == "lasso":
                model=Lasso(alpha)
                model.fit(X_train,y_train)
                lasso_coefficients.append(model.coef_)

    ridge_coefficients = np.array(ridge_coefficients) 
    lasso_coefficients = np.array(lasso_coefficients)

    plt.figure(figsize=(10, 6))  

    for coef_index in range(lasso_coefficients.shape[1]):
        plt.plot(alphas, lasso_coefficients[:, coef_index], label=f'Feature {coef_index + 1}')

    plt.xscale('log')  
    plt.xlabel('Alpha (Log10 Scale)', fontsize=13)
    plt.ylabel('Coefficient Value',fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True)

    filename = f'lasso_coef_{prepro}.png'
    save_path = os.path.join(save_folder, filename)
    plt.tight_layout()
    plt.savefig(save_path)

    plt.figure(figsize=(10, 6))  
    

    for coef_index in range(ridge_coefficients.shape[1]):
        plt.plot(alphas, ridge_coefficients[:, coef_index], label=f'Feature {coef_index + 1}')

    plt.xscale('log')
    plt.xlabel('Alpha (Log10 Scale)', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True)

    filename = f'ridge_coef_{prepro}.png'
    save_path = os.path.join(save_folder, filename)
    plt.tight_layout()
    plt.savefig(save_path)


    
    