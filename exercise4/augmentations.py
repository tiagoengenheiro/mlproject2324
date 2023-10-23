import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import cross_validate
import pandas as pd
import matplotlib.pyplot as plt
import colorsys
import random


def self_augmentation(X_array,y_array,func=np.copy):
    class_proportions=np.array(np.bincount(np.array(y_array,dtype=np.int64)),dtype=np.float32)/X_array.shape[0]
    #print(class_proportions)
    class_proportions=np.round(np.max(class_proportions)/class_proportions)-1
    #print("number of agumentation",class_proportions)
    X_augmentations=[]
    y_augmentations=[]
    for i,n_aug in enumerate(class_proportions): #class,number of augmentations
        if n_aug!=0:
            X_label=X_array[y_array==i]
            #print(f"Class {i}",X_label.shape[0],n_aug)
            for _ in np.arange(n_aug): #percorrer por augmentation ou percorrer por labels
                X_label_augmented=np.apply_along_axis(func,1,X_label)
                X_augmentations.append(X_label_augmented)
            y_augmentations=np.concatenate((y_augmentations,i*np.ones(X_label.shape[0]*np.int32(n_aug))))
    X_augmentations=np.concatenate(X_augmentations,axis=0)
    X_augmented=np.concatenate((X_array,X_augmentations))
    y_augmented=np.concatenate((y_array,y_augmentations))
    return X_augmented,y_augmented