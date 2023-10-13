import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import cross_validate
import pandas as pd
import matplotlib.pyplot as plt


def self_augmentation_rotate_flip(X_train,y_train):
    X_train=X_train.reshape(X_train.shape[0],28,28,3)
    X_train_neg=X_train[y_train==0] 
    X_train_pos=X_train[y_train==1] 

    X_train_pos_extra=[]
    for image in X_train_pos:
        #Rotation to 90 and 180
        for k in [1,2,3]:
            X_train_pos_extra.append(np.rot90(image,k=k))
        for axis in [0,1]:
            X_train_pos_extra.append(np.flip(image,axis=axis))
    #print(np.array(X_train_pos_extra).shape)
    X_train_pos=np.concatenate((X_train_pos,X_train_pos_extra),axis=0)
    #print(X_train_pos.shape)
    X_train=np.concatenate((X_train_pos,X_train_neg),axis=0)
    #print(X_train.shape)
    y_train=np.concatenate((np.ones(X_train_pos.shape[0]),np.zeros(X_train_neg.shape[0])))
    return X_train,y_train


def self_augmentation_shift(X_train,y_train,shift_n=2):
    X_train=X_train.reshape(X_train.shape[0],28,28,3)
    X_train_neg=X_train[y_train==0] 
    X_train_pos=X_train[y_train==1] 

    X_train_pos_extra=[]
    for image in X_train_pos:
        #Rotation to 90 and 180
        for axis in [0,1]:
            for shift in [-shift_n,shift_n]:
                X_train_pos_extra.append(np.roll(image,shift=shift,axis=axis))
        X_train_pos_extra.append(np.flip(image,axis=0))
    #print(np.array(X_train_pos_extra).shape)
    X_train_pos=np.concatenate((X_train_pos,X_train_pos_extra),axis=0)
    #print(X_train_pos.shape)
    X_train=np.concatenate((X_train_pos,X_train_neg),axis=0)
    #print(X_train.shape)
    y_train=np.concatenate((np.ones(X_train_pos.shape[0]),np.zeros(X_train_neg.shape[0])))
    print(X_train.shape,y_train.shape)
    return X_train,y_train

def self_augmentation(X_train,y_train):
    X_train=X_train.reshape(X_train.shape[0],28,28,3)
    X_train_neg=X_train[y_train==0] 
    X_train_pos=X_train[y_train==1] 

    X_train_pos_extra=[]
    for image in X_train_pos:
        #Rotation to 90 and 180
        for k in [0,0,0,0,0]:
            X_train_pos_extra.append(np.rot90(image,k=k))
    X_train_pos=np.concatenate((X_train_pos,X_train_pos_extra),axis=0)
    #print(X_train_pos.shape)
    X_train=np.concatenate((X_train_pos,X_train_neg),axis=0)
    #print(X_train.shape)
    y_train=np.concatenate((np.ones(X_train_pos.shape[0]),np.zeros(X_train_neg.shape[0])))
    return X_train,y_train