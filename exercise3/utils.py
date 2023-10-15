import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import cross_validate
import pandas as pd
import matplotlib.pyplot as plt
import colorsys


def self_augmentation_rotate_flip(X_train,y_train):
    X_train=X_train.reshape(X_train.shape[0],28,28,3)
    X_train_neg=X_train[y_train==0] 
    X_train_pos=X_train[y_train==1] 

    X_train_pos_extra=[]
    for image in X_train_pos:
        #Rotation to 90 and 180
        for k in [1,2,2]:
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


def rgb_to_hsv(v):
    return np.array(colorsys.rgb_to_hsv(v[0], v[1], v[2]))
def hsv_to_rgb(v):
    return np.array(colorsys.hsv_to_rgb(v[0], v[1], v[2]))

def apply_saturation (image,saturation_factor):
    image_t=(image).astype(np.float32)
    image_t=image_t/255.0
    # Convert RGB to HSV
    hsv_image = np.apply_along_axis(rgb_to_hsv,2,image_t)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * (1+saturation_factor), 0, 1)
    rgb_image = np.apply_along_axis(hsv_to_rgb,2,hsv_image)
    rgb_image = (rgb_image * 255.0).astype(np.uint8)
    return rgb_image

def apply_hue(image,saturation_factor):
    image_t=(image).astype(np.float32)
    image_t=image_t/255.0
    # Convert RGB to HSV
    hsv_image = np.apply_along_axis(rgb_to_hsv,2,image_t)

    #Modify the hue component by adding the hue shift in degrees
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + (saturation_factor*100 / 360)) % 1.0 #because it's degrees it goes around
    #Convert HSV back to RGB
    rgb_image = np.apply_along_axis(hsv_to_rgb,2,hsv_image)
    rgb_image = (rgb_image * 255).astype(np.uint8)
    return rgb_image

def apply_brightness(image,brightness_factor):
    return np.clip(image*(1+brightness_factor), 0, 255).astype(np.uint8)

def apply_contrast(image,constrast_factor):
    adjusted_image_array = (image - 127.5) * (1+constrast_factor) + 127.5
    return np.clip(adjusted_image_array, 0, 255).astype(np.uint8)

def self_augmentation_combined(X_train,y_train,std,th):
    print("Using Combined")
    X_train=X_train.reshape(X_train.shape[0],28,28,3)
    #Separate X_train based on class
    X_train_neg=X_train[y_train==0] 
    X_train_pos=X_train[y_train==1] 
    print("Total shape",X_train.shape)
    print("Neg shape",X_train_neg.shape)
    print("Pos shape",X_train_pos.shape)
    #Augment the positive class
    X_train_pos_extra=[]
    for image in X_train_pos:
        #Rotation to 90 and 180
        values_for_k=np.clip(std*np.random.randn(5),-th,th)
        X_train_pos_extra.append(apply_hue(image,np.random.choice(values_for_k)))
        X_train_pos_extra.append(apply_hue(image,np.random.choice(values_for_k)))
        X_train_pos_extra.append(apply_hue(image,np.random.choice(values_for_k)))
        X_train_pos_extra.append(apply_hue(image,np.random.choice(values_for_k)))
        X_train_pos_extra.append(apply_hue(image,np.random.choice(values_for_k)))
        
    
    X_train_pos=np.concatenate((X_train_pos,X_train_pos_extra),axis=0)
    print("Final X_train pos",X_train_pos.shape)
    #Add X_train pos with X_train neg
    X_train=np.concatenate((X_train_pos,X_train_neg),axis=0)
    print("Final X_train shape",X_train.shape)
    #Add the classes
    y_train=np.concatenate((np.ones(X_train_pos.shape[0]),np.zeros(X_train_neg.shape[0])))
    return X_train,y_train

def self_augmentation_merged(X_train,y_train,std,th):
    print("Using Merged")
    X_train=X_train.reshape(X_train.shape[0],28,28,3)
    #Separate X_train based on class
    X_train_neg=X_train[y_train==0] 
    X_train_pos=X_train[y_train==1] 

    #Augment the positive class
    X_train_pos_extra=[]
    for image in X_train_pos:
        #Rotation to 90 and 180
        values_for_k=np.clip(std*np.random.randn(5),-th,th)
        # for i in range(5):
        #     image_transformed=apply_hue(image,values_for_k[i])
        #     image_transformed=apply_saturation(image,values_for_k[i])
        #     image_transformed=apply_brightness(image,values_for_k[i])
        #     image_transformed=np.rot90(image,k=np.random.choice([1,2,3]))
        #     image_transformed=np.flip(image,axis=np.random.choice([0,1]))
        #     X_train_pos_extra.append(image_transformed)
        for _ in range(2):
            image_transformed=apply_hue(image,np.random.choice(values_for_k))
            image_transformed=np.rot90(image,k=np.random.choice([1,2]))
            X_train_pos_extra.append(image_transformed)
        for _ in range(2):
            image_transformed=apply_brightness(image,np.random.choice(values_for_k))
            image_transformed=np.flip(image,axis=np.random.choice([0,1]))
            X_train_pos_extra.append(image_transformed)
        image_transformed=apply_saturation(image,np.random.choice(values_for_k))
        image_transformed=np.rot90(image,k=3)
        X_train_pos_extra.append(image_transformed)
    X_train_pos=np.concatenate((X_train_pos,X_train_pos_extra),axis=0)
    print("Final X_train pos",X_train_pos.shape)
    #Add X_train pos with X_train neg
    X_train=np.concatenate((X_train_pos,X_train_neg),axis=0)
    print("Final X_train shape",X_train.shape)
    #Add the classes
    y_train=np.concatenate((np.ones(X_train_pos.shape[0]),np.zeros(X_train_neg.shape[0])))
    return X_train,y_train
def self_augmentation_hue(X_train,y_train,std,th):
    X_train=X_train.reshape(X_train.shape[0],28,28,3)
    print("Using Hue")
    #Separate X_train based on class
    X_train_neg=X_train[y_train==0] 
    X_train_pos=X_train[y_train==1] 

    #Augment the positive class
    X_train_pos_extra=[]
    for image in X_train_pos:
        #Rotation to 90 and 180
        for k in np.clip(std*np.random.randn(5),-th,th):
            X_train_pos_extra.append(apply_hue(image,k))
    #print(np.array(X_train_pos_extra).shape)
    X_train_pos=np.concatenate((X_train_pos,X_train_pos_extra),axis=0)
    #Add X_train pos with X_train neg
    X_train=np.concatenate((X_train_pos,X_train_neg),axis=0)
    #Add the classes
    y_train=np.concatenate((np.ones(X_train_pos.shape[0]),np.zeros(X_train_neg.shape[0])))
    return X_train,y_train

def self_augmentation_contrast(X_train,y_train,std,th):
    X_train=X_train.reshape(X_train.shape[0],28,28,3)
    print("Using Contrast")
    #Separate X_train based on class
    X_train_neg=X_train[y_train==0] 
    X_train_pos=X_train[y_train==1] 

    #Augment the positive class
    X_train_pos_extra=[]
    for image in X_train_pos:
        #Rotation to 90 and 180
        for k in np.clip(std*np.random.randn(5),-th,th):
            X_train_pos_extra.append(apply_contrast(image,k))
    #print(np.array(X_train_pos_extra).shape)
    X_train_pos=np.concatenate((X_train_pos,X_train_pos_extra),axis=0)
    #Add X_train pos with X_train neg
    X_train=np.concatenate((X_train_pos,X_train_neg),axis=0)
    #Add the classes
    y_train=np.concatenate((np.ones(X_train_pos.shape[0]),np.zeros(X_train_neg.shape[0])))
    return X_train,y_train

def self_augmentation_brightness(X_train,y_train,std,th):
    X_train=X_train.reshape(X_train.shape[0],28,28,3)
    print("Using Brightness")
    #Separate X_train based on class
    X_train_neg=X_train[y_train==0] 
    X_train_pos=X_train[y_train==1] 

    #Augment the positive class
    X_train_pos_extra=[]
    for image in X_train_pos:
        #Rotation to 90 and 180
        for k in np.clip(std*np.random.randn(5),-th,th):
            X_train_pos_extra.append(apply_brightness(image,k))
    #print(np.array(X_train_pos_extra).shape)
    X_train_pos=np.concatenate((X_train_pos,X_train_pos_extra),axis=0)
    #Add X_train pos with X_train neg
    X_train=np.concatenate((X_train_pos,X_train_neg),axis=0)
    #Add the classes
    y_train=np.concatenate((np.ones(X_train_pos.shape[0]),np.zeros(X_train_neg.shape[0])))
    return X_train,y_train

def self_augmentation_saturation(X_train,y_train,std,th):
    X_train=X_train.reshape(X_train.shape[0],28,28,3)
    print("Using Saturation")
    #Separate X_train based on class
    X_train_neg=X_train[y_train==0] 
    X_train_pos=X_train[y_train==1] 

    #Augment the positive class
    X_train_pos_extra=[]
    for image in X_train_pos:
        #Rotation to 90 and 180
        for k in np.clip(std*np.random.randn(5),-th,th):
            X_train_pos_extra.append(apply_saturation(image,k))
    #print(np.array(X_train_pos_extra).shape)
    X_train_pos=np.concatenate((X_train_pos,X_train_pos_extra),axis=0)
    #Add X_train pos with X_train neg
    X_train=np.concatenate((X_train_pos,X_train_neg),axis=0)
    #Add the classes
    y_train=np.concatenate((np.ones(X_train_pos.shape[0]),np.zeros(X_train_neg.shape[0])))
    return X_train,y_train