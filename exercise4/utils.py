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

def apply_hue(image,hue_factor):
    image_t=(image).astype(np.float32)
    image_t=image_t/255.0
    # Convert RGB to HSV
    hsv_image = np.apply_along_axis(rgb_to_hsv,2,image_t)

    #Modify the hue component by adding the hue shift in degrees
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + (hue_factor*100 / 360)) % 1.0 #because it's degrees it goes around
    #Convert HSV back to RGB
    rgb_image = np.apply_along_axis(hsv_to_rgb,2,hsv_image)
    rgb_image = (rgb_image * 255).astype(np.uint8)
    return rgb_image

def apply_brightness(image,brightness_factor):
    return np.clip(image*(1+brightness_factor), 0, 255).astype(np.uint8)

def apply_contrast(image,constrast_factor):
    adjusted_image_array = (image - 127.5) * (1+constrast_factor) + 127.5
    return np.clip(adjusted_image_array, 0, 255).astype(np.uint8)

def pre_processing_contrast(X_array,contrast_factor):
    X_array_reshaped=X_array.reshape(X_array.shape[0],28,28,3)
    #np.apply_over_axis()
    X_preprocessed=np.array([apply_contrast(image,contrast_factor) for image in X_array_reshaped])
    X_preprocessed=X_preprocessed.reshape(X_array.shape[0],-1)
    return X_preprocessed

        

