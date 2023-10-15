import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageChops
import colorsys

X=np.load("Xtrain_Classification1.npy")
y=np.load("ytrain_Classification1.npy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) #0.8 - 0.2 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2


X_train=X_train.reshape(X_train.shape[0],28,28,3)
print(X_train[0][0][0])

print(X_train.shape)
X_train_neg=X_train[y_train==0] #3218
X_train_pos=X_train[y_train==1] #531

def rgb_to_hsv(v):
    return np.array(colorsys.rgb_to_hsv(v[0], v[1], v[2]))
def hsv_to_rgb(v):
    return np.array(colorsys.hsv_to_rgb(v[0], v[1], v[2]))

def apply_saturation (image,saturation_factor):
    image=(image).astype(np.float32)
    image=image/255.0
    hsv_image = np.apply_along_axis(rgb_to_hsv,2,image) 
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * (1+saturation_factor), 0, 1)
    rgb_image = np.apply_along_axis(hsv_to_rgb,2,hsv_image)
    rgb_image = (rgb_image * 255.0).astype(np.uint8)
    return rgb_image

def apply_hue(image,saturation_factor):
    image=(image).astype(np.float32)
    image=image/255.0
    # Convert RGB to HSV
    hsv_image = np.apply_along_axis(rgb_to_hsv,2,image)

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

image=X_train_pos[0]
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

std=0.15
th=0.3
brightness_factor = np.clip(std*np.random.randn(6),-th,th)
for idx,ax in enumerate(axs.flat):
    ax.imshow(apply_brightness(image,brightness_factor[idx]))
    ax.set_title(f"{round(brightness_factor[idx]*100)}%")
plt.tight_layout()
plt.savefig("images/Brightness")






#Rotation an Invertion
# X_train_pos_extra=[]
# for image in X_train_pos:
#     #Rotation to 90 and 180
#     for k in [1,2,3]:
#         X_train_pos_extra.append(np.rot90(image,k=k))
#     for axis in [0,1]:
#         X_train_pos_extra.append(np.flip(image,axis=axis))
# #print(np.array(X_train_pos_extra).shape)
# X_train_pos=np.concatenate((X_train_pos,X_train_pos_extra),axis=0)
# #print(X_train_pos.shape)
# X_train=np.concatenate((X_train_pos,X_train_neg),axis=0)
# #print(X_train.shape)
# y_train=np.concatenate((np.ones(X_train_pos.shape[0]),np.zeros(X_train_neg.shape[0])))
# #print(y_train.shape)
#




# Shifting
# 1
# image=X_train_pos[0]
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 6),
#                         subplot_kw={'xticks': [], 'yticks': []})

# for idx,_ in enumerate(axs.flat):
#     print(idx)
#     if idx<3:
#         axis=0
#     else:
#         axis=1
#     for p,s in enumerate([0,3,-3]):
#         axs[axis,p].imshow(np.roll(image,shift=s,axis=axis))
#         axs[axis,p].set_title(f"Shifted {s} positions in axis {axis}")
# plt.tight_layout()
# plt.show()
